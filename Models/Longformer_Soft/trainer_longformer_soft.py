import datetime
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import LongformerForSequenceClassification, AdamW

from Models.Base.trainer_base import Trainer_Base
from Models.Longformer_Classify.dataset_for_long import Dataset_Long, Collect_FN
from Models.Longformer_Classify.eval_model import Eval_Model_For_Long
from Models.Longformer_Classify.predict_unlabel import Predicter
from compent.checkpoint import CheckPointer_Normal
from compent.comm import is_main_process, synchronize, get_rank
from compent.metric_logger import MetricLogger
from compent.utils import move_to_device, reduce_loss_dict

device = torch.device('cuda')

class Trainer_Longformer_Soft(Trainer_Base):
    def __init__(self, cfg, logger, distributed, sentences_all):
        super(Trainer_Longformer_Soft, self).__init__(cfg = cfg, logger = logger, distributed = distributed)
        self.checkpointer = CheckPointer_Normal(cfg = cfg, logger = logger, rank = get_rank())
        self.model = None
        dataloader_eval = self.__build_dataloader__(sentences_all.val_sentence, sentences_all.val_GT_label,
                                                    for_train = False, soft_labels = None)
        self.evaler = Eval_Model_For_Long(self.cfg, self.logger, distributed = True, rank = self.rank,
                                          dataloader_eval = dataloader_eval)

    def __build_dataloader__(self, sentences, hard_label, soft_labels, for_train):
        collect_fn = Collect_FN(hard_label is not None, soft_labels is not None)
        dataset = Dataset_Long(sentences, labels_hard = hard_label, labels_soft = soft_labels)
        sampler = DistributedSampler(dataset, shuffle = for_train)
        dataloader = DataLoader(dataset, batch_size = self.cfg.classifier.batch_size, sampler = sampler,
                                collate_fn = collect_fn)

        return dataloader

    def train_model(self, sentences, hard_label, soft_labels, ITR, finetune_from_pretrain = True):
        self.logger.info('start longformer training')

        sentences, hard_label, soft_labels = self.upsample_balance_with_one_extra(sentences, hard_label, soft_labels)
        sample_number_per_class = self.get_classes_count(hard_label)
        self.logger.info('sample_number_per_class:{}'.format(sample_number_per_class))

        dataloader_train = self.__build_dataloader__(sentences, hard_label, soft_labels, for_train = True)

        if (finetune_from_pretrain):
            self.logger.info('finetune from pretrain, load pretrain model')
            self.model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                                             num_labels = self.cfg.model.number_classes,
                                                                             gradient_checkpointing = True)
            self.model.train()
            self.model = self.model.to(device)
            if (self.distributed):
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.model = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
                )
        else:
            self.logger.info('finetune from last model parameters')

        self.logger.info('rank:{},build eval dataset...'.format(self.rank))

        acc = self.__do_train__(dataloader = dataloader_train, ITR = ITR)
        self.logger.info('acc:{}'.format(acc))

    def __do_train__(self, dataloader, ITR):
        self.logger.info('start training')
        self.model.train()

        meters = MetricLogger(delimiter = "  ")
        end = time.time()

        optimizer = AdamW(self.model.parameters(), lr = 1e-5)
        log_soft_op = torch.nn.LogSoftmax(dim = 1)
        loss_func = torch.nn.KLDivLoss(reduction = 'batchmean', log_target = False)

        best_res = 0

        total_epoch = 1
        total_itr = 0
        for epoch in range(total_epoch):
            self.logger.info('total epoch:{}, cur epoch:{}'.format(total_epoch, epoch))
            if (self.distributed):
                dataloader.sampler.set_epoch(epoch)
            for iteration, batch in enumerate(dataloader):
                total_itr += 1
                data_time = time.time() - end
                batch = move_to_device(batch)
                optimizer.zero_grad()
                # input_ids: [16,128]   label_id:[16]
                output = self.model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'])
                logits = output.logits
                logits = log_soft_op(logits)
                loss = loss_func(logits, batch['labels_soft'])

                loss_dict_reduced = reduce_loss_dict({'loss_all': loss})
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                meters.update(loss = losses_reduced, **loss_dict_reduced)
                loss.backward()
                optimizer.step()
                batch_time = time.time() - end
                end = time.time()
                meters.update(time = batch_time, data = data_time)

                eta_seconds = meters.time.global_avg * (self.cfg.classifier.total_steps - iteration)
                eta_string = str(datetime.timedelta(seconds = int(eta_seconds)))

                if iteration % 10 == 0:
                    memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                    self.logger.info(
                            meters.delimiter.join(
                                    [
                                        "eta: {eta}",
                                        "iter: {iter}",
                                        "total_itr: {total_itr}",
                                        "{meters}",
                                        "lr: {lr:.6f}",
                                        "max mem: {memory:.0f}",
                                    ]
                            ).format(
                                    eta = eta_string,
                                    iter = iteration,
                                    total_itr = len(dataloader),
                                    meters = str(meters),
                                    lr = optimizer.param_groups[0]["lr"],
                                    memory = memory,
                            ),
                            show_one = True
                    )
                    self.logger.plot_record(value = memory, win_name = 'memory')
                    self.logger.plot_record(value = meters.loss.median, win_name = 'classifier loss,itr:{}'.format(ITR),
                                            X_value = total_itr)
                if (iteration + 0) % 100 == 0:
                    self.logger.info('start eval...'.format(self.rank))
                    synchronize()
                    res_dict = self.evaler(self.model)
                    if (is_main_process()):
                        f11 = res_dict['f1_micro']
                        self.logger.plot_record(f11, win_name = 'classifier eval f1,itr:{}'.format(ITR),
                                                X_value = total_itr)
                        if (f11 > best_res):
                            best_res = f11
                            self.checkpointer.save_to_checkpoint_file_with_name(model = self.model,
                                                                                filename = 'longformer',
                                                                                other_info = res_dict)
                        self.logger.info('best f1_micro:{}'.format(best_res))
                        self.logger.visdom_text(text = 'best f1_micro:{}'.format(best_res), win_name = 'best_f1')
                    self.logger.info('eval over')
                    synchronize()

            synchronize()
            res_dict = self.evaler(self.model)
            if (is_main_process()):
                f11 = res_dict['f1_micro']
                self.logger.plot_record(f11, win_name = 'classifier eval f1,itr:{}'.format(ITR), X_value = total_itr)
                if (f11 > best_res):
                    best_res = f11
                    self.checkpointer.save_to_checkpoint_file_with_name(model = self.model, filename = 'longformer',
                                                                        other_info = res_dict)
                self.logger.info('best f1_micro:{}'.format(best_res))
                self.logger.visdom_text(text = 'best f1_micro:{}'.format(best_res), win_name = 'best_f1')
            synchronize()
        self.logger.plot_record(value = best_res, win_name = 'itr classifier best f1')
        return best_res

    def do_label_sentences(self, sentences):
        self.logger.info('start do_label_sentences'.format(self.rank))
        if (self.model is None):
            self.model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                                             num_labels = self.cfg.model.number_classes,
                                                                             gradient_checkpointing = True)
        self.checkpointer.load_from_filename(self.model, filename = 'longformer', strict = True)
        self.model = self.model.to(device)
        if (self.distributed):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
            )
        self.model.eval()
        self.logger.info('eval to check before do_label_sentences')
        res = self.evaler(self.model)
        self.logger.info('load from best model, model res:{}'.format(res))
        self.logger.info('do_label_sentences total unlabeled sentences:{}'.format(self.rank, len(sentences)))
        dataloader_sentence = self.__build_dataloader__(sentences, soft_labels = None, hard_label = None,
                                                        for_train = False)
        predicter = Predicter(cfg = self.cfg, logger = self.logger, distributed = True, rank = self.rank,
                              dataloader_sentence = dataloader_sentence, model = self.model)
        sentences_all, labels = predicter()
        self.model.train()
        return sentences_all, labels
