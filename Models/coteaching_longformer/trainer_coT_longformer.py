import datetime
import time

import numpy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import LongformerForSequenceClassification, AdamW

from Models.Base.trainer_base import Trainer_Base
from Models.Longformer_Classify.eval_model import Eval_Model_For_Long
from Models.Longformer_Classify.predict_unlabel import Predicter
from Models.coteaching_longformer.dataset_for_long import Dataset_Long, Collect_FN_CoT
from compent.checkpoint import CheckPointer_Normal
from compent.comm import is_main_process, synchronize, get_rank
from compent.metric_logger import MetricLogger
from compent.utils import move_to_device, reduce_loss_dict

device = torch.device('cuda')

batch_size = 3


class Trainer_Longformer_CoT(Trainer_Base):
    def __init__(self, cfg, logger, distributed, sentences_all):
        super(Trainer_Longformer_CoT, self).__init__(cfg = cfg, logger = logger, distributed = distributed)
        self.checkpointer = CheckPointer_Normal(cfg = cfg, logger = logger, rank = get_rank())
        dataloader_eval = self.__build_dataloader_eval_longformer__(sentences_all.val_sentence,
                                                                    sentences_all.val_GT_label,
                                                                    for_train = False)
        self.evaler = Eval_Model_For_Long(self.cfg, self.logger, distributed = True, rank = self.rank,
                                          dataloader_eval = dataloader_eval)

    def __build_dataloader_coteaching__(self, sentences, labels, GT_label, for_train):
        collect_fn = Collect_FN_CoT()
        dataset = Dataset_Long(sentences, labels, GT_label)
        sampler = DistributedSampler(dataset, shuffle = for_train)
        dataloader = DataLoader(dataset, batch_size = batch_size, sampler = sampler,
                                collate_fn = collect_fn)
        return dataloader

    def train_model(self, sentences, labels, GT_label = None, finetune_from_pretrain = True):
        self.logger.info('finetune distributed:{}'.format(self.distributed))

        sentences, labels, GT_label = self.upsample_balance_with_one_extra(sentences, labels, GT_label)
        # sample_number_per_class = self.get_classes_count(labels)
        # self.logger.info('sample_number_per_class:{}'.format(sample_number_per_class))

        dataloader_train = self.__build_dataloader_coteaching__(sentences, labels, GT_label, for_train = True)

        self.logger.info('finetune from pretrain, load pretrain model')
        self.model1 = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                                          num_labels = self.cfg.model.number_classes,
                                                                          gradient_checkpointing = True)
        self.model1.train()
        self.model1 = self.model1.to(device)
        if (self.distributed):
            model1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model1)
            self.model1 = torch.nn.parallel.DistributedDataParallel(
                    model1, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
            )

        self.model2 = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                                          num_labels = self.cfg.model.number_classes,
                                                                          gradient_checkpointing = True)
        self.model2.train()
        self.model2 = self.model2.to(device)
        if (self.distributed):
            model2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model2)
            self.model2 = torch.nn.parallel.DistributedDataParallel(
                    model2, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
            )

        self.logger.info('rank:{},build eval dataset...'.format(self.rank))

        acc = self.__do_train__(dataloader = dataloader_train)
        self.logger.info('acc:{}'.format(acc))

    def get_num_remember(self):
        tail_keep = 1
        ans = numpy.ones(9000, dtype = numpy.int) * tail_keep
        ans[:300] = batch_size
        ans[300:800] = 2
        self.logger.info(ans)
        self.logger.visdom_text('remember number:{}'.format(str(ans)), win_name = 'remember number')
        return ans

    def __do_train__(self, dataloader):
        self.logger.info('start training')
        self.model1.train()
        self.model2.train()

        meters = MetricLogger(delimiter = "  ")
        end = time.time()

        optimizer1 = AdamW(self.model1.parameters(), lr = 1e-5)
        optimizer2 = AdamW(self.model2.parameters(), lr = 1e-5)

        remember_num = self.get_num_remember()

        best_res_1 = 0
        best_res_2 = 0
        all_best = 0

        total_epoch = 3
        total_itr = 0

        for epoch in range(total_epoch):
            self.logger.info('total epoch:{}, cur epoch:{}'.format(total_epoch, epoch))
            dataloader.sampler.set_epoch(epoch)
            for iteration, batch in enumerate(dataloader):
                total_itr += 1
                data_time = time.time() - end
                batch = move_to_device(batch)
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                # input_ids: [16,128]   label_id:[16]
                output1 = self.model1(input_ids = batch['input_ids'],
                                      attention_mask = batch['attention_mask'])
                pred_1 = output1.logits

                output2 = self.model2(input_ids = batch['input_ids'],
                                      attention_mask = batch['attention_mask'])
                pred_2 = output2.logits

                loss_1, loss_2, pure_rate_1, pure_rate_2 = self.coteaching_loss(pred_1, pred_2,
                                                                                labels = batch['labels'],
                                                                                GT_labels = batch['GT_labels'],
                                                                                num_remember = remember_num[total_itr])

                loss_dict_reduced = reduce_loss_dict({'loss_1': loss_1, 'loss_2': loss_2})
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                meters.update(loss = losses_reduced, **loss_dict_reduced)
                loss_1.backward()
                loss_2.backward()
                optimizer1.step()
                optimizer2.step()

                batch_time = time.time() - end
                end = time.time()
                meters.update(time = batch_time, data = data_time)
                meters.update(pure_rate_1 = pure_rate_1, pure_rate_2 = pure_rate_2)

                eta_seconds = meters.time.global_avg * (self.cfg.classifier.total_steps - iteration)
                eta_string = str(datetime.timedelta(seconds = int(eta_seconds)))

                if iteration % 10 == 0:
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
                                    lr = optimizer1.param_groups[0]["lr"],
                                    memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                            ),
                            show_one = True
                    )
                    self.logger.plot_record(value = meters.loss.median, win_name = 'total loss')
                    self.logger.plot_record(value = meters.loss_1.median, win_name = 'loss_1')
                    self.logger.plot_record(value = meters.loss_2.median, win_name = 'loss_2')
                    self.logger.plot_record(value = meters.pure_rate_1.median, win_name = 'pure_rate_1')
                    self.logger.plot_record(value = meters.pure_rate_2.median, win_name = 'pure_rate_2')
                    self.logger.plot_record(value = remember_num[total_itr].tolist(), win_name = 'remember_num')
                    self.logger.visdom_text(text = 'epoch:{}, itr:{}'.format(epoch, iteration), win_name = 'epoch_itr')

                if (iteration + 0) % 100 == 0:
                    self.logger.info('start eval...'.format(self.rank))
                    synchronize()
                    res_dict_1 = self.evaler(self.model1)
                    res_dict_2 = self.evaler(self.model2)
                    if (is_main_process()):
                        f11 = res_dict_1['f1_micro']
                        f22 = res_dict_2['f1_micro']
                        self.logger.plot_record(f11, win_name = 'classifier_1 eval f1')
                        self.logger.plot_record(f22, win_name = 'classifier_2 eval f1')
                        if (f11 > best_res_1):
                            best_res_1 = f11
                            if (f11 > all_best):
                                all_best = f11
                                self.checkpointer.save_to_best_model_file(model = self.model1, other_info = res_dict_1)
                        if (f22 > best_res_2):
                            best_res_2 = f22
                            if (f22 > all_best):
                                all_best = f22
                                self.checkpointer.save_to_best_model_file(model = self.model2, other_info = res_dict_2)
                        self.logger.info('best classifier 1 f1_micro:{}'.format(best_res_1))
                        self.logger.info('best classifier 2 f1_micro:{}'.format(best_res_2))

                        self.logger.visdom_text(text = 'best classifier 1 f1_micro:{}'.format(best_res_1),
                                                win_name = 'best_f1_cls1')
                        self.logger.visdom_text(text = 'best classifier 2 f1_micro:{}'.format(best_res_2),
                                                win_name = 'best_f1_cls2')

                    self.logger.info('eval over')
                    synchronize()

            # synchronize()
            # res_dict_1 = self.evaler(self.model1)
            # if (is_main_process()):
            #     f11 = res_dict_1['f1_micro']
            #     self.logger.plot_record(f11, win_name = 'classifier eval f1')
            #     if (f11 > best_res_1):
            #         best_res_1 = f11
            #         self.checkpointer.save_to_best_model_file(model = self.model1, other_info = res_dict_1)
            #     self.logger.info('best f1_micro:{}'.format(best_res_1))
            #     self.logger.visdom_text(text = 'best f1_micro:{}'.format(best_res_1), win_name = 'best_f1')
            synchronize()
        self.logger.plot_record(value = best_res_1, win_name = 'itr classifier_1 best f1')
        self.logger.plot_record(value = best_res_2, win_name = 'itr classifier_2 best f1')
        return best_res_1

    def do_label_sentences(self, sentences):
        self.logger.info('start do_label_sentences'.format(self.rank))
        if (self.model1 is None):
            self.model1 = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                                              num_labels = self.cfg.model.number_classes,
                                                                              gradient_checkpointing = True)
        self.checkpointer.load_from_best_model(self.model1)
        self.model1 = self.model1.to(device)
        if (self.distributed):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model1)
            self.model1 = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
            )
        self.model1.eval()
        self.logger.info('eval to check before do_label_sentences')
        res = self.evaler(self.model1)
        self.logger.info('load from best model, model res:{}'.format(res))
        self.logger.info('do_label_sentences total unlabeled sentences:{}'.format(self.rank, len(sentences)))
        dataloader_sentence = self.__build_dataloader_eval_longformer__(sentences, labels = None, for_train = False)
        predicter = Predicter(cfg = self.cfg, logger = self.logger, distributed = True, rank = self.rank,
                              dataloader_sentence = dataloader_sentence, model = self.model1)
        sentences_all, labels = predicter()
        self.model1.train()
        return sentences_all, labels
