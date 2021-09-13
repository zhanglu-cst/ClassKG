import datetime
import time

import numpy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import LongformerForSequenceClassification, AdamW

from Models.Base.trainer_base import Trainer_Base
from Models.Longformer_Classify.dataset_for_long import Dataset_Long, Collect_FN
from Models.Longformer_Classify.eval_model import Eval_Model_For_Long
from compent.checkpoint import CheckPointer_Normal
from compent.comm import synchronize, get_rank
from compent.metric_logger import MetricLogger
from compent.utils import move_to_device, reduce_loss_dict

device = torch.device('cuda')


class Trainer_Longformer(Trainer_Base):
    def __init__(self, cfg, logger, distributed, sentences_all):
        super(Trainer_Longformer, self).__init__(cfg = cfg, logger = logger, distributed = distributed)
        self.checkpointer = CheckPointer_Normal(cfg = cfg, logger = logger, rank = get_rank())
        dataloader_eval = self.__build_dataloader__(sentences_all.unlabeled_sentence, sentences_all.unlabeled_GT_label,
                                                    for_train = False)
        self.evaler_on_all = Eval_Model_For_Long(self.cfg, self.logger, distributed = True, rank = self.rank,
                                                 dataloader_eval = dataloader_eval)
        self.global_best_longformer = 0
        self.stop_itr_list = cfg.classifier.stop_itr

    def get_stop_itr(self, ITR):
        if (ITR < len(self.stop_itr_list)):
            return self.stop_itr_list[ITR]
        else:
            return self.stop_itr_list[-1]

    def __build_dataloader__(self, sentences, labels, for_train, GT_labels = None):
        collect_fn = Collect_FN(labels is not None, GT_labels is not None)
        dataset = Dataset_Long(sentences, labels, GT_labels)
        sampler = DistributedSampler(dataset, shuffle = for_train)
        dataloader = DataLoader(dataset, batch_size = self.cfg.classifier.batch_size, sampler = sampler,
                                collate_fn = collect_fn)
        return dataloader

    def train_model(self, sentences, labels, GT_labels, ITR, finetune_from_pretrain = True):
        assert finetune_from_pretrain == True

        # train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels,
        #                                                                               test_size = 0.1)
        # dataloader_pseudo_eval = self.__build_dataloader__(test_sentences, test_labels, for_train = False)
        # self.eval_on_pseudo = Eval_Model_For_Long(self.cfg, self.logger, distributed = True, rank = self.rank,
        #                                           dataloader_eval = dataloader_pseudo_eval)

        self.logger.info('start longformer training')

        sentences, labels, GT_labels = self.upsample_balance_with_one_extra(sentences, labels, GT_labels)

        sample_number_per_class = self.get_classes_count(labels)
        self.logger.info('sample_number_per_class:{}'.format(sample_number_per_class))

        dataloader_train = self.__build_dataloader__(sentences, labels, for_train = True, GT_labels = GT_labels)

        if (finetune_from_pretrain):
            self.logger.info('finetune from pretrain, load pretrain model')
            self.model1 = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                                              num_labels = self.cfg.model.number_classes,
                                                                              gradient_checkpointing = True)
            self.model1.train()
            self.model1 = self.model1.to(device)
            if (self.distributed):
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model1)
                self.model1 = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
                )

            self.model2 = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                                              num_labels = self.cfg.model.number_classes,
                                                                              gradient_checkpointing = True)
            self.model2.train()
            self.model2 = self.model2.to(device)
            if (self.distributed):
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model2)
                self.model2 = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
                )

        else:
            self.logger.info('finetune from last model parameters')

        self.logger.info('rank:{},build eval dataset...'.format(self.rank))

        res_sentences, res_preds = self.__do_train__(dataloader = dataloader_train, ITR = ITR)
        return res_sentences, res_preds

    def get_num_remember(self):
        tail_keep = 3
        ans = numpy.ones(3000, dtype = numpy.int) * tail_keep
        ans[:300] = self.cfg.classifier.batch_size
        # ans[300:800] = int(self.cfg.classifier.batch_size * 0.75)
        self.logger.info(ans)
        self.logger.visdom_text('remember number:{}'.format(str(ans)), win_name = 'remember number')
        return ans

    def __do_train__(self, dataloader, ITR):
        self.logger.info('start training')
        self.model1.train()
        self.model2.train()

        meters = MetricLogger(delimiter = "  ")
        end = time.time()

        stop_itr = self.get_stop_itr(ITR)

        remember_num = self.get_num_remember()

        optimizer_1 = AdamW(self.model1.parameters(), lr = 2e-6)
        optimizer_2 = AdamW(self.model2.parameters(), lr = 2e-6)

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
                optimizer_1.zero_grad()
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
                optimizer_1.step()
                optimizer_2.step()

                batch_time = time.time() - end
                end = time.time()
                meters.update(time = batch_time, data = data_time)
                meters.update(pure_rate_1 = pure_rate_1, pure_rate_2 = pure_rate_2)

                eta_seconds = meters.time.global_avg * (self.cfg.classifier.total_steps - iteration)
                eta_string = str(datetime.timedelta(seconds = int(eta_seconds)))

                if total_itr % 10 == 0:
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
                                    lr = optimizer_1.param_groups[0]["lr"],
                                    memory = memory,
                            ),
                            show_one = True
                    )
                    self.logger.plot_record(value = memory, win_name = 'memory')
                    self.logger.plot_record(value = meters.loss.median,
                                            win_name = 'classifier total loss,itr:{}'.format(ITR),
                                            X_value = total_itr)
                    self.logger.plot_record(value = meters.loss_1.median, win_name = 'classifier loss_1')
                    self.logger.plot_record(value = meters.loss_2.median, win_name = 'classifier loss_2')
                    self.logger.plot_record(value = meters.pure_rate_1.median, win_name = 'pure_rate_1')
                    self.logger.plot_record(value = meters.pure_rate_2.median, win_name = 'pure_rate_2')
                    self.logger.plot_record(value = remember_num[total_itr].tolist(), win_name = 'remember_num')

                if (total_itr == stop_itr):
                    break

                if (total_itr % 100 == 0):
                    synchronize()
                    res_dict = self.evaler_on_all(self.model1)
                    f1_micro = res_dict['f1_micro']
                    f1_macro = res_dict['f1_macro']
                    self.logger.plot_record(f1_micro, win_name = 'cls on all micro itr_{}'.format(ITR),
                                            X_value = total_itr)
                    self.logger.plot_record(f1_macro, win_name = 'cls on all macro itr_{}'.format(ITR),
                                            X_value = total_itr)
                    synchronize()

        synchronize()
        self.logger.visdom_text(text = 'start evaling longformer on all data', win_name = 'state')
        res_dict = self.evaler_on_all(self.model1)
        f1_micro = res_dict['f1_micro']
        f1_macro = res_dict['f1_macro']
        self.logger.plot_record(f1_micro, win_name = 'classifier eval on all f1_micro',
                                X_value = ITR)
        self.logger.plot_record(f1_macro, win_name = 'classifier eval on all f1_macro',
                                X_value = ITR)

        self.logger.plot_record(f1_micro, win_name = 'cls on all micro itr_{}'.format(ITR),
                                X_value = total_itr)
        self.logger.plot_record(f1_macro, win_name = 'cls on all macro itr_{}'.format(ITR),
                                X_value = total_itr)

        synchronize()

        return res_dict['sentences'], res_dict['preds']

    # def do_label_sentences(self, sentences):
    #     self.logger.info('start do_label_sentences'.format(self.rank))
    #     if (self.model is None):
    #         self.model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
    #                                                                          num_labels = self.cfg.model.number_classes,
    #                                                                          gradient_checkpointing = True)
    #     self.checkpointer.load_from_filename(self.model, filename = 'longformer', strict = True)
    #     self.model = self.model.to(device)
    #     if (self.distributed):
    #         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
    #         self.model = torch.nn.parallel.DistributedDataParallel(
    #                 model, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
    #         )
    #     self.model.eval()
    #     # self.logger.info('eval to check before do_label_sentences')
    #     # res = self.evaler(self.model)
    #     # self.logger.info('load from best model, model res:{}'.format(res))
    #     self.logger.info('do_label_sentences total unlabeled sentences:{}'.format(self.rank, len(sentences)))
    #     dataloader_sentence = self.__build_dataloader__(sentences, labels = None, for_train = False)
    #     predicter = Predicter(cfg = self.cfg, logger = self.logger, distributed = True, rank = self.rank,
    #                           dataloader_sentence = dataloader_sentence, model = self.model)
    #     sentences_all, labels = predicter()
    #     self.model.train()
    #     return sentences_all, labels
