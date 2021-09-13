import datetime
import time

import numpy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW

from Models.Base.trainer_base import Trainer_Base
from Models.Graph.GCN_model import GCN_Classifier
from Models.Graph.eval_graph import Eval_Model_For_Graph, Eval_Model_On_Labeling_Quality
from Models.coteaching_GCN.dataset_graph_withGT import Graph_Keywords_Dataset_CoT, collate_fn
from compent.checkpoint import CheckPointer_Normal
from compent.comm import is_main_process, synchronize, get_rank
from compent.metric_logger import MetricLogger
from compent.utils import move_to_device, reduce_loss_dict

device = torch.device('cuda')

batch_size = 256


class Trainer_GCN_CoT(Trainer_Base):
    def __init__(self, cfg, logger, distributed, sentences_all, keywords):
        super(Trainer_GCN_CoT, self).__init__(cfg = cfg, logger = logger, distributed = distributed)
        self.checkpointer = CheckPointer_Normal(cfg = cfg, logger = logger, rank = get_rank())
        self.keywords = keywords
        self.sentences_all = sentences_all

    def __build_dataloader__(self, sentences, labels, GT_labels, for_train):
        dataset = Graph_Keywords_Dataset_CoT(cfg = self.cfg, logger = self.logger, keywords = self.keywords,
                                             sentences_vote = sentences, labels_vote = labels,
                                             GT_labels = GT_labels,
                                             sentences_eval = self.sentences_all.val_sentence,
                                             labels_eval = self.sentences_all.val_GT_label, for_train = for_train)
        sampler = DistributedSampler(dataset, shuffle = for_train)
        dataloader = DataLoader(dataset, batch_size = batch_size, sampler = sampler,
                                collate_fn = collate_fn)
        return dataloader

    def train_model(self, sentences, labels, GT_label = None, finetune_from_pretrain = True):
        self.logger.info('finetune distributed:{}'.format(self.distributed))

        sentences, labels, GT_label = self.upsample_balance_with_one_extra(sentences, labels, GT_label)
        # sample_number_per_class = self.get_classes_count(labels)
        # self.logger.info('sample_number_per_class:{}'.format(sample_number_per_class))

        dataloader_train = self.__build_dataloader__(sentences, labels, GT_label, for_train = True)
        dataloader_eval = self.__build_dataloader__(sentences, labels, GT_label, for_train = False)
        self.evaler = Eval_Model_For_Graph(cfg = self.cfg, logger = self.logger, distributed = True, rank = self.rank,
                                           dataloader_eval = dataloader_eval)
        self.evaler_labeling_quality = Eval_Model_On_Labeling_Quality(cfg = self.cfg, logger = self.logger,
                                                                      distributed = True, rank = self.rank,
                                                                      dataloader_train = dataloader_train)

        self.logger.info('finetune from pretrain, load pretrain model')
        self.model1 = GCN_Classifier(in_dim = len(self.keywords) + self.number_classes + 1, hidden_dim = 256,
                                     n_classes = self.number_classes)
        self.model1.train()
        self.model1 = self.model1.to(device)
        if (self.distributed):
            model1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model1)
            self.model1 = torch.nn.parallel.DistributedDataParallel(
                    model1, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
            )

        self.model2 = GCN_Classifier(in_dim = len(self.keywords) + self.number_classes + 1, hidden_dim = 256,
                                     n_classes = self.number_classes)
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

    def get_forget_rate_schedule(self):
        forget_rate = 0.25
        zero_point = 20
        num_gradual = 90
        self.logger.visdom_text(
                text = 'forget rate:{}\n zero_point:{} \n num_gradual:{}'.format(forget_rate, zero_point, num_gradual),
                win_name = 'run_info')
        forget_rate_schedule = numpy.ones(5000) * forget_rate
        forget_rate_schedule[:zero_point] = 0
        forget_rate_schedule[zero_point:num_gradual] = numpy.linspace(0, forget_rate, num_gradual - zero_point)
        return forget_rate_schedule

    def __do_train__(self, dataloader):
        self.logger.info('start training')
        self.model1.train()
        self.model2.train()

        meters = MetricLogger(delimiter = "  ")
        end = time.time()

        optimizer1 = AdamW(self.model1.parameters(), lr = 1e-3)
        optimizer2 = AdamW(self.model2.parameters(), lr = 1e-3)

        forget_rate_schedule = self.get_forget_rate_schedule()

        best_res_1 = 0
        best_res_2 = 0
        all_best = 0

        total_epoch = 30
        total_itr = 0

        for epoch in range(total_epoch):
            self.logger.info('total epoch:{}, cur epoch:{}'.format(total_epoch, epoch))
            dataloader.sampler.set_epoch(epoch)
            for iteration, batch in enumerate(dataloader):
                total_itr += 1
                data_time = time.time() - end
                batch = move_to_device(batch, rank = self.rank)
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                # print('graph dim:{}'.format(batch['graphs'].ndata['nf'].shape))
                output1 = self.model1(graphs = batch['graphs'], labels = None, return_loss = False)
                output2 = self.model2(graphs = batch['graphs'], labels = None, return_loss = False)

                num_remember = int((1 - forget_rate_schedule[total_itr]) * len(batch['labels']))
                loss_1, loss_2, pure_rate_1, pure_rate_2 = self.coteaching_loss(output1, output2,
                                                                                labels = batch['labels'],
                                                                                GT_labels = batch['GT_labels'],
                                                                                num_remember = num_remember)

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

                if iteration % 2 == 0:
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
                    self.logger.plot_record(value = meters.loss.median, win_name = 'total loss', X_value = total_itr)
                    self.logger.plot_record(value = meters.loss_1.median, win_name = 'loss_1', X_value = total_itr)
                    self.logger.plot_record(value = meters.loss_2.median, win_name = 'loss_2', X_value = total_itr)
                    self.logger.plot_record(value = meters.pure_rate_1.median, win_name = 'pure_rate_1',
                                            X_value = total_itr)
                    self.logger.plot_record(value = meters.pure_rate_2.median, win_name = 'pure_rate_2',
                                            X_value = total_itr)
                    self.logger.plot_record(value = num_remember, win_name = 'remember_num', X_value = total_itr)
                    self.logger.visdom_text(text = 'epoch:{}, itr:{}, total_itr:{}'.format(epoch, iteration, total_itr),
                                            win_name = 'epoch_itr')

                if (iteration + 1) % 10 == 0:
                    self.logger.info('start eval...'.format(self.rank))
                    synchronize()
                    res_dict_1 = self.evaler(self.model1)
                    res_dict_2 = self.evaler(self.model2)
                    if (is_main_process()):
                        f11 = res_dict_1['f1_micro']
                        f22 = res_dict_2['f1_micro']
                        self.logger.plot_record(f11, win_name = 'graph_1 eval f1', X_value = total_itr)
                        self.logger.plot_record(f22, win_name = 'graph_2 eval f1', X_value = total_itr)
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
                        self.logger.info('best graph 1 f1_micro:{}'.format(best_res_1))
                        self.logger.info('best graph 2 f1_micro:{}'.format(best_res_2))

                        self.logger.visdom_text(text = 'best graph 1 f1_micro:{}'.format(best_res_1),
                                                win_name = 'best_f1_cls1')
                        self.logger.visdom_text(text = 'best graph 2 f1_micro:{}'.format(best_res_2),
                                                win_name = 'best_f1_cls2')

                    self.logger.info('eval over')
                    synchronize()

                if iteration == 0:
                    synchronize()
                    res_dict_1 = self.evaler_labeling_quality(self.model1)
                    if (is_main_process()):
                        f11 = res_dict_1['f1_micro']
                        self.logger.plot_record(f11, win_name = 'eval_unlabel_quality', X_value = total_itr)

            synchronize()
        self.logger.plot_record(value = best_res_1, win_name = 'itr graph_1 best f1')
        self.logger.plot_record(value = best_res_2, win_name = 'itr graph_2 best f1')
        return best_res_1
