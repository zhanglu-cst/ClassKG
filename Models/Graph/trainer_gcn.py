import datetime
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW

from Models.Base.trainer_base import Trainer_Base
from Models.Graph.GCN_model import GCN_Classifier
from Models.Graph.GIN_model import UnsupervisedGIN
from Models.Graph.eval_graph import Eval_Model_For_Graph, Eval_Model_On_Labeling_Quality
from Models.coteaching_GCN.dataset_graph_withGT import Graph_Keywords_Dataset_CoT, collate_fn
from compent.checkpoint import CheckPointer_Normal
from compent.comm import is_main_process, synchronize, get_rank
from compent.metric_logger import MetricLogger
from compent.utils import move_to_device, reduce_loss_dict

device = torch.device('cuda')


class Trainer_GCN(Trainer_Base):
    def __init__(self, cfg, logger, distributed, sentences_all, keywords):
        super(Trainer_GCN, self).__init__(cfg = cfg, logger = logger, distributed = distributed)
        self.checkpointer = CheckPointer_Normal(cfg = cfg, logger = logger, rank = get_rank())
        self.model = None
        self.keywords = keywords
        self.sentences_all = sentences_all

    def __build_dataloader__(self, sentences, labels, GT_labels, for_train):
        dataset = Graph_Keywords_Dataset_CoT(cfg = self.cfg, logger = self.logger, keywords = self.keywords,
                                             sentences_vote = sentences, labels_vote = labels, GT_labels = GT_labels,
                                             sentences_eval = self.sentences_all.val_sentence,
                                             labels_eval = self.sentences_all.val_GT_label, for_train = for_train)
        sampler = DistributedSampler(dataset, shuffle = for_train)
        dataloader = DataLoader(dataset, batch_size = 256, sampler = sampler,
                                collate_fn = collate_fn)
        return dataloader

    def train_model(self, sentences, labels, finetune_from_pretrain = False, GT_labels = None):
        self.logger.info('Start Training distributed:{}'.format(self.distributed))

        sentences, labels, GT_labels = self.upsample_balance_with_one_extra(sentences, labels, GT_labels)

        dataloader_train = self.__build_dataloader__(sentences = sentences, labels = labels, GT_labels = GT_labels,
                                                     for_train = True)
        dataloader_eval = self.__build_dataloader__(sentences = sentences, labels = labels, GT_labels = GT_labels,
                                                    for_train = False)

        # self.model = GCN_Classifier(in_dim = 90, hidden_dim = 256, n_classes = self.number_classes)
        self.model = UnsupervisedGIN()

        self.model.train()
        self.model = self.model.to(device)
        if (self.distributed):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
            )

        self.evaler = Eval_Model_For_Graph(cfg = self.cfg, logger = self.logger, distributed = True, rank = self.rank,
                                           dataloader_eval = dataloader_eval)
        self.eval_labeling_quality = Eval_Model_On_Labeling_Quality(cfg = self.cfg, logger = self.logger,
                                                                    distributed = True, rank = self.rank,
                                                                    dataloader_train = dataloader_train)

        acc = self.__do_train__(dataloader = dataloader_train)
        self.logger.info('acc:{}'.format(acc))

    def __do_train__(self, dataloader):
        self.logger.info('start training graphs')
        self.model.train()

        meters = MetricLogger(delimiter = "  ")
        end = time.time()

        optimizer = AdamW(self.model.parameters(), lr = 1e-3)

        best_res = 0

        # save_best_model = cfg.save_best_model
        total_epoch = 20
        total_itr = 0
        for epoch in range(total_epoch):
            self.logger.info('total epoch:{}, cur epoch:{}'.format(total_epoch, epoch))
            if (self.distributed):
                dataloader.sampler.set_epoch(epoch)
            for iteration, batch in enumerate(dataloader):
                total_itr += 1
                data_time = time.time() - end
                batch = move_to_device(batch, rank = self.rank)
                # print(batch)
                optimizer.zero_grad()
                # input_ids: [16,128]   label_id:[16]
                loss = self.model(graphs = batch['graphs'], labels = batch['labels'])
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

                if iteration % 100 == 0:
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
                                    memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                            ),
                            show_one = True
                    )
                    self.logger.plot_record(value = meters.loss.median, win_name = 'classifier loss')
                if (iteration + 0) % 100 == 0:
                    self.logger.info('start eval...'.format(self.rank))
                    synchronize()
                    res_dict = self.evaler(self.model)
                    if (is_main_process()):
                        f11 = res_dict['f1_micro']
                        self.logger.plot_record(f11, win_name = 'classifier eval f1')
                        if (f11 > best_res):
                            best_res = f11
                            self.checkpointer.save_to_best_model_file(model = self.model, other_info = res_dict)
                        self.logger.info('best f1_micro:{}'.format(best_res))
                    self.logger.info('eval over')

                    res_labeling = self.eval_labeling_quality(self.model)
                    if (is_main_process()):
                        f11_labeling = res_labeling['f1_micro']
                        self.logger.plot_record(f11_labeling, win_name = 'labeling quality')
                    synchronize()

            # synchronize()
            # res_dict = self.evaler(self.model)
            # if (is_main_process()):
            #     f11 = res_dict['f1_micro']
            #     self.logger.plot_record(f11, win_name = 'classifier eval f1')
            #     if (f11 > best_res):
            #         best_res = f11
            #         self.checkpointer.save_to_best_model_file(model = self.model, other_info = res_dict)
            #     self.logger.info('best f1_micro:{}'.format(best_res))
            synchronize()
        self.logger.plot_record(value = best_res, win_name = 'itr classifier best f1')
        return best_res
