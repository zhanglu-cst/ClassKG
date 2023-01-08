import datetime
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW

from Models.Base.trainer_base import Trainer_Base
from Models.Graph.eval_graph import Eval_Model_For_Graph, Eval_Model_On_Labeling_Quality
from Models.Graph_SSL_Soft.GIN_model import UnsupervisedGIN
from Models.Graph_SSL_Soft.dataset_graph_SSL import Graph_Keywords_Dataset_SSL_Soft, Collate_FN
from Models.SSL.trainer_SSL import Trainer_SSL
from compent.checkpoint import CheckPointer_Normal
from compent.comm import is_main_process, synchronize, get_rank
from compent.metric_logger import MetricLogger
from compent.utils import move_to_device, reduce_loss_dict

device = torch.device('cuda')


class Trainer_GCN(Trainer_Base):
    def __init__(self, cfg, logger, distributed, sentences_all, keywords):
        super(Trainer_GCN, self).__init__(cfg = cfg, logger = logger, distributed = distributed)
        self.checkpointer = CheckPointer_Normal(cfg = cfg, logger = logger, rank = get_rank())
        self.keywords = keywords
        self.sentences_all = sentences_all

    def __build_dataloader__(self, sentences, soft_labels, GT_labels, for_train):
        dataset = Graph_Keywords_Dataset_SSL_Soft(cfg = self.cfg, logger = self.logger, keywords = self.keywords,
                                                  sentences_vote = sentences, labels_soft = soft_labels,
                                                  GT_labels = GT_labels,
                                                  sentences_eval = self.sentences_all.val_sentence,
                                                  labels_eval = self.sentences_all.val_GT_label, for_train = for_train)
        sampler = DistributedSampler(dataset, shuffle = for_train)
        collate_fn = Collate_FN(for_train = for_train)
        dataloader = DataLoader(dataset, batch_size = 256, sampler = sampler,
                                collate_fn = collate_fn)
        return dataloader

    def pretrain_model(self, graph, evaler, evaler_labeling_quality, model):
        if (self.cfg.SSL.enable):
            trainer = Trainer_SSL(cfg = self.cfg, logger = self.logger, keywords = self.keywords,
                                  graph = graph, evaler = evaler, evaler_labeling_quality = evaler_labeling_quality,
                                  model = model)
            trainer.do_train()

    def train_model(self, sentences, soft_labels, hard_labels, ITR, GT_labels = None):

        sentences, hard_labels, soft_labels, GT_labels = self.upsample_balance_with_GT_soft(sentences = sentences,
                                                                                            hard_labels = hard_labels,
                                                                                            GT_labels = GT_labels,
                                                                                            soft_labels = soft_labels)

        dataloader_train = self.__build_dataloader__(sentences = sentences, soft_labels = soft_labels,
                                                     GT_labels = GT_labels,
                                                     for_train = True)
        dataloader_eval = self.__build_dataloader__(sentences = sentences, soft_labels = soft_labels,
                                                    GT_labels = GT_labels,
                                                    for_train = False)

        # self.model = GCN_Classifier(in_dim = 90, hidden_dim = 256, n_classes = self.number_classes)
        self.model = UnsupervisedGIN(self.cfg, input_dim = len(self.keywords) + self.number_classes)

        synchronize()
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

        self.pretrain_model(dataloader_train.dataset.Large_G, self.evaler, self.eval_labeling_quality, self.model)

        labeled_sentences, labeled_pred, soft_pred = self.__do_train__(dataloader = dataloader_train, ITR = ITR)
        return labeled_sentences, labeled_pred, soft_pred

    def __do_train__(self, dataloader, ITR):
        self.logger.info('start training graphs')
        self.model.train()

        meters = MetricLogger(delimiter = "  ")
        end = time.time()

        if (self.cfg.SSL.enable):
            optimizer = AdamW(self.model.parameters(), lr = 1e-4)
        else:
            optimizer = AdamW(self.model.parameters(), lr = 1e-3)

        log_soft_op = torch.nn.LogSoftmax(dim = 1)
        loss_func = torch.nn.KLDivLoss(reduction = 'batchmean', log_target = False)

        best_res = 0

        # ------------------------- #
        synchronize()
        res_dict = self.evaler(self.model)
        if (is_main_process()):
            f11 = res_dict['f1_micro']
            self.logger.plot_record(f11, win_name = 'GIN eval f1,itr:{}'.format(ITR), X_value = 0)
            if (f11 > best_res):
                best_res = f11
                self.checkpointer.save_to_checkpoint_file_with_name(model = self.model,
                                                                    filename = 'GIN_best',
                                                                    other_info = res_dict)
            self.logger.info('best f1_micro:{}'.format(best_res))
        self.logger.info('eval over')

        res_labeling = self.eval_labeling_quality(self.model)
        if (is_main_process()):
            f11_labeling = res_labeling['f1_micro']
            self.logger.plot_record(f11_labeling, win_name = 'labeling quality,itr:{}'.format(ITR),
                                    X_value = 0)
        synchronize()
        # ------------------------- #

        total_epoch = self.cfg.trainer_Graph.epoch
        total_itr = 0
        for epoch in range(total_epoch):
            self.logger.info('total epoch:{}, cur epoch:{}'.format(total_epoch, epoch))
            if (self.distributed):
                dataloader.sampler.set_epoch(epoch)
            for iteration, batch in enumerate(dataloader):
                total_itr += 1
                data_time = time.time() - end
                batch = move_to_device(batch, rank = self.rank)
                # batch = move_to_device(batch)

                # print(batch)
                optimizer.zero_grad()
                # input_ids: [16,128]   label_id:[16]
                out = self.model(graphs = batch['graphs'])
                out = log_soft_op(out)
                loss = loss_func(out, batch['labels'])

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
                    self.logger.plot_record(value = meters.loss.median, win_name = 'GIN loss,itr:{}'.format(ITR),
                                            X_value = total_itr)
                    self.logger.plot_record(value = memory, win_name = 'memory')

                if (iteration + 0) % 100 == 0:
                    self.logger.info('start eval...'.format(self.rank))
                    synchronize()
                    res_dict = self.evaler(self.model)
                    if (is_main_process()):
                        f11 = res_dict['f1_micro']
                        self.logger.plot_record(f11, win_name = 'GIN eval f1,itr:{}'.format(ITR), X_value = total_itr)
                        if (f11 > best_res):
                            best_res = f11
                            self.checkpointer.save_to_checkpoint_file_with_name(model = self.model,
                                                                                filename = 'GIN_best',
                                                                                other_info = res_dict)
                        self.logger.info('best f1_micro:{}'.format(best_res))
                    self.logger.info('eval over')

                    res_labeling = self.eval_labeling_quality(self.model)
                    if (is_main_process()):
                        f11_labeling = res_labeling['f1_micro']
                        self.logger.plot_record(f11_labeling, win_name = 'labeling quality,itr:{}'.format(ITR),
                                                X_value = total_itr)
                    synchronize()

            synchronize()
            res_dict = self.evaler(self.model)
            if (is_main_process()):
                f11 = res_dict['f1_micro']
                self.logger.plot_record(f11, win_name = 'GIN eval f1,itr:{}'.format(ITR), X_value = total_itr)
                if (f11 > best_res):
                    best_res = f11
                    self.checkpointer.save_to_checkpoint_file_with_name(model = self.model, filename = 'GIN_best',
                                                                        other_info = res_dict)
                self.logger.info('best f1_micro:{}'.format(best_res))
            synchronize()

        self.logger.plot_record(value = best_res, win_name = 'itr GIN best f1', X_value = ITR)

        # ------------------------------------- #
        self.logger.info('labeling the results')
        self.checkpointer.load_from_filename(self.model, filename = 'GIN_best', strict = True)
        res_labeling = self.eval_labeling_quality(self.model)
        res_eval_last = self.evaler(self.model)
        if (is_main_process()):
            labeling_quality_to_train_longfomer = res_labeling['f1_micro']
            self.logger.plot_record(value = labeling_quality_to_train_longfomer,
                                    win_name = 'labeling_quality_to_train_longfomer', X_value = ITR)
            labeled_sentences = res_labeling['sentences']
            labeled_pred = res_labeling['pred']
            soft_pred = res_labeling['soft_pred']
            eval_last = res_eval_last['f1_micro']
            self.logger.plot_record(value = eval_last, win_name = 'GIN_eval_highest_per_ITR', X_value = ITR)

            return labeled_sentences, labeled_pred, soft_pred
        else:
            return None, None, None

    # def do_label_sentences(self, sentences):
    #     self.logger.info('start do_label_sentences'.format(self.rank))
    #     if (self.model is None):
    #         self.model = UnsupervisedGIN(self.cfg, input_dim = len(self.keywords) + self.number_classes)
    #     self.checkpointer.load_from_best_model(self.model)
    #     self.model = self.model.to(device)
    #     if (self.distributed):
    #         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
    #         self.model = torch.nn.parallel.DistributedDataParallel(
    #                 model, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
    #         )
    #
