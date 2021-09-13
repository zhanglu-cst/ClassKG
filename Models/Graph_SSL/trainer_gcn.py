import datetime
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW

from Models.Base.trainer_base import Trainer_Base
from Models.Graph.eval_graph import Eval_Model_On_Labeling_Quality
from Models.Graph_SSL.GAT_model import GAT_Classifier
from Models.Graph_SSL.GCN_model import GCN_Classifier
from Models.Graph_SSL.GIN_model import GIN
from Models.Graph_SSL.dataset_graph_SSL import Graph_Keywords_Dataset_SSL, collate_fn
from Models.SSL.trainer_SSL import Trainer_SSL
from compent.checkpoint import CheckPointer_Normal
from compent.comm import is_main_process, synchronize, get_rank
from compent.metric_logger import MetricLogger
from compent.utils import move_to_device, reduce_loss_dict, get_memory_used

device = torch.device('cuda')


class Trainer_GCN(Trainer_Base):
    def __init__(self, cfg, logger, distributed, sentences_all, keywords):
        super(Trainer_GCN, self).__init__(cfg = cfg, logger = logger, distributed = distributed)
        self.checkpointer = CheckPointer_Normal(cfg = cfg, logger = logger, rank = get_rank())
        self.keywords = keywords
        self.sentences_all = sentences_all

    def __build_dataloader__(self, sentences, labels, GT_labels, for_train):
        dataset = Graph_Keywords_Dataset_SSL(cfg = self.cfg, logger = self.logger, keywords = self.keywords,
                                             sentences_vote = sentences, labels_vote = labels, GT_labels = GT_labels,
                                             )
        sampler = DistributedSampler(dataset, shuffle = for_train)
        dataloader = DataLoader(dataset, batch_size = self.cfg.trainer_Graph.batch_size, sampler = sampler,
                                collate_fn = collate_fn)
        return dataloader

    def pretrain_model(self, graph, model):
        if (self.cfg.SSL.enable):
            trainer = Trainer_SSL(cfg = self.cfg, logger = self.logger, keywords = self.keywords,
                                  graph = graph, evaler_labeling_quality = self.eval_labeling_quality,
                                  eval_on_origin = self.eval_on_origin_data,
                                  model = model)
            trainer.do_train()

    def train_model(self, sentences, vote_labels, ITR, GT_labels = None):
        dataloader_origin_data = self.__build_dataloader__(sentences = sentences, labels = vote_labels,
                                                           GT_labels = GT_labels, for_train = False)
        self.eval_on_origin_data = Eval_Model_On_Labeling_Quality(cfg = self.cfg, logger = self.logger,
                                                                  distributed = True, rank = self.rank,
                                                                  dataloader_train = dataloader_origin_data)

        sentences, vote_labels, GT_labels = self.upsample_balance_with_one_extra(sentences = sentences,
                                                                                 labels = vote_labels,
                                                                                 GT_labels = GT_labels,
                                                                                 )

        dataloader_train = self.__build_dataloader__(sentences = sentences, labels = vote_labels, GT_labels = GT_labels,
                                                     for_train = True)
        # dataloader_eval = self.__build_dataloader__(sentences = sentences, labels = vote_labels, GT_labels = GT_labels,
        #                                             for_train = False)

        # self.model = GCN_Classifier(in_dim = 90, hidden_dim = 256, n_classes = self.number_classes)
        self.logger.visdom_text('GNN model:{}'.format(self.cfg.GNN_model), win_name = 'state')
        if (self.cfg.GNN_model == 'GIN'):
            self.model = GIN(self.cfg, input_dim = len(self.keywords) + self.number_classes)
        elif (self.cfg.GNN_model == 'GCN'):
            self.model = GCN_Classifier(self.cfg, input_dim = len(self.keywords) + self.number_classes)
        elif (self.cfg.GNN_model == 'GAT'):
            self.model = GAT_Classifier(self.cfg, input_dim = len(self.keywords) + self.number_classes)
        else:
            raise NotImplementedError
        # self.model = UnsupervisedGIN(self.cfg, input_dim = self.keywords.max_number_per_class + self.number_classes)

        synchronize()
        self.model.train()
        self.model = self.model.to(device)
        if (self.distributed):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
            )

        self.eval_labeling_quality = Eval_Model_On_Labeling_Quality(cfg = self.cfg, logger = self.logger,
                                                                    distributed = True, rank = self.rank,
                                                                    dataloader_train = dataloader_train)

        self.pretrain_model(dataloader_train.dataset.Large_G, self.model)

        res_dict = self.__do_train__(dataloader = dataloader_train, ITR = ITR)
        return res_dict

    def get_total_epoch(self, ITR):
        list_epoch = self.cfg.trainer_Graph.epoch
        if (ITR < len(list_epoch)):
            return list_epoch[ITR]
        else:
            return list_epoch[-1]

    def __do_train__(self, dataloader, ITR):
        self.logger.info('start training graphs')
        self.model.train()

        meters = MetricLogger(delimiter = "  ")
        end = time.time()

        if (self.cfg.SSL.enable):
            optimizer = AdamW(self.model.parameters(), lr = 1e-4)
        else:
            optimizer = AdamW(self.model.parameters(), lr = 1e-3)

        loss_func = torch.nn.CrossEntropyLoss()

        # ------------------------- #
        synchronize()


        total_epoch = self.get_total_epoch(ITR)

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

                out = self.model(graphs = batch['graphs'])
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

                del batch

                if iteration % 100 == 0:
                    GPU_memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                    memory = get_memory_used()
                    self.logger.info(
                            meters.delimiter.join(
                                    [
                                        "eta: {eta}",
                                        "iter: {iter}",
                                        "total_itr: {total_itr}",
                                        "{meters}",
                                        "lr: {lr:.6f}",
                                        "max mem: {GPU_memory:.0f}",
                                        "memory: {memory:.1f}",
                                    ]
                            ).format(
                                    eta = eta_string,
                                    iter = iteration,
                                    total_itr = len(dataloader),
                                    meters = str(meters),
                                    lr = optimizer.param_groups[0]["lr"],
                                    GPU_memory = GPU_memory,
                                    memory = memory,
                            ),
                            show_one = True
                    )
                    self.logger.plot_record(value = meters.loss.median, win_name = 'GIN loss,itr:{}'.format(ITR),
                                            X_value = total_itr)
                    self.logger.plot_record(value = GPU_memory, win_name = 'GPU')
                    self.logger.plot_record(value = memory, win_name = 'memory')
                    # self.logger.plot_record(value = optimizer.param_groups[0]["lr"], win_name = 'lr')

            synchronize()

        # ------------------------------------- #

        res_labeling = self.eval_labeling_quality(self.model)
        # res_on_origin = self.eval_on_origin_data(self.model)
        if (is_main_process()):
            labeled_sentences = res_labeling['sentences']
            labeled_pred = res_labeling['pred']
            GT_labels = res_labeling['GT_labels']

            self.checkpointer.save_to_checkpoint_file_with_name(self.model, filename = 'GIN_itr_{}'.format(ITR))
            return {'sentences': labeled_sentences, 'pred_labels': labeled_pred, 'GT_labels': GT_labels}
        else:
            return None

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
