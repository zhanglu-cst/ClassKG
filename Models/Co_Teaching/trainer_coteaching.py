import datetime
import time

import numpy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import LongformerForSequenceClassification, AdamW

from Models.Base.trainer_base import Trainer_Base
from Models.Co_Teaching.dataset_coteaching import Graph_Keywords_Dataset, Collect_FN_Combine
from Models.Graph.GCN_model import GCN_Classifier
from Models.Graph.eval_graph import Eval_Model_For_Graph
from Models.Longformer_Classify.dataset_for_long import Dataset_Long, Collect_FN
from Models.Longformer_Classify.eval_model import Eval_Model_For_Long
from Models.Longformer_Classify.predict_unlabel import Predicter
from compent.checkpoint import CheckPointer_Normal
from compent.comm import is_main_process, synchronize, get_rank
from compent.metric_logger import MetricLogger
from compent.utils import move_to_device, reduce_loss_dict

device = torch.device('cuda')


class Trainer_CoTeaching(Trainer_Base):
    def __init__(self, cfg, logger, distributed, sentences_all, keywords):
        super(Trainer_CoTeaching, self).__init__(cfg = cfg, logger = logger, distributed = distributed)
        self.checkpointer = CheckPointer_Normal(cfg = cfg, logger = logger, rank = get_rank())
        self.classifier = None

        self.keywords = keywords
        self.sentences_all = sentences_all

    def __build_dataloader_combine__(self, sentences, labels, GT_labels, for_train):
        collate_fn = Collect_FN_Combine()
        dataset = Graph_Keywords_Dataset(cfg = self.cfg, logger = self.logger, keywords = self.keywords,
                                         sentences_vote = sentences, labels_vote = labels, GT_labels = GT_labels,
                                         sentences_eval = self.sentences_all.val_sentence,
                                         labels_eval = self.sentences_all.val_GT_label, for_train = for_train)
        sampler = DistributedSampler(dataset, shuffle = for_train)
        dataloader = DataLoader(dataset, batch_size = 4, sampler = sampler,
                                collate_fn = collate_fn)
        return dataloader

    def __build_dataloader_eval_longformer__(self, sentences, labels, for_train):
        collect_fn = Collect_FN(labels is not None)
        dataset = Dataset_Long(sentences, labels)
        sampler = DistributedSampler(dataset, shuffle = for_train)
        dataloader = DataLoader(dataset, batch_size = 4, sampler = sampler,
                                collate_fn = collect_fn)
        return dataloader

    def train_model(self, vote_sentences, vote_labels, GT_labels = None, finetune_from_pretrain = True):

        self.logger.info('finetune distributed:{}'.format(self.distributed))

        # vote_sentences, vote_labels, GT_labels = self.upsample_balance_with_GT(vote_sentences, vote_labels, GT_labels)

        dataloader_train = self.__build_dataloader_combine__(vote_sentences, vote_labels, GT_labels, for_train = True)

        dataloader_eval_classifier = self.__build_dataloader_eval_longformer__(self.sentences_all.val_sentence,
                                                                               self.sentences_all.val_GT_label,
                                                                               for_train = False)
        self.evaler_classifier = Eval_Model_For_Long(self.cfg, self.logger, distributed = True, rank = self.rank,
                                                     dataloader_eval = dataloader_eval_classifier)
        dataloader_eval_graph = self.__build_dataloader_combine__(vote_sentences, vote_labels, GT_labels,
                                                                  for_train = False)
        self.evaler_graph = Eval_Model_For_Graph(cfg = self.cfg, logger = self.logger, distributed = True,
                                                 rank = self.rank, dataloader_eval = dataloader_eval_graph)

        self.logger.info('build longformer')
        self.classifier = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                                              num_labels = self.cfg.model.number_classes,
                                                                              gradient_checkpointing = True)
        self.classifier.train()
        self.classifier = self.classifier.to(device)
        if (self.distributed):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.classifier)
            self.classifier = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
            )

        self.graph = GCN_Classifier(in_dim = 90, hidden_dim = 256, n_classes = self.number_classes)
        self.graph.train()
        self.graph = self.graph.to(device)
        if (self.distributed):
            graph = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.graph)
            self.graph = torch.nn.parallel.DistributedDataParallel(
                    graph, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
            )

        self.logger.info('rank:{},build eval dataset...'.format(self.rank))

        acc = self.__do_train__(dataloader = dataloader_train)
        self.logger.info('f1:{}'.format(acc))

    def get_forget_rate_schedule(self):
        forget_rate = 0.75
        zero_point = 100
        num_gradual = 500
        self.logger.visdom_text(
            text = 'forget rate:{}\n zero_point:{} \n num_gradual:{}'.format(forget_rate, zero_point, num_gradual),win_name = 'run_info')
        forget_rate_schedule = numpy.ones(3000) * forget_rate
        forget_rate_schedule[:zero_point] = 0
        forget_rate_schedule[zero_point:num_gradual] = numpy.linspace(0, forget_rate, num_gradual - zero_point)
        return forget_rate_schedule

    def __do_train__(self, dataloader):
        self.logger.info('start training')
        self.classifier.train()
        self.graph.train()

        meters = MetricLogger(delimiter = "  ")
        end = time.time()

        optimizer_classifier = AdamW(self.classifier.parameters(), lr = 1e-5)
        optimizer_graph = AdamW(self.graph.parameters(), lr = 1e-3)

        best_res_classifier = 0
        best_res_graph = 0

        total_epoch = 2
        total_itr = 0

        forget_rate_schedule = self.get_forget_rate_schedule()
        self.logger.info(forget_rate_schedule)

        for epoch in range(total_epoch):
            self.logger.info('total epoch:{}, cur epoch:{}'.format(total_epoch, epoch))
            if (self.distributed):
                dataloader.sampler.set_epoch(epoch)
            for iteration, batch in enumerate(dataloader):
                total_itr += 1
                data_time = time.time() - end
                batch = move_to_device(batch, rank = self.rank)

                # input_ids: [16,128]   label_id:[16]

                output_classifier = self.classifier(input_ids = batch['input_ids'],
                                                    attention_mask = batch['attention_mask'])
                pred_classifier = output_classifier.logits
                pred_graph = self.graph(graphs = batch['graphs'], labels = None, return_loss = False)
                labels = batch['labels']
                GT_labels = batch['GT_labels']


                loss1, loss2, pure_rate_graph, pure_rate_cls = self.coteaching_loss(pred_graph, pred_classifier,
                                                                                    labels = labels,
                                                                                    GT_labels = GT_labels,
                                                                                    forget_rate = forget_rate_schedule[
                                                                                        total_itr])
                optimizer_classifier.zero_grad()
                optimizer_graph.zero_grad()
                loss1.backward()
                loss2.backward()
                optimizer_classifier.step()
                optimizer_graph.step()

                loss_dict_reduced = reduce_loss_dict({'loss_graph': loss1, 'loss_classifier': loss2})
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                meters.update(loss = losses_reduced, **loss_dict_reduced)

                pure_rate = reduce_loss_dict({'pure_rate_graph': pure_rate_graph, 'pure_rate_cls': pure_rate_cls})
                batch_time = time.time() - end
                end = time.time()
                meters.update(time = batch_time, data = data_time)
                meters.update(pure_rate_graph = pure_rate['pure_rate_graph'],
                              pure_rate_classifier = pure_rate['pure_rate_cls'])
                meters.update(forget_rate = forget_rate_schedule[total_itr])

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
                                        "lr_classifier: {lr_classifier:.6f}",
                                        "lr_graph: {lr_graph:.6f}",
                                        "max mem: {memory:.0f}",
                                    ]
                            ).format(
                                    eta = eta_string,
                                    iter = iteration,
                                    total_itr = len(dataloader),
                                    meters = str(meters),
                                    lr_classifier = optimizer_classifier.param_groups[0]["lr"],
                                    lr_graph = optimizer_graph.param_groups[0]['lr'],
                                    memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                            ),
                            show_one = True
                    )
                    self.logger.plot_record(value = meters.loss.median, win_name = 'loss total')
                    self.logger.plot_record(value = meters.loss_graph.median, win_name = 'loss graph')
                    self.logger.plot_record(value = meters.loss_classifier.median, win_name = 'loss classifier')
                    self.logger.plot_record(value = meters.pure_rate_graph.median, win_name = 'pure_rate_graph')
                    self.logger.plot_record(value = meters.pure_rate_classifier.median,
                                            win_name = 'pure_rate_classifier')
                    self.logger.plot_record(value = forget_rate_schedule[total_itr], win_name = 'forget_rate')

                if (iteration + 0) % 100 == 0:
                    self.logger.info('start eval...'.format(self.rank))
                    synchronize()
                    res_classifier = self.evaler_classifier(self.classifier)
                    res_graph = self.evaler_graph(self.graph)
                    if (is_main_process()):
                        f11_classifier = res_classifier['f1_micro']
                        self.logger.plot_record(f11_classifier, win_name = 'classifier eval f1')
                        self.logger.info('classifier f1:{}'.format(f11_classifier))
                        if (f11_classifier > best_res_classifier):
                            best_res_classifier = f11_classifier
                            self.checkpointer.save_to_checkpoint_file_with_name(model = self.classifier,
                                                                                filename = 'classifier',
                                                                                other_info = res_classifier)
                        self.logger.info('classifier best f1_micro:{}'.format(best_res_classifier))

                        f11_graph = res_graph['f1_micro']
                        self.logger.info('graph f1:{}'.format(f11_graph))
                        self.logger.plot_record(f11_graph, win_name = 'graph eval f1')
                        if (f11_graph > best_res_graph):
                            best_res_graph = f11_graph
                            self.checkpointer.save_to_checkpoint_file_with_name(model = self.graph,
                                                                                filename = 'graph',
                                                                                other_info = res_classifier)
                        self.logger.info('graph best f1_micro:{}'.format(best_res_graph))

                    self.logger.info('eval over')
                    synchronize()

            synchronize()

        self.logger.plot_record(value = best_res_classifier, win_name = 'itr classifier best f1')
        self.logger.plot_record(value = best_res_graph, win_name = 'itr graph best f1')
        return best_res_classifier, best_res_graph



    def do_label_sentences(self, sentences):
        self.logger.info('start do_label_sentences'.format(self.rank))
        if (self.classifier is None):
            self.classifier = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                                                  num_labels = self.cfg.model.number_classes,
                                                                                  gradient_checkpointing = True)
        self.checkpointer.load_from_filename(self.classifier, 'classifier')
        self.classifier = self.classifier.to(device)
        if (self.distributed):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.classifier)
            self.classifier = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
            )
        self.classifier.eval()
        self.logger.info('eval to check before do_label_sentences')
        res = self.evaler_classifier(self.classifier)
        self.logger.info('load from best model, model res:{}'.format(res))
        self.logger.info('do_label_sentences total unlabeled sentences:{}'.format(self.rank, len(sentences)))
        dataloader_sentence = self.__build_dataloader__(sentences, labels = None, for_train = False)
        predicter = Predicter(cfg = self.cfg, logger = self.logger, distributed = True, rank = self.rank,
                              dataloader_sentence = dataloader_sentence, model = self.classifier)
        sentences_all, labels = predicter()
        self.classifier.train()
        return sentences_all, labels
