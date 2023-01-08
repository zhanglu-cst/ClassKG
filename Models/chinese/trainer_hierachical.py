import datetime
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, BertForSequenceClassification

from Models.Base.trainer_base import Trainer_Base
from Models.chinese.dataset_for_BERT import Dataset_BERT, Collect_FN
from Models.chinese.eval_model import Eval_Model_For_BERT
from compent.checkpoint import CheckPointer_Normal
from compent.comm import synchronize, get_rank
from compent.metric_logger import MetricLogger
from compent.utils import move_to_device, reduce_loss_dict, get_memory_used

device = torch.device('cuda')


def build_model(number_class, cfg):
    rank = get_rank()
    model = BertForSequenceClassification.from_pretrained(cfg.model.model_name,
                                                          num_labels = number_class,
                                                          gradient_checkpointing = True,
                                                          )
    model.train()
    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids = [rank], output_device = rank,  # find_unused_parameters=True
    )
    model.train()
    return model


class Trainer_HIERA(Trainer_Base):
    def __init__(self, cfg, logger, hiera_keywords, father_road):
        super(Trainer_HIERA, self).__init__(cfg = cfg, logger = logger, distributed = True)
        self.checkpointer = CheckPointer_Normal(cfg = cfg, logger = logger, rank = get_rank())

        self.hiera_keywords = hiera_keywords
        self.father_road = father_road
        self.father_road_str = hiera_keywords.road_list_to_str(self.father_road)
        test_sentence, test_labels = hiera_keywords.get_test_sentence_labels_cur_father_road(father_road)
        dataloader_eval = self.__build_dataloader__(test_sentence, test_labels, for_train = False)
        self.evaler_on_all = Eval_Model_For_BERT(self.cfg, self.logger, distributed = True, rank = self.rank,
                                                 dataloader_eval = dataloader_eval)

    def __build_dataloader__(self, sentences, labels, for_train):
        collect_fn = Collect_FN(self.cfg, labels is not None)
        dataset = Dataset_BERT(sentences, labels)
        sampler = DistributedSampler(dataset, shuffle = for_train)
        dataloader = DataLoader(dataset, batch_size = self.cfg.classifier.batch_size, sampler = sampler,
                                collate_fn = collect_fn)
        return dataloader

    def get_loader_from_origin_sentence(self, sentences, labels):
        sentences, labels = self.upsample_balance(sentences, labels)
        sample_number_per_class = self.get_classes_count(labels)
        self.logger.info('sample_number_per_class:{}'.format(sample_number_per_class))
        dataloader_train = self.__build_dataloader__(sentences, labels, for_train = True)
        return dataloader_train

    def train_model(self, model, sentences_belong_cur_father):
        self.model = model
        sentences, labels = self.hiera_keywords.get_pseudo_train_sentences_cur_father_road(self.father_road,
                                                                                           sentences_belong_cur_father)
        itr_self_training = 0
        last_global_best = 0
        while itr_self_training < 5:
            dataloader_train = self.get_loader_from_origin_sentence(sentences, labels)
            sentences, labels, global_best = self.__do_train__(dataloader = dataloader_train,
                                                               itr_self_training = itr_self_training)
            itr_self_training += 1
            if (global_best < last_global_best):
                break
            else:
                last_global_best = global_best

        children = self.hiera_keywords.get_children_cur_road(self.father_road)
        sentences_each_child = [[] for i in range(len(children))]
        for one_sentence, one_label in zip(sentences, labels):
            sentences_each_child[one_label].append(one_sentence)
        return sentences_each_child

    def __do_train__(self, dataloader, itr_self_training):
        self.logger.info('start training')

        self.logger.info('finetune from pretrain, load pretrain model')

        meters = MetricLogger(delimiter = "  ")
        end = time.time()

        # stop_itr = self.get_stop_itr(ITR)

        global_best = 0

        optimizer = AdamW(self.model.parameters(), lr = self.cfg.classifier.lr)

        total_epoch = self.cfg.classifier.n_epochs
        total_itr = 0
        train_over_flag = False
        stop_itr = 500
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
                del batch['sentences']
                output = self.model(**batch)
                loss = output.loss
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

                if total_itr % 10 == 0:
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
                                        "memory:{memory:.1f}"
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
                    self.logger.plot_record(value = GPU_memory, win_name = 'GPU')
                    self.logger.plot_record(value = meters.loss.median,
                                            win_name = 'loss,road:{},ST:{}'.format(self.father_road_str,
                                                                                   itr_self_training),
                                            X_value = total_itr)
                    self.logger.plot_record(value = memory, win_name = 'memory')
                    # self.logger.plot_record(value = optimizer.param_groups[0]["lr"], win_name = 'lr')

                    if (total_itr == stop_itr):
                        train_over_flag = True
                        break

                    if (total_itr % self.cfg.classifier.eval_interval == 0):
                        synchronize()
                        res_dict = self.evaler_on_all(self.model)
                        f1_micro = res_dict['f1_micro']
                        f1_macro = res_dict['f1_macro']
                        acc = res_dict['acc']
                        self.logger.plot_record(f1_micro,
                                                win_name = 'eval micro road:{},ST:{}'.format(self.father_road_str,
                                                                                             itr_self_training),
                                                X_value = total_itr)
                        self.logger.plot_record(f1_macro,
                                                win_name = 'eval macro road:{},ST:{}'.format(self.father_road_str,
                                                                                             itr_self_training),
                                                X_value = total_itr)
                        self.logger.plot_record(acc, win_name = 'eval acc road:{},ST:{}'.format(self.father_road_str,
                                                                                                itr_self_training),
                                                X_value = total_itr)
                        if (acc > global_best):
                            global_best = acc
                            self.checkpointer.save_to_checkpoint_file_with_name(model = self.model,
                                                                                filename = 'road_{}'.format(
                                                                                        self.father_road_str))
                            synchronize()
                        if (train_over_flag):
                            break

        synchronize()

        self.logger.visdom_text(text = 'start evaling last', win_name = 'state')
        res_dict = self.evaler_on_all(self.model)
        f1_micro = res_dict['f1_micro']
        f1_macro = res_dict['f1_macro']
        acc = res_dict['acc']
        if (acc > global_best):
            global_best = acc
            self.checkpointer.save_to_checkpoint_file_with_name(model = self.model,
                                                                filename = 'road_{}'.format(self.father_road_str))
        self.logger.plot_record(f1_micro, win_name = 'eval micro road:{},ST:{}'.format(self.father_road_str,
                                                                                       itr_self_training),
                                X_value = total_itr)
        self.logger.plot_record(f1_macro, win_name = 'eval macro road:{},ST:{}'.format(self.father_road_str,
                                                                                       itr_self_training),
                                X_value = total_itr)
        self.logger.plot_record(acc, win_name = 'eval acc road:{},ST:{}'.format(self.father_road_str,
                                                                                itr_self_training),
                                X_value = total_itr)
        synchronize()

        self.checkpointer.load_from_filename(model = self.model,
                                             filename = 'road_{}'.format(self.father_road_str))
        self.logger.visdom_text(text = 'start evaling last', win_name = 'state')
        res_dict = self.evaler_on_all(self.model)
        f1_micro = res_dict['f1_micro']
        f1_macro = res_dict['f1_macro']
        acc = res_dict['acc']
        self.logger.plot_record(f1_micro, win_name = 'eval last micro,road_{}'.format(self.father_road_str),
                                )
        self.logger.plot_record(f1_macro, win_name = 'eval last macro,road_{}'.format(self.father_road_str),
                                )
        self.logger.plot_record(acc, win_name = 'eval last acc,road_{}'.format(self.father_road_str))

        return res_dict['sentences'], res_dict['preds'], global_best

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
