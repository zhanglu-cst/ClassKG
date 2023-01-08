import datetime
import time

import numpy
import torch
from torch.utils.data import DataLoader
from compent.comm import get_world_size

from Models.Bert_Classify import sentence_process
from Models.Bert_Classify.eval_model import Eval_Model
from compent.checkpoint import CheckPointer_Bert
from compent.comm import is_main_process, synchronize
from compent.metric_logger import MetricLogger
from compent.optim import optim4GPU
from compent.utils import move_to_device, reduce_loss_dict
from model.BERT import BERT_Classifier

device = torch.device('cuda')


class Trainer_Bert():
    def __init__(self, cfg, logger,):
        super(Trainer_Bert, self).__init__()
        self.cfg = cfg
        self.logger = logger
        self.world_size = get_world_size()

    def __build_dataloader__(self, sentences, labels, shuffle = True):
        self.logger.info('building dataset...')
        dataset = sentence_process.Dataset_For_Single_Sentence_Predict(self.cfg, sentences, labels)
        dataloader = DataLoader(dataset, batch_size = self.cfg.classifier.batch_size, shuffle = shuffle)
        self.logger.info('finish building dataset')
        return dataloader

    def finetune_model(self, sentences, labels, sentences_for_eval, labels_for_eval, finetune_from_bert = True):
        dataloader_train = self.__build_dataloader__(sentences, labels)
        dataloader_eval = self.__build_dataloader__(sentences_for_eval, labels_for_eval)
        if (finetune_from_bert):
            self.model = BERT_Classifier(self.cfg)
            self.checkpointer = CheckPointer_Bert(self.model, self.cfg, self.logger, rank = 0)
            self.checkpointer.load_from_pretrainBERT()

            self.model = self.model.to(device)

        evaler = Eval_Model(self.cfg, self.logger, distributed = False, rank = 0,
                            dataloader_eval = dataloader_eval, model = self.model)
        acc = self.__do_train__(dataloader = dataloader_train, evaler = evaler, distributed = False)
        print('acc:{}'.format(acc))

    def do_label_sentences(self, sentences):
        pass

    def __do_train__(self, dataloader, evaler, distributed = False):
        self.model.train()

        meters = MetricLogger(delimiter = "  ")
        end = time.time()

        optimizer = optim4GPU(self.cfg, self.model)

        best_res = 0

        # save_best_model = cfg.save_best_model
        total_epoch = self.cfg.classifier.n_epochs
        total_itr = 0
        for epoch in range(total_epoch):
            self.logger.info('total epoch:{}, cur epoch:{}'.format(total_epoch, epoch))
            if (distributed):
                dataloader.sampler.set_epoch(epoch)
            record_loss = []
            for iteration, batch in enumerate(dataloader):
                total_itr += 1
                data_time = time.time() - end
                batch = move_to_device(batch)

                optimizer.zero_grad()
                # input_ids: [16,128]   label_id:[16]
                loss_dict = self.model(batch)
                losses = sum(loss for loss in loss_dict.values())

                loss_dict_reduced = reduce_loss_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                record_loss.append(losses_reduced.data.item())

                meters.update(loss = losses_reduced, **loss_dict_reduced)
                losses.backward()

                optimizer.step()

                batch_time = time.time() - end
                end = time.time()
                meters.update(time = batch_time, data = data_time)

                eta_seconds = meters.time.global_avg * (self.cfg.classifier.total_steps - iteration)
                eta_string = str(datetime.timedelta(seconds = int(eta_seconds)))

                if iteration % 20 == 0 or iteration == self.cfg.classifier.total_steps:
                    self.logger.info(
                            meters.delimiter.join(
                                    [
                                        "eta: {eta}",
                                        "iter: {iter}",
                                        "{meters}",
                                        "lr: {lr:.6f}",
                                        "max mem: {memory:.0f}",
                                    ]
                            ).format(
                                    eta = eta_string,
                                    iter = iteration,
                                    meters = str(meters),
                                    lr = optimizer.param_groups[0]["lr"],
                                    memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                            )
                    )

            mean_loss = numpy.mean(record_loss)
            self.logger.info('epoch:{}, loss mean:{}'.format(epoch, mean_loss))
            if (is_main_process()):
                res = evaler()
                f11 = res['f1_micro']
                if (f11 > best_res):
                    best_res = f11
                    self.checkpointer.save_to_best_model_file()
                self.logger.info('best f11:{}'.format(best_res))
            synchronize()
        return best_res

