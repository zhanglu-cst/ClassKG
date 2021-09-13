import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from Models.Base.trainer_base import Trainer_Base
from Models.HAN_classify.HAN_model import HierAttNet
from Models.HAN_classify.dataset_HAN import HAN_Dataset
from Models.HAN_classify.eval_for_HAN import Evaler_HAN
from Models.HAN_classify.utils import get_evaluation
from compent.comm import synchronize
from compent.saver import Saver

dict_path = r"/remote-home/XXX/glove.6B.50d.txt"


class Trainer_HAN(Trainer_Base):
    def __init__(self, cfg, logger, distributed, sentences_all):
        super(Trainer_HAN, self).__init__(cfg = cfg, logger = logger, distributed = distributed)
        # self.max_word_length, self.max_sent_length = 25, 14
        # self.max_word_length, self.max_sent_length = get_max_lengths(sentences_all.unlabeled_sentence)

        # self.logger.visdom_text(
        #         text = 'max word length:{},max sent len:{}'.format(self.max_word_length, self.max_sent_length),
        #         win_name = 'state')
        self.saver = Saver(save_dir = cfg.file_path.save_dir, logger = logger)
        loader_test = self.__build_loader__(sentences = sentences_all.unlabeled_sentence,
                                            labels = sentences_all.unlabeled_GT_label, for_train = False)
        self.evaler = Evaler_HAN(cfg = cfg, logger = logger, loader = loader_test)

    def __build_loader__(self, sentences, labels, for_train):
        dataset = HAN_Dataset(logger = self.logger, cfg = self.cfg, setences = sentences, labels = labels)
        self.max_word_length = dataset.max_length_word
        self.max_sent_length = dataset.max_length_sentences
        sampler = DistributedSampler(dataset, shuffle = for_train)
        loader = DataLoader(dataset, sampler = sampler, batch_size = self.cfg.classifier.batch_size,
                            drop_last = for_train)
        return loader

    def train_model(self, sentences, labels, ITR):
        sentences, labels = self.upsample_balance(sentences, labels)
        sample_number_per_class = self.get_classes_count(labels)
        self.logger.info('sample_number_per_class:{}'.format(sample_number_per_class))

        train_loader = self.__build_loader__(sentences = sentences, labels = labels, for_train = True)

        model = HierAttNet(word_hidden_size = 256, sent_hidden_size = 256, batch_size = self.cfg.classifier.batch_size,
                           num_classes = self.cfg.model.number_classes,
                           pretrained_word2vec_path = dict_path,
                           max_sent_length = self.max_sent_length, max_word_length = self.max_word_length)
        model.cuda()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.2,
                                    momentum = 0.9)
        epoch_number = self.cfg.classifier.n_epochs
        stop_itr = self.get_stop_itr(ITR)
        total_itr = 0
        finish = False
        for epoch in range(epoch_number):
            train_loader.sampler.set_epoch(epoch)
            for iter, (feature, label, batch_sentence) in enumerate(train_loader):
                total_itr += 1
                feature = feature.cuda()
                label = label.cuda()
                optimizer.zero_grad()
                model.module._init_hidden_state()
                predictions = model(feature)
                loss = criterion(predictions, label)
                loss.backward()
                optimizer.step()
                training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(),
                                                  list_metrics = ["accuracy"])
                self.logger.info('epoch:{},itr:{},loss:{}'.format(epoch, iter, loss))

                self.logger.plot_record(value = training_metrics["accuracy"], win_name = 'train_acc_itr:{}'.format(ITR))
                self.logger.plot_record(value = training_metrics["f1_micro"],
                                        win_name = 'train_f1_micro_itr:{}'.format(ITR))
                self.logger.plot_record(value = loss, win_name = 'loss_itr:{}'.format(ITR))

                if (total_itr % self.cfg.classifier.eval_interval == 0):
                    synchronize()
                    self.logger.info('start eval,epoch:{}'.format(epoch))
                    res = self.evaler(model)
                    model.train()
                    f1_micro = res['f1_micro']
                    f1_macro = res['f1_macro']
                    self.logger.plot_record(value = f1_micro, win_name = 'test_f1_micro_itr:{}'.format(ITR),
                                            X_value = total_itr)
                    self.logger.plot_record(value = f1_macro, win_name = 'test_f1_macro_itr:{}'.format(ITR),
                                            X_value = total_itr)

                if (total_itr == stop_itr):
                    finish = True
                    break

            if (finish):
                break

        synchronize()
        res = self.evaler(model)
        model.train()
        f1_micro = res['f1_micro']
        f1_macro = res['f1_macro']
        self.logger.plot_record(value = f1_micro, win_name = 'test_f1_micro_itr:{}'.format(ITR),
                                X_value = total_itr)
        self.logger.plot_record(value = f1_macro, win_name = 'test_f1_macro_itr:{}'.format(ITR),
                                X_value = total_itr)
        self.logger.plot_record(value = f1_micro, win_name = 'cls itr res micro')
        self.logger.plot_record(value = f1_macro, win_name = 'cls itr res macro')

        return res['sentence'], res['all_preds']
