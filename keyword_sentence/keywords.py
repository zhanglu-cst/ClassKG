import json
import os
import pickle

import numpy
import torch
from sklearn.metrics import classification_report, f1_score

from PROJECT_ROOT import ROOT_DIR
from compent.comm import get_rank
from compent.comm import synchronize, accumulate_results_from_multiple_gpus, broadcast_data
from compent.utils import class_index_to_one_hot
from compent.vote import Voter
from keyword_sentence.sentence import Sentence_ALL


class KeyWords():
    def __init__(self, cfg, logger):
        self.cfg = cfg
        data_dir_name = cfg.data_dir_name
        self.data_dir = os.path.join(ROOT_DIR, 'data', 'processed', data_dir_name)
        self.logger = logger
        self.number_classes = cfg.model.number_classes
        self.rank = get_rank()

        with open(os.path.join(self.data_dir, 'keywords.json'), 'r') as f:
            label_to_keywords = json.load(f)
            self.label_to_keywords = {}
            # key to int
            for str_label in label_to_keywords:
                int_label = int(str_label)
                keywords_cur_label = label_to_keywords[str_label]
                keywords_cur_label = [item.lower() for item in keywords_cur_label]
                self.label_to_keywords[int_label] = keywords_cur_label

        self.keywords_to_label = {}
        self.keywords_to_score = {}
        for label in self.label_to_keywords:
            for word in self.label_to_keywords[label]:
                self.keywords_to_label[word] = label
                self.keywords_to_score[word] = cfg.keywords_update.seed_word_score

        self.keywords_to_index = {}
        self.index_to_keywords = {}
        for index, keyword in enumerate(self.keywords_to_label):
            self.keywords_to_index[keyword] = index
            self.index_to_keywords[index] = keyword

    def dump_keyworks(self, filename):
        exclude = ['logger']
        path = os.path.join(self.cfg.file_path.save_dir, filename)
        with open(path, 'wb') as f:
            D = dict()
            for k, v in self.__dict__.items():
                if (k not in exclude):
                    D[k] = v
            pickle.dump(D, f)

    def load_keywords(self, filename):
        path = os.path.join(self.cfg.file_path.save_dir, filename)
        with open(path, 'rb') as f:
            D = pickle.load(f)
            for k, v in D.items():
                self.__dict__[k] = v

    def syn_across_GPUs(self):
        self.logger.info('syn keywords before broadcast,rank')
        exclude = ['logger']
        synchronize()
        for key in sorted(self.__dict__):
            synchronize()
            keys_differ_GPUs = accumulate_results_from_multiple_gpus([key])
            key_set = set(keys_differ_GPUs)
            assert len(key_set) == 1
            if (key not in exclude):
                self.__dict__[key] = broadcast_data(self.__dict__[key])
        self.logger.info('syn keywords finish')

    @property
    def feature(self):
        keywords = []
        for i in range(len(self.index_to_keywords)):
            keywords.append(self.index_to_keywords[i])

        # -----------word vector feature--------------
        # path_word_to_emb = os.path.join(self.data_dir, 'word_to_emb.json')
        # with open(path_word_to_emb, 'r') as f:
        #     all_word_to_embedding = json.load(f)
        # keywords_embedding = []
        # cnt_has_no_embedding_words = 0
        # for word in keywords:
        #     if (word in all_word_to_embedding):
        #         emb_cur_word = all_word_to_embedding[word]
        #     else:
        #         emb_cur_word = torch.ones(300)
        #         cnt_has_no_embedding_words += 1
        #         self.logger.info('cnt_has_no_embedding_words:{}'.format(cnt_has_no_embedding_words))
        #     keywords_embedding.append(emb_cur_word)


        # -----------word class feature--------------
        keywords_class_index = []
        for word in keywords:
            keywords_class_index.append(self.keywords_to_label[word])
        label_one_hot = class_index_to_one_hot(keywords_class_index, self.cfg.model.number_classes)
        # [length, num_classes]
        # -----------word class feature--------------

        # -----------word index feature--------------
        keywords_indexs = []
        for word in keywords:
            keywords_indexs.append(self.keywords_to_index[word])
        index_one_hot = class_index_to_one_hot(keywords_indexs, len(keywords_indexs))
        # [length, length]
        # -----------word index feature--------------

        # -----------score---------------------
        # keywords_scores = []
        # for word in keywords:
        #     keywords_scores.append(self.keywords_to_score[word])
        #     # [length]

        # add other feature

        label_one_hot = torch.tensor(label_one_hot).float()
        index_one_hot = torch.tensor(index_one_hot).float()
        # keywords_scores = torch.tensor(keywords_scores).float().unsqueeze(1)
        # keywords_embedding = torch.tensor(keywords_embedding).float()

        # print('label_one_hot shape:{}'.format(label_one_hot.shape))
        # print('index_one_hot shape:{}'.format(index_one_hot.shape))
        # feature = torch.cat((label_one_hot, index_one_hot), dim = 1)
        # feature = torch.cat((label_one_hot, index_one_hot, keywords_scores), dim = 1)
        feature = torch.cat((label_one_hot, index_one_hot), dim = 1)

        # feature = torch.cat((label_one_hot, index_one_hot, keywords_embedding), dim = 1)
        # print('keywords origin feature shape:{}'.format(feature.shape))
        return feature

    def get_keywords_number(self):
        max_keywords_number_per_classes_list = self.cfg.keywords_update.keywords_set_keep_max_num
        cur_ITR = self.logger.get_value('ITR')
        if (cur_ITR < len(max_keywords_number_per_classes_list)):
            return max_keywords_number_per_classes_list[cur_ITR]
        else:
            return max_keywords_number_per_classes_list[-1]

    def update_keywords(self, _keywords_to_label, _keywords_to_score, incremental = True):

        overwrite_conflict = self.cfg.keywords_update.overwrite_conflict
        count_origin = len(self.keywords_to_label)
        if (incremental):
            if (overwrite_conflict):
                self.keywords_to_label.update(_keywords_to_label)  # may exceed max limit
                self.keywords_to_score.update(_keywords_to_score)
            else:
                for keyword in _keywords_to_label:
                    if (keyword not in self.keywords_to_label):
                        self.keywords_to_label[keyword] = _keywords_to_label[keyword]
                        self.keywords_to_score[keyword] = _keywords_to_score[keyword]
                    else:
                        score_new = _keywords_to_score[keyword]
                        score_old = self.keywords_to_score[keyword]
                        if (score_new > score_old):
                            self.keywords_to_score[keyword] = _keywords_to_score[keyword]
                            self.keywords_to_label[keyword] = _keywords_to_label[keyword]

            label_to_pairs = {}
            for keyword in self.keywords_to_label:
                cur_label = self.keywords_to_label[keyword]
                score = self.keywords_to_score[keyword]
                if (cur_label not in label_to_pairs):
                    label_to_pairs[cur_label] = []
                label_to_pairs[cur_label].append([keyword, score])
            assert len(label_to_pairs) == self.number_classes, 'some classes has not keywords'

            keywords_set_keep_max_num = self.get_keywords_number()
            for class_index in range(self.number_classes):
                pairs_cur_class = label_to_pairs[class_index]
                sorted_pairs = sorted(pairs_cur_class, key = lambda x: x[1], reverse = True)
                keep = min(len(sorted_pairs), keywords_set_keep_max_num)
                sorted_pairs = sorted_pairs[:keep]
                label_to_pairs[class_index] = sorted_pairs
                print('class:{}, keywords:{}'.format(class_index, len(label_to_pairs[class_index])))

            self.keywords_to_label = {}
            self.keywords_to_score = {}
            for class_index in range(self.number_classes):
                for keyword, score in label_to_pairs[class_index]:
                    self.keywords_to_label[keyword] = class_index
                    self.keywords_to_score[keyword] = score

            print(
                    'Incremental add, before add:{}, after add:{}'.format(count_origin, len(self.keywords_to_label)))
            # print('keywords_to_score:{}'.format(self.keywords_to_score))
        else:
            self.keywords_to_label = _keywords_to_label
            self.keywords_to_score = _keywords_to_score
            self.logger.info(
                    'Non_incremental, before add:{}, after add:{}'.format(count_origin, len(self.keywords_to_label)))

        self.label_to_keywords = {}
        for keyword in self.keywords_to_label:
            label = self.keywords_to_label[keyword]
            if (label not in self.label_to_keywords):
                self.label_to_keywords[label] = []
            self.label_to_keywords[label].append(keyword)
        assert len(self.label_to_keywords) == self.number_classes, 'some classes does not have keywords'

        self.keywords_to_index = {}
        self.index_to_keywords = {}
        for index, keyword in enumerate(self.keywords_to_label):
            self.keywords_to_index[keyword] = index
            self.index_to_keywords[index] = keyword

    def analyse_on_GTunlabel(self, sentence_all: Sentence_ALL):
        voter = Voter(self.cfg, self)
        y_true = []
        y_pred = []
        for sentence, label in zip(sentence_all.unlabeled_sentence, sentence_all.unlabeled_GT_label):
            pred = voter(sentence, need_no_confuse = self.cfg.keywords_update.vote_need_no_confuse)
            if (pred is not None):
                y_true.append(label)
                y_pred.append(pred)
        class_statistics_GT = numpy.zeros(self.number_classes)
        class_statistics_Pred = numpy.zeros(self.number_classes)
        for one_GT in y_true:
            class_statistics_GT[one_GT] += 1
        for one_pred in y_pred:
            class_statistics_Pred[one_pred] += 1
        for class_index, (GT_count, Pred_count) in enumerate(zip(class_statistics_GT, class_statistics_Pred)):
            print('class:{}, GT count:{}, Pred count:{}'.format(class_index, GT_count, Pred_count))
        print('rank:{},analyse of keywords:'.format(self.rank))
        print(classification_report(y_true = y_true, y_pred = y_pred, ))
        print('rank:{},analyse of keywords:, keywords count:{}'.format(self.rank, self.__len__()))
        print('rank:{},cover:{}, all:{}'.format(self.rank, len(y_pred), len(sentence_all.unlabeled_sentence)))
        print('rank:{}, cover%:{}'.format(self.rank, len(y_pred) / len(sentence_all.unlabeled_sentence)))
        f1_micro = f1_score(y_true = y_true, y_pred = y_pred, average = 'micro')
        f1_macro = f1_score(y_true = y_true, y_pred = y_pred, average = 'macro')
        self.logger.info('rank:{}, keywords f1_micro:{}'.format(self.rank, f1_micro))
        self.logger.info('rank:{}, keywords f1_macro:{}'.format(self.rank, f1_macro))

        coverage = len(y_pred) / len(sentence_all.unlabeled_sentence)
        # self.logger.plot_record(coverage, win_name = 'keywords cover')
        # self.logger.plot_record(self.__len__(), win_name = 'keywords number')
        # self.logger.plot_record(f1_micro, win_name = 'keywords f1_micro')
        # self.logger.plot_record(f1_macro, win_name = 'keywords f1_macro')
        return {'f1_micro': f1_micro, 'f1_macro': f1_macro, 'coverage': coverage}

        # self.logger.visdom_text(text = classification_report(y_true = y_true, y_pred = y_pred,),win_name = 'keywords cls report',append = True)

        # P = precision_score(y_true = y_true, y_pred = y_pred, average = 'micro')
        # print('cover:{}, all:{}'.format(len(y_pred), len(sentence_all.unlabeled_sentence)))
        # print('cover%:{}'.format(len(y_pred) / len(sentence_all.unlabeled_sentence)))
        # print('P:{}'.format(P))
        # F1 = f1_score(y_true = y_true, y_pred = y_pred, pos_label = 1, average = 'micro')
        # print('f1:{}'.format(F1))
        # acc = accuracy_score(y_true = y_true, y_pred = y_pred)
        # print('ACC:{}'.format(acc))

    def __len__(self):
        return len(self.keywords_to_index)

    def __str__(self):

        s = 'number classes:{}, total keywords:{}  \n'.format(len(self.label_to_keywords), len(self.keywords_to_label))
        for label in self.label_to_keywords:
            s += 'labels:{}, count:{}, keywords:{} '.format(label, len(self.label_to_keywords[label]),
                                                            self.label_to_keywords[label], ) + '\n'
        return s
