import json
import os

import numpy
from sklearn.metrics import classification_report, accuracy_score, precision_score

from PROJECT_ROOT import ROOT_DIR


class LFs():
    def __init__(self, cfg, logger):
        super(LFs, self).__init__()
        self.cfg = cfg
        self.logger = logger
        self.data_dir = os.path.join(ROOT_DIR, 'preprocess_data', 'luodi')

        with open(os.path.join(self.data_dir, 'LF_to_CID.json'), 'r') as f:
            strLFs_to_CID = json.load(f)

        self.LF_CIDs = []
        for str_LF in strLFs_to_CID:
            CID_cur = strLFs_to_CID[str_LF]
            keywords = str_LF.split('、')
            self.LF_CIDs.append([keywords, CID_cur])
            assert isinstance(CID_cur, int)

    def judge_keywords_combine_in_sentence(self, keywords_set, sentence):
        for one_keyword in keywords_set:
            if (one_keyword not in sentence):
                return False
        return True

    def vote_for_one_sentence(self, sentence):
        count_vector = [0] * self.cfg.model.number_classes
        for ky_com, CID in self.LF_CIDs:
            if (self.judge_keywords_combine_in_sentence(ky_com, sentence) == True):
                count_vector[CID] += 1
        count_vector = numpy.array(count_vector)
        if (numpy.sum(count_vector) == 0):
            return None
        else:
            max_label = numpy.argmax(count_vector, axis = 0)
            return max_label

    def vote_for_all_sentences(self, sentence_all):
        assert isinstance(sentence_all, list)
        all_voted_labels = []
        all_voted_sentence = []
        for one_s in sentence_all:
            label = self.vote_for_one_sentence(one_s)
            if (label is not None):
                all_voted_sentence.append(one_s)
                all_voted_labels.append(label)
        self.logger.info(
                'sentence total number:{}, voted:{}, rate:{}'.format(len(sentence_all), len(all_voted_sentence),
                                                                     len(all_voted_sentence) / len(sentence_all)))
        return all_voted_sentence, all_voted_labels

    def analyse_on_labeled(self, sentence_all, labels_all):
        all_preds = []
        all_GTs = []
        for one_s, one_gt in zip(sentence_all, labels_all):
            label = self.vote_for_one_sentence(one_s)
            if (label is not None):
                all_preds.append(label)
                all_GTs.append(one_gt)
        self.logger.info(classification_report(y_true = all_GTs, y_pred = all_preds))
        self.logger.info('ACC={}'.format(accuracy_score(y_true = all_GTs, y_pred = all_preds)))
        self.logger.info(
                'precision micro={}'.format(precision_score(y_true = all_GTs, y_pred = all_preds, average = 'micro')))
        self.logger.info(
                'precision macro={}'.format(precision_score(y_true = all_GTs, y_pred = all_preds, average = 'macro')))
