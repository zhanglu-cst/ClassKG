import json
import os
import re

from PROJECT_ROOT import ROOT_DIR


class Sentence_ALL():
    def __init__(self, cfg):
        taskname = cfg.task_name
        self.data_dir = os.path.join(ROOT_DIR, 'preprocess_data', 'luodi')
        read_file_list = ['unlabeled.json', 'val.json', 'test.json']  # 'test.json'

        # try:
        #     with open(os.path.join(self.data_dir, 'labeled.json'), 'r') as f:
        #         list_labeled = json.load(f)
        #         self.labeled_sentence, self.labeled_GT_label = list(zip(*list_labeled))
        #         # print('labels type:{}'.format(type(self.labeled_GT_label[0])))
        # except:
        #     print('load labeled error')

        with open(os.path.join(self.data_dir, 'unlabeled.json'), 'r') as f:
            self.unlabeled_sentence = json.load(f)
            # self.unlabeled_sentence, self.unlabeled_GT_label = list(zip(*list_unlabeled))  # list, list
        print('loading unlabeled sentence:{}'.format(len(self.unlabeled_sentence)))
        self.unlabeled_sentence = self.preprocess_sentence(self.unlabeled_sentence)

        # with open(os.path.join(self.data_dir, 'val.json'), 'r') as f:
        #     list_val = json.load(f)
        #     self.val_sentence, self.val_GT_label = list(zip(*list_val))
        # print('loading val sentence:{}'.format(len(self.val_sentence)))
        #

        # with open(os.path.join(self.data_dir, 'test.json'), 'r') as f:
        #     list_test = json.load(f)
        #     self.test_sentence, self.test_label = list(zip(*list_test))
        #     self.test_sentence = self.preprocess_sentence(self.test_sentence)
        with open(os.path.join(self.data_dir, 'test_text_class_name.json'), 'r') as f:
            list_text_to_classname = json.load(f)
            # print('type:')
            # print(type(list_text_to_classname))
            # print(list_text_to_classname[0])
            self.test_sentence, self.test_classname = list(zip(*list_text_to_classname))
            self.test_sentence = self.preprocess_sentence(self.test_sentence)

    def preprocess_sentence(self, sentences):
        # remve digit
        assert isinstance(sentences, list) or isinstance(sentences, tuple)
        res = []
        for one_text in sentences:
            res_one = re.sub('[0-9]+', '', one_text)
            res.append(res_one)
        return res
