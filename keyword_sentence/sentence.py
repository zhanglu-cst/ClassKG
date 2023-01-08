import json
import os

from PROJECT_ROOT import ROOT_DIR


class Sentence_ALL():
    def __init__(self, cfg):
        data_dir_name = cfg.data_dir_name
        self.data_dir = os.path.join(ROOT_DIR, 'data', 'processed', data_dir_name)

        # try:
        #     with open(os.path.join(self.data_dir, 'labeled.json'), 'r') as f:
        #         list_labeled = json.load(f)
        #         self.labeled_sentence, self.labeled_GT_label = list(zip(*list_labeled))
        #         # print('labels type:{}'.format(type(self.labeled_GT_label[0])))
        # except:
        #     print('load labeled error')

        with open(os.path.join(self.data_dir, 'unlabeled.json'), 'r') as f:
            list_unlabeled = json.load(f)
            self.unlabeled_sentence, self.unlabeled_GT_label = list(zip(*list_unlabeled))  # list, list
        print('loading unlabeled sentence:{}'.format(len(self.unlabeled_sentence)))

        # with open(os.path.join(self.data_dir, 'val.json'), 'r') as f:
        #     list_val = json.load(f)
        #     self.val_sentence, self.val_GT_label = list(zip(*list_val))
        # print('loading val sentence:{}'.format(len(self.val_sentence)))
        #
        try:
            with open(os.path.join(self.data_dir, 'test.json'), 'r') as f:
                list_test = json.load(f)
                self.test_sentence, self.test_label = list(zip(*list_test))
                self.has_test_data = True
        except:
            print('test not exist')
            self.has_test_data = False
