import os
import pickle

import numpy

from PROJECT_ROOT import ROOT_DIR


class Hierarchical_Keywords():
    def __init__(self, cfg, logger, sentence_all):
        super(Hierarchical_Keywords, self).__init__()
        self.cfg = cfg
        self.logger = logger
        self.sentence_all = sentence_all
        self.data_dir = os.path.join(ROOT_DIR, 'preprocess_data', 'luodi')
        with open(os.path.join(self.data_dir, 'hierarchical_class_dict.pkl'), 'rb') as f:
            self.hierarchical_class_dict = pickle.load(f)

        with open(os.path.join(self.data_dir, 'class_each_hierarchical.pkl'), 'rb') as f:
            self.class_each_hierarchical = pickle.load(f)
        with open(os.path.join(self.data_dir, 'keywords_in_all_deep.pkl'), 'rb') as f:
            self.keywords_in_all_deep = pickle.load(f)

        self.middle_labelname_to_index = {}
        self.leaf_labelname_to_index = {}
        self.leaf_index = 0
        assert len(self.class_each_hierarchical[0]) == 1
        self.build_node_labelname_to_index(list(self.class_each_hierarchical[0]))

    def build_node_labelname_to_index(self, cur_road):
        str_road_name = self.road_list_to_str(cur_road)
        if (len(cur_road) == 5):
            self.leaf_labelname_to_index[str_road_name] = self.leaf_index
            self.leaf_index += 1
            return
        else:
            children = self.get_children_cur_road(cur_road)
            for index, one_child in enumerate(children):
                cur_road.append(one_child)
                child_road_str = self.road_list_to_str(cur_road)
                self.build_node_labelname_to_index(cur_road)
                cur_road.pop()
                self.middle_labelname_to_index[child_road_str] = index

    def get_keywords_set_with_road(self, cur_road):
        str_cur_label_name = '_'.join(cur_road)
        keywords_cur_label_name = self.keywords_in_all_deep[str_cur_label_name]
        return list(keywords_cur_label_name)

    def get_children_cur_road(self, cur_road):
        # str_cur_label_name = '_'.join(cur_road)
        last = cur_road[-1]
        children = self.hierarchical_class_dict[last]
        assert len(children) >= 1 and isinstance(children, list)
        return children

    def road_list_to_str(self, road):
        str_res = '_'.join(road)
        return str_res

    def get_pseudo_train_sentences_with_keywords_each_child(self, all_children_keywords, sentences_belong_cur_father):
        # for one_child_roadname, keywords_cur_child in all_children_keywords.items():
        res_sentences = []
        res_labels = []
        num_child = len(all_children_keywords)
        for one_sentence in sentences_belong_cur_father:
            count_vector = [0] * num_child
            count_vector = numpy.array(count_vector)
            for index_class, (child_roadname, keywords_cur_children) in enumerate(all_children_keywords):
                for one_keyword in keywords_cur_children:
                    if (one_keyword in one_sentence):
                        count_vector[index_class] += 1
            if (numpy.sum(count_vector) == 0):
                continue
            else:
                max_label_index = numpy.argmax(count_vector, axis = 0)
                res_sentences.append(one_sentence)
                res_labels.append(all_children_keywords[max_label_index][0])
        self.logger.info(
                'total sentences:{}, hit setences:{}'.format(len(sentences_belong_cur_father), len(res_sentences)))
        return res_sentences, res_labels

    def get_pseudo_train_sentences_cur_father_road(self, cur_father_road, sentences_belong_cur_father):
        children = self.get_children_cur_road(cur_road = cur_father_road)
        children_road = cur_father_road
        all_children_keywords = []
        for one_child in children:
            children_road.append(one_child)
            keywords_cur_children = self.get_keywords_set_with_road(children_road)
            child_roadname = self.road_list_to_str(children_road)
            all_children_keywords.append([child_roadname, keywords_cur_children])
            children_road.pop()
        train_sentences, train_labels = self.get_pseudo_train_sentences_with_keywords_each_child(all_children_keywords,
                                                                                                 sentences_belong_cur_father)
        return train_sentences, train_labels

    def get_test_sentence_labels_cur_father_road(self, cur_father_road):
        children = self.get_children_cur_road(cur_road = cur_father_road)
        res_sentence = []
        res_labels_str = []
        children_road = cur_father_road
        for one_child in children:
            children_road.append(one_child)
            child_roadname = self.road_list_to_str(children_road)
            for one_test_sentence, one_test_classname in zip(self.sentence_all.test_sentence,
                                                             self.sentence_all.test_classname):
                if (child_roadname in one_test_classname):
                    res_sentence.append(one_test_sentence)
                    res_labels_str.append(child_roadname)

        int_labels = []
        for str_label in res_labels_str:
            int_labels.append(self.middle_labelname_to_index[str_label])
        return res_sentence, int_labels

#
# class LFs():
#     def __init__(self, cfg, logger):
#         super(LFs, self).__init__()
#         self.cfg = cfg
#         self.logger = logger
#         self.data_dir = os.path.join(ROOT_DIR, 'preprocess_data', 'luodi')
#
#         with open(os.path.join(self.data_dir, 'LF_to_CID.json'), 'r') as f:
#             strLFs_to_CID = json.load(f)
#
#         self.LF_CIDs = []
#         for str_LF in strLFs_to_CID:
#             CID_cur = strLFs_to_CID[str_LF]
#             keywords = str_LF.split('、')
#             self.LF_CIDs.append([keywords, CID_cur])
#             assert isinstance(CID_cur, int)
#
#     def judge_keywords_combine_in_sentence(self, keywords_set, sentence):
#         for one_keyword in keywords_set:
#             if (one_keyword not in sentence):
#                 return False
#         return True
#
#     def vote_for_one_sentence(self, sentence):
#         count_vector = [0] * self.cfg.model.number_classes
#         for ky_com, CID in self.LF_CIDs:
#             if (self.judge_keywords_combine_in_sentence(ky_com, sentence) == True):
#                 count_vector[CID] += 1
#         count_vector = numpy.array(count_vector)
#         if (numpy.sum(count_vector) == 0):
#             return None
#         else:
#             max_label = numpy.argmax(count_vector, axis = 0)
#             return max_label
#
#     def vote_for_all_sentences(self, sentence_all):
#         assert isinstance(sentence_all, list)
#         all_voted_labels = []
#         all_voted_sentence = []
#         for one_s in sentence_all:
#             label = self.vote_for_one_sentence(one_s)
#             if (label is not None):
#                 all_voted_sentence.append(one_s)
#                 all_voted_labels.append(label)
#         self.logger.info(
#                 'sentence total number:{}, voted:{}, rate:{}'.format(len(sentence_all), len(all_voted_sentence),
#                                                                      len(all_voted_sentence) / len(sentence_all)))
#         return all_voted_sentence, all_voted_labels
#
#     def analyse_on_labeled(self, sentence_all, labels_all):
#         all_preds = []
#         all_GTs = []
#         for one_s, one_gt in zip(sentence_all, labels_all):
#             label = self.vote_for_one_sentence(one_s)
#             if (label is not None):
#                 all_preds.append(label)
#                 all_GTs.append(one_gt)
#         self.logger.info(classification_report(y_true = all_GTs, y_pred = all_preds))
#         self.logger.info('ACC={}'.format(accuracy_score(y_true = all_GTs, y_pred = all_preds)))
#         self.logger.info(
#                 'precision micro={}'.format(precision_score(y_true = all_GTs, y_pred = all_preds, average = 'micro')))
#         self.logger.info(
#                 'precision macro={}'.format(precision_score(y_true = all_GTs, y_pred = all_preds, average = 'macro')))
