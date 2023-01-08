import numpy

from keyword_sentence.sentence_process import split_sentence_into_words


class Keywords_Updater_Base():
    def __init__(self, keywords, cfg, logger):
        super(Keywords_Updater_Base, self).__init__()
        self.keywords = keywords
        self.cfg = cfg
        self.logger = logger
        self.number_classes = cfg.model.number_classes

    def update_keywords(self, sentences, labels, incremental):
        pass

    def get_words_list(self, sentences):
        words_all = set()
        for one_sentence in sentences:
            words_cur_sentence = split_sentence_into_words(one_sentence)
            words_all.update(words_cur_sentence)
        word_list = list(words_all)
        word_to_index = dict()
        for index, word in enumerate(word_list):
            word_to_index[word] = index
        return word_list, word_to_index

    def cal_IDF(self, sentences, word_list, word_to_index):

        n_sentences = len(sentences)
        doc_frequency = numpy.zeros(len(word_list))

        for one_doc in sentences:
            words = split_sentence_into_words(one_doc)
            words = set(words)
            for one_word_cur_doc in words:
                index = word_to_index[one_word_cur_doc]
                doc_frequency[index] += 1

        tmp = n_sentences / doc_frequency
        IDF = numpy.log(tmp)

        return IDF

    def cal_LI(self, sentences, labels, word_list, word_to_index):

        doc_num_each_class_each_word = numpy.zeros((self.number_classes, len(word_list)))
        doc_num_each_class = self.get_doc_number_each_class(labels).reshape(-1, 1)

        for class_index in range(self.number_classes):
            for one_sentence, one_label in zip(sentences, labels):
                if (one_label == class_index):
                    words = split_sentence_into_words(one_sentence)
                    words = set(words)
                    for one_word in words:
                        word_index = word_to_index[one_word]
                        doc_num_each_class_each_word[class_index][word_index] += 1

        LI = doc_num_each_class_each_word / doc_num_each_class
        return LI

    def cal_TF_normal_with_doc_num(self, sentences, labels, word_list, word_to_index):

        doc_num_each_class_each_word = numpy.zeros((self.number_classes, len(word_list)))
        doc_num_each_class = self.get_doc_number_each_class(labels).reshape(-1, 1)

        for class_index in range(self.number_classes):
            for one_sentence, one_label in zip(sentences, labels):
                if (one_label == class_index):
                    words = split_sentence_into_words(one_sentence)
                    for one_word in words:
                        word_index = word_to_index[one_word]
                        doc_num_each_class_each_word[class_index][word_index] += 1

        TF = doc_num_each_class_each_word / doc_num_each_class
        TF = numpy.tanh(TF)
        return TF

    def cal_TF_normal_with_words_num(self, sentences, labels, word_list, word_to_index):
        doc_num_each_class_each_word = numpy.zeros((self.number_classes, len(word_list)))
        words_num_each_class = self.get_words_number_each_class(sentences, labels).reshape(-1, 1)
        words_num_each_class = words_num_each_class + 1

        for class_index in range(self.number_classes):
            for one_sentence, one_label in zip(sentences, labels):
                if (one_label == class_index):
                    words = split_sentence_into_words(one_sentence)
                    for one_word in words:
                        word_index = word_to_index[one_word]
                        doc_num_each_class_each_word[class_index][word_index] += 1
        TF = doc_num_each_class_each_word / words_num_each_class * 1
        TF = numpy.tanh(TF)

        return TF

    def cal_exclusivity(self, sentences, labels, word_list, word_to_index):
        pass

    def cal_keywords_diff(self, origin_keywords_to_label, new_keywords_to_label):
        fenmu = len(new_keywords_to_label)
        fenzi = 0
        for word in new_keywords_to_label:
            if (word not in origin_keywords_to_label):
                fenzi += 1
            else:
                label_new = new_keywords_to_label[word]
                label_origin = origin_keywords_to_label[word]
                if (label_new != label_origin):
                    fenzi += 1
        return fenzi / fenmu

    def __update_keywords_with_indicator_M1__(self, indicator, word_list, incremental):
        assert indicator.ndim == 2
        max_keywords_number_per_classes = self.cfg.keywords_update.extract_keywords_per_class
        score_thr = self.cfg.keywords_update.score_thr

        class_to_keywords = {}
        for class_index in range(self.number_classes):
            scores_for_words = indicator[class_index]
            pair = [(s, w) for s, w in zip(scores_for_words, word_list)]
            sort_pair = sorted(pair, key = lambda x: x[0], reverse = True)

            class_to_keywords[class_index] = []
            for j in range(max_keywords_number_per_classes):
                score = sort_pair[j][0]
                word = sort_pair[j][1]
                if (score > score_thr):
                    class_to_keywords[class_index].append(word)
                    print('class index:{}, keyword:{}, score:{}'.format(class_index, word, score))
                else:
                    break
            self.logger.info(
                    'class:{}, keyword number:{}, keywords:{}'.format(class_index, len(class_to_keywords[class_index]),
                                                                      class_to_keywords[class_index]))

        words_count = {}
        for class_index in range(self.number_classes):
            for word in class_to_keywords[class_index]:
                if (word not in words_count):
                    words_count[word] = 1
                else:
                    words_count[word] += 1

        keyword_to_label = {}
        for class_index in range(self.number_classes):
            for word in class_to_keywords[class_index]:
                if (words_count[word] == 1):
                    keyword_to_label[word] = class_index

        print('total keywords count:{}'.format(len(keyword_to_label)))
        for keyword in keyword_to_label:
            print('keyword:{}, labels:{}'.format(keyword, keyword_to_label[keyword]))

        self.keywords.update_keywords(keyword_to_label, incremental = incremental)

    def get_keywords_number(self):
        max_keywords_number_per_classes_list = self.cfg.keywords_update.extract_keywords_per_class
        cur_ITR = self.logger.get_value('ITR')
        if (cur_ITR < len(max_keywords_number_per_classes_list)):
            return max_keywords_number_per_classes_list[cur_ITR]
        else:
            return max_keywords_number_per_classes_list[-1]

    def __update_keywords_with_indicator_M2__(self, indicator, word_list, incremental):

        assert indicator.ndim == 2
        score_thr = self.cfg.keywords_update.score_thr
        max_keywords_number_per_classes = self.get_keywords_number()

        classes_per_keywords = numpy.argmax(indicator, axis = 0)
        cor_max_scores = numpy.max(indicator, axis = 0)

        class_to_words = {}

        for class_index in range(self.number_classes):
            class_to_words[class_index] = []

        for keyword, score, class_index in zip(word_list, cor_max_scores, classes_per_keywords):
            if (score > score_thr):
                class_to_words[class_index].append([keyword, score])

        filtered_class_to_words = {}
        for class_index in range(self.number_classes):
            words_cur_class = class_to_words[class_index]
            if (len(words_cur_class) == 0):
                filtered_class_to_words[class_index] = []
                continue
            sort_pair = sorted(words_cur_class, key = lambda x: x[1], reverse = True)
            if (isinstance(max_keywords_number_per_classes, float)):
                keep = int(len(sort_pair) * max_keywords_number_per_classes)
            else:
                keep = min(len(sort_pair), max_keywords_number_per_classes)
            print('class index:{}, extract word count:{}'.format(class_index, keep))
            sort_pair = sort_pair[:keep]
            filtered_class_to_words[class_index] = sort_pair

        keywords_to_label = {}
        keyword_to_score = {}
        for class_index in range(self.number_classes):
            for item in filtered_class_to_words[class_index]:
                keyword, score = item
                keywords_to_label[keyword] = class_index
                keyword_to_score[keyword] = score

        diff = self.cal_keywords_diff(self.keywords.keywords_to_label, keywords_to_label)
        self.keywords.update_keywords(keywords_to_label, keyword_to_score, incremental = incremental)
        return diff

    def get_doc_number_each_class(self, labels):
        labels = numpy.array(labels)
        doc_num_each_class = numpy.zeros(self.number_classes)
        for class_index in range(self.number_classes):
            pred_docs_cur_class = numpy.sum(labels == class_index)
            doc_num_each_class[class_index] = pred_docs_cur_class
        return doc_num_each_class

    def get_words_number_each_class(self, sentences, labels):
        labels = numpy.array(labels)
        words_num_each_class = numpy.zeros(self.number_classes)
        for cur_sentences, cur_label in zip(sentences, labels):
            words = split_sentence_into_words(cur_sentences)
            num_word = len(words)
            words_num_each_class[cur_label] += num_word
        return words_num_each_class
