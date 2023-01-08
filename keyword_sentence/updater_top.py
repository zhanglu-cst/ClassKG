import numpy


from keyword_sentence.keywords import KeyWords
from keyword_sentence.updater_base import Keywords_Updater_Base


class Keywords_Updater_Conwea(Keywords_Updater_Base):
    def __init__(self, keywords: KeyWords, cfg, logger):
        super(Keywords_Updater_Conwea, self).__init__(keywords, cfg, logger)

    def update_keywords(self, sentences, labels, incremental):
        self.logger.info('update keywords')

        word_list, word_to_index = self.get_words_list(sentences)

        IDF_vector = self.cal_IDF(sentences, word_list, word_to_index)
        LI = self.cal_LI(sentences, labels, word_list, word_to_index)
        TF = self.cal_TF_normal_with_doc_num(sentences, labels, word_list, word_to_index)

        indicator = TF * LI * IDF_vector

        self.__update_keywords_with_indicator_M1__(indicator, word_list, incremental = incremental)


class Keywords_Updater_TFIDF(Keywords_Updater_Base):
    def __init__(self, keywords: KeyWords, cfg, logger):
        super(Keywords_Updater_TFIDF, self).__init__(keywords, cfg, logger)
        self.IDF_n = cfg.keywords_update.IDF_n

    def get_classes_count(self, labels):  # Move to base
        ans = numpy.zeros(self.number_classes, dtype = numpy.int32)
        arr_labels = numpy.array(labels)
        for class_index in range(self.number_classes):
            cnt_cur = numpy.sum(arr_labels == class_index)
            ans[class_index] = cnt_cur
        return ans


    def upsample_balance(self, sentences, labels):
        sample_number_per_class = self.get_classes_count(labels)
        self.logger.info('berfor balance, sample number each class:{}'.format(sample_number_per_class))
        self.logger.visdom_text(str(sample_number_per_class), win_name = 'sample_number_per_class', append = True)
        max_number = numpy.max(sample_number_per_class)
        fill_number_each_class = max_number - sample_number_per_class
        sentence_each_class = [[] for i in range(self.number_classes)]
        for s, l in zip(sentences, labels):
            sentence_each_class[l].append(s)
        for class_index, (sentences_cur_class, fill_num_cur_class) in enumerate(
                zip(sentence_each_class, fill_number_each_class)):
            append_cur_class = []
            for i in range(fill_num_cur_class):
                append_cur_class.append(sentences_cur_class[i % len(sentences_cur_class)])
            sentence_each_class[class_index] = sentences_cur_class + append_cur_class
        ans_sentences = []
        ans_labels = []
        for class_index in range(self.number_classes):
            for s in sentence_each_class[class_index]:
                ans_sentences.append(s)
                ans_labels.append(class_index)
        sample_number_per_class = self.get_classes_count(ans_labels)
        self.logger.info('after balance, sample number each class:{}'.format(sample_number_per_class))
        self.logger.visdom_text(sample_number_per_class, win_name = 'sample_number_per_class', append = True)
        return ans_sentences, ans_labels


    def update_keywords(self, sentences, labels, incremental):
        self.logger.info('update keywords, keywords number per class:{}, IDF_n:{}'.format(
                self.cfg.keywords_update.extract_keywords_per_class, self.IDF_n))

        # sentences, labels = self.upsample_balance(sentences, labels)

        word_list, word_to_index = self.get_words_list(sentences)

        IDF_vector = self.cal_IDF(sentences, word_list, word_to_index)
        # LI = self.cal_LI(sentences, labels, word_list, word_to_index)
        # TF = self.cal_TF_normal_with_doc_num(sentences, labels, word_list, word_to_index)
        TF = self.cal_TF_normal_with_words_num(sentences, labels, word_list,
                                               word_to_index)

        # indicator = TF * IDF_vector
        indicator = TF * numpy.power(IDF_vector, self.IDF_n)

        diff = self.__update_keywords_with_indicator_M2__(indicator, word_list, incremental = incremental)

        self.keywords.syn_across_GPUs()

        return diff
