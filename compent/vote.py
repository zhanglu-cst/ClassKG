import numpy

from keyword_sentence.sentence_process import get_sentence_hit_keywords


class Voter():
    def __init__(self, cfg, keywords):
        super(Voter, self).__init__()
        self.number_classes = cfg.model.number_classes
        self.keywords = keywords

    def __call__(self, sentence, need_no_confuse = False, return_hit_counts = False):
        words, labels = get_sentence_hit_keywords(sentence, self.keywords, return_labels = True)
        count_vector = [0] * self.number_classes
        count_vector = numpy.array(count_vector)
        for per_label in labels:
            count_vector[per_label] += 1
        if (numpy.sum(count_vector) == 0):
            if (return_hit_counts):
                return None, None
            else:
                return None
        else:
            max_label = numpy.argmax(count_vector, axis = 0)
            if (need_no_confuse):
                if numpy.sum(count_vector == 0) == (len(count_vector) - 1):
                    if (return_hit_counts):
                        return max_label, len(labels)
                    else:
                        return max_label
                else:
                    if (return_hit_counts):
                        return None, None
                    else:
                        return None
            else:
                if (return_hit_counts):
                    return max_label, len(labels)
                else:
                    return max_label


class Vote_All_Unlabeled():
    def __init__(self, cfg, logger, keywords, need_no_confuse, need_static_hit_words = True):
        super(Vote_All_Unlabeled, self).__init__()
        self.cfg = cfg
        self.logger = logger
        self.voter = Voter(cfg, keywords)
        self.need_no_confuse = need_no_confuse
        self.number_classes = cfg.model.number_classes
        self.need_static_hit_words = need_static_hit_words

    def __call__(self, unlabeled_sentences, GT_label):
        res_sentences = []
        res_labels = []
        res_GT = []
        hit_counts = []
        for sentence, cur_GT in zip(unlabeled_sentences, GT_label):
            sentence = sentence.lower()
            if (self.need_static_hit_words):
                label, count_hit_words = self.voter(sentence, need_no_confuse = self.need_no_confuse,
                                                    return_hit_counts = True)
                if (label is not None):
                    hit_counts.append(count_hit_words)
            else:
                label = self.voter(sentence, need_no_confuse = self.need_no_confuse)

            if (label is not None):
                res_sentences.append(sentence)
                res_labels.append(label)
                res_GT.append(cur_GT)
        self.logger.info(
                'vote generate sentences:{}. total count:{}, cover:{}'.format(len(res_sentences),
                                                                              len(unlabeled_sentences),
                                                                              len(res_sentences) / len(
                                                                                      unlabeled_sentences)))

        if (self.need_static_hit_words):
            loc = numpy.mean(hit_counts)
            std = numpy.std(hit_counts)
            self.logger.set_value('loc', loc)
            self.logger.set_value('std', std)
            self.logger.plot_record(value = loc, win_name = 'keywords mean per s')
            self.logger.plot_record(value = std, win_name = 'keywords std  per s')

        array_label = numpy.array(res_labels)
        for label in range(self.number_classes):
            count_cur_label = numpy.sum(array_label == label)
            self.logger.info('labels:{}, count:{}'.format(label, count_cur_label))

        return res_sentences, res_labels, res_GT


class Voter_Soft(Voter):
    def __init__(self, cfg, keywords):
        super(Voter_Soft, self).__init__(cfg, keywords)

    def __call__(self, sentence, need_no_confuse = False, return_hit_counts = True):
        assert need_no_confuse == False and return_hit_counts == True
        words, labels = get_sentence_hit_keywords(sentence, self.keywords, return_labels = True)
        count_vector = [0] * self.number_classes
        count_vector = numpy.array(count_vector)
        for per_label in labels:
            count_vector[per_label] += 1
        if (numpy.sum(count_vector) == 0):
            return None, None, None
        else:
            sum_count = numpy.sum(count_vector)
            soft_label = count_vector / sum_count
            hard_label = numpy.argmax(count_vector, axis = 0)
            return soft_label, hard_label, len(labels)


class Vote_All_Unlabeled_Soft(Vote_All_Unlabeled):
    def __init__(self, cfg, logger, keywords, need_no_confuse = False, need_static_hit_words = True):
        assert need_static_hit_words == True and need_no_confuse == False
        super(Vote_All_Unlabeled_Soft, self).__init__(cfg, logger, keywords, need_no_confuse, need_static_hit_words)
        self.voter = Voter_Soft(cfg, keywords)

    def __call__(self, unlabeled_sentences, GT_label):
        res_sentences = []
        res_soft_labels = []
        res_hard_labels = []
        res_GT = []
        hit_counts = []
        for sentence, cur_GT in zip(unlabeled_sentences, GT_label):
            label_soft, hard_label, count_hit_words = self.voter(sentence, need_no_confuse = False,
                                                                 return_hit_counts = True)
            if (label_soft is not None):
                hit_counts.append(count_hit_words)
                res_sentences.append(sentence)
                res_soft_labels.append(label_soft)
                res_hard_labels.append(hard_label)
                res_GT.append(cur_GT)
        self.logger.info(
                'vote generate sentences:{}. total count:{}, cover:{}'.format(len(res_sentences),
                                                                              len(unlabeled_sentences),
                                                                              len(res_sentences) / len(
                                                                                      unlabeled_sentences)))

        if (self.need_static_hit_words):
            loc = numpy.mean(hit_counts)
            std = numpy.std(hit_counts)
            self.logger.set_value('loc', loc)
            self.logger.set_value('std', std)
            self.logger.plot_record(value = loc, win_name = 'keywords mean per s')
            self.logger.plot_record(value = std, win_name = 'keywords std  per s')

        array_label = numpy.array(res_hard_labels)
        for label in range(self.number_classes):
            count_cur_label = numpy.sum(array_label == label)
            self.logger.info('vote hard labels:{}, count:{}'.format(label, count_cur_label))

        return res_sentences, res_soft_labels, res_hard_labels, res_GT


def build_voter_all(cfg, logger, keywords, need_no_confuse = False, need_static_hit_words = True):
    if (cfg.soft_label):
        return Vote_All_Unlabeled_Soft(cfg, logger, keywords, need_no_confuse, need_static_hit_words)
    else:
        return Vote_All_Unlabeled(cfg, logger, keywords, need_no_confuse, need_static_hit_words)
