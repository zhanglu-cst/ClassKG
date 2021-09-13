import numpy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from Models.Longformer_Classify.dataset_for_long import Collect_FN, Dataset_Long
from compent.comm import get_world_size, get_rank


class Trainer_Base():
    def __init__(self, cfg, logger, distributed):
        super(Trainer_Base, self).__init__()
        self.cfg = cfg
        self.logger = logger
        self.world_size = get_world_size()
        self.distributed = distributed
        self.number_classes = cfg.model.number_classes
        self.stop_itr_list = cfg.classifier.stop_itr
        if (self.distributed == True):
            assert self.world_size > 1
            self.rank = get_rank()

    def get_stop_itr(self, ITR):
        if (ITR < len(self.stop_itr_list)):
            return self.stop_itr_list[ITR]
        else:
            return self.stop_itr_list[-1]

    def __build_dataloader_eval_longformer__(self, sentences, labels, for_train):
        collect_fn = Collect_FN(labels is not None)
        dataset = Dataset_Long(sentences, labels)
        sampler = DistributedSampler(dataset, shuffle = for_train)
        dataloader = DataLoader(dataset, batch_size = 4, sampler = sampler,
                                collate_fn = collect_fn)
        return dataloader

    # def train_model(self, sentences, labels, ITR, GT_label = None, finetune_from_pretrain = True):
    # raise NotImplementedError

    def get_classes_count(self, labels):  # Move to base
        ans = numpy.zeros(self.number_classes, dtype = numpy.int32)
        arr_labels = numpy.array(labels)
        for class_index in range(self.number_classes):
            cnt_cur = numpy.sum(arr_labels == class_index)
            ans[class_index] = cnt_cur
        return ans

    def data_aug(self, sentences, labels):
        pass

    def upsample_balance(self, sentences, labels):
        self.logger.visdom_text(text = 'upsample for classifier', win_name = 'state', append = True)
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

    def upsample_balance_with_one_extra(self, sentences, labels, GT_labels):

        sample_number_per_class = self.get_classes_count(labels)
        self.logger.info('berfor balance, sample number each class:{}'.format(sample_number_per_class))
        self.logger.visdom_text('label:' + str(sample_number_per_class), win_name = 'sample_number_per_class',
                                append = True)
        max_number = numpy.max(sample_number_per_class)
        fill_number_each_class = max_number - sample_number_per_class
        sentence_gt_each_class = [[] for i in range(self.number_classes)]
        for s, l, gt in zip(sentences, labels, GT_labels):
            sentence_gt_each_class[l].append((s, gt))
        for class_index, (sentences_gt_cur_class, fill_num_cur_class) in enumerate(
                zip(sentence_gt_each_class, fill_number_each_class)):
            append_cur_class = []
            for i in range(fill_num_cur_class):
                append_cur_class.append(sentences_gt_cur_class[i % len(sentences_gt_cur_class)])
            sentence_gt_each_class[class_index] = sentences_gt_cur_class + append_cur_class
        ans_sentences = []
        ans_labels = []
        ans_GT = []
        for class_index in range(self.number_classes):
            for s_gt in sentence_gt_each_class[class_index]:
                ans_sentences.append(s_gt[0])
                ans_GT.append(s_gt[1])
                ans_labels.append(class_index)
        sample_number_per_class = self.get_classes_count(ans_labels)
        self.logger.info('after balance, sample number each class:{}'.format(sample_number_per_class))
        self.logger.visdom_text(sample_number_per_class, win_name = 'sample_number_per_class', append = True)
        # sample_number_GT_per_class = self.get_classes_count(ans_GT)
        # self.logger.visdom_text(sample_number_GT_per_class, win_name = 'sample_number_per_class')
        return ans_sentences, ans_labels, ans_GT

    def upsample_balance_with_GT_soft(self, sentences, hard_labels, GT_labels, soft_labels):
        sample_number_per_class = self.get_classes_count(hard_labels)
        self.logger.info('berfor balance, sample number each class:{}'.format(sample_number_per_class))
        self.logger.visdom_text('label:' + str(sample_number_per_class), win_name = 'sample_number_per_class',
                                append = True)
        max_number = numpy.max(sample_number_per_class)
        fill_number_each_class = max_number - sample_number_per_class
        sentence_gts_each_class = [[] for i in range(self.number_classes)]
        for s, hl, gt, sl in zip(sentences, hard_labels, GT_labels, soft_labels):
            sentence_gts_each_class[hl].append((s, gt, sl))
        for class_index, (sentences_gts_cur_class, fill_num_cur_class) in enumerate(
                zip(sentence_gts_each_class, fill_number_each_class)):
            append_cur_class = []
            for i in range(fill_num_cur_class):
                append_cur_class.append(sentences_gts_cur_class[i % len(sentences_gts_cur_class)])
            sentence_gts_each_class[class_index] = sentences_gts_cur_class + append_cur_class
        ans_sentences = []
        ans_hard = []
        ans_GT = []
        ans_soft = []
        for class_index in range(self.number_classes):
            for s_gts in sentence_gts_each_class[class_index]:
                ans_sentences.append(s_gts[0])
                ans_GT.append(s_gts[1])
                ans_soft.append(s_gts[2])
                ans_hard.append(class_index)
        sample_number_per_class = self.get_classes_count(ans_hard)
        self.logger.info('after balance, sample number each class:{}'.format(sample_number_per_class))
        self.logger.visdom_text(sample_number_per_class, win_name = 'sample_number_per_class', append = True)
        sample_number_GT_per_class = self.get_classes_count(ans_GT)
        self.logger.visdom_text(sample_number_GT_per_class, win_name = 'sample_number_per_class')
        return ans_sentences, ans_hard, ans_soft, ans_GT

    def coteaching_loss(self, pred_1, pred_2, labels, GT_labels, num_remember):
        loss_1 = F.cross_entropy(pred_1, labels, reduction = 'none')
        sort_index_1 = torch.argsort(loss_1)

        loss_2 = F.cross_entropy(pred_2, labels, reduction = 'none')
        sort_index_2 = torch.argsort(loss_2)

        # print('num_remember:{}'.format(num_remember))
        index_1_update = sort_index_1[:num_remember].long()
        index_2_update = sort_index_2[:num_remember].long()

        # print('index_1_update:{}'.format(index_1_update))
        # print('index_1_update dtype:{}'.format(index_1_update.dtype))
        # print('index_1_update shape:{}'.format(index_1_update.shape))
        # print('labels shape:{}'.format(labels.shape))

        pure_rate_1 = torch.sum(labels[index_1_update] == GT_labels[index_1_update]) / float(num_remember)
        pure_rate_2 = torch.sum(labels[index_2_update] == GT_labels[index_2_update]) / float(num_remember)

        # loss_1_update = F.cross_entropy(pred_1[index_2_update], labels[index_2_update])
        # loss_2_update = F.cross_entropy(pred_2[index_1_update], labels[index_1_update])
        # self.logger.info('updated labels:{}, GT_labels:{}, labels2:{}, GT_labels2:{}'.format(labels[index_1_update],
        #                                                                                      GT_labels[index_1_update],
        #                                                                                      labels[index_2_update],
        #                                                                                      GT_labels[index_2_update]))

        loss_1_update = F.cross_entropy(pred_1[index_2_update], labels[index_2_update])
        loss_2_update = F.cross_entropy(pred_2[index_1_update], labels[index_1_update])

        return loss_1_update, loss_2_update, pure_rate_1, pure_rate_2

    def coteaching_loss_GT_test(self, pred_1, pred_2, labels, GT_labels, num_remember):
        loss_1 = F.cross_entropy(pred_1, labels, reduction = 'none')
        sort_index_1 = torch.argsort(loss_1)

        loss_2 = F.cross_entropy(pred_2, labels, reduction = 'none')
        sort_index_2 = torch.argsort(loss_2)

        # print('num_remember:{}'.format(num_remember))
        index_1_update = sort_index_1[:num_remember].long()
        index_2_update = sort_index_2[:num_remember].long()

        # print('index_1_update:{}'.format(index_1_update))
        # print('index_1_update dtype:{}'.format(index_1_update.dtype))
        # print('index_1_update shape:{}'.format(index_1_update.shape))
        # print('labels shape:{}'.format(labels.shape))

        pure_rate_1 = torch.sum(labels[index_1_update] == GT_labels[index_1_update]) / float(num_remember)
        pure_rate_2 = torch.sum(labels[index_2_update] == GT_labels[index_2_update]) / float(num_remember)

        # loss_1_update = F.cross_entropy(pred_1[index_2_update], labels[index_2_update])
        # loss_2_update = F.cross_entropy(pred_2[index_1_update], labels[index_1_update])

        loss_1_update = F.cross_entropy(pred_1[index_2_update], GT_labels[index_2_update])
        loss_2_update = F.cross_entropy(pred_2[index_1_update], GT_labels[index_1_update])

        return loss_1_update, loss_2_update, pure_rate_1, pure_rate_2

    def do_label_sentences(self, sentences):

        raise NotImplementedError
