import os
import sys

import torch

sys.path.append('..')

from torch.multiprocessing import spawn

GPUs = '0,1'
cfg_file = 'chinese.yaml'
visdom_env_name = 'chinese'
os.environ['CUDA_VISIBLE_DEVICES'] = GPUs

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '10999'
world_size = len(GPUs.split(','))
device = torch.device('cuda')

# from Classifier.BERT.trainer_bert import Trainer_BERT
from Models.chinese.trainer_BERT_ST import Trainer_BERT

from PROJECT_ROOT import ROOT_DIR
from compent.logger import Logger
from config import cfg
from keyword_sentence.LFs import LFs
from keyword_sentence.sentence import Sentence_ALL
from compent.set_multi_GPUs import set_multi_GPUs_envs
from compent.comm import broadcast_data
from compent.utils import set_seed_all
from compent.saver import Saver

TOTAL_ITR = 1


def main(rank):
    set_seed_all(seed = 999)
    set_multi_GPUs_envs(rank, world_size)
    cfg_file_path = os.path.join(ROOT_DIR, 'config', cfg_file)
    cfg.merge_from_file(cfg_file_path)

    logger = Logger(name = visdom_env_name, save_dir = cfg.file_path.log_dir, distributed_rank = rank,
                    only_main_rank = False, visdom_port = 8888)
    logger.info('batch size:{}, number epoch:{}'.format(cfg.classifier.batch_size, cfg.classifier.n_epochs))
    logger.visdom_text(text = str(cfg), win_name = 'cfg')

    sentence_all = Sentence_ALL(cfg)
    # keywords = KeyWords(cfg = cfg, logger = logger)
    lfs = LFs(cfg, logger)
    # print(keywords)
    # keywords.load_keywords('keywords_itrOK_0')

    # GCN_trainer = Trainer_GCN(cfg = cfg, logger = logger, distributed = True, sentences_all = sentence_all,
    #                           keywords = keywords)
    bert_trainer = Trainer_BERT(cfg, logger, distributed = True, sentences_all = sentence_all)
    # updater = Keywords_Updater_TFIDF(keywords = keywords, cfg = cfg, logger = logger)

    saver = Saver(save_dir = cfg.file_path.save_dir, logger = logger)

    for cur_itr in range(TOTAL_ITR):
        logger.set_value(key = 'ITR', value = cur_itr)
        logger.info('iteration:{}, start'.format(cur_itr))
        logger.visdom_text(text = 'start ITR:{}'.format(cur_itr), win_name = 'state')
        logger.info('vote for test set')
        lfs.analyse_on_labeled(sentence_all = sentence_all.test_sentence, labels_all = sentence_all.test_label)
        logger.info('vote for unlabeled')
        vote_sentences, vote_labels = lfs.vote_for_all_sentences(sentence_all.unlabeled_sentence)

        # test
        # voted_sentences, voted_label, GT_labels = vote_all_unlabeled(sentence_all.unlabeled_sentence,
        #                                                              GT_label = sentence_all.unlabeled_GT_label)

        # logger.visdom_text(text = 'start training GIN', win_name = 'state')
        # res_dict = GCN_trainer.train_model(sentences = voted_sentences,
        #                                    vote_labels = voted_label,
        #                                    GT_labels = GT_labels, ITR = cur_itr)
        # res_dict = broadcast_data(res_dict)
        vote_sentences, vote_labels = broadcast_data(vote_sentences), broadcast_data(vote_labels)
        logger.visdom_text(text = 'start training BERT', win_name = 'state')
        sentences, labels = bert_trainer.train_model(sentences = vote_sentences,
                                                     labels = vote_labels,
                                                     finetune_from_pretrain = True, ITR = cur_itr)
        # sentences, labels = bert_trainer.train_model(sentences = res_dict['sentences'],
        #                                                    labels = res_dict['pred_labels'],
        #                                                    finetune_from_pretrain = True, ITR = cur_itr)
        # logger.visdom_text(text = 'start labeling with longformer', win_name = 'state')
        # sentences, labels = bert_trainer.do_label_sentences(sentences = sentence_all.unlabeled_sentence)
        # logger.info('main loop, do_label_sentences over'.format(rank))
        saver.save_to_file(obj = [sentences, labels], filename = 'sentence_label_itr_{}'.format(cur_itr))

        # logger.visdom_text(text = 'start update keywords', win_name = 'state')
        # diff = updater.update_keywords(sentences = sentences, labels = labels, incremental = False)
        # logger.plot_record(value = diff, win_name = 'keywords_diff')
        # logger.info('main loop, update_keywords over'.format(rank))
        # keywords.analyse_on_GTunlabel(sentence_all)
        #
        # keywords.dump_keyworks('keywords_{}'.format(cur_itr))
        # logger.info('main loop, analyse_on_GTunlabel over'.format(rank))
        # logger.info('----------------------------------------------------------------------\n\n\n\n\n')


if __name__ == '__main__':
    spawn(main, args = (), nprocs = world_size, join = True)
    print('finish')
