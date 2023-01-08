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

from PROJECT_ROOT import ROOT_DIR
from compent.logger import Logger
from config import cfg
from keyword_sentence.sentence import Sentence_ALL
from compent.set_multi_GPUs import set_multi_GPUs_envs
from compent.utils import set_seed_all
from keyword_sentence.keywords_hierarchical import Hierarchical_Keywords
from Models.chinese.trainer_hierachical import build_model, Trainer_HIERA

TOTAL_ITR = 1


def train(cur_road, cfg, logger, hiera_keywords, sentences_belong_cur_father):
    print('start training:{}'.format(cur_road))
    if (len(cur_road) == 5):
        return
    else:
        childrens = hiera_keywords.get_children_cur_road(cur_road = cur_road)
        if (len(childrens) == 1):
            cur_road.append(childrens[0])
            train(cur_road, cfg, logger, hiera_keywords, sentences_belong_cur_father)
            cur_road.pop()
        else:
            model_cur_road = build_model(number_class = len(childrens), cfg = cfg)
            trainer = Trainer_HIERA(cfg = cfg, logger = logger, hiera_keywords = hiera_keywords,
                                    father_road = cur_road)
            sentences_each_child = trainer.train_model(model = model_cur_road,
                                                       sentences_belong_cur_father = sentences_belong_cur_father)
            for one_child, sentence_cur_child in zip(childrens, sentences_each_child):
                cur_road.append(one_child)
                train(cur_road, cfg, logger, hiera_keywords, sentence_cur_child)
                cur_road.pop()


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
    hiera_keywords = Hierarchical_Keywords(cfg, logger, sentence_all)
    top = list(hiera_keywords.class_each_hierarchical[0])
    train(cur_road = top, cfg = cfg, logger = logger, hiera_keywords = hiera_keywords,
          sentences_belong_cur_father = sentence_all.unlabeled_sentence)

    # saver = Saver(save_dir = cfg.file_path.save_dir, logger = logger)
    #     saver.save_to_file(obj = [sentences, labels], filename = 'sentence_label_itr_{}'.format(cur_itr))


if __name__ == '__main__':
    spawn(main, args = (), nprocs = world_size, join = True)
    print('finish')
