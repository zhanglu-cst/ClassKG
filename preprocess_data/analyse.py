import os

from PROJECT_ROOT import ROOT_DIR
from compent.logger import Logger
from compent.vote import Voter
from config import cfg
from keyword_sentence.keywords import KeyWords, Sentence_ALL

cfg_file = 'amazon.yaml'
cfg_file_path = os.path.join(ROOT_DIR, 'config_files', cfg_file)
cfg.merge_from_file(cfg_file_path)
logger = Logger(name = 'analyse', save_dir = cfg.file_path.log_dir, distributed_rank = 0)

keywords = KeyWords(cfg = cfg, logger = logger)
sentence_all = Sentence_ALL(cfg)
voter = Voter(cfg, keywords)

keywords.analyse_on_GTunlabel(sentence_all)
