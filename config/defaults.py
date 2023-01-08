import os

from yacs.config import CfgNode as CN

_C = CN()

_C.save_to_file = False
_C.save_best_model = True
_C.seed = 43
_C.task_name = 'XXX'
_C.filter_thr_to_train_graph = 0.9
_C.self_loop = True
_C.use_edge_cat_feature = True
_C.max_len = 128

_C.soft_label = False

_C.SSL = CN()
_C.SSL.enable = False
_C.SSL.batch_size = 50
_C.SSL.number_itr = 400000
_C.SSL.restart_prob = 0.1
_C.SSL.queue_size = 1600

_C.trainer_Graph = CN()
_C.trainer_Graph.epoch = [9]
_C.trainer_Graph.batch_size = 256

_C.GNN_model = 'GIN'  # 'GCN', 'GAT'

_C.GIN = CN()
_C.GIN.num_layers = 3
_C.GIN.num_mlp_layers = 2
_C.GIN.hidden_dim = 128
_C.GIN.final_dropout = 0.5
_C.GIN.learn_eps = False
_C.GIN.graph_pooling_type = "sum"
_C.GIN.neighbor_pooling_type = "sum"
_C.GIN.use_selayer = False

_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (700, 1500)
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 100
_C.SOLVER.WARMUP_METHOD = "linear"

_C.keywords_update = CN()
_C.keywords_update.IDF_n = 4
_C.keywords_update.extract_keywords_per_class = [100]
_C.keywords_update.keywords_set_keep_max_num = [100]
_C.keywords_update.overwrite_conflict = False
_C.keywords_update.vote_need_no_confuse = False
_C.keywords_update.score_thr = 0.0
_C.keywords_update.seed_word_score = 3

_C.train = CN()
# _C.train.batch_size = 16
# _C.train.n_epochs = 5
_C.train.save_steps = 100
# _C.train.total_steps = 345

# _C.finetune = CN()
# _C.finetune.n_epochs = 5

_C.classifier = CN()
_C.classifier.batch_size = 8
_C.classifier.lr = 2e-6
_C.classifier.finetune_bert = True
_C.classifier.total_steps = 345
_C.classifier.n_epochs = 1
_C.classifier.stop_itr = [700]
_C.classifier.eval_interval = 100
# _C.classifier.upsample_balance = True
# _C.classifier.lr = 2e-05
# _C.classifier.warmup = 0.1


_C.coteaching = CN()
_C.coteaching.n_epochs = 3

_C.model = CN()
_C.model.vocab_size = 30522
_C.model.dim = 768
_C.model.n_layers = 12
_C.model.n_heads = 12
_C.model.dim_ff = 3072
_C.model.p_drop_hidden = 0.1
_C.model.p_drop_attn = 0.1
_C.model.max_len = 512
_C.model.n_segments = 2
_C.model.number_classes = 20
_C.model.model_name = 'uer/chinese_roberta_L-12_H-768'

_C.data = CN()
# _C.data.input_mode = 'pair'  # pair single
_C.data.max_len = 128

_C.file_path = CN()

_C.file_path.pretrain_bert_dir = r'/home/zhanglu/tmp/uncased_L-12_H-768_A-12/'
_C.do_lower_case = True
_C.file_path.save_dir = 'exp'
_C.file_path.log_dir = r'log'
_C.file_path.glove = r'/home/zhanglu/glove_all.txt'

_C.file_path.best_model_name = 'best_model'
_C.file_path.pretrain_model_path = os.path.join(_C.file_path.pretrain_bert_dir, 'bert_model.ckpt')
_C.file_path.vocab_path = os.path.join(_C.file_path.pretrain_bert_dir, 'vocab.txt')
