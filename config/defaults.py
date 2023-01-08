from yacs.config import CfgNode as CN

_C = CN()

_C.save_to_file = False
_C.save_best_model = True
_C.seed = 999
_C.data_dir_name = 'XXX'
_C.self_loop = True
_C.use_edge_cat_feature = True

_C.SSL = CN()
_C.SSL.enable = True
_C.SSL.batch_size = 50
_C.SSL.number_itr = 400000
_C.SSL.restart_prob = 0.1
_C.SSL.queue_size = 1600

_C.trainer_Graph = CN()
_C.trainer_Graph.epoch = [9]
_C.trainer_Graph.batch_size = 256

_C.GNN_model = 'GIN'  # 'GCN', 'GAT'

_C.GNN = CN()
_C.GNN.num_layers = 3
_C.GNN.num_mlp_layers = 2
_C.GNN.hidden_dim = 128
_C.GNN.final_dropout = 0.5
_C.GNN.learn_eps = False
_C.GNN.graph_pooling_type = "sum"
_C.GNN.neighbor_pooling_type = "sum"
_C.GNN.use_selayer = False

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
_C.train.save_steps = 100

_C.classifier = CN()
_C.classifier.type = 'long'  # 'short'
_C.classifier.batch_size = 8
_C.classifier.lr = 2e-6
_C.classifier.finetune_bert = True
_C.classifier.total_steps = 345
_C.classifier.n_epochs = 1
_C.classifier.stop_itr = [700]
_C.classifier.eval_interval = 100

_C.coteaching = CN()
_C.coteaching.n_epochs = 3

_C.model = CN()
_C.model.number_classes = 20

_C.file_path = CN()
_C.file_path.save_dir = 'exp'
_C.file_path.log_dir = r'log'

_C.file_path.best_model_name = 'best_model'
