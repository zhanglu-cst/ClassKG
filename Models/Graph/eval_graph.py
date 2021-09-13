import torch
from sklearn.metrics import f1_score, classification_report, accuracy_score

from compent.comm import accumulate_results_from_multiple_gpus, synchronize, is_main_process
from compent.utils import move_to_device

device = torch.device('cuda')


#
# class Eval_Model_For_Graph():
#     def __init__(self, cfg, logger, distributed, rank, dataloader_eval):
#         self.logger = logger
#         self.dataloader = dataloader_eval
#         self.rank = rank
#         self.distributed = distributed
#         assert self.distributed == True
#
#     def __call__(self, model):
#         self.logger.info('start eval graphs')
#         model.eval()
#         # self.dataloader.dataset.eval()
#
#         with torch.no_grad():
#             label_cur_GPU = []
#             pred_cur_GPU = []
#             for batch in self.dataloader:
#                 batch = move_to_device(batch, rank = self.rank)
#                 labels = batch['labels'].tolist()
#                 graph = batch['graphs']
#                 outputs = model(graph)
#                 pred = torch.argmax(outputs, dim = 1).tolist()
#                 label_cur_GPU += labels
#                 pred_cur_GPU += pred
#
#         model.train()
#         # self.dataloader.dataset.train()
#
#         synchronize()
#         all_labels = accumulate_results_from_multiple_gpus(label_cur_GPU)
#         all_preds = accumulate_results_from_multiple_gpus(pred_cur_GPU)
#
#         if not is_main_process():
#             return None
#
#         self.logger.info('accumulate labels:{}, preds:{}'.format(len(all_labels), len(all_preds)))
#
#         # acc = accuracy_score(y_true = all_labels, y_pred = all_preds)
#         # f1_macro = f1_score(y_true = all_labels, y_pred = all_preds, average = 'macro')
#         f1_micro = f1_score(y_true = all_labels, y_pred = all_preds, average = 'micro')
#         f1_macro = f1_score(y_true = all_labels, y_pred = all_preds, average = 'macro')
#         # self.logger.info('ACC:{}'.format(acc))
#         # self.logger.info('f1_macro:{}'.format(f1_macro))
#         self.logger.info('f1_micro:{}'.format(f1_micro))
#         print('Eval Model On Eval Set:')
#         eval_text = classification_report(y_true = all_labels, y_pred = all_preds)
#         # self.logger.visdom_text(text = 'eval model on eval set\n{}'.format(eval_text), win_name = 'eval_text')
#         print(eval_text)
#         print('Eval Model On Eval Set:')
#         return {'f1_micro': f1_micro, 'f1_macro': f1_macro}


class Eval_Model_On_Labeling_Quality():
    def __init__(self, cfg, logger, distributed, rank, dataloader_train):
        self.logger = logger
        self.dataloader = dataloader_train
        self.rank = rank
        self.distributed = distributed
        assert self.distributed == True

    def __call__(self, model):
        self.logger.info('start Eval_Model_On_Labeling_Quality')
        model.eval()

        with torch.no_grad():
            label_cur_GPU = []
            pred_cur_GPU = []
            sentences_cur_GPU = []
            soft_label_cur_GPU = []
            for batch in self.dataloader:
                batch = move_to_device(batch, rank = self.rank)
                GT_labels = batch['GT_labels'].tolist()
                graph = batch['graphs']
                sentences = batch['sentences']
                outputs = model(graph)
                pred = torch.argmax(outputs, dim = 1).tolist()
                pred_soft = outputs.cpu().tolist()
                label_cur_GPU += GT_labels
                pred_cur_GPU += pred
                soft_label_cur_GPU += pred_soft
                sentences_cur_GPU += sentences

        model.train()

        synchronize()
        all_labels = accumulate_results_from_multiple_gpus(label_cur_GPU)
        all_preds = accumulate_results_from_multiple_gpus(pred_cur_GPU)
        all_sentences = accumulate_results_from_multiple_gpus(sentences_cur_GPU)
        all_soft_preds = accumulate_results_from_multiple_gpus(soft_label_cur_GPU)
        if not is_main_process():
            return None

        self.logger.info(
                'accumulate on all unlabeled sentences labels:{}, preds:{}'.format(len(all_labels), len(all_preds)))

        acc = accuracy_score(y_true = all_labels, y_pred = all_preds)
        f1_macro = f1_score(y_true = all_labels, y_pred = all_preds, average = 'macro')
        f1_micro = f1_score(y_true = all_labels, y_pred = all_preds, average = 'micro')
        self.logger.info('ACC:{}'.format(acc))
        self.logger.info('f1_micro:{}'.format(f1_micro))
        # self.logger.plot_record(value = acc, win_name = 'ACC on unlabeled labeling quality')
        # self.logger.plot_record(value = f1_micro, win_name = 'f1_micro on unlabeled labeling quality')

        print('Eval Model On All unlabeled:')
        eval_text = classification_report(y_true = all_labels, y_pred = all_preds)
        print(eval_text)
        print('Eval Model On All unlabeled:')
        return {'f1_micro': f1_micro, 'f1_macro': f1_macro, 'sentences': all_sentences, 'pred': all_preds,
                'soft_pred': all_soft_preds, 'GT_labels': all_labels}
