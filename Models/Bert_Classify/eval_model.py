import torch
from sklearn.metrics import accuracy_score, f1_score

from compent.utils import move_to_device

device = torch.device('cuda')


class Eval_Model():
    def __init__(self, cfg, logger, distributed, rank, dataloader_eval, model):
        self.logger = logger
        self.dataloader = dataloader_eval
        self.rank = rank
        self.distributed = distributed
        self.model = model

    def __call__(self):
        if (self.rank != 0):
            return
        self.model.eval()
        with torch.no_grad():
            all_label = []
            all_pred = []
            for batch in self.dataloader:
                batch = move_to_device(batch)
                label_id = batch['label_id'].tolist()
                pred_score = self.model(batch)
                pred = torch.argmax(pred_score, dim = 1).tolist()
                all_label += label_id
                all_pred += pred
        acc = accuracy_score(y_true = all_label, y_pred = all_pred)
        f1_macro = f1_score(y_true = all_label, y_pred = all_pred, average = 'macro')
        f1_micro = f1_score(y_true = all_label, y_pred = all_pred, average = 'micro')
        self.logger.info('ACC:{}'.format(acc))
        self.logger.info('f1_macro:{}'.format(f1_macro))
        self.logger.info('f1_micro:{}'.format(f1_micro))
        self.model.train()
        return {'acc': acc, 'f1_macro': f1_macro, 'f1_micro': f1_micro}
        # return {'acc': acc, 'f11': f11, 'f10': f10}
