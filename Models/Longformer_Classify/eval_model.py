import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

from compent.comm import accumulate_results_from_multiple_gpus, synchronize, is_main_process
from compent.utils import move_to_device

device = torch.device('cuda')


class Eval_Model_For_Long():
    def __init__(self, cfg, logger, distributed, rank, dataloader_eval):
        self.logger = logger
        self.dataloader = dataloader_eval
        self.rank = rank
        self.distributed = distributed
        assert self.distributed == True

    def __call__(self, model):
        self.logger.info('start eval')
        model.eval()
        with torch.no_grad():
            label_cur_GPU = []
            pred_cur_GPU = []
            sentence_cur_GPU = []
            for batch in self.dataloader:
                batch = move_to_device(batch)
                label_id = batch['labels'].tolist()
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                sentence_cur_GPU += batch['sentences']

                outputs = model(input_ids = input_ids, attention_mask = attention_mask)
                pred_score = outputs.logits
                pred = torch.argmax(pred_score, dim = 1).tolist()
                label_cur_GPU += label_id
                pred_cur_GPU += pred

        model.train()

        synchronize()
        all_labels = accumulate_results_from_multiple_gpus(label_cur_GPU)
        all_preds = accumulate_results_from_multiple_gpus(pred_cur_GPU)
        all_sentences = accumulate_results_from_multiple_gpus(sentence_cur_GPU)


        self.logger.info('accumulate labels:{}, preds:{}'.format(len(all_labels), len(all_preds)))

        acc = accuracy_score(y_true = all_labels, y_pred = all_preds)
        f1_macro = f1_score(y_true = all_labels, y_pred = all_preds, average = 'macro')
        f1_micro = f1_score(y_true = all_labels, y_pred = all_preds, average = 'micro')
        self.logger.info('ACC:{}'.format(acc))
        self.logger.info('f1_macro:{}'.format(f1_macro))
        self.logger.info('f1_micro:{}'.format(f1_micro))
        print('Eval Model On Eval Set:')
        print(classification_report(y_true = all_labels, y_pred = all_preds))
        print('Eval Model On Eval Set:')
        return {'f1_micro': f1_micro, 'f1_macro': f1_macro, 'sentences': all_sentences, 'preds': all_preds}
