import torch
from sklearn.metrics import f1_score, classification_report

from compent.comm import get_rank, accumulate_results_from_multiple_gpus


class Evaler_HAN():
    def __init__(self, cfg, logger, loader):
        self.cfg = cfg
        self.logger = logger
        self.loader = loader
        self.rank = get_rank()

    def __call__(self, model):
        self.logger.info('start eval')
        model.eval()
        with torch.no_grad():
            label_cur_GPU = []
            pred_cur_GPU = []
            sentence_cur_GPU = []
            for batch_input, batch_label, sentence in self.loader:
                num_sample = len(batch_label)
                batch_input = batch_input.cuda()
                # batch_label = batch_label.cuda()

                model.module._init_hidden_state(num_sample)

                outputs = model(batch_input)
                # pred_score = outputs.logits
                pred = torch.argmax(outputs, dim = 1).tolist()

                label_cur_GPU += batch_label.tolist()
                pred_cur_GPU += pred
                sentence_cur_GPU += sentence

        model.train()
        self.logger.visdom_text(text = 'before' + str(len(label_cur_GPU)), win_name = 'test GPU count')
        all_labels = accumulate_results_from_multiple_gpus(label_cur_GPU)
        self.logger.visdom_text(text = 'after' + str(len(all_labels)), win_name = 'test GPU count')
        all_preds = accumulate_results_from_multiple_gpus(pred_cur_GPU)
        all_sentences = accumulate_results_from_multiple_gpus(sentence_cur_GPU)

        self.logger.info('accumulate labels:{}, preds:{}'.format(len(all_labels), len(all_preds)))

        print(classification_report(y_true = all_labels, y_pred = all_preds))

        f1_macro = f1_score(y_true = all_labels, y_pred = all_preds, average = 'macro')
        f1_micro = f1_score(y_true = all_labels, y_pred = all_preds, average = 'micro')
        test_metrics = {}
        test_metrics['f1_macro'] = f1_macro
        test_metrics['f1_micro'] = f1_micro
        test_metrics['sentence'] = all_sentences
        test_metrics['all_preds'] = all_preds
        return test_metrics
