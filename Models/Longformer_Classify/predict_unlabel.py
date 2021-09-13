import numpy
import torch

from compent.comm import accumulate_results_from_multiple_gpus, synchronize
from compent.utils import move_to_device

device = torch.device('cuda')


class Predicter():
    def __init__(self, cfg, logger, distributed, rank, dataloader_sentence, model):
        super(Predicter, self).__init__()
        self.logger = logger
        self.dataloader = dataloader_sentence
        self.rank = rank
        self.distributed = distributed
        self.model = model
        self.number_classes = cfg.model.number_classes
        assert self.distributed == True

    def __call__(self):

        print('rank:{},start predict'.format(self.rank))
        self.model.eval()
        with torch.no_grad():
            sentence_cur_GPU = []
            pred_cur_GPU = []
            for batch in self.dataloader:
                batch = move_to_device(batch)
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
                pred_score = outputs.logits
                pred = torch.argmax(pred_score, dim = 1).tolist()
                sentence_cur_GPU += batch['sentences']
                pred_cur_GPU += pred
        self.model.train()

        synchronize()
        all_sentences = accumulate_results_from_multiple_gpus(sentence_cur_GPU)
        all_pred_labels = accumulate_results_from_multiple_gpus(pred_cur_GPU)
        print('rank:{}, predict over, total count:{}'.format(self.rank, len(all_pred_labels)))
        print('longformer predict count:')
        array_pred = numpy.array(all_pred_labels)
        for class_index in range(self.number_classes):
            print('class:{}, pred count:{}'.format(class_index, numpy.sum(array_pred == class_index)))

        return all_sentences, all_pred_labels
