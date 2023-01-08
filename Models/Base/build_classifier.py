from Models.BERT.trainer_bert_ST import Trainer_BERT
from Models.Longformer_Classify.trainer_longformer_ST import Trainer_Longformer


def build_classifier_with_cfg(cfg):
    if (cfg.classifier.type == 'long'):
        return Trainer_Longformer
    elif (cfg.classifier.type == 'short'):
        return Trainer_BERT
    else:
        raise NotImplementedError
