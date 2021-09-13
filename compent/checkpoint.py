# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Load a checkpoint file of pretrained transformer to a model in pytorch """

import os
import pickle
import time

import numpy as np
import tensorflow as tf
import torch

from compent.utils import make_dirs


# import ipdb
# from models import *

def load_param(checkpoint_file, conversion_table):
    """
    Load parameters in pytorch model from checkpoint file according to conversion_table
    checkpoint_file : pretrained checkpoint model file in tensorflow
    conversion_table : { pytorch tensor in a model : checkpoint variable name }
    """
    for pyt_param, tf_param_name in conversion_table.items():
        tf_param = tf.train.load_variable(checkpoint_file, tf_param_name)

        # for weight(kernel), we should do transpose
        if tf_param_name.endswith('kernel'):
            tf_param = np.transpose(tf_param)

        assert pyt_param.size() == tf_param.shape, \
            'Dim Mismatch: %s vs %s ; %s' % \
            (tuple(pyt_param.size()), tf_param.shape, tf_param_name)

        # assign pytorch tensor from tensorflow param
        pyt_param.data = torch.from_numpy(tf_param)


def load_model(model, checkpoint_file):
    """ Load the pytorch model from checkpoint file """

    # Embedding layer
    e, p = model.embed, 'bert/embeddings/'
    load_param(checkpoint_file, {
        e.tok_embed.weight: p + "word_embeddings",
        e.pos_embed.weight: p + "position_embeddings",
        e.seg_embed.weight: p + "token_type_embeddings",
        e.norm.gamma: p + "LayerNorm/gamma",
        e.norm.beta: p + "LayerNorm/beta"
    })

    # Transformer blocks
    for i in range(len(model.blocks)):
        b, p = model.blocks[i], "bert/encoder/layer_%d/" % i
        load_param(checkpoint_file, {
            b.attn.proj_q.weight: p + "attention/self/query/kernel",
            b.attn.proj_q.bias: p + "attention/self/query/bias",
            b.attn.proj_k.weight: p + "attention/self/key/kernel",
            b.attn.proj_k.bias: p + "attention/self/key/bias",
            b.attn.proj_v.weight: p + "attention/self/value/kernel",
            b.attn.proj_v.bias: p + "attention/self/value/bias",
            b.proj.weight: p + "attention/output/dense/kernel",
            b.proj.bias: p + "attention/output/dense/bias",
            b.pwff.fc1.weight: p + "intermediate/dense/kernel",
            b.pwff.fc1.bias: p + "intermediate/dense/bias",
            b.pwff.fc2.weight: p + "output/dense/kernel",
            b.pwff.fc2.bias: p + "output/dense/bias",
            b.norm1.gamma: p + "attention/output/LayerNorm/gamma",
            b.norm1.beta: p + "attention/output/LayerNorm/beta",
            b.norm2.gamma: p + "output/LayerNorm/gamma",
            b.norm2.beta: p + "output/LayerNorm/beta",
        })


class CheckPointer_Normal():
    def __init__(self, cfg, logger, rank):
        super(CheckPointer_Normal, self).__init__()
        self.cfg = cfg
        self.logger = logger
        self.rank = rank
        self.save_dir = cfg.file_path.save_dir
        self.best_model_path = os.path.join(self.save_dir, self.cfg.file_path.best_model_name)
        make_dirs(self.save_dir)

    # def load_from_best_model(self, model, strict = True):
    #     path = self.best_model_path
    #     self.__load_from_file__(model, path, strict = strict)

    def load_from_filename(self, model, filename, strict = False):
        path = os.path.join(self.save_dir, filename)
        self.__load_from_file__(model, path, strict = strict)

    def __load_from_file__(self, model, path, strict):
        data = torch.load(path, map_location = 'cpu')
        model_time = data.pop('time')
        self.logger.info('Loading model from:{}, model save time:{}'.format(path, model_time))
        model_para = data.pop('model')
        keep_para = model_para

        # self.logger.info('load strict:{}'.format(strict))
        # if (strict == False):
        #     keep_para = {}
        #     for k in list(model_para.keys()):
        #         if (k.startswith('linears_prediction') == False):
        #             keep_para[k] = model_para[k]
        #     assert len(keep_para) + self.cfg.GIN.num_layers * 2 == len(model_para)
        # else:
        #     keep_para = model_para

        if (hasattr(model, 'module')):
            model.module.load_state_dict(keep_para, strict = strict)
        else:
            model.load_state_dict(keep_para, strict = strict)
        self.logger.info('Load model success, other info:{}'.format(data))

    def __save_to_path__(self, model, path, other_info):
        if (self.rank != 0):
            return
        self.logger.info('save to file path:{}'.format(path))
        data = {}
        if (hasattr(model, 'module')):
            data['model'] = model.module.state_dict()
        else:
            data['model'] = model.state_dict()
        data['time'] = time.ctime()
        if (other_info):
            data.update(other_info)
        # data['optimizer'] = self.optimizer.state_dict()
        torch.save(data, path)

    # def save_to_best_model_file(self, model, other_info = None):
    #     self.__save_to_path__(model, self.best_model_path, other_info)

    def save_to_checkpoint_file_with_name(self, model, filename, other_info = None):
        path = os.path.join(self.save_dir, str(filename))
        self.__save_to_path__(model, path, other_info)


class CheckPointer_Bert(CheckPointer_Normal):
    def __init__(self, cfg, logger, rank):
        super(CheckPointer_Bert, self).__init__(cfg, logger, rank)
        self.pretrain_filepath = cfg.file_path.pretrain_model_path

    def load_from_pretrainBERT(self, model):
        self.logger.info('Loading the pretrained BERT model from:{}'.format(self.pretrain_filepath))
        if self.pretrain_filepath.endswith('.ckpt'):  # checkpoint file in tensorflow
            load_model(model.transformer, self.pretrain_filepath)
        elif self.pretrain_filepath.endswith('.pt'):  # pretrain model file in pytorch
            model.transformer.load_state_dict(
                    {key[12:]: value
                     for key, value in torch.load(self.pretrain_filepath).items()
                     if key.startswith('transformer')}
            )
        else:
            raise Exception('error ending')


class Checkpointer_For_Sentence_Labels():
    def __init__(self, cfg, logger, rank):
        self.cfg = cfg
        self.logger = logger
        self.rank = rank
        self.save_dir = cfg.file_path.save_dir

    def save_to_file(self, sentences, labels, ITR):
        ans = {'sentences': sentences, 'labels': labels}
        path = os.path.join(self.save_dir, str(ITR))
        self.logger.info('save sentences_labels to file:{}'.format(path))
        with open(path, 'wb') as f:
            pickle.dump(ans, f)

    def load_from_file(self, ITR):
        path = os.path.join(self.save_dir, str(ITR))
        self.logger.info('load sentences_labels from file:{}'.format(path))
        with open(path, 'rb') as f:
            ans = pickle.load(f)
        return ans['sentences'], ans['labels']
