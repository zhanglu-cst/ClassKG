import csv
import os
import pickle

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

dict_path = r"/remote-home/XXX/glove.6B.50d.txt"


class HAN_Dataset(Dataset):

    def __init__(self, logger, cfg, setences, labels):
        super(HAN_Dataset, self).__init__()
        self.logger = logger
        self.texts = setences
        self.labels = labels
        self.cfg = cfg
        self.dict = pd.read_csv(filepath_or_buffer = dict_path, header = None, sep = " ", quoting = csv.QUOTE_NONE,
                                usecols = [0]).values
        self.dict = [word[0] for word in self.dict]

        self.num_classes = len(set(self.labels))

        self.process_text = []
        self.logger.visdom_text('init data', win_name = 'state')

        save_dir = self.cfg.file_path.save_dir
        path_sentence_to_token = os.path.join(save_dir, 'dict_s_token.pkl')

        with open(path_sentence_to_token, 'rb') as f:
            s_to_token = pickle.load(f)

        self.max_length_sentences = s_to_token['max_sentence']
        self.max_length_word = s_to_token['max_word']

        for text in self.texts:
            token_one = s_to_token[text]
            self.process_text.append(token_one)

        # for text in self.texts:
        #     document_encode = [
        #         [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text = sentences)] for
        #         sentences
        #         in
        #         sent_tokenize(text = text)]
        #
        #     for sentences in document_encode:
        #         if len(sentences) < self.max_length_word:
        #             extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
        #             sentences.extend(extended_words)
        #
        #     if len(document_encode) < self.max_length_sentences:
        #         extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
        #                               range(self.max_length_sentences - len(document_encode))]
        #         document_encode.extend(extended_sentences)
        #
        #     document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
        #                       :self.max_length_sentences]
        #
        #     document_encode = np.stack(arrays = document_encode, axis = 0)
        #     document_encode += 1
        #     cur_res = document_encode.astype(np.int64)
        #     self.process_text.append(cur_res)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text_processed = self.process_text[index]
        sentence = self.texts[index]
        return text_processed, label, sentence


def collate_fn(batchs):
    text_processed, label, sentence = map(list, zip(*batchs))
    text_processed = torch.tensor(text_processed).long()
    label = torch.tensor(label).long()
    return text_processed, label, sentence

