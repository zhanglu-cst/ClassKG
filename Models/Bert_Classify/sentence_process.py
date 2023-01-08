import torch
from torch.utils.data import Dataset

from Models.Bert_Classify import tokenization_bert


class Dataset_For_Single_Sentence_Predict(Dataset):
    def __init__(self, cfg, sentences, labels):
        super(Dataset_For_Single_Sentence_Predict, self).__init__()
        max_len = cfg.max_len
        print('max sentence length:{}'.format(max_len))
        tokenizer = tokenization_bert.FullTokenizer(vocab_file = cfg.file_path.vocab_path, do_lower_case = cfg.do_lower_case)

        pipeline = [
            Tokenizing(tokenizer, max_len = max_len),
            TokenIndexing(tokenizer, max_len = max_len)
        ]

        if (labels is None):
            lines = sentences
        else:
            lines = [(s, l) for s, l in zip(sentences, labels)]
        data = []

        for instance in lines:
            for proc in pipeline:
                instance = proc(instance)
            data.append(instance)

        self.tensors = [torch.tensor(x, dtype = torch.long) for x in zip(*data)]

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return {'input_ids': self.tensors[0][index], 'segment_ids': self.tensors[1][index],
                'input_mask': self.tensors[2][index], 'label_id': self.tensors[3][index]}


class Tokenizing():
    """ Tokenizing sentence pair """

    def __init__(self, tokenizer, max_len):
        super(Tokenizing, self).__init__()
        self.preprocessor = tokenizer.convert_to_unicode  # e.graphs. text normalization
        self.tokenize = tokenizer.tokenize  # tokenize function
        self.max_len = max_len

    def __call__(self, instance):
        if (isinstance(instance, str)):
            label = -1
            text_a = instance
        else:
            text_a, label = instance

        # labels = self.preprocessor(labels)
        tokens_a = self.tokenize(self.preprocessor(text_a))

        if (len(tokens_a) > self.max_len):
            tokens_a = tokens_a[:self.max_len]

        return tokens_a, label


class TokenIndexing():
    """ Convert tokens into token indexes and do zero-padding """

    def __init__(self, tokenizer, max_len):
        super(TokenIndexing, self).__init__()
        self.indexer = tokenizer.convert_tokens_to_ids  # function : tokens to indexes
        # map from a labels name to a labels index
        # self.label_map = {name: i for i, name in enumerate(labels)}
        self.max_len = max_len

    def __call__(self, instance):
        tokens_a, label = instance
        input_ids = self.indexer(tokens_a)
        segment_ids = [0] * len(tokens_a)  # token type ids
        input_mask = [1] * len(tokens_a)

        # label_id = self.label_map[labels]

        # zero padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)
        input_mask.extend([0] * n_pad)

        return (input_ids, segment_ids, input_mask, label)
