import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class Dataset_BERT(Dataset):
    def __init__(self, sentences, labels_hard = None, GT_labels = None):
        super(Dataset_BERT, self).__init__()
        self.sentences = sentences
        self.labels_hard = labels_hard
        self.GT_labels = GT_labels

    def __getitem__(self, index):
        if (self.labels_hard and self.GT_labels is None):
            return self.sentences[index], self.labels_hard[index]
        elif (self.labels_hard and self.GT_labels):
            return self.sentences[index], self.labels_hard[index], self.GT_labels[index]
        else:
            return self.sentences[index]

    def __len__(self):
        return len(self.sentences)


class Collect_FN():
    def __init__(self, with_label_hard, with_GT_labels = False):
        super(Collect_FN, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.with_label = with_label_hard
        self.with_GT_labels = with_GT_labels

    def __call__(self, batchs):
        # print(batchs)
        if (self.with_label and self.with_GT_labels == False):
            sentences, labels = map(list, zip(*batchs))
        elif (self.with_label == True and self.with_GT_labels == True):
            sentences, labels, GT_labels = map(list, zip(*batchs))
        else:
            sentences = batchs
        encoding = self.tokenizer(sentences, return_tensors = 'pt', padding = True, truncation = True)
        # input_ids = encoding['input_ids']
        # attention_mask = encoding['attention_mask']
        # ans = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if (self.with_label):
            labels = torch.tensor(labels).long()
            encoding['labels'] = labels
        if (self.with_GT_labels):
            GT_labels = torch.tensor(GT_labels).long()
            encoding['GT_labels'] = GT_labels
        encoding['sentences'] = sentences
        return encoding
