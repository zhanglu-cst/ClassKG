import torch
from torch.utils.data import Dataset
from transformers import LongformerTokenizer


class Dataset_Long(Dataset):
    def __init__(self, sentences, labels, GT_labels):
        super(Dataset_Long, self).__init__()
        self.sentences = sentences
        self.labels = labels
        self.GT_labels = GT_labels

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index], self.GT_labels[index]

    def __len__(self):
        return len(self.sentences)


class Collect_FN_CoT():
    def __init__(self):
        super(Collect_FN_CoT, self).__init__()
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

    def __call__(self, batchs):
        # print(batchs)
        sentences, labels, GT_labels = map(list, zip(*batchs))
        encoding = self.tokenizer(sentences, return_tensors = 'pt', padding = True, truncation = True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        ans = {'input_ids': input_ids, 'attention_mask': attention_mask}
        labels = torch.tensor(labels).long()
        GT_labels = torch.tensor(GT_labels).long()
        ans['labels'] = labels
        ans['GT_labels'] = GT_labels
        return ans
