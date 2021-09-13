import json

import numpy
from transformers import BertTokenizer

taskname = '20News_x'

target_unlabel_path = '/remote-home/xxx/xxx/data/processed/' + taskname + '/unlabeled.json'

with open(target_unlabel_path, 'r') as f:
    unlabeled = json.load(f)
#
# def show_max_mean(token_func):
#     token_lengths = []
#     for s, l in unlabeled:
#         words = token_func(s)
#         token_lengths.append(len(words))
#         print(len(words))
#         break
#     print(numpy.max(token_lengths))
#     print(numpy.mean(token_lengths))
#     print()
#
# show_max_mean(split_sentence_into_words)


tokens_func = BertTokenizer.from_pretrained('bert-base-uncased')
all_lens = []
for index, (s, l) in enumerate(unlabeled):
    words = tokens_func(s)
    all_lens.append(len(words['input_ids']))
    if (index % 5000 == 0):
        print(index)
        print(numpy.mean(all_lens))
        print(numpy.max(all_lens))

print(numpy.mean(all_lens))
print(numpy.max(all_lens))
print()

# #
# def token_bert(s):
#     tokens_a = BertTokenizer.from_pretrained('bert-base-uncased')
#     return tokens_a
#
# show_max_mean(token_bert)
