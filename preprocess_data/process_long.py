import json
import os
import pickle

import numpy

from PROJECT_ROOT import ROOT_DIR
from compent.utils import make_dirs

path_origin = r'/XXX/dataset/nyt/fine/'
cur_task = 'NYT25'

path_keyword = os.path.join(path_origin, 'seedwords.json')
path_data = os.path.join(path_origin, 'df.pkl')

target_task_dir = os.path.join(ROOT_DIR, 'data', 'processed', cur_task)
make_dirs(target_task_dir)

with open(path_keyword, 'r') as f:
    label_to_keywords = json.load(f)

print(label_to_keywords)

label_to_index = {}
index_to_keywords = {}

for index, label in enumerate(label_to_keywords):
    label_to_index[label] = index
    index_to_keywords[index] = label_to_keywords[label]

# with open('label_to_index.json','w') as f:
#     json.dump(label_to_index,f)


with open(os.path.join(target_task_dir, 'keywords.json'), 'w') as f:
    json.dump(index_to_keywords, f)

print(index_to_keywords)

with open(path_data, 'rb') as f:
    df = pickle.load(f)
all_sentence = df['sentence'].tolist()
origin_label = df['label'].tolist()
print(len(all_sentence), len(origin_label))

s_length = []
for s in all_sentence:
    s_length.append(len(s))
print('mean sentence length:{}'.format(numpy.mean(s_length)))
print('max sentence length:{}'.format(numpy.max(s_length)))

all_label = []
for str_label in origin_label:
    int_label = label_to_index[str_label]
    all_label.append(int_label)

all_data_unlabel = []
for s, l in zip(all_sentence, all_label):
    all_data_unlabel.append((s, l))

target_unlabel_path = os.path.join(ROOT_DIR, 'data', 'processed', cur_task, 'unlabeled.json')
with open(target_unlabel_path, 'w') as f:
    json.dump(all_data_unlabel, f)
