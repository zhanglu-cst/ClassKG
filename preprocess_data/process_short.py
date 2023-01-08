import json
import os

from PROJECT_ROOT import ROOT_DIR
from compent.utils import make_dirs

root_dir_origin = r'/xxx/dataset'

cur_task = 'imdb'

target_task_dir = os.path.join(ROOT_DIR, 'data', 'processed', cur_task)
make_dirs(target_task_dir)

origin_task_dir = os.path.join(root_dir_origin, cur_task)

print('origin_task_dir:{}'.format(origin_task_dir))
print('target_task_dir:{}'.format(target_task_dir))

label_name_path = os.path.join(origin_task_dir, 'label_names.txt')
with open(label_name_path, 'r') as f:
    labelnames = f.read().splitlines()

print(labelnames)


keywords_to_index = {}
index_to_keywords = {}

for index, one_labelname in enumerate(labelnames):
    keywords = one_labelname.split()
    for one_keyword in keywords:
        keywords_to_index[one_keyword] = index
    index_to_keywords[index] = keywords

print('keywords_to_index:{}'.format(keywords_to_index))
print('index_to_keywords:{}'.format(index_to_keywords))

with open(os.path.join(target_task_dir, 'keywords.json'), 'w') as f:
    json.dump(index_to_keywords, f)


def solve(pre):
    text_path = os.path.join(origin_task_dir, pre + '.txt')
    label_path = os.path.join(origin_task_dir, pre + '_labels.txt')
    print('solve:{}'.format(pre))
    with open(text_path, 'r') as f:
        # text = f.read().splitlines()
        temp = f.readlines()
        print(temp[0])
        text = []
        for item in temp:
            text.append(item.strip())
        print(text[0])
    with open(label_path, 'r') as f:
        label = f.read().split()

    print('len text:{}, len label:{}'.format(len(text), len(label)))
    assert len(text) == len(label), 'len text:{}, len label:{}'.format(len(text), len(label))

    res = []
    for item_text, item_label in zip(text, label):
        item_label = int(item_label)
        res.append([item_text, item_label])
    filename = 'unlabeled.json' if pre == 'train' else 'test.json'
    target_path = os.path.join(target_task_dir, filename)
    print('target_path:{}'.format(target_path))
    with open(target_path, 'w') as f:
        json.dump(res, f)


solve('train')
solve('test')
