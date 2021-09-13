import json
import os

origin_dir = r'/remote-home/xxx/EMNLP_origin/data/x_class'
target_dir = r'/remote-home/xxx/EMNLP_origin/data/processed'

all_items = os.listdir(origin_dir)

for item in all_items:
    print('starting:{}'.format(item))
    target_item_dir = os.path.join(target_dir, item + '_x')
    try:
        os.mkdir(target_item_dir)
    except Exception as e:
        pass

    origin_item_dir = os.path.join(origin_dir, item)
    label_name_path = os.path.join(origin_item_dir, 'classes.txt')
    with open(label_name_path, 'r') as f:
        labelnames = f.read().splitlines()
    dict_labelID_to_keywords = {}
    for index_class, keyword in enumerate(labelnames):
        dict_labelID_to_keywords[index_class] = [keyword]
    target_keywords = os.path.join(target_item_dir, 'keywords.json')
    with open(target_keywords, 'w') as f:
        json.dump(dict_labelID_to_keywords, f)

    datasets_text_path = os.path.join(origin_item_dir, 'dataset.txt')
    with open(datasets_text_path, 'r') as f:
        datasets = f.readlines()
        datasets = [item.strip() for item in datasets]
    label_path = os.path.join(origin_item_dir, 'labels.txt')
    with open(label_path, 'r') as f:
        labels = f.readlines()
        labels = [item.strip() for item in labels]

    print('len:{},{}'.format(len(datasets), len(labels)))
    assert len(datasets) == len(labels), '{}:{}'.format(len(datasets), len(labels))
    unlabeled = []
    for one_line, one_label in zip(datasets, labels):
        one_label = int(one_label)
        unlabeled.append([one_line, one_label])
    target_unlabeled_path = os.path.join(target_item_dir, 'unlabeled.json')
    with open(target_unlabeled_path, 'w') as f:
        json.dump(unlabeled, f)
    print('finish:{}'.format(item))
