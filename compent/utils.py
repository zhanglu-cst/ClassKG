import logging
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from compent.comm import get_world_size
import transformers
import psutil


import dgl

def set_seed_all(seed = None):
    if (seed is None):
        seed = int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    dgl.seed(seed)
    dgl.random.seed(seed)
    transformers.set_seed(seed)


def get_memory_used():
    mem = psutil.virtual_memory()
    used = mem.used / 1024 / 1024 / 1024
    return used

# def set_seeds(seed):
#     "set random seeds"
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)



    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def get_device():
    "get device (CPU or GPU)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def make_dirs(dir):
    if (os.path.exists(dir) == False):
        try:
            os.makedirs(dir)
        except Exception as e:
            print(str(e))


def find_sublist(haystack, needle):
    """Return the index at which the sequence needle appears in the
    sequence haystack, or -1 if it is not found, using the Boyer-
    Moore-Horspool algorithm. The elements of needle and haystack must
    be hashable.
    https://codereview.stackexchange.com/questions/19627/finding-sub-list
    """
    h = len(haystack)
    n = len(needle)
    skip = {needle[i]: n - i - 1 for i in range(n - 1)}
    i = n - 1
    while i < h:
        for j in range(n):
            if haystack[i - j] != needle[-j - 1]:
                i += skip.get(haystack[i], n)
                break
        else:
            return i - n + 1
    return -1


def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_random_word(vocab_words):
    i = random.randint(0, len(vocab_words) - 1)
    return vocab_words[i]


def get_logger(name, log_path):
    "get logger"
    logger = logging.getLogger(name)
    fomatter = logging.Formatter(
            '[ %(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

    if not os.path.isfile(log_path):
        f = open(log_path, "w+")

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(fomatter)
    logger.addHandler(fileHandler)

    # streamHandler = logging.StreamHandler()
    # streamHandler.setFormatter(fomatter)
    # logger.addHandler(streamHandler)

    logger.setLevel(logging.DEBUG)
    return logger


def move_to_device(batch, rank = None):
    ans = {}
    if (rank is None):
        device = 'cuda'
    else:
        device = 'cuda:{}'.format(rank)
    for key in batch:
        try:
            ans[key] = batch[key].to(device = device)
        except Exception as e:
            # print(str(e))
            ans[key] = batch[key]
    return ans


def class_index_to_one_hot(class_indexs, number_of_classes):
    length = len(class_indexs)
    ans = torch.zeros((length, number_of_classes))
    for i in range(length):
        ans[i][class_indexs[i]] = 1
    return ans


def reduce_loss_dict(loss_dict):
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim = 0)
        dist.reduce(all_losses, dst = 0)
        if dist.get_rank() == 0:
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

# if __name__ == '__main__':
#     class_index_to_one_hot([0,1,3,2],5)
