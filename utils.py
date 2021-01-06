import pickle

import torch
import torch.nn as nn
import copy
import numpy as np
import json


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def read_pickle(path):
    return pickle.load(open(path, "rb"))


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
