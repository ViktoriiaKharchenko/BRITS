import os
import time

import ujson as json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MySet(Dataset):
    def __init__(self):
        super(MySet, self).__init__()
        self.content = open('./json/json').readlines()

        indices = np.arange(len(self.content))
        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])

        if idx in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec

class MyTrainSet(Dataset):
    def __init__(self):
        super(MyTrainSet, self).__init__()
        self.content = open('./json/train.json').readlines()

        indices = np.arange(len(self.content))

        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        if idx in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec


class MyTestSet(Dataset):
    def __init__(self):
        super(MyTestSet, self).__init__()
        self.content = open('./json/test.json').readlines()

        indices = np.arange(len(self.content))

        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())


    def __len__(self):
        return len(self.content)


def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))

    def to_tensor_dict(recs):
        values = torch.FloatTensor([r['values'] for r in recs])
        masks = torch.FloatTensor([r['masks'] for r in recs])
        deltas = torch.FloatTensor([r['deltas'] for r in recs])

        evals = torch.FloatTensor([r['evals'] for r in recs])
        eval_masks = torch.FloatTensor([r['eval_masks'] for r in recs])
        forwards = torch.FloatTensor([r['forwards'] for r in recs])


        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    ret_dict['labels'] = torch.FloatTensor(list(map(lambda x: x['label'], recs)))
    ret_dict['is_train'] = torch.FloatTensor(list(map(lambda x: x['is_train'], recs)))

    return ret_dict

def get_loader(batch_size = 64, shuffle = True):
    data_set = MySet()
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 4, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter


def get_train_loader(batch_size=100, shuffle=True):
    data_set = MyTrainSet()
    data_iter = DataLoader(dataset=data_set, \
                           batch_size=batch_size, \
                           num_workers=1, \
                           shuffle=shuffle, \
                           pin_memory=True, \
                           collate_fn=collate_fn)

    return data_iter


def get_test_loader(batch_size=100, shuffle=False):
    data_set = MyTestSet()
    data_iter = DataLoader(dataset=data_set, \
                           batch_size=batch_size, \
                           num_workers=1, \
                           shuffle=shuffle, \
                           pin_memory=True, \
                           collate_fn=collate_fn)

    return data_iter