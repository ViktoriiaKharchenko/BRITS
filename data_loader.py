import os
import time

import ujson as json
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
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
        self.content = open('./json/DACMI_train.json').readlines()

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
        self.content = open('./json/DACMI_test.json').readlines()

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        rec['is_train'] = 0
        return rec


def collate_fn(recs):

    forward_values = [torch.FloatTensor(rec['forward']['values']) for rec in recs]
    forward_masks = [torch.FloatTensor(rec['forward']['masks']) for rec in recs]
    forward_deltas = [torch.FloatTensor(rec['forward']['deltas']) for rec in recs]
    forward_evals = [torch.FloatTensor(rec['forward']['evals']) for rec in recs]
    forward_eval_masks = [torch.FloatTensor(rec['forward']['eval_masks']) for rec in recs]
    forward_forwards = [torch.FloatTensor(rec['forward']['forwards']) for rec in recs]

    backward_values = [torch.FloatTensor(rec['backward']['values']) for rec in recs]
    backward_masks = [torch.FloatTensor(rec['backward']['masks']) for rec in recs]
    backward_deltas = [torch.FloatTensor(rec['backward']['deltas']) for rec in recs]
    backward_evals = [torch.FloatTensor(rec['backward']['evals']) for rec in recs]
    backward_eval_masks = [torch.FloatTensor(rec['backward']['eval_masks']) for rec in recs]
    backward_forwards = [torch.FloatTensor(rec['backward']['forwards']) for rec in recs]

    forward_lengths = torch.tensor([len(x) for x in forward_values], dtype=torch.long)
    backward_lengths = torch.tensor([len(x) for x in backward_values], dtype=torch.long)

    # Pad sequences
    padded_forward_values = pad_sequence(forward_values, batch_first=True)
    padded_forward_masks = pad_sequence(forward_masks, batch_first=True)
    padded_forward_deltas = pad_sequence(forward_deltas, batch_first=True)
    padded_forward_evals = pad_sequence(forward_evals, batch_first=True)
    padded_forward_eval_masks = pad_sequence(forward_eval_masks, batch_first=True)
    padded_forward_forwards = pad_sequence(forward_forwards, batch_first=True)

    # Pad sequences
    padded_backward_values = pad_sequence(backward_values, batch_first=True)
    padded_backward_masks = pad_sequence(backward_masks, batch_first=True)
    padded_backward_deltas = pad_sequence(backward_deltas, batch_first=True)
    padded_backward_evals = pad_sequence(backward_evals, batch_first=True)
    padded_backward_eval_masks = pad_sequence(backward_eval_masks, batch_first=True)
    padded_backward_forwards = pad_sequence(backward_forwards, batch_first=True)

    ret_dict = {'forward': {
        'values': padded_forward_values,
        'masks': padded_forward_masks,
        'deltas': padded_forward_deltas,
        'evals': padded_forward_evals,
        'eval_masks': padded_forward_eval_masks,
        'forwards': padded_forward_forwards,
        'lengths': forward_lengths,  # Include original lengths of sequences

    }, 'backward': {
        'values': padded_backward_values,
        'masks': padded_backward_masks,
        'deltas': padded_backward_deltas,
        'evals': padded_backward_evals,
        'eval_masks': padded_backward_eval_masks,
        'forwards': padded_backward_forwards,
        'lengths': backward_lengths,

    }, 'is_train': torch.FloatTensor(list(map(lambda x: x['is_train'], recs))),
        'labels': torch.FloatTensor(list(map(lambda x: x['label'], recs)))}

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