import os
import pandas as pd
import numpy as np
import torch
from time import time
import warnings

warnings.filterwarnings("ignore")

def collate_fn_train(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    lens = [len(hist) for _, hist, _ in data]
    users, labels = [], []
    padded_seq = torch.zeros(len(data), max(lens)).long()
    for i, (user, hist, label) in enumerate(data):
        users.append(user)
        padded_seq[i, :lens[i]] = torch.LongTensor(hist)
        labels.append(label)
    # padded_sesss = padded_sesss.transpose(0, 1)

    return users, padded_seq, torch.tensor(labels).long(), lens

def collate_fn_vali(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    lens = [len(hist) for _, hist, _, _ in data]
    users, labels = [], []
    padded_seq = torch.zeros(len(data), max(lens)).long()
    click_mask_matrix = []
    for i, (user, hist, label, cmm) in enumerate(data):
        users.append(user)
        padded_seq[i, :lens[i]] = torch.LongTensor(hist)
        labels.append(label)
        click_mask_matrix.append(cmm.reshape(1, -1))
        # click_mask_matrix[i] = cmm
    click_mask_matrix = torch.cat(click_mask_matrix, dim=0)

    return users, padded_seq, torch.tensor(labels).long(), lens, click_mask_matrix

def collate_fn_test(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    lens = [len(hist) for _, hist, _ in data]
    users = []
    padded_seq = torch.zeros(len(data), max(lens)).long()
    click_mask_matrix = []
    for i, (user, hist, cmm) in enumerate(data):
        users.append(user)
        padded_seq[i, :lens[i]] = torch.LongTensor(hist)
        click_mask_matrix.append(cmm.reshape(1, -1))
    # padded_sesss = padded_sesss.transpose(0, 1)
    click_mask_matrix = torch.cat(click_mask_matrix, dim=0)

    return users, padded_seq, lens, click_mask_matrix