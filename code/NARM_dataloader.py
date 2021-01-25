import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from NARM_utils import *
from tqdm import tqdm
from time import time
import gc
import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index], self.data[2][index]

    def __len__(self):
        return len(self.data[0])

def TrainDataLoader(data, bs=512):
    data_set = TrainDataset(data)
    data_loader = DataLoader(data_set, batch_size=bs, shuffle=True, collate_fn=collate_fn_train, drop_last=True)

    return data_loader


class ValiDataset(Dataset):

    def __init__(self, data, clicked_mask=True, click=None, item_nuniq=None):
        users, hists, targets = data
        df = pd.DataFrame(np.array(users).reshape(-1, 1), columns=['user'])

        self.clicked_mask = clicked_mask
        if self.clicked_mask:
            click = click[click.user_id.isin(np.unique(users))]
            mask_value = -1e10
            tmp = click.groupby('user_id')['click_article_id'].agg(lambda x: list(x)[:-1]).reset_index()
            tmp.columns = ['user', 'total_history']
            df = df.merge(tmp, on='user', how='left')
            clicked_mask_matrix = torch.ones((df.shape[0], item_nuniq))
            for i, row in tqdm(enumerate(df['total_history'].values)):
                for item in row:
                    clicked_mask_matrix[i, item - 1] = mask_value
            del tmp
            gc.collect()
            self.data = (users, hists, targets, clicked_mask_matrix)
        else:
            self.data = (users, hists, targets)
        del data, df
        gc.collect()

    def __getitem__(self, idx):
        if self.clicked_mask:
            return self.data[0][idx], self.data[1][idx], self.data[2][idx], self.data[3][idx]
        else:
            return self.data[0][idx], self.data[1][idx], self.data[2][idx]

    def __len__(self):
        return len(self.data[0])

def ValiDataLoader(data, clicked_mask=True, click=None, item_nuniq=None, bs=512):
    data_set = ValiDataset(data, clicked_mask, click, item_nuniq)
    data_loader = DataLoader(data_set, batch_size=bs, shuffle=False, collate_fn=collate_fn_vali)

    return data_loader


class TestDataset(Dataset):

    def __init__(self, data, clicked_mask=True, click=None, item_nuniq=None):
        users, hists = data
        df = pd.DataFrame(np.array(users).reshape(-1, 1), columns=['user'])

        self.clicked_mask = clicked_mask
        if self.clicked_mask:
            mask_value = -1e10
            tmp = click.groupby('user_id')['click_article_id'].agg(list).reset_index()
            tmp.columns = ['user', 'total_history']
            df = df.merge(tmp, on='user', how='left')
            clicked_mask_matrix = torch.ones((df.shape[0], item_nuniq))
            for i, row in tqdm(enumerate(df['total_history'].values)):
                for item in row:
                    clicked_mask_matrix[i, item - 1] = mask_value
            del tmp
            gc.collect()
            self.data = (users, hists, clicked_mask_matrix)
        else:
            self.data = (users, hists)
        del data, df
        gc.collect()

    def __getitem__(self, idx):
        if self.clicked_mask:
            return self.data[0][idx], self.data[1][idx], self.data[2][idx]
        else:
            return self.data[0][idx], self.data[1][idx]

    def __len__(self):
        return len(self.data[0])

def TestDataLoader(data, clicked_mask=True, click=None, item_nuniq=None, bs=512):
    data_set = TestDataset(data, clicked_mask, click, item_nuniq)
    data_loader = DataLoader(data_set, batch_size=bs, shuffle=False, collate_fn=collate_fn_test)

    return data_loader