import os
import argparse
import pandas as pd
import numpy as np
from NARM import pretrain_narm
from time import time
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--offline', action='store_true', help='offline')
parser.add_argument('--epochs', type=int, default=1, help='the number of epochs to train for')
opt = parser.parse_args()
print(opt)

def load_data(train_path):
    if len(train_path) == 2:
        train_click1 = pd.read_csv(train_path[0])
        train_click2 = pd.read_csv(train_path[1])
        train_click = pd.concat([train_click1, train_click2], axis=0)
    else:
        train_click = pd.read_csv(train_path)

    return train_click

if __name__ == '__main__':
    if opt.offline:
        data_path = 'offline_data'
    else:
        data_path = '../tcdata'

    train_path = ['{}/train_click_log.csv'.format(data_path), '{}/testA_click_log.csv'.format(data_path)]
    train_click = load_data(train_path)

    # NARM
    pretrain_narm(train_click, opt.epochs)
    print('[-] PRETRAINED OF NARM FINISHED.\n')