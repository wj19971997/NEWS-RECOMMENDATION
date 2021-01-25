import os
import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm
import gc
import warnings

warnings.filterwarnings("ignore")

np.random.seed(1997)

def construct_samples_train(train_click, valid_num=20000, max_len=19, model_name='srgnn'):
    train_click.sort_values("click_timestamp", inplace=True)

    # users = train_click.user_id.unique()
    # np.random.shuffle(users)
    # valid_users = users[:valid_num]
    valid_users = np.load('valid_users.npy')
    valid_click = train_click[train_click.user_id.isin(valid_users)]
    train_click = train_click[~train_click.user_id.isin(valid_users)]

    train_users, train_seqs, train_targets = [], [], []
    for user, val in tqdm(train_click.groupby('user_id')):
        hist_item = val['click_article_id'].values.tolist()
        # hist_time = val['click_timestamp'].values.tolist()
        hist_len = len(hist_item)
        if hist_len == 1:
            continue
        elif hist_len <= max_len + 1:
            for i in range(hist_len - 1):
                train_users.append(user), train_seqs.append(hist_item[:i + 1]), train_targets.append(hist_item[i + 1])
        else:
            for i in range(hist_len - max_len):
                train_users.append(user), train_seqs.append(hist_item[i:i + max_len]), train_targets.append(
                        hist_item[i + max_len])

    valid_users, valid_seqs, valid_targets = [], [], []
    for user, val in tqdm(valid_click.groupby('user_id')):
        hist_item = val['click_article_id'].values.tolist()
        # hist_time = val['click_timestamp'].values.tolist()
        hist_len = len(hist_item)
        if hist_len == 1:
            continue
        elif hist_len <= max_len + 1:
            for i in range(hist_len - 2):
                train_users.append(user), train_seqs.append(hist_item[:i + 1]), train_targets.append(hist_item[i + 1])
            valid_users.append(user), valid_seqs.append(hist_item[:-1]), valid_targets.append(hist_item[-1])
        else:
            for i in range(hist_len - max_len - 1):
                train_users.append(user), train_seqs.append(hist_item[i:i + max_len]), train_targets.append(
                        hist_item[i + max_len])
            valid_users.append(user), valid_seqs.append(hist_item[-(max_len + 1):-1]), valid_targets.append(hist_item[-1])

    save_path = '../samples_tmp/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train = (train_users, train_seqs, train_targets)
    np.savez(save_path + 'train_{}.npz'.format(model_name), users=train_users, seqs=train_seqs, targets=train_targets)
    del train_users, train_seqs, train_targets

    valid = (valid_users, valid_seqs, valid_targets)
    np.savez(save_path + 'valid_{}.npz'.format(model_name), users=valid_users, seqs=valid_seqs, targets=valid_targets)
    del valid_users, valid_seqs, valid_targets

    gc.collect()

    print(len(train[0]), len(valid[0]))

    return train, valid

def construct_samples_test(train, test_click, max_len=19):
    test_click.sort_values("click_timestamp", inplace=True)
    # train_users, train_seqs, train_targets = train[0], train[1], train[2]
    train_users, train_seqs, train_targets = list(train[0]), list(train[1]), list(train[2])
    test_users, test_seqs = [], []
    for user, val in tqdm(test_click.groupby('user_id')):
        hist_item = val['click_article_id'].values.tolist()
        # hist_time = val['click_timestamp'].values.tolist()
        hist_len = len(hist_item)
        if hist_len == 1:
            test_users.append(user), test_seqs.append(hist_item)
        elif hist_len == 2:
            train_users.append(user), train_seqs.append(hist_item[:-1]), train_targets.append(hist_item[-1])
            test_users.append(user), test_seqs.append(hist_item)
        elif hist_len <= max_len + 1:
            for i in range(hist_len - 1):
                train_users.append(user), train_seqs.append(hist_item[:i + 1]), train_targets.append(hist_item[i + 1])
            # train_users.append(user), train_seqs.append(hist_item[:-2]), train_targets.append(hist_item[-2])
            test_users.append(user), test_seqs.append(hist_item)
        else:
            for i in range(hist_len - max_len):
                train_users.append(user), train_seqs.append(hist_item[i:i + max_len]), train_targets.append(
                        hist_item[i + max_len])
            test_users.append(user), test_seqs.append(hist_item[-max_len:])

    train = (train_users, train_seqs, train_targets)
    del train_users, train_seqs, train_targets

    test = (test_users, test_seqs)
    del test_users, test_seqs
    gc.collect()

    print(len(train[0]), len(test[0]))

    return train, test
