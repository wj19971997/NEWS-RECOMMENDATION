import os
import pandas as pd
import numpy as np
from time import time
import warnings

warnings.filterwarnings("ignore")

def label_enc(click, col):
    uniq, nuniq = click[col].unique(), click[col].nunique()
    match = dict(zip(uniq, range(1, nuniq + 1)))
    click[col] = click[col].map(match)

    return click, match, nuniq

np.random.seed(1997)
valid_num = 50000
train_click = pd.read_csv('../tcdata/train_click_log.csv')
train_click, user_match, user_nuniq = label_enc(train_click, 'user_id')
users = train_click.user_id.unique()
np.random.shuffle(users)
valid_users = users[:valid_num]

np.save('valid_users.npy', valid_users)