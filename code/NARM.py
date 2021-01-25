import os
import pandas as pd
import numpy as np
from GET_SAMPLES import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from NARM_dataloader import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from tqdm import tqdm
from time import time
from copy import deepcopy
from seed import seed_torch
import gc
import warnings

warnings.filterwarnings("ignore")

seed_torch(1997)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def label_enc(click, col):
    uniq, nuniq = click[col].unique(), click[col].nunique()
    match = dict(zip(uniq, range(1, nuniq + 1)))
    click[col] = click[col].map(match)

    return click, match, nuniq

def label_enc_test(dic, click, col):
    init_uniq, init_max = dic.keys(), max(dic.values())
    new_uniq = list(set(click[col].unique()).difference(set(init_uniq)))
    dic_ = dict(zip(new_uniq, range(init_max + 1, init_max + 1 + len(new_uniq))))
    new_dic = dic.copy()
    new_dic.update(dic_)
    click[col] = click[col].map(new_dic)
    nuniq = len(new_dic.items())

    return click, new_dic, nuniq, len(new_uniq)

class Model(nn.Module):

    def __init__(self, n_items, hidden_size, embedding_dim, n_layers=1):
        super(Model, self).__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.emb = nn.Embedding(self.n_items + 1, self.embedding_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.25) # 0.25
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers, batch_first=True)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(0.5) # 0.5
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size, bias=False)
        # self.sf = nn.Softmax()
        self.device = device

        # self.tanh = nn.Tanh()

    def forward(self, seq, lengths):
        hidden = self.init_hidden(seq.size(0))
        embs = self.emb_dropout(self.emb(seq))
        embs = pack_padded_sequence(embs, lengths, batch_first=True)
        gru_out, hidden = self.gru(embs, hidden)
        gru_out, lengths = pad_packed_sequence(gru_out, batch_first=True)

        # fetch the last hidden state of last timestamp
        ht = hidden[-1]
        # gru_out = gru_out.permute(1, 0, 2)

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())
        q2 = self.a_2(ht)
        mask = torch.where(seq > 0, torch.tensor([1.], device=self.device),
                           torch.tensor([0.], device=self.device))
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size)).view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)

        # c_t = self.tanh(c_t)

        item_embs = self.emb(torch.arange(self.n_items + 1).to(self.device))
        scores = torch.matmul(c_t, self.b(item_embs).permute(1, 0))
        # scores = self.sf(scores)

        return scores

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)

class Loss(nn.Module):
    def __init__(self, reg=0, eps=1e-6):
        super(Loss, self).__init__()
        self.reg = reg
        self.eps = eps

    def forward(self, p, n):
        p = torch.exp(p)
        n = torch.exp(n)
        prob = - torch.log(p / (p + torch.sum(n, dim=1, keepdim=True)) + self.eps)

        return prob.sum() + self.reg


def evaluate(rec_matrix, targets, match_num):
    target_repeats = torch.repeat_interleave(targets.view(-1, 1), dim=1, repeats=match_num)
    judge = torch.where(rec_matrix - target_repeats == 0)
    hit = len(judge[0])
    mrr = 0
    ndcg = 0
    for pos in judge[1]:
        mrr += 1 / (pos.float() + 1)
        ndcg += 1 / torch.log2(pos.float() + 2)

    return hit, ndcg, mrr

def Train(train_loader, vali_loader, item_nuniq,
          emb_dim=64, epochs=10, lr=1e-4,
          hidden_size=100, n_layers=1,
          match_num=5, gamma=1e-5, mix_recall_num=50,
          pretrained=False
          ):
    tmp_save_path = '../tmp/NARM/'
    model_save = '../Model/'
    if not os.path.exists(model_save):
        os.makedirs(model_save)

    print_freq = 500

    if pretrained:
        model = torch.load(model_save + 'narm.pkl')
        # emb_lookup = nn.Embedding(item_nuniq + 1, emb_dim, padding_idx=0)
        pret_emb = torch.zeros(item_nuniq + 1, emb_dim)
        pret_emb[:model.emb.weight.data.shape[0], :] = model.emb.weight.data
        # emb_lookup.weight.data.copy_(pret_emb)
        model.emb = nn.Embedding(item_nuniq + 1, emb_dim, padding_idx=0).to(device)
        model.emb.weight.data.copy_(pret_emb)
        model.n_items = item_nuniq
    else:
        model = Model(item_nuniq, hidden_size, emb_dim, n_layers).to(device)
    num_params = sum(param.numel() for param in model.parameters())

    reg = gamma * num_params
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # candidates = torch.Tensor(range(1, item_nuniq + 1)).long().to(device)

    best_mrr = 0
    for epoch in range(epochs):
        if epoch == 0:
            st = time()

        # if epoch == 30 or epoch == 50:
        #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
        print('========================')
        print('lr:%.4e' % optimizer.param_groups[0]['lr'])

        model.train()
        for i, (user, hist_click, target, lens) in enumerate(train_loader):
            hist_click, target = hist_click.to(device), target.to(device)
            output = model(hist_click, lens)

            loss = criterion(output, target)

            loss.backward()

            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

            optimizer.step()

            optimizer.zero_grad()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss:.4f}\t'.format(epoch, i, len(train_loader), loss=loss.item()))

        model.eval()
        HIT, NDCG, MRR = 0, 0, 0
        length = 0
        for user, hist_click, target, lens, clicked_mask in vali_loader:
            hist_click, target = hist_click.to(device), target.to(device)
            candidates_score = F.softmax(model(hist_click, lens)[:, 1:], dim=1)
            # mask clicked
            candidates_score *= clicked_mask.to(device)

            candidate_argsort = candidates_score.argsort(dim=1, descending=True)
            rec_matrix = candidate_argsort[:, :match_num] + 1
            hit, ndcg, mrr = evaluate(rec_matrix, target, match_num)
            length += len(rec_matrix)
            HIT += hit
            NDCG += ndcg
            MRR += mrr

        HIT /= length
        NDCG /= length
        MRR /= length
        print('[+] HIT@{} : {}'.format(match_num, HIT))
        print('[+] NDCG@{} : {}'.format(match_num, NDCG))
        print('[+] MRR@{} : {}'.format(match_num, MRR))
        if best_mrr < MRR:
            best_model = deepcopy(model)
            best_mrr = MRR
            corr_hit, corr_ndcg = HIT, NDCG
            torch.save(best_model, model_save + 'narm.pkl')
            print('Model updated.')
        if epoch == 0:
            et = time()
            print('The last time of 1 epoch: {} min'.format((et - st) / 60))

    print('\n--- Validation Set ---')
    print('[+] HIT@{} : {}'.format(match_num, corr_hit))
    print('[+] NDCG@{} : {}'.format(match_num, corr_ndcg))
    print('[+] Best MRR@{} : {}'.format(match_num, best_mrr))

    # save for offline model fusion
    with torch.no_grad():
      best_model.eval()
      users, targets, match_lst, scores_lst = [], [], [], []
      for user, hist_click, target, lens, clicked_mask in tqdm(vali_loader):
          users += user
          targets += list(target.numpy())
          hist_click, target = hist_click.to(device), target.to(device)
          candidates_score = F.softmax(best_model(hist_click, lens)[:, 1:], dim=1)
          # mask clicked
          candidates_score *= clicked_mask.to(device)

          candidate_argsort = candidates_score.argsort(dim=1, descending=True)
          match_lst.append(candidate_argsort[:, :mix_recall_num] + 1)
          scores_lst.append(candidates_score.sort(dim=1, descending=True)[0][:, :mix_recall_num])

      match_lst = torch.cat(match_lst, dim=0).cpu().numpy()
      scores_lst = torch.cat(scores_lst, dim=0).cpu().numpy()
      np.savez(tmp_save_path + 'vali.npz', users=users, recall_lst=match_lst, scores_lst=scores_lst, targets=targets)
      print('vali prob saved')
      del match_lst, scores_lst
      gc.collect()

    return best_model


def Inference(model, loader, match_num=50, mix_recall_num=50):
    tmp_save_path = '../tmp/NARM/'
    rec_result = []
    users = []
    match_lst, scores_lst = [], []
    with torch.no_grad():
      model.eval()
      for user, hist_click, lens, clicked_mask in tqdm(loader):
          hist_click = hist_click.to(device)
          candidates_score = F.softmax(model(hist_click, lens)[:, 1:], dim=1)
          # mask clicked
          candidates_score *= clicked_mask.to(device)

          candidate_argsort = candidates_score.argsort(dim=1, descending=True)
          rec_matrix = candidate_argsort[:, :match_num] + 1
          rec_result.append(rec_matrix)
          match_lst.append(candidate_argsort[:, :mix_recall_num] + 1)
          scores_lst.append(candidates_score.sort(dim=1, descending=True)[0][:, :mix_recall_num])
          users += user
      rec_result = torch.cat(rec_result, dim=0).cpu().numpy()
      users = np.array(users).reshape(-1, 1)
      match_lst = torch.cat(match_lst, dim=0).cpu().numpy()
      scores_lst = torch.cat(scores_lst, dim=0).cpu().numpy()
      np.savez(tmp_save_path + 'test.npz', users=users, recall_lst=match_lst, scores_lst=scores_lst)
      print('test prob saved')
      del match_lst, scores_lst
      gc.collect()

    rec_result = pd.DataFrame(rec_result, columns=['article_{}'.format(i) for i in range(1, match_num + 1)])
    users = pd.DataFrame(users, columns=['user_id'])
    rec_result = pd.concat([users, rec_result], axis=1)

    return rec_result

def ptrain(train_click, # tmp_save_path,
           max_len=3, bs=512, epochs=10,
           lr=1e-4, emb_dim=64, match_num=5,
           gamma=1e-5, hidden_size=100,
           n_layers=1, mix_recall_num=50, valid_num=20000,
          ):
    # train_click = pd.read_csv(train_path)
    global tmp_save_path
    tmp_save_path = '../tmp/NARM/'

    dict_save = '../dict_tmp/'
    if not os.path.exists(dict_save):
        os.makedirs(dict_save)

    train_click, user_match, user_nuniq = label_enc(train_click, 'user_id')
    train_click, item_match, item_nuniq = label_enc(train_click, 'click_article_id')
    np.save(dict_save + 'user_dict.npy', user_match)
    np.save(dict_save + 'item_dict.npy', item_match)

    if not os.path.exists(tmp_save_path):
        os.makedirs(tmp_save_path)

    train, vali = construct_samples_train(train_click, max_len=max_len, valid_num=valid_num, model_name='narm')
    train_loader = TrainDataLoader(train, bs)
    vali_loader = ValiDataLoader(vali, clicked_mask=True, click=train_click, item_nuniq=item_nuniq, bs=bs)

    print(torch.cuda.get_device_name(0))

    model = Train(train_loader, vali_loader, item_nuniq, emb_dim=emb_dim,
                  epochs=epochs, lr=lr, match_num=match_num, gamma=gamma,
                  hidden_size=hidden_size, n_layers=n_layers,
                  mix_recall_num=mix_recall_num, pretrained=False
                  )
    del train_loader, vali_loader
    gc.collect()

def ContinueTrainInference(train_click, test_click, emb_dim=64,
              max_len=3, bs=512, epochs=10,
              lr=1e-4, match_num=5, gamma=1e-5,
              hidden_size=100, n_layers=1, mix_recall_num=50
              ):
    dict_save = '../dict_tmp/'
    init_user_match, init_item_match = np.load(dict_save + 'user_dict.npy', allow_pickle=True).item(), \
                                       np.load(dict_save + 'item_dict.npy', allow_pickle=True).item()
    test_click, user_match, user_nuniq, new_user_nuniq = label_enc_test(init_user_match, test_click, 'user_id')
    test_click, item_match, item_nuniq, new_item_nuniq = label_enc_test(init_item_match, test_click, 'click_article_id')
    np.save(dict_save + 'user_dict.npy', user_match)
    np.save(dict_save + 'item_dict.npy', item_match)
    train_click['user_id'] = train_click['user_id'].map(user_match)
    train_click['click_article_id'] = train_click['click_article_id'].map(item_match)

    print('New users: {}, items: {}'.format(new_user_nuniq, new_item_nuniq))

    train_ = np.load('../samples_tmp/train_narm.npz', allow_pickle=True)
    train = (train_['users'], train_['seqs'], train_['targets'])
    train, test = construct_samples_test(train, test_click, max_len=max_len)
    valid_ = np.load('../samples_tmp/valid_narm.npz', allow_pickle=True)
    valid = (valid_['users'], valid_['seqs'], valid_['targets'])
    del train_, valid_
    gc.collect()

    train_loader = TrainDataLoader(train, bs)
    vali_loader = ValiDataLoader(valid, clicked_mask=True, click=train_click, item_nuniq=item_nuniq, bs=bs)
    print(torch.cuda.get_device_name(0))

    model = Train(train_loader, vali_loader, item_nuniq, emb_dim=emb_dim,
                  epochs=epochs, lr=lr, match_num=match_num, gamma=gamma,
                  hidden_size=hidden_size, n_layers=n_layers,
                  mix_recall_num=mix_recall_num, pretrained=True
                  )
    del train_loader, vali_loader
    gc.collect()

    test_loader = TestDataLoader(test, clicked_mask=True, click=test_click, item_nuniq=item_nuniq, bs=bs)

    rec_result = Inference(model, test_loader, match_num=match_num, mix_recall_num=mix_recall_num)

    user_rev_match = {v: k for k, v in user_match.items()}
    item_rev_match = {v: k for k, v in item_match.items()}
    rec_result['user_id'] = rec_result.user_id.map(user_rev_match)
    for col in ['article_{}'.format(i) for i in range(1, match_num + 1)]:
        rec_result[col] = rec_result[col].map(item_rev_match)

    rec_result.to_csv('sub_NARM.csv', index=False)
    print(rec_result.head(5))

def pretrain_narm(train_click, epochs):
    emb_dim = 256 # 256
    hidden_size = 512 # 512
    max_len = 18 # 9
    valid_num = 50000
    mix_recall_num = 50

    ptrain(train_click, # tmp_save_path,
           max_len=max_len, bs=512, epochs=epochs,
           lr=1e-4, emb_dim=emb_dim, match_num=5, gamma=1e-5,
           hidden_size=hidden_size, n_layers=1,
           mix_recall_num=mix_recall_num, valid_num=valid_num,
           )

def inference_narm(train_click, test_click, epochs):
  emb_dim = 256 # 256
  hidden_size = 512 # 512
  max_len = 18 # 9
  valid_num = 50000
  mix_recall_num = 50
  
  ContinueTrainInference(train_click, test_click,
                           max_len=max_len, bs=512, epochs=epochs,
                           lr=1e-4, emb_dim=emb_dim, match_num=5, gamma=1e-5,
                           hidden_size=hidden_size, n_layers=1, mix_recall_num=mix_recall_num
                           )

'''
def load_data(train_path):
  if len(train_path) == 2:
    train_click1 = pd.read_csv(train_path[0])
    train_click2 = pd.read_csv(train_path[1])
    train_click = pd.concat([train_click1, train_click2], axis=0)
  else:
    train_click = pd.read_csv(train_path)

  return train_click

if __name__ == '__main__':
    # train_path = 'data/train_click_log.csv'
    # train_path = ['data/train_click_log.csv', 'data/testA_click_log.csv']
    offline = True
    if offline:
      data_path = 'offline_data'
    else:
      data_path = 'data'

    train_path = ['{}/train_click_log.csv'.format(data_path), '{}/testA_click_log.csv'.format(data_path)]
    train_click = load_data(train_path)

    # global tmp_save_path
    # tmp_save_path = 'tmp/NARM/'

    emb_dim = 256 # 256
    hidden_size = 512 # 512
    max_len = 18 # 9
    valid_num = 50000
    mix_recall_num = 50

    # Only use train set
    ptrain(train_click, # tmp_save_path,
           max_len=max_len, bs=512, epochs=10,
           lr=1e-4, emb_dim=emb_dim, match_num=5, gamma=1e-5,
           hidden_size=hidden_size, n_layers=1,
           mix_recall_num=mix_recall_num, valid_num=valid_num,
           )
   
    # test set
    # test_path = 'data/testA_click_log.csv'
    test_path = 'data/testB_click_log.csv'
    test_click = pd.read_csv(test_path)
    ContinueTrainInference(train_click, test_click,
                           max_len=max_len, bs=512, epochs=40,
                           lr=1e-4, emb_dim=emb_dim, match_num=5, gamma=1e-5,
                           hidden_size=hidden_size, n_layers=1, mix_recall_num=mix_recall_num
                           )
'''