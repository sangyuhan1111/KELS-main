import torch
import pandas as pd
import sklearn
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
import copy

from torch.utils.data import Dataset,DataLoader


class Embedding(nn.Module):
    """
    각 년차별로 쓸 mlp
    """

    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.ReLU,dropout=0.1):
        super().__init__()
        out_features = out_features or in_features
        self.fc1=nn.Linear(in_features,hidden_features)
        self.act = act_layer()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def batch_to_embbedings(datas,networks):
    """
    !!제일 중요!!
    batches : years of data. seperated outputs of dataloader. each element of datas has different feature size.
    network : embedding linear networks that matches feature size of each data
    batches에 저장된 nan 데이터를 모두 0으로 바꾸고 각자 맞는 network에 통과시킴. 통과시킨 데이터를 emb_list에 저장
    batches에 있는 nan을 토대로 각 데이터에 어떤 년도가 nan인지 저장.
    저장한 것을 바탕으로 transformer에 쓸 attention mask 만듬.
    return : emb_list, attn_mask
    emb_list : list of embbedings. embeddings = (batch, emb_features eg. 72)
    attn_mask : 트랜스포머에 쓸 마스크. (batch,1,seq_len+1,seq_len+1). seq_len에 1을 더하는건 트랜스포머의 cls_token 때문.


    """
    emb_list = []
    batch_nan_list = []
    batch_size = datas[0].shape[0]
    device = datas[0].device
    for i,net in enumerate(networks):
        datas_nan = torch.isnan(datas[i])
        _nan = datas_nan[:,0].clone().detach()
        x = torch.nan_to_num(datas[i])
        batch_nan_list.append(_nan)
        emb = net(x)
        emb[_nan] = 0 
        emb_list.append(emb)
    
    attn_mask = torch.stack(batch_nan_list,dim=1)
    seq_len = len(batch_nan_list)
    temp = torch.BoolTensor(batch_size).to(device)

    temp[:] = False
    attn_mask = torch.concat((temp.unsqueeze(1),attn_mask),dim=1)
    attn_mask = attn_mask.unsqueeze(1).expand(-1,seq_len+1,-1)
    attn_mask = attn_mask.unsqueeze(1)
    return emb_list, attn_mask




def make_split_list(year_datas):
    """make split list used for spliting batches. batches must be splitted with torch.tensor_split with split_list"""
    split_list = []
    split = 0
    for data in year_datas:
        split += data.shape[1]
        split_list.append(split)
    split_list.pop() # 
    return split_list


def batch_to_splited_datas(batch,split_list):
    """
    batch : 모든 feature들이 concat된 행렬 하나. 얘를 다시 년차별 데이터 리스트로 나눠줘야함.
    split_list : feature 나누는 번호가 기록된 리스트.
    return : 년차별 행렬이 담긴 리스트
    """


    list = torch.tensor_split(batch,split_list,dim=1)
    return list


class KELSDataSet(Dataset):
    """
    
    make dataset with list of dataframe.
    input : list of dataframe

    __getitem__ returns (batch, concated featres eg. 233 )
    
    """
    def __init__(self,input,label,is_regression=False):
        
        
        self.is_regression = is_regression
        self.label = label.to_numpy()
        self.seq_len = len(input)
        self.data_len = input.shape[0]
        self.data = input


    def __len__(self):
        return self.data_len

    def __getitem__(self,idx):

        x = torch.FloatTensor(self.data[idx])
        if self.is_regression == True:
            y_E,y_K,y_M  = torch.FloatTensor(self.label[idx])[0],torch.FloadTensor(self.label[idx])[1],torch.FloadTensor(self.label[idx])[2]
        else:
            y_E,y_K,y_M = torch.LongTensor(self.label[idx])[0],torch.LongTensor(self.label[idx])[1],torch.LongTensor(self.label[idx])[2]

        return (x,(y_E,y_K,y_M))

def make_embbeding_networks(sample_datas,hidden_features = 100, out_features = 72, dropout=0.1):
    """
    make embedding networks based on train_dataset. batch size must be same with dataloaders.
    transformer 내부 __init__에서 사용함.
    """

    #modulelist : 리스트는 리스트인데 모듈이 들어가는 리스트.
    embbeding_networks = nn.ModuleList() #모듈리스트에 등록하지 않으면 학습이 안됨. 원래는 자동으로 되지만, 우리는 6개의 임베딩네트워크를 리스트로 묶을 거라서 그냥 리스트에다 넣으면 안됨.
    # embbeding networks : 총 6개의 인코딩 네트워크. 흠.. nan 들어오면 batch x feature 사이즈의 nan true false 내놔야..?
    # batch x seq 의 nanlist도 필요..
    
    for sample_data in sample_datas:
        in_features = sample_data.shape[1]
        emb_net = Embedding(in_features,hidden_features=hidden_features,out_features=out_features,dropout=dropout)
        embbeding_networks.append(emb_net)
    return embbeding_networks

# def make_attn_mask(emb_seq_batch):
#     """
#     make attention mask from embedding batch. 
#     batch = (batch, seq_len,embedding_size)
#     return = (batch, seq_len, seq_len)
#     """
#     batch_size = emb_seq_batch.shape[0]
#     seq_len = emb_seq_batch.shape[1]
#     emb_seq_batch_isnan = torch.isnan(emb_seq_batch)
#     torch.nan_to_num(emb_seq_batch) # emb_seq_batch 내용물의 nan을 0으로


#     attn_mask = emb_seq_batch_isnan[:,:,0]
#     temp = torch.BoolTensor(batch_size)

#     temp[:] = False
#     attn_mask = torch.concat((temp.unsqueeze(1),attn_mask),dim=1)

#     attn_mask = attn_mask.unsqueeze(1).expand(-1,seq_len+1,-1)
#     attn_mask = attn_mask.unsqueeze(1)
#     return attn_mask
