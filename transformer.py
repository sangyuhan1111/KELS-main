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
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from utils.nn_utils import *
from models.ViT import ViT_LRP_nan_excluded

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(233,80),
            nn.ReLU(),
            nn.Linear(80,30),
            nn.ReLU(),
            nn.Linear(30,9)
        )
    def forward(self,x):
        x[torch.isnan(x)] = 0
        return self.mlp(x)



def accuracy_roughly(y_pred, y_label):
    if len(y_pred) != len(y_label):
        print("not available, fit size first")
        return
    cnt = 0
    correct = 0
    for pred, label in zip(y_pred, y_label):
        cnt += 1
        if abs(pred-label) <= 1:
            correct += 1
    return correct / cnt



def train_net(model,train_loader,test_loader,optimizer_cls = optim.AdamW, criterion = nn.CrossEntropyLoss(),
n_iter=10,device='cpu',lr = 0.001,weight_decay = 0.01,mode = None):
        
        train_losses = []
        train_acc = []
        val_accs = []
        positive_accs = []
        #optimizer = optimizer_cls(model.parameters(),lr=lr,weight_decay=weight_decay)
        optimizer = optimizer_cls(model.parameters(),lr=lr)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[25,40,60,80], gamma=0.5,last_epoch=-1)
        

        for epoch in range(n_iter):
                running_loss = 0.0
                model.train()
                n = 0
                n_acc = 0
                ys = []
                ypreds = []
                for i,(xx,(label_E,label_K,label_M)) in tqdm(enumerate(train_loader)):

                
                
                        xx = xx.to(device)
                        if mode == 'E':
                                yy = label_E
                        elif mode == 'K':
                                yy = label_K
                        elif mode == 'M':
                                yy = label_M
                        else:
                                assert True
                        
                        yy = yy.to(device)
                        

                        
                        
                
                        optimizer.zero_grad()
                        outputs = model(xx)
                        _,y_pred = outputs.max(1)

                        loss1 = criterion(outputs,yy)
                        loss2 = criterion(outputs,(yy+1).clamp(max=8))
                        loss3 = criterion(outputs,(yy-1).clamp(min=0))
                        loss = loss1 + loss2 + loss3

                        # Getting gradients w.r.t. parameters
                        loss.backward()

                        # Updating parameters
                        optimizer.step()
                        ys.append(yy)
                        ypreds.append(y_pred)
                        
                        
                        i += 1
                        n += len(xx)
                        _, y_pred = outputs.max(1)
                        n_acc += (yy == y_pred).float().sum().item()
                #scheduler.step()
                train_losses.append(running_loss/i)
                train_acc.append(n_acc/n)
                ys = torch.cat(ys)
                ypreds = torch.cat(ypreds)
                train_positive_acc = accuracy_roughly(ypreds,ys)
                acc, positive_acc = eval_net(model,test_loader,device,mode = mode)
                val_accs.append(acc)
                positive_accs.append(positive_acc)

                print(f'epoch : {epoch},train_positive_acc : {train_positive_acc} train_acc : {train_acc[-1]}, acc : {val_accs[-1]}. positive_acc : {positive_accs[-1]}',flush = True)

        return np.array(val_accs), np.array(positive_accs)

def eval_net(model,data_loader,device,mode=None):
    model.eval()
    ys = []
    ypreds = []
    for xx,(label_E,label_K,label_M) in data_loader:

                
                
        xx = xx.to(device)
        if mode == 'E':
            y = label_E
        elif mode == 'K':
            y = label_K
        elif mode == 'M':
            y = label_M
        else:
            assert True
        
        y = y.to(device)

        with torch.no_grad():
                score = model(xx)
                _,y_pred = score.max(1)
        ys.append(y)
        ypreds.append(y_pred)

    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    positive_acc = accuracy_roughly(ypreds,ys)
    acc= (ys == ypreds).float().sum() / len(ys)

    # print(sklearn.metrics.confusion_matrix(ys.numpy(),ypreds.numpy()))


    # print(sklearn.metrics.classification_report(ys.numpy(),ypreds.numpy()))
    

    return acc, positive_acc
    #return acc.item()



X_datapaths = ['./preprocessed/prepared/nan/L2Y1.pkl','./preprocessed/prepared/nan/L2Y2.pkl','./preprocessed/prepared/nan/L2Y3.pkl','./preprocessed/prepared/nan/L2Y4.pkl','./preprocessed/prepared/nan/L2Y5.pkl','./preprocessed/prepared/nan/L2Y6.pkl']
label_datapath = './preprocessed/prepared/nan/label.pkl'
#X_datapaths = ['./preprocessed/prepared/fill/L2Y1.pkl','./preprocessed/prepared/fill/L2Y2.pkl','./preprocessed/prepared/fill/L2Y3.pkl','./preprocessed/prepared/fill/L2Y4.pkl','./preprocessed/prepared/fill/L2Y5.pkl','./preprocessed/prepared/fill/L2Y6.pkl',]
#label_datapath = './preprocessed/prepared/fill/label.pkl'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# read pickle
input_datas = [] # list of each input pandas dataframe
for datapath in X_datapaths:
    temp = pd.read_pickle(datapath)
    temp = temp.reset_index()
    temp = temp.drop(columns=['index'])
    input_datas.append(temp)


label_data = pd.read_pickle(label_datapath)
label_data = label_data.reset_index()
label_data = label_data.drop(columns=['index'])



split_list = make_split_list(input_datas)
input_concated = np.concatenate(input_datas,axis=1) # concated input. (number of instance x number of features) will be splited with kfold
seq_len = len(input_datas)
label_data = label_data - 1



CLS2IDX = {
    0 : '1등급',
    1 : '2등급',
    2 : '3등급',
    3 : '4등급',
    4 : '5등급',
    5 : '6등급',
    6 : '7등급',
    7 : '8등급',
    8 : '9등급'
}
is_regression = False

batch_size = 32
hidden_features = 100
embbed_dim = 72
n_splits = 10
kfold = KFold(n_splits=n_splits)
fold_acc_dict = {}
for fold,(train_idx,test_idx) in enumerate(kfold.split(input_concated)):
    print('------------fold no---------{}----------------------'.format(fold))
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
    dataset = KELSDataSet(input_concated,label_data)
    train_loader = DataLoader(
                        dataset, 
                        batch_size=batch_size, sampler=train_subsampler)
    test_loader = DataLoader(
                        dataset,
                        batch_size=batch_size, sampler=test_subsampler)
    sample_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    assert batch_size
    (sample,label) = next(iter(sample_loader))
    sample_datas = batch_to_splited_datas(sample,split_list)
    model_E = ViT_LRP_nan_excluded.VisionTransformer(sample_datas,split_list,seq_len=6, num_classes=9, embed_dim=16*3, depth=8,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, mlp_head=False, drop_rate=0.2, attn_drop_rate=0.2)
    model_E = model_E.to(device)
    val_accs , positive_accs = train_net(model_E,train_loader,test_loader,n_iter=100,device=device,mode='E',lr=0.0001,optimizer_cls = optim.AdamW)
    temp_dict = {}
    temp_dict['val_accs'] = val_accs
    temp_dict['positive_accs'] = positive_accs
    fold_acc_dict[fold] = temp_dict

 
    

#embedding_networks : 년차별로 맞는 mlp 리스트. 리스트 내용물에 따라 인풋 채널 개수 다름.
 # not used in traing; only used to initialize embbeding layer

val_acc_mean = np.zeros_like(fold_acc_dict[0]['val_accs'])
pos_acc_mean = np.zeros_list(fold_acc_dict[0]['positive_accs'])
for i in len(range(fold_acc_dict)):
    val_acc_mean += fold_acc_dict[i]['val_accs']
    pos_acc_mean += fold_acc_dict[i]['positive_accs']

val_acc_mean = val_acc_mean / n_splits
pos_acc_mean = pos_acc_mean / n_splits

print(f"mean          accuracy across {n_splits} fold : {val_acc_mean}")
print(f"mean positive accuracy across {n_splits} fold : {pos_acc_mean}")




