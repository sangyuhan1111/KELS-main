import torch
import pandas as pd
import sklearn
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from utils.nn_utils import *
from models.ViT import ViT_LRP_nan_excluded

X_datapaths = ['./preprocessed/prepared/nan/L2Y1.pkl','./preprocessed/prepared/nan/L2Y2.pkl','./preprocessed/prepared/nan/L2Y3.pkl','./preprocessed/prepared/nan/L2Y4.pkl','./preprocessed/prepared/nan/L2Y5.pkl','./preprocessed/prepared/nan/L2Y6.pkl',]
label_datapath = './preprocessed/prepared/nan/label.pkl'
#X_datapaths = ['./preprocessed/prepared/fill/L2Y1.pkl','./preprocessed/prepared/fill/L2Y2.pkl','./preprocessed/prepared/fill/L2Y3.pkl','./preprocessed/prepared/fill/L2Y4.pkl','./preprocessed/prepared/fill/L2Y5.pkl','./preprocessed/prepared/fill/L2Y6.pkl',]
#label_datapath = './preprocessed/prepared/fill/label.pkl'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# read pickle
input_datas = [] # list of each input pandas dataframe
for datapath in X_datapaths:
    temp = pd.read_pickle(datapath)
    temp = temp.reset_index()
    input_datas.append(temp)

label_data = pd.read_pickle(label_datapath)
label_data = label_data.reset_index()



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
X_trains, X_tests, y_train, y_test = make_splited_data(input_datas,label_data,is_regression=is_regression)

train_dataset = KELSDataSet(X_trains,y_train,is_regression=is_regression)
test_dataset = KELSDataSet(X_tests,y_test,is_regression=is_regression)


batch_size = 32
hidden_features = 100
embbed_dim = 72



train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
split_list = train_dataset.split_list
#embedding_networks : 년차별로 맞는 mlp 리스트. 리스트 내용물에 따라 인풋 채널 개수 다름.

sample_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
assert batch_size
(sample,label) = next(iter(sample_loader))
sample_datas = batch_to_splited_datas(sample,split_list)


#embbeding_networks = make_embbeding_networks(sample_datas,batch_size=batch_size,hidden_features = hidden_features,out_features=embbed_dim)
#embbeding network 사용법 : 
#1. train_loader에서 배치를 받는다. 이 배치는 (batch, 학생수, features)인데, features는 모든 년차별 feature가 concat된 상태.
#2. 이 배치를 batch_to_splited_datas에 넣는다. datas = batch_to_splited_datas(배치,split_list)
# 그러면 datas에 년차별 feature가 나누어진 리스트가 생기게 된다. 리스트 길이는 년차 길이고, 내용물은 행렬
# 3. 이 datas를 batch_to_embbedings 에 넣는다. 그럼 각 행렬이 embbeding_networks에 든 mlp를 통과하여 같은 크기의 행렬들을 리턴한다. 
# emb_batch_list= batch_to_embbedings(datas,embbeding_networks)
#4. emb_batch_list의 내용물이 바로 contrastive loss에 들어갈 벡터들이 된다. 

#5. lstm에 넣으려면 얘들을 seq_len 방향으로 쌓아야 한다. 
#emb_batched_seq = torch.stack(emb_batch_list).transpose(0,1) 으로 쌓고 lstm에 넣으면 완성..?



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
        val_acc = []
        #optimizer = optimizer_cls(model.parameters(),lr=lr,weight_decay=weight_decay)
        optimizer = optimizer_cls(model.parameters(),lr=lr)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[25,40,60,80], gamma=0.5,last_epoch=-1)
        

        for epoch in range(n_iter):
                running_loss = 0.0
                model.train()
                n = 0
                n_acc = 0
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

                        # Calculate Loss: softmax --> cross entropy loss
                        loss = criterion(outputs, yy)


                        # Getting gradients w.r.t. parameters
                        loss.backward()

                        # Updating parameters
                        optimizer.step()
                        
                        
                        
                        i += 1
                        n += len(xx)
                        _, y_pred = outputs.max(1)
                        n_acc += (yy == y_pred).float().sum().item()
                #scheduler.step()
                train_losses.append(running_loss/i)
                train_acc.append(n_acc/n)

                val_acc.append(eval_net(model,test_loader,device,mode = mode))

                print(f'epoch : {epoch}, train_acc : {train_acc[-1]}, validation_acc : {val_acc[-1]}',flush = True)



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
    acc = accuracy_roughly(ypreds,ys)
    #acc= (ys == ypreds).float().sum() / len(ys)

    # print(sklearn.metrics.confusion_matrix(ys.numpy(),ypreds.numpy()))


    # print(sklearn.metrics.classification_report(ys.numpy(),ypreds.numpy()))
    

    return acc
    #return acc.item()



model_E = ViT_LRP_nan_excluded.VisionTransformer(sample_datas,split_list,seq_len=6, num_classes=9, embed_dim=16*3, depth=8,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, mlp_head=False, drop_rate=0.2, attn_drop_rate=0.2)
model_E = model_E.to(device)

train_net(model_E,train_loader,test_loader,n_iter=100,device=device,mode='E',lr=0.0001,optimizer_cls = optim.AdamW)



model_K = ViT_LRP_nan_excluded.VisionTransformer(sample_datas,split_list,seq_len=6, num_classes=9, embed_dim=16*3, depth=8,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, mlp_head=False, drop_rate=0.2, attn_drop_rate=0.2)
model_K = model_K.to(device)

train_net(model_K,train_loader,test_loader,n_iter=100,device=device,mode='K',lr=0.0001,optimizer_cls = optim.AdamW)



xx,(label_E,label_K,label_M) = next(iter(train_loader))

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



MLP = MLP()


train_net(MLP,train_loader,test_loader,n_iter=100,device=device,mode='E',lr=0.0001)





