import torch
import torch.nn as nn
import argparse
import pretty_errors
import pandas as pd
import numpy as np
import pandas as pd
import os
from utils.utils import *
import torch.optim as optim
from algorithms.RNN import VanillaRNN
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

#https://data-science-hi.tistory.com/190

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true',help='run in cpu') 
args = parser.parse_args()

GPU_NUM = 0 

if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda')
    print(torch.cuda.get_device_name(GPU_NUM), "allocated in ", torch.cuda.current_device())


path_merge = '../KELS_data/preprocessed/merge/inner'
df_input = pd.read_csv(os.path.join(path_merge, 'input_merge.csv'))
df_input = df_input.set_index('L2SID', drop=True)
df_label = pd.read_csv(os.path.join(path_merge, 'label_merge.csv'))
df_label = df_label.set_index('L2SID', drop=True)
df_label = df_label.astype(int)


sss = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
X, y = df_input, df_label["L2Y6_K_CS"]
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

y_test = y_test.to_numpy()
print("train size:", X_train.shape)
print("test size:", X_test.shape)

print(X_train)
print(X_test)


def seq_data(x, y, sequence_length):
  x_seq = []
  y_seq = []
  for i in range(len(x) - sequence_length):
    x_seq.append(x[i: i+sequence_length])
    y_seq.append(y[i+sequence_length])

  return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view([-1, 1])


split = 200
sequence_length = 5

x_seq, y_seq = seq_data(X_train, y_train, sequence_length)

x_train_seq = x_seq[:split]
y_train_seq = y_seq[:split]
x_test_seq = x_seq[split:]
y_test_seq = y_seq[split:]
print(x_train_seq.size(), y_train_seq.size())
print(x_test_seq.size(), y_test_seq.size())



batch_size = 20
train_loader = torch.utils.data.DataLoader(dataset=X_train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=X_test, batch_size=batch_size, shuffle=False)

input_size = X_train.size(2) 
num_layers = 2
hidden_size = 8



# print(input_size)


model = VanillaRNN(input_size=input_size,
                   hidden_size=hidden_size,
                   sequence_length=sequence_length,
                   num_layers=num_layers,
                   device=device).to(device)


criterion = nn.MSELoss()

lr = 1e-3
num_epochs = 200
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_graph = [] # 그래프 그릴 목적인 loss.
n = len(train_loader)

for epoch in range(num_epochs):
  running_loss = 0.0

  for data in train_loader:
    # seq_ = len(data['year'])
    

    seq, target = data # 배치 데이터.
    ###
    # seq <- input_size , 
    ###
    out = model(seq)   # 모델에 넣고,
    loss = criterion(out, target) # output 가지고 loss 구하고,

    optimizer.zero_grad() # 
    loss.backward() # loss가 최소가 되게하는 
    optimizer.step() # 가중치 업데이트 해주고,
    running_loss += loss.item() # 한 배치의 loss 더해주고,

  loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
  if epoch % 100 == 0:
    print('[epoch: %d] loss: %.4f'%(epoch, running_loss/n))

plt.figure(figsize=(20,10))
plt.plot(loss_graph)
plt.savefig('loss.png', dpi=300)


def plotting(train_loader, test_loader, actual):
  with torch.no_grad():
    train_pred = []
    test_pred = []

    for data in train_loader:
      seq, target = data
      out = model(seq)
      train_pred += out.cpu().numpy().tolist()

    for data in test_loader:
      seq, target = data
      out = model(seq)
      test_pred += out.cpu().numpy().tolist()
      
  total = train_pred + test_pred
  plt.figure(figsize=(20,10))
  plt.plot(np.ones(100)*len(train_pred), np.linspace(0,1,100), '--', linewidth=0.6)
  plt.plot(actual, '--')
  plt.plot(total, 'b', linewidth=0.6)

  plt.legend(['train boundary', 'actual', 'prediction'])
  plt.savefig('fig1.png', dpi=300)

plotting(train_loader, test_loader, df['Close'][sequence_length:]
         
         
