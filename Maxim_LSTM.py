import pandas as pd
import argparse
from rich import print
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision import models as models
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pretty_errors
from torch.utils.data import Dataset, DataLoader

dataPath = '../MAXIM_data/sample_pre.pkl'

dataPre = pd.read_pickle(dataPath)
dataPre = dataPre.reset_index()


print(dataPre)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.cuda.get_device_name(0))

train_train_ratio = 0.80
train_test_ratio = 0.10
train_valid_ratio = 0.10


dataPreGT = dataPre['class_gt']
# print(dataPre.keys())
# dataPreValues = dataPre.drop(columns=['class_gt', 'index', 'Time','Sample Count','Activity', 'SpO2 State', 'SCD State ', 'SAMPLE Time'])
dataPreValues = dataPre.drop(columns=['index','Time',' Sample Count', ' Activity', ' SpO2 Confidence (%)', ' SpO2 (%)', ' SpO2 Percent Complete',
       ' Low Signal Quality', ' Motion Flag', ' WSPO2 Low Pi', ' Unreliable R',
       ' SpO2 State', ' SCD State', ' SAMPLE Time', ' Walk Steps',
       ' Run Steps', ' KCal', ' Tot. Act. Energy', ' Ibi Offset', ' R Value', ' HR Confidence (%)',' RR', ' RR Confidence (%)', ' Operating Mode'])
# print(dataPreValues.keys())

min_max_scaler = MinMaxScaler()

def minMax(dfIn, scaler):
    dfgt = dfIn['class_gt'].reset_index()
    dfval = dfIn.drop(columns=['class_gt']).reset_index()
    fitted = min_max_scaler.fit(dfval)
    output = scaler.transform(dfval)
    output = pd.DataFrame(output, columns=dfval.columns, index=list(dfval.index.values))
    # output = pd.DataFrame(output, columns=dfval.columns)
    output = pd.concat([output.reset_index() , dfgt.reset_index()], axis=1)
    return output



dataPreValues = minMax(dataPreValues, min_max_scaler)

print(len(dataPreValues))
# 훈련/테스트 분할
dfTrain, dfTest = train_test_split(dataPreValues, train_size = train_train_ratio, shuffle=False)

# 훈련/검증 분할
dfTest, dfValid = train_test_split(dfTest, train_size = 0.5, shuffle=False)


dfTrain =  dfTrain.drop(columns = ['level_0', 'index'])
dfTest =  dfTest.drop(columns = ['level_0', 'index'])
dfValid =  dfValid.drop(columns = ['level_0', 'index'])



print(len(dfTrain))
print(len(dfValid))
print(len(dfTest))


dfTrain = dfTrain.iloc[0:496800, :]
dfValid = dfValid.iloc[0:61200, :]
dfTest = dfTest.iloc[0:61200, :]


print(len(dfTrain))
print(len(dfValid))
print(len(dfTest))


# BATCH_SIZE = 64

#Load an iterator
# train_iterator, valid_iterator = data.BucketIterator.splits(
#     (dfTrain, dfValid), 
#     batch_size = BATCH_SIZE,
#     sort_key = lambda x: len(x.class_gt),
#     sort_within_batch=True,
#     device = device)





# feature_cols = [' Green Count', ' Green2 Count', ' IR Count',
#        ' Red Count', ' X Axis Acceleration (g)', ' Y Axis Acceleration (g)',
#        ' Z Axis Acceleration (g)', ' Heart Rate (bpm)',
#        'class_gt']
# label_cols = ['class_gt']


def labeling(dfIn):
    dfIn = dfIn.replace('SLEEP-S0', 0)
    dfIn = dfIn.replace('SLEEP-S1', 1)
    dfIn = dfIn.replace('SLEEP-S2', 2)
    dfIn = dfIn.replace('SLEEP-S3', 3)
    dfIn = dfIn.replace('SLEEP-REM', 4)
    dfIn = dfIn.reset_index(drop=True)
    return dfIn


# def labeling(dfIn):
#     dfIn = dfIn.replace('SLEEP-S0', 0)
#     dfIn = dfIn.replace('SLEEP-S1', 0)
#     dfIn = dfIn.replace('SLEEP-S2', 0)
#     dfIn = dfIn.replace('SLEEP-S3', 0)
#     dfIn = dfIn.replace('SLEEP-REM', 1)
#     dfIn = dfIn.reset_index(drop=True)
#     return dfIn


X_train = dfTrain.drop(columns='class_gt').reset_index(drop=True) 
y_train = dfTrain.iloc[:, 8:9]
y_train = labeling(y_train)

# one_hot_encoding(y_train)


# print(X_train)
print(y_train)


X_val = dfValid.drop(columns='class_gt').reset_index(drop=True) 
y_val = dfValid.iloc[:, 8:9]
y_val = labeling(y_val)


X_test = dfTest.drop(columns='class_gt').reset_index(drop=True) 
y_test = dfTest.iloc[:, 8:9]
y_test = labeling(y_test)


# print("Training Shape", X_train.shape, y_train.shape) 
# print("Testing Shape", X_test.shape, y_test.shape)





list_y = list(range(0,len(y_train), 3600))
print("y_val.index")
print(list_y)

y_train = y_train.iloc[list_y, :]

print(y_train)

X_train_tensors = Variable(torch.Tensor(X_train.values).float())
X_test_tensors = Variable(torch.Tensor(X_test.values).float())

y_train_tensors = Variable(torch.Tensor(y_train.values).float()).to(device)
y_test_tensors = Variable(torch.Tensor(y_test.values).float()).to(device)

print(X_test_tensors.shape)

# batch  = 138 / 17, seq length = 3600 , features  = 8 

X_train_tensors_final = torch.reshape(X_train_tensors, (138, 3600, 8)).to(device)
X_test_tensors_final = torch.reshape(X_test_tensors, (17, 3600 , 8)).to(device) 


# def sort3600(dfIn):
#     dfIn = dfIn

# print(len(X_train_tensors))
# print(X_train_tensors.shape)

# print(len(X_test_tensors_final))
# print(X_test_tensors_final.shape)

# print("y_train.shape")
# print(y_train.shape)
# print(y_train)

# print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape) 
# print("Training Shape", X_train_tensors_final.dtype, y_train_tensors.dtype) 
# print("Testing Shape", X_test_tensors_final.dtype, y_test_tensors.dtype)


class LSTM1(nn.Module): 
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length): 
        super(LSTM1, self).__init__() 
        self.num_classes = num_classes #number of classes 
        self.num_layers = num_layers #number of layers 
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state 
        self.seq_length = seq_length #sequence length 
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 = nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer
        self.softmax = nn.LogSoftmax()
        self.relu = nn.ReLU() 
    # def init_hidden(self, batch_size):
    #     return(Variable(torch.randn(1, batch_size, self.hidden_dim)),
	# 					Variable(torch.randn(1, batch_size, self.hidden_dim)))

    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float()).to(device) #hidden state 
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float()).to(device) #internal state # Propagate input through LSTM
        # print(h_0)
        # print(c_0)
        # self.hidden = self.init_hidden(x.size(0))
        output, (hn, cn) =  self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state 
        # print('hn')
        # print(hn)
        
        # print('cn')
        # print(cn)
        
        # print('output')
        # print(output)

        # hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next 
        # out = self.relu(output) 
        # print("output before")
        # print(output)
        output = output[:, -1 , : ]
        print(output.shape)
        # print("output after")
        # print("output")
        # print(output.shape)
        out = self.fc_1(output) #first Dense
        # print("out 1")
        # print(out.shape)
        out = self.relu(out) #relu 
        # print("out 2")
        # print(out.shape)
        out = self.fc(out) #Final Output 
        # print("out 3")
        # print(out.shape)
        out = self.softmax(out) #Final Output
        
        return out




num_epochs = 1000 #1000 epochs
learning_rate = 0.001 #0.    01 lr

input_size = 8 #number of features
hidden_size = 8 #number of features in hidden state
num_layers = 3 #number of stacked lstm layers

num_classes = 5 #number of output classes 
seq_length = 1000
# lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]).to(device)

# print(X_train_tensors_final)
# print(X_train_tensors_final[1])

lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final[1]).to(device)

# loss_function = torch.nn.MSELoss(reduction="sum")    # mean-squared error for regression
# optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)  # adam optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm1.parameters(), lr=0.01)


class TrainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)

print(X_train_tensors_final.shape)
print(y_train.shape)

# train data loader
train_data = TrainData(X_train_tensors_final, 
                       torch.FloatTensor(y_train.to_numpy()))     
train_loader = DataLoader(dataset=train_data, batch_size=138, shuffle=False)

# test data loader
# test_data = TestData(torch.FloatTensor(X_test_tensors_final))
# test_loader = DataLoader(dataset=test_data, batch_size=17, shuffle=False)

def binary_acc(y_pred, y_test):
    print("y_pred")
    print(y_pred)
    # print(y_test)
    # y_pred_tag = torch.round(torch.sigmoid(y_pred))
    y_pred_tag = torch.argmax(y_pred, dim=1)
    y_test = torch.argmax(y_test, dim=1)
    print("y_pred_tag")
    print(y_pred_tag)
    print("y_test")
    print(y_test)
    # y_pred_tag= F.one_hot(y_pred_tag.to(torch.int64), num_classes=5).to(device)
    # print(y_pred_tag)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/len(y_test)
    acc = torch.round(acc * 100)
    
    return acc


for e in range(1, 99+1):
    epoch_loss = 0
    epoch_acc = 0

    for X_batch, y_batch in train_loader:
        # print(X_batch)
        # print(y_batch)
        # y_batch = y_batch()
        y_one_hot = F.one_hot(y_batch.to(torch.int64), num_classes=5)
        # y_one_hot[:,y_batch] = 1
        # X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        X_batch, y_batch = X_batch.to(device), y_one_hot.to(device)
        # print(y_batch)
        y_batch =  y_batch.squeeze()
        y_batch = y_batch.to(torch.float)
        optimizer.zero_grad()
        
        y_pred = lstm1.forward(X_batch)
        # y_pred = y_pred.squeeze()
        # print("y_batch.shape")
        # print(y_batch.shape)
        # print("y_pred.shape")
        # print(y_pred.shape)
        loss = criterion(y_pred, y_batch)
        acc = binary_acc(y_pred, y_batch)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')


# lstm1 = LSTM()
# loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(num_epochs): 
#     outputs = lstm1.forward(X_train_tensors_final) #forward pass 
#     print("final")
#     print(outputs.shape)
#     print(len(outputs))
#     optimizer.zero_grad() #caluclate the gradient, manually setting to 0 
#     # obtain the loss function 
#     # print(outputs)
#     # print(outputs.shape)
#     # print(y_train_tensors)
#     # print(y_train_tensors.shape)
#     print("y")
#     print(len(y_train_tensors))
#     # print(y_train_tensors)
#     loss = loss_function(outputs, y_train_tensors) 
#     loss.backward() #calculates the loss of the loss function 
#     optimizer.step() #improve from loss, i.e backprop 
#     # if epoch % 100 == 0: print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
#     print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

# df_X_ss = ss.transform(df.drop(columns='Volume'))
# df_y_mm = mm.transform(df.iloc[:, 5:6])

# df_X_ss = Variable(torch.Tensor(sA1_rs)) #converting to Tensors
# df_y_mm = Variable(torch.Tensor(sA1_rs))

# #reshaping the dataset

# df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))
# train_predict = lstm1(df_X_ss.to(device))#forward pass
# data_predict = train_predict.data.detach().cpu().numpy() #numpy conversion
# dataY_plot = df_y_mm.data.numpy()

# data_predict = mm.inverse_transform(data_predict) #reverse transformation
# dataY_plot = mm.inverse_transform(dataY_plot)

# plt.figure(figsize=(10,6)) #plotting
# # plt.axvline(x=tr_len, c='r', linestyle='--') #size of the training set
# print(data_predict)
# # print(sA1)
# plt.scatter(tA1, sA1, label='Actual Data') #actual plot
# plt.plot(data_predict, label='Predicted Data') #predicted plot
# plt.title('Time-Series Prediction')
# plt.legend()
# plt.show()


# plt.savefig("")