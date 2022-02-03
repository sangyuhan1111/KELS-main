import torch
import torch.nn as nn
import torch.nn.functional as F

class KELS_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, device):
        super(KELS_LSTM, self).__init__()
        
        # batch size is 1 (sequence length is variable)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device
        self.layer_num = 1
        
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, batch_first=True)
        
        
    def forward(self, input):
        input = torch.tensor(input, dtype=torch.float32).to(self.device)
        input = input.unsqueeze(0)
        h0 = torch.zeros(self.layer_num, self.batch_size, self.hidden_size, requires_grad=True).to(self.device)
        c0 = torch.zeros(self.layer_num, self.batch_size, self.hidden_size, requires_grad=True).to(self.device)
        
        output, _ = self.lstm(input, (h0, c0))
        output = F.softmax(output[0], dim=1)
        
        max_idx = torch.argmax(output, dim=1, keepdim=True)
        pred = torch.zeros_like(output)
        pred.scatter_(1, max_idx, 1)
        pred = torch.tensor(pred, dtype=torch.float32, requires_grad=True).to(self.device)
        
        ##ADD FC LAYER TO LABEL DETECT
        
        return pred


class KELS_LSTM_(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, device):
        super(KELS_LSTM_, self).__init__()
        
        # batch size is 1 (sequence length is variable)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device
        self.layer_num = 1
        
        # self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, batch_first=True)
        self.lstm = nn.LSTM(input_size = 5, hidden_size = self.hidden_size, batch_first=True)
        # self.fc1 = nn.Linear(hidden_size, )
        
        
    def forward(self, input, year):
        input = torch.tensor(input, dtype=torch.float32).to(self.device)
        input = input.unsqueeze(0)
        h0 = torch.zeros(self.layer_num, self.batch_size, self.hidden_size, requires_grad=True).to(self.device)
        c0 = torch.zeros(self.layer_num, self.batch_size, self.hidden_size, requires_grad=True).to(self.device)
        nn.init.xavier_uniform_(h0)
        nn.init.xavier_uniform_(c0)
        hidden = (h0, c0)
        
        for idx, _ in enumerate(range(len(year))):
            ## MODIFY ENCODER ##
            input_ = input[0, idx].reshape((5, 5))
            output, hidden = self.lstm(input_.unsqueeze(0), hidden)
            
        h_t, c_t = hidden
        
        
        
        
        h_t = h_t.flatten()
        
        pred = F.softmax(h_t)
        pred = torch.tensor(pred, dtype=torch.float32, requires_grad=True).to(self.device)
                
        # output = F.softmax(output, dim=1)
        # output = torch.tensor(output, dtype=torch.float32, requires_grad=True).to(self.device)
        
        # max_idx = torch.argmax(output, dim=1, keepdim=True)
        # pred = torch.zeros_like(output)
        # pred.scatter_(1, max_idx, 1)
        # pred = torch.tensor(pred, dtype=torch.float32, requires_grad=True).to(self.device)
        
        return pred
    
