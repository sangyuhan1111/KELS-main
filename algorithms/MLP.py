import torch
import torch.nn as nn
import torch.nn.functional as F

class KELS_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, target_size, batch_size, device):
        super(KELS_MLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.target_size = hidden_size
        self.batch_size = batch_size
        self.device = device
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        # self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, target_size)
        
    def forward(self, input):
        input = torch.tensor(input, dtype=torch.float32).to(self.device)
        input = input.flatten()
        
        x = F.softmax(self.layer1(input))

        x = F.softmax(self.layer2(x))
        output = F.softmax(x)

        
        return output
