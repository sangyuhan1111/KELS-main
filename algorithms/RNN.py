import torch
import torch.nn as nn

inputs_test = torch.Tensor(1, 10, 5)
# print(inputs_test.shape[2])

# one way
def oneRNN(input, hidden_size) : 
  input_size = (input.shape[2]) # input shape
  # (batch_size, time_steps, input_size)
  cell = nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = 2, batch_first=True)
  outputs, _status = cell(input)
  return outputs

# bi direc.
def biRNN(input, hidden_size) : 
  input_size = (input.shape[2]) # input shape
  # (batch_size, time_steps, input_size)
  cell = nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = 2, batch_first=True, bidirectional = True)
  outputs, _status = cell(input)
  return outputs


class VanillaRNN(nn.Module):
  
  def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
    super(VanillaRNN, self).__init__()
    self.device = device
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length, 1), nn.Sigmoid())

  def forward(self, x):
    ###
    # Encoder
    ###
    h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device) # 초기 hidden state 설정하기.
    out, _ = self.rnn(x, h0) # out: RNN의 마지막 레이어로부터 나온 output feature 를 반환한다. hn: hidden state를 반환한다.
    out = out.reshape(out.shape[0], -1) # many to many 전략
    out = self.fc(out)
    return out


# class EncoderRNN(nn.Module):
#   def __init__(self, device):
#     self.device = device
