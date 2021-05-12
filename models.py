import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
class MLP(nn.Module):
    def __init__(self, input_size=10, layer1_size=64, layer2_size=16, output_size=1):

        super(MLP, self).__init__()

        self.input_size = input_size       
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.output_size = output_size

        self.layer1 = nn.Linear(input_size, layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.output = nn.Linear(layer2_size, output_size)

    def forward(self, x):

        x = F.dropout(self.layer1(x), p=0.1)
        x = F.tanh(x)
        x = self.layer2(x)
        x = F.tanh(x)
        x = self.output(x).squeeze(1)
        return x


class CNN(nn.Module):
    def __init__(self,dropout=0.1):
        super(CNN, self).__init__()
        self.emb_layer = nn.Linear(1, 3)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU())
            #nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU())
            #nn.Dropout(p=dropout),
            #nn.MaxPool1d(2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(2))
        self.conv_last = nn.Conv1d(8, 1, kernel_size=1, padding=0)
        self.fc = nn.Linear(10, 1)
        #self.gamma = torch.nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        #print(x)
        x_ = x.unsqueeze(2)
        x_emb = x_.transpose(1,2)
        out1 = self.layer1(x_emb) 
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = self.conv_last(out3) + x_emb 
        out_last = self.fc(out.squeeze())

        return out_last

class RNN(nn.Module):
    def __init__(self, rnn_layer=2, input_size=1, hidden_size=4):
        super(RNN, self).__init__()
        self.rnn_layer = rnn_layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size = self.input_size,  #每个字母的向量长度
            hidden_size=self.hidden_size,  # RNN隐藏神经元个数
            num_layers=self.rnn_layer,  # RNN隐藏层个数
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, 1)
    def init_hidden(self, x):
        batch_size = x.shape[0]
        init_h = torch.zeros(self.rnn_layer, batch_size, self.hidden_size, device=x.device).requires_grad_()
        return init_h
    def forward(self, x, h=None):
        x = x.unsqueeze(2)
        h = h if h else self.init_hidden(x)
        out, h = self.rnn(x, h)
        out = self.fc(out[:,-1,:]).squeeze(1)
        return out

class LSTM(nn.Module):
    def __init__(self, lstm_layer=2, input_dim=1, hidden_size=8):
        super(LSTM, self).__init__()
        self.hidden_size=hidden_size
        self.lstm_layer = lstm_layer
        self.emb_layer = nn.Linear(input_dim, hidden_size)
        self.out_layer = nn.Linear(hidden_size, input_dim)
        self.lstm = nn.LSTM(input_size=rnn_unit, hidden_size=hidden_size, num_layers=self.lstm_layer, batch_first=True)
    
    def init_hidden(self, x):
        batch_size = x.shape[0]
        init_h = (torch.zeros(self.lstm_layer, batch_size, self.hidden_size, device=x.device),
                torch.zeros(self.lstm_layer, batch_size, self.hidden_size, device=x.device))
        return init_h

    def forward(self, x, h=None):
        # batch x time x dim
        x = x.unsqueeze(2)
        h = h if h else self.init_hidden(x)
        x = self.emb_layer(x)        
        output, hidden = self.lstm(x, h)        
        out = self.out_layer(output[:,-1,:]).squeeze(1)
        return out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #print('##posi',position.shape)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #print('##div_term',div_term.shape)
        #print((position * div_term).shape)
        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Transformer(nn.Module):
    def __init__(self,feature_size=64,num_layers=4,dropout=0.1):
        super(Transformer, self).__init__()
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        src = src.unsqueeze(2)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)

        #print('##src',src.shape,self.src_mask.shape)
        output_1 = self.transformer_encoder(src)   #, self.src_mask)

        output = self.decoder(output_1[-1]).squeeze(1)
        return output
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask