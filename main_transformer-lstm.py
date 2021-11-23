import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset 
import tqdm
from torch.autograd import Variable
import argparse
import math
import torch.nn.functional as F

torch.random.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser("Transformer-LSTM")
parser.add_argument("-data_path", type=str, default="/home/cjd/jinrong/stocks/shangzheng.csv", help="dataset path")

args = parser.parse_args()
time_step = 10

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(0), :]
class TransAm(nn.Module):
    def __init__(self,feature_size=64,num_layers=6,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
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
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class AttnDecoder(nn.Module):

    def __init__(self, code_hidden_size, hidden_size, time_step):
        super(AttnDecoder, self).__init__()
        self.code_hidden_size = code_hidden_size
        self.hidden_size = hidden_size
        self.T = time_step

        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=code_hidden_size)
        self.attn2 = nn.Linear(in_features=code_hidden_size, out_features=code_hidden_size)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=code_hidden_size, out_features=1)
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size,num_layers=1)
        self.tilde = nn.Linear(in_features=self.code_hidden_size + 1, out_features=1)
        self.fc1 = nn.Linear(in_features=code_hidden_size + hidden_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, h, y_seq):
        h_ = h.transpose(0,1) 
        batch_size = h.size(0)
        d = self.init_variable(1, batch_size, self.hidden_size)
        s = self.init_variable(1, batch_size, self.hidden_size)
        h_0 = self.init_variable(1,batch_size, self.hidden_size)
        h_ = torch.cat((h_0,h_),dim=0)

        for t in range(self.T):
            x = torch.cat((d,h_[t,:,:].unsqueeze(0)), 2)
            h1 = self.attn1(x)
            _, states = self.lstm(y_seq[:,t].unsqueeze(0).unsqueeze(2), (h1, s))
            d = states[0]
            s = states[1]
        y_res = self.fc2(self.fc1(torch.cat((d.squeeze(0), h_[-1,:,:]), dim=1)))
        return y_res

    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        return Variable(zero_tensor)

    def embedding_hidden(self, x):
        return x.permute(1, 0, 2)


class StockDataset(Dataset):
    def __init__(self, file_path, T=time_step, train_flag=True):
        # read data
        with open(file_path, "r", encoding="GB2312") as fp:
            data_pd = pd.read_csv(fp)
        self.train_flag = train_flag
        self.data_train_ratio = 0.9
        self.T = T # use 10 data to pred
        if train_flag:
            self.data_len = int(self.data_train_ratio * len(data_pd))
            data_all = np.array(data_pd['close'])
            data_all = (data_all-np.mean(data_all))/np.std(data_all)
            self.data = data_all[:self.data_len]
        else:
            self.data_len = int((1-self.data_train_ratio) * len(data_pd))
            data_all = np.array(data_pd['close'])
            data_all = (data_all-np.mean(data_all))/np.std(data_all)
            self.data = data_all[-self.data_len:]
        print("data len:{}".format(self.data_len))


    def __len__(self):
        return self.data_len-self.T

    def __getitem__(self, idx):
        return self.data[idx:idx+self.T], self.data[idx+self.T]


def l2_loss(pred, label):
    loss = torch.nn.functional.mse_loss(pred, label, size_average=True)
    return loss

def train_once(encoder,decoder, dataloader, encoder_optim,decoder_optim):
    encoder.train()
    decoder.train()
    loader = tqdm.tqdm(dataloader)
    loss_epoch = 0
    for idx, (data, label) in enumerate(loader):
        data_x = data.unsqueeze(2)
        data_tran = data_x.transpose(0,1)
        data_x, label ,data_y= data_tran.float(), label.float() ,data.float()
        code_hidden = encoder(data_x)
        code_hidden = code_hidden.transpose(0,1)
        output = decoder(code_hidden, data_y)
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
        loss = l2_loss(output.squeeze(1), label)
        loss.backward()
        encoder_optim.step()
        decoder_optim.step()
        loss_epoch += loss.detach().item()
    loss_epoch /= len(loader)
    return loss_epoch


def eval_once(encoder,decoder ,dataloader):
    encoder.eval()
    decoder.eval()
    loader = tqdm.tqdm(dataloader)
    loss_epoch = 0
    preds = []
    labels = []
    for idx, (data, label) in enumerate(loader):
        # data: batch, time x 1
        data_x = data.unsqueeze(2)
        data_x, label ,data_y= data_x.float(), label.float(),data.float()
        code_hidden = encoder(data_x)
        output = decoder(code_hidden, data_y).squeeze(1)
        loss = l2_loss(output, label)
        #print('##loss',loss)
        loss_epoch += loss.detach().item()
        preds+=(output.detach().tolist())
        labels+=(label.detach().tolist())
    preds = torch.Tensor(preds)
    labels = torch.Tensor(labels)
    pred1 = preds[:-1]
    pred2 = preds[1:]
    pred_ = preds[1:]>preds[:-1]
    label1 = labels[:-1]
    label2 = labels[1:]
    label_ = labels[1:]>labels[:-1]
    accuracy = (label_ == pred_).sum()/len(pred1)
    loss_epoch /= len(loader)
    return loss_epoch,accuracy

def eval_plot(encoder,decoder ,dataloader):
    dataloader.shuffle = False
    preds = []
    labels = []
    encoder.eval()
    decoder.eval()
    loader = tqdm.tqdm(dataloader)
    for idx, (data, label) in enumerate(loader):
        data_x = data.unsqueeze(2)
        data_x, label ,data_y= data_x.float(), label.float(),data.float()
        code_hidden = encoder(data_x)
        output = decoder(code_hidden, data_y)
        preds+=(output.detach().tolist())
        labels+=(label.detach().tolist())
    fig, ax = plt.subplots()
    data_x = list(range(len(preds)))
    ax.plot(data_x, preds, label='predict', color='red')
    ax.plot(data_x, labels,label='ground truth', color='blue')
    plt.savefig('results/shangzheng-tran-lstm.png' )
    plt.legend()
    plt.show()
def main():
    dataset_train = StockDataset(file_path=args.data_path)
    dataset_val = StockDataset(file_path=args.data_path, train_flag=False)
    #print('###1',len(dataset_train))
    
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False)
    encoder = TransAm()
    decoder = AttnDecoder(code_hidden_size=64, hidden_size=64, time_step=time_step)
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optim = torch.optim.Adam(decoder.parameters(), lr=0.001)
    
    total_epoch = 201
    #eval_plot(model, val_loader)
    for epoch_idx in range(total_epoch):
        train_loss = train_once(encoder,decoder, train_loader, encoder_optim,decoder_optim)
        print("stage: train, epoch:{:5d}, loss:{}".format(epoch_idx, train_loss))
        if epoch_idx%5==0:
            eval_loss,accuracy = eval_once(encoder,decoder, val_loader)
            print("####stage: test, epoch:{:5d}, loss:{},accuracy:{}".format(epoch_idx, eval_loss,accuracy))
            eval_plot(encoder,decoder, val_loader)

if __name__ == "__main__":
    main()


