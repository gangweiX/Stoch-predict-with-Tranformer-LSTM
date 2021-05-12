import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import tqdm
import math
from models import MLP, CNN, RNN, LSTM, Transformer
from dataset import StockDataset
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
#checkpointdir = './checkpoints/'
stock_file = '/home/cjd/jinrong/stocks/shangzheng.csv'
loadckpt = '/home/cjd/jinrong/checkpoints/checkpoint_190.ckpt'

def plot():
    # dataset
    dataset_test = StockDataset(file_path = stock_file, time_step = 10, train_flag=False)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)
    loader = tqdm.tqdm(test_loader)
    ##################################################################
    ########  model is one of MLP,CNN,RNN,LSTM,Transformer
    #model = MLP()
    #model = CNN()
    model = RNN(rnn_layer=2, input_size=1, hidden_size=4)
    #model = LSTM(lstm_layer=2, input_dim=1, hidden_size=8)
    #model = Transformer(feature_size=64, num_layers=4, dropout=0.1)
    ##################################################################
    model.load_state_dict(torch.load(loadckpt))
    preds = []
    labels = []
    for idx, (data, label) in enumerate(loader):
        data, label = data.float(), label.float()
        output = model(data)
        preds += (output.detach().tolist())
        labels += (label.detach().tolist())
    fig, ax = plt.subplots()
    data_x = list(range(len(preds)))
    ax.plot(data_x[-60:], preds[-60:], label='predict', color='red')
    ax.plot(data_x[-60:], labels[-60:],label='ground truth', color='blue')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    plot()