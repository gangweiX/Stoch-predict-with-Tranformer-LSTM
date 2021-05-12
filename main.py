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
from tensorboardX import SummaryWriter
checkpointdir = './checkpoints/'
stock_file = '/home/cjd/jinrong/stocks/shangzheng.csv'
logger = SummaryWriter(checkpointdir)

def l2_loss(pred, label):
    loss = torch.nn.functional.mse_loss(pred, label, size_average=True)
    return loss
def train(model, dataloader, optimizer):
    model.train()
    loader = tqdm.tqdm(dataloader)
    loss_epoch = 0
    for idx, (data, label) in enumerate(loader):
        data, label = data.float(), label.float()
        output = model(data)
        optimizer.zero_grad()
        loss = l2_loss(output, label)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.detach().item()
        #print('loss',loss)

    loss_epoch /= len(loader)
    return loss_epoch
def eval(model, dataloader):
	model.eval()
	loader = tqdm.tqdm(dataloader)
	loss_epoch = 0
	for idx, (data, label) in enumerate(loader):
		data, label = data.float(), label.float()
		output = model(data)
		loss = l2_loss(output, label)
		loss_epoch += loss.detach().item()
	loss_epoch /= len(loader)
	return loss_epoch
def main():
	# dataset
	dataset_train = StockDataset(file_path = stock_file,time_step = 10)
	dataset_test = StockDataset(file_path = stock_file,time_step = 10, train_flag=False)
	###if MLP,CNN,batch_size = 1
	train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
	test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)
	####################################################################
	# model is one of MLP,CNN,RNN,LSTM,Transformer
	#model = MLP()
	#model = CNN()
	model = RNN(rnn_layer=2, input_size=1, hidden_size=4)
	#model = LSTM(lstm_layer=2, input_dim=1, hidden_size=8)
	#model = Transformer(feature_size=64, num_layers=4, dropout=0.1)
	####################################################################
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	total_epoch = 200
	for epoch_idx in range(total_epoch):
		train_loss = train(model, train_loader, optimizer)
		print("stage: train, epoch:{:5d}, loss:{}".format(epoch_idx, train_loss))
		logger.add_scalar('Train/Loss', train_loss, epoch_idx)
		if epoch_idx%10==0:
			eval_loss = eval(model, test_loader)
			print("stage: test, epoch:{:5d}, loss:{}".format(epoch_idx, eval_loss))
			torch.save(model.state_dict(), "{}/checkpoint_{:0>3}.ckpt".format(checkpointdir, epoch_idx))
			logger.add_scalar('Test/Loss', eval_loss, epoch_idx)

if __name__ == '__main__':
	main()
