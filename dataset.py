import pandas as pd
import numpy as np
import torch
import math
from torch.utils.data import DataLoader, Dataset
class StockDataset(Dataset):
    def __init__(self, file_path, time_step=10, train_flag=True):
        # read data
        with open(file_path, "r", encoding="GB2312") as fp:
            data_pd = pd.read_csv(fp)
        self.train_flag = train_flag
        self.data_train_ratio = 0.9
        self.T = time_step # use 10 data to pred
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