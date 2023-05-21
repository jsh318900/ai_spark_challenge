import random
import torch
from torch.utils.data import Dataset, Sampler, DataLoader, Subset

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def combine_df(root_dir, metadata_path):
    dfs = []
    meta_df = pd.read_csv(metadata_path)
    for i in range(meta_df.shape[0]):
        csv_path = meta_df.iloc[i]['Location'] + '.csv'
        measure_df = pd.read_csv(root_dir + csv_path)
        measure_df['Latitude'] = meta_df.iloc[i]['Latitude']
        measure_df['Longitude'] = meta_df.iloc[i]['Longitude']
        dfs.append(measure_df)
    return pd.concat(dfs, axis=0)

class AWSTrainDataset(Dataset):
    def __init__(self, data, num_aws, time_size, neighbors):
        self.data = data
        self.num_aws = num_aws
        self.time_size = time_size
        self.neighbors = neighbors

    def __len__(self):
        T, A, _ = self.data.shape
        return (T - 2 * self.time_size + 1) * A

    def __getitem__(self, idx):
        time_idx = idx // 30 + self.time_size
        aws_idx = idx % 30
        aws_loc = self.data[0, aws_idx, -2:]
        data1 = self.data[time_idx - self.time_size:time_idx, self.neighbors[aws_idx], :]
        data1[:, :, -2:] -= aws_loc
        data2 = self.data[time_idx: time_idx + self.time_size, aws_idx, :-2]
        return data1, data2

    def get(self, time_idx, aws_idx):
        aws_loc = self.data[0, aws_idx, -2:]
        data1 = self.data[time_idx - self.time_size:time_idx, self.neighbors[aws_idx], :]
        data1[:, :, -2:] -= aws_loc
        return data1
    
    def get_pm(self, time_idx, neighbors, pm_pos):
        data1 = self.data[time_idx - self.time_size:time_idx, neighbors, :]
        data1[:, :, -2:] -= pm_pos
        return data1

class PMTrainDataset(Dataset):
    def __init__(self, data, num_aws, time_size, neighbors):
        self.data = data
        self.num_aws = num_aws
        self.time_size = time_size
        self.neighbors = neighbors

    def __len__(self):
        T, A, _ = self.data.shape
        return (T - 3 * self.time_size + 1) * A

    def __getitem__(self, idx):
        time_idx = idx // 17 + self.time_size + 10
        aws_idx = idx % 17
        aws_loc = self.data[0, aws_idx, -2:]
        data1 = self.data[time_idx - self.time_size:time_idx, self.neighbors[aws_idx], :]
        data1[:, :, -2:] -= aws_loc
        data2 = self.data[time_idx: time_idx + self.time_size, aws_idx, :-2]
        return data1, data2

    def get(self, time_idx, aws_idx):
        aws_loc = self.data[0, aws_idx, -2:]
        data1 = self.data[time_idx - self.time_size:time_idx, self.neighbors[aws_idx], :]
        data1[:, :, -2:] -= aws_loc
        return data1

class NonNullSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return self
    
    def __next__(self):
        n = len(self.dataset)
        rnd_idx = np.random.randint(n)
        x0, x1 = self.dataset[rnd_idx]
        while np.isnan(x0).sum() > 0 or np.isnan(x1).sum() > 0:
            rnd_idx = np.random.randint(n)
            x0, x1 = self.dataset[rnd_idx]
        return rnd_idx
