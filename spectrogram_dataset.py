import pandas as pd
import polars as pl
import numpy as np
import math
import random

from sklearn.model_selection import StratifiedGroupKFold

import matplotlib.pyplot as plt

import gc

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import datasets, transforms
import torch.nn as nn

from efficientnet_pytorch import EfficientNet

from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)


SPECTROGRAMS = ['LL','LP','RP','RL']

SP_LEN = 300
SP_NUM_FREQ = 100
IMAGE_SIZE = 456
OFFSET_X = (IMAGE_SIZE - SP_LEN) // 2
OFFSET_Y = (IMAGE_SIZE - SP_NUM_FREQ*4) // 2

SPECTROGRAM_COLS = [
['LL_0.59', 'LL_0.78', 'LL_0.98', 'LL_1.17', 'LL_1.37', 'LL_1.56', 'LL_1.76', 'LL_1.95', 'LL_2.15', 'LL_2.34', 'LL_2.54', 'LL_2.73', 'LL_2.93', 'LL_3.13', 'LL_3.32', 'LL_3.52', 'LL_3.71', 'LL_3.91', 'LL_4.1', 'LL_4.3', 'LL_4.49', 'LL_4.69', 'LL_4.88', 'LL_5.08', 'LL_5.27', 'LL_5.47', 'LL_5.66', 'LL_5.86', 'LL_6.05', 'LL_6.25', 'LL_6.45', 'LL_6.64', 'LL_6.84', 'LL_7.03', 'LL_7.23', 'LL_7.42', 'LL_7.62', 'LL_7.81', 'LL_8.01', 'LL_8.2', 'LL_8.4', 'LL_8.59', 'LL_8.79', 'LL_8.98', 'LL_9.18', 'LL_9.38', 'LL_9.57', 'LL_9.77', 'LL_9.96', 'LL_10.16', 'LL_10.35', 'LL_10.55', 'LL_10.74', 'LL_10.94', 'LL_11.13', 'LL_11.33', 'LL_11.52', 'LL_11.72', 'LL_11.91', 'LL_12.11', 'LL_12.3', 'LL_12.5', 'LL_12.7', 'LL_12.89', 'LL_13.09', 'LL_13.28', 'LL_13.48', 'LL_13.67', 'LL_13.87', 'LL_14.06', 'LL_14.26', 'LL_14.45', 'LL_14.65', 'LL_14.84', 'LL_15.04', 'LL_15.23', 'LL_15.43', 'LL_15.63', 'LL_15.82', 'LL_16.02', 'LL_16.21', 'LL_16.41', 'LL_16.6', 'LL_16.8', 'LL_16.99', 'LL_17.19', 'LL_17.38', 'LL_17.58', 'LL_17.77', 'LL_17.97', 'LL_18.16', 'LL_18.36', 'LL_18.55', 'LL_18.75', 'LL_18.95', 'LL_19.14', 'LL_19.34', 'LL_19.53', 'LL_19.73', 'LL_19.92'],
['LP_0.59', 'LP_0.78', 'LP_0.98', 'LP_1.17', 'LP_1.37', 'LP_1.56', 'LP_1.76', 'LP_1.95', 'LP_2.15', 'LP_2.34', 'LP_2.54', 'LP_2.73', 'LP_2.93', 'LP_3.13', 'LP_3.32', 'LP_3.52', 'LP_3.71', 'LP_3.91', 'LP_4.1', 'LP_4.3', 'LP_4.49', 'LP_4.69', 'LP_4.88', 'LP_5.08', 'LP_5.27', 'LP_5.47', 'LP_5.66', 'LP_5.86', 'LP_6.05', 'LP_6.25', 'LP_6.45', 'LP_6.64', 'LP_6.84', 'LP_7.03', 'LP_7.23', 'LP_7.42', 'LP_7.62', 'LP_7.81', 'LP_8.01', 'LP_8.2', 'LP_8.4', 'LP_8.59', 'LP_8.79', 'LP_8.98', 'LP_9.18', 'LP_9.38', 'LP_9.57', 'LP_9.77', 'LP_9.96', 'LP_10.16', 'LP_10.35', 'LP_10.55', 'LP_10.74', 'LP_10.94', 'LP_11.13', 'LP_11.33', 'LP_11.52', 'LP_11.72', 'LP_11.91', 'LP_12.11', 'LP_12.3', 'LP_12.5', 'LP_12.7', 'LP_12.89', 'LP_13.09', 'LP_13.28', 'LP_13.48', 'LP_13.67', 'LP_13.87', 'LP_14.06', 'LP_14.26', 'LP_14.45', 'LP_14.65', 'LP_14.84', 'LP_15.04', 'LP_15.23', 'LP_15.43', 'LP_15.63', 'LP_15.82', 'LP_16.02', 'LP_16.21', 'LP_16.41', 'LP_16.6', 'LP_16.8', 'LP_16.99', 'LP_17.19', 'LP_17.38', 'LP_17.58', 'LP_17.77', 'LP_17.97', 'LP_18.16', 'LP_18.36', 'LP_18.55', 'LP_18.75', 'LP_18.95', 'LP_19.14', 'LP_19.34', 'LP_19.53', 'LP_19.73', 'LP_19.92'],
['RP_0.59', 'RP_0.78', 'RP_0.98', 'RP_1.17', 'RP_1.37', 'RP_1.56', 'RP_1.76', 'RP_1.95', 'RP_2.15', 'RP_2.34', 'RP_2.54', 'RP_2.73', 'RP_2.93', 'RP_3.13', 'RP_3.32', 'RP_3.52', 'RP_3.71', 'RP_3.91', 'RP_4.1', 'RP_4.3', 'RP_4.49', 'RP_4.69', 'RP_4.88', 'RP_5.08', 'RP_5.27', 'RP_5.47', 'RP_5.66', 'RP_5.86', 'RP_6.05', 'RP_6.25', 'RP_6.45', 'RP_6.64', 'RP_6.84', 'RP_7.03', 'RP_7.23', 'RP_7.42', 'RP_7.62', 'RP_7.81', 'RP_8.01', 'RP_8.2', 'RP_8.4', 'RP_8.59', 'RP_8.79', 'RP_8.98', 'RP_9.18', 'RP_9.38', 'RP_9.57', 'RP_9.77', 'RP_9.96', 'RP_10.16', 'RP_10.35', 'RP_10.55', 'RP_10.74', 'RP_10.94', 'RP_11.13', 'RP_11.33', 'RP_11.52', 'RP_11.72', 'RP_11.91', 'RP_12.11', 'RP_12.3', 'RP_12.5', 'RP_12.7', 'RP_12.89', 'RP_13.09', 'RP_13.28', 'RP_13.48', 'RP_13.67', 'RP_13.87', 'RP_14.06', 'RP_14.26', 'RP_14.45', 'RP_14.65', 'RP_14.84', 'RP_15.04', 'RP_15.23', 'RP_15.43', 'RP_15.63', 'RP_15.82', 'RP_16.02', 'RP_16.21', 'RP_16.41', 'RP_16.6', 'RP_16.8', 'RP_16.99', 'RP_17.19', 'RP_17.38', 'RP_17.58', 'RP_17.77', 'RP_17.97', 'RP_18.16', 'RP_18.36', 'RP_18.55', 'RP_18.75', 'RP_18.95', 'RP_19.14', 'RP_19.34', 'RP_19.53', 'RP_19.73', 'RP_19.92'],
['RL_0.59', 'RL_0.78', 'RL_0.98', 'RL_1.17', 'RL_1.37', 'RL_1.56', 'RL_1.76', 'RL_1.95', 'RL_2.15', 'RL_2.34', 'RL_2.54', 'RL_2.73', 'RL_2.93', 'RL_3.13', 'RL_3.32', 'RL_3.52', 'RL_3.71', 'RL_3.91', 'RL_4.1', 'RL_4.3', 'RL_4.49', 'RL_4.69', 'RL_4.88', 'RL_5.08', 'RL_5.27', 'RL_5.47', 'RL_5.66', 'RL_5.86', 'RL_6.05', 'RL_6.25', 'RL_6.45', 'RL_6.64', 'RL_6.84', 'RL_7.03', 'RL_7.23', 'RL_7.42', 'RL_7.62', 'RL_7.81', 'RL_8.01', 'RL_8.2', 'RL_8.4', 'RL_8.59', 'RL_8.79', 'RL_8.98', 'RL_9.18', 'RL_9.38', 'RL_9.57', 'RL_9.77', 'RL_9.96', 'RL_10.16', 'RL_10.35', 'RL_10.55', 'RL_10.74', 'RL_10.94', 'RL_11.13', 'RL_11.33', 'RL_11.52', 'RL_11.72', 'RL_11.91', 'RL_12.11', 'RL_12.3', 'RL_12.5', 'RL_12.7', 'RL_12.89', 'RL_13.09', 'RL_13.28', 'RL_13.48', 'RL_13.67', 'RL_13.87', 'RL_14.06', 'RL_14.26', 'RL_14.45', 'RL_14.65', 'RL_14.84', 'RL_15.04', 'RL_15.23', 'RL_15.43', 'RL_15.63', 'RL_15.82', 'RL_16.02', 'RL_16.21', 'RL_16.41', 'RL_16.6', 'RL_16.8', 'RL_16.99', 'RL_17.19', 'RL_17.38', 'RL_17.58', 'RL_17.77', 'RL_17.97', 'RL_18.16', 'RL_18.36', 'RL_18.55', 'RL_18.75', 'RL_18.95', 'RL_19.14', 'RL_19.34', 'RL_19.53', 'RL_19.73', 'RL_19.92'],
]

eps = 10**-10


class Spectrogram_4channels_Dataset(Dataset):

    def __init__(self, df_info) -> None:
        super().__init__()

        self.df_info = df_info.reset_index(drop=True)
        self.spectrogram_path = 'data/train_spectrograms/'

    def __len__(self):
        return self.df_info.shape[0]


    def __getitem__(self, index):

        row = self.df_info.loc[index]

        spectrogram_id = row['spectrogram_id']
        spectrograms_offset = int(row['spectrogram_label_offset_seconds'] * 0.5)

        labels = row.iloc[-6:] / row.iloc[-6:].sum()

        spectrogram = []       

        raw_data = pl.read_parquet(self.spectrogram_path + f'{spectrogram_id}.parquet')
        for i, prefix in enumerate(SPECTROGRAMS):
            df_sp = raw_data.slice(spectrograms_offset, SP_LEN).select(SPECTROGRAM_COLS[i])
            cur = np.log10(np.nan_to_num(df_sp.to_numpy(), nan=1) + eps).clip(-2, 5)
            # cur /= np.linalg.norm(cur)
            spectrogram.append(cur)
  
        return torch.Tensor(spectrogram), torch.Tensor(labels.values.astype('float32'))



class SpectrogramDataset(Dataset):

    def __init__(self, df_info) -> None:
        super().__init__()

        self.df_info = df_info.reset_index(drop=True)
        self.spectrogram_path = 'data/train_spectrograms/'

    def __len__(self):
        return self.df_info.shape[0]


    def __getitem__(self, index):

        row = self.df_info.loc[index]

        spectrogram_id = row['spectrogram_id']
        spectrograms_offset = int(row['spectrogram_label_offset_seconds'] * 0.5)

        labels = row.iloc[-6:] / row.iloc[-6:].sum()

        spectrogram = np.zeros((SP_LEN, 4*SP_NUM_FREQ), dtype='float32')        

        raw_data = pl.read_parquet(self.spectrogram_path + f'{spectrogram_id}.parquet')
        for i, prefix in enumerate(SPECTROGRAMS):
            df_sp = raw_data.slice(spectrograms_offset, SP_LEN).select(SPECTROGRAM_COLS[i])
            cur = np.log10(np.nan_to_num(df_sp.to_numpy(), nan=0) + eps).clip(-2, 5)
            # cur /= np.linalg.norm(cur)
            # if i % 2 == 0:
            #     cur = np.flip(cur, axis=1)
            spectrogram[:,  SP_NUM_FREQ*i:SP_NUM_FREQ*i+SP_NUM_FREQ] = cur

        return torch.Tensor([spectrogram]), torch.Tensor(labels.values.astype('float32'))
    


class SpectrogramNormalizedDataset(Dataset):

    def __init__(self, df_info) -> None:
        super().__init__()

        self.df_info = df_info.reset_index(drop=True)
        self.spectrogram_path = 'data/train_spectrograms/'

    def __len__(self):
        return self.df_info.shape[0]


    def __getitem__(self, index):

        row = self.df_info.loc[index]

        spectrogram_id = row['spectrogram_id']
        spectrograms_offset = int(row['spectrogram_label_offset_seconds'] * 0.5)

        labels = row.iloc[-6:] / row.iloc[-6:].sum()

        spectrogram = np.zeros((SP_LEN, 4*SP_NUM_FREQ), dtype='float32')        

        raw_data = pl.read_parquet(self.spectrogram_path + f'{spectrogram_id}.parquet')
        for i, prefix in enumerate(SPECTROGRAMS):
            df_sp = raw_data.slice(spectrograms_offset, SP_LEN).select(SPECTROGRAM_COLS[i])
            cur = np.log10(np.nan_to_num(df_sp.to_numpy(), nan=0) + eps)
            cur /= np.linalg.norm(cur)
            spectrogram[:,  SP_NUM_FREQ*i:SP_NUM_FREQ*i+SP_NUM_FREQ] = cur

        return torch.Tensor([spectrogram]), torch.Tensor(labels.values.astype('float32'))



        # if np.any(np.isnan(spectrogram)):
        #     print('!'*10, index, '!'*10)
        #     print(spectrogram)

        # spectrogram -= spectrogram.min()
        # spectrogram /= spectrogram.max()
        


        # return torch.Tensor(np.array([spectrogram, spectrogram, spectrogram])), torch.Tensor(labels.values.astype('float32'))