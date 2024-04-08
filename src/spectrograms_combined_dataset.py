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

eps = 10**-10
SP_LEN = 300
SP_NUM_FREQ = 100

class SpectrogramsCombinedDataset(Dataset):

    def __init__(self, df_info) -> None:
        super().__init__()

        self.df_info = df_info.reset_index(drop=True)
        
        spectrogram_path = 'data/spectrogram_0.5sec.npy'
        tmp = np.load(spectrogram_path, allow_pickle=True).item()
        self.all_spectrograms = {key: tmp[key] for key in df_info['eeg_id']}
        del tmp

    def __len__(self):
        return self.df_info.shape[0]


    def __getitem__(self, index):

        row = self.df_info.loc[index]

        eeg_id = row['eeg_id']
        labels = row.iloc[-6:] / row.iloc[-6:].sum()

        spectrogram_eeg = self.all_spectrograms[eeg_id] 

        spectrogram_id = row['spectrogram_id']
        spectrograms_offset = int(row['spectrogram_label_offset_seconds'] * 0.5)

        raw_data = pl.read_parquet(f'data/train_spectrograms/{spectrogram_id}.parquet')
        df_sp = raw_data.slice(spectrograms_offset, SP_LEN)
        spectrogram_kaggle = np.log10(np.nan_to_num(df_sp.to_numpy(), nan=0) + eps).clip(-2, 5)
        
        spectrogram = np.zeros((SP_LEN + 125, 400), dtype='float32')       
        spectrogram[:SP_LEN, :] = spectrogram_kaggle[:300, :400]

        for i in range(5):
            spectrogram[SP_LEN + i*25:SP_LEN + (i+1)*25, :] = spectrogram_eeg[i][:, :400]
        
        labels = row.iloc[-6:] / row.iloc[-6:].sum()

        return torch.Tensor([spectrogram]), torch.Tensor(labels.values.astype('float32'))
