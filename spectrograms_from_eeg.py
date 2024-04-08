import pandas as pd
import polars as pl
import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset



class SpectrogramFromEEGDataset(Dataset):

    def __init__(self, df_info, time) -> None:
        super().__init__()

        if not time in [0.5, 1, 5]:
            raise ValueError('Value of <time> should be one of [0.5, 1, 5].')

        self.time = time
        self.df_info = df_info
        spectrogram_path = f'data/spectrogram_{time}sec.npy'
        tmp = np.load(spectrogram_path, allow_pickle=True).item()
        self.all_spectrograms = {key: tmp[key] for key in df_info['eeg_id']}
        del tmp

    def __len__(self):
        return self.df_info.shape[0]


    def __getitem__(self, index):

        row = self.df_info.loc[index]

        eeg_id = row['eeg_id']
        labels = row.iloc[-6:] / row.iloc[-6:].sum()
        
        spectrogram = self.all_spectrograms[eeg_id]

        return torch.Tensor(spectrogram), torch.Tensor(labels.values.astype('float32'))
    

    # ----------- visualization ---------------------

