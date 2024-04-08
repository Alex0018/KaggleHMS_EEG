import pandas as pd
import polars as pl
import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset



class SpectrogramFromEEGCombinedDataset(Dataset):

    def __init__(self, df_info) -> None:
        super().__init__()

        self.df_info = df_info
        spectrogram_path_5 = f'data/spectrogram_5sec.npy'
        spectrogram_path_1 = f'data/spectrogram_1sec.npy'
        spectrogram_path_05 = f'data/spectrogram_0.5sec.npy'

        self.spectrograms_5 = np.load(spectrogram_path_5, allow_pickle=True).item()
        # self.spectrograms_1 = np.load(spectrogram_path_1, allow_pickle=True).item()
        # self.spectrograms_05 = np.load(spectrogram_path_05, allow_pickle=True).item()

        # tmp = np.load(spectrogram_path, allow_pickle=True).item()
        # self.all_spectrograms = {key: tmp[key] for key in df_info['eeg_id']}
        # del tmp

    def __len__(self):
        return self.df_info.shape[0]


    def __getitem__(self, index):

        row = self.df_info.loc[index]

        eeg_id = row['eeg_id']
        labels = row.iloc[-6:] / row.iloc[-6:].sum()
        
        spectrogram_5 = self.spectrograms_5[eeg_id]
        # spectrogram_1 = self.spectrograms_1[eeg_id]
        # spectrogram_05 = self.spectrograms_05[eeg_id]

        spectrogram = np.zeros((406, 375), dtype='float32')

        return torch.Tensor(spectrogram), torch.Tensor(labels.values.astype('float32'))
    

    # ----------- visualization ---------------------

