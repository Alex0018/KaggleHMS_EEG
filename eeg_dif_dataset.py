import pandas as pd
import polars as pl
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from scipy.signal import butter, lfilter


SR = 200
LEN_SEC = 50


def butter_lowpass_filter(data, cutoff_freq: int = 50, sampling_rate: int = 200, order: int = 4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data



class EEG_Dif_Dataset(Dataset):

    def __init__(self, df_info, normalizer=None) -> None:
        super().__init__()

        self.df_info = df_info.reset_index(drop=True)
        self.path = 'data/train_eegs/'
        self.normalizer = normalizer

    def __len__(self):
        return self.df_info.shape[0]  

    def __getitem__(self, index):

        row = self.df_info.loc[index]

        eeg_id = row['eeg_id']
        eeg_offset = int(row['eeg_label_offset_seconds'] * SR)

        labels = row.iloc[-6:] / row.iloc[-6:].sum()    

        raw_data = pl.read_parquet(self.path + f'{eeg_id}.parquet')
        df = raw_data.slice(eeg_offset, LEN_SEC*SR).select(['Fp1','T3','C3','O1','Fp2','C4','T4','O2']).to_pandas()

        # === Feature engineering ===
        X = np.zeros((10_000, 8), dtype='float32')

        X[:,0] = df['Fp1'] - df['T3']
        X[:,1] = df['T3'] - df['O1']

        X[:,2] = df['Fp1'] - df['C3']
        X[:,3] = df['C3'] - df['O1']

        X[:,4] = df['Fp2'] - df['C4']
        X[:,5] = df['C4'] - df['O2']

        X[:,6] = df['Fp2'] - df['T4']
        X[:,7] = df['T4'] - df['O2']

        # === Standarize ===
        X = np.clip(X,-1024, 1024)
        X = np.nan_to_num(X, nan=0) / 32.0

        # === Butter Low-pass Filter ===
        X = butter_lowpass_filter(X)[::5]
        
        return torch.Tensor(np.array(X)), torch.Tensor(labels.values.astype('float32'))