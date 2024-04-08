import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedGroupKFold

from src.spectrogram_dataset import SpectrogramDataset, Spectrogram_4channels_Dataset
from src.eeg_dif_dataset import EEG_Dif_Dataset
from src.eeg_dataset import EEG_Dataset
from src.melspectrogram_hfreq15_dataset import Melspectrogram_hfreq15_Dataset, Melspectrogram_hfreq15_4channels_Dataset
from src.spectrograms_combined_dataset import SpectrogramsCombinedDataset
from src.dual_spectrograms_dataset import Dual_Spectrogram_4channels_Dataset
from src.spectrograms_from_eeg import SpectrogramFromEEGDataset
from src.eeg_avg_dataset import EEG_RegionalMean_Dataset, EEG_RegionalMean_Difference_Dataset
from src.spectrograms_from_eeg_combined_dataset import SpectrogramFromEEGCombinedDataset

def get_train_val(path, fold):
    df = pd.read_csv(path).sort_values(['eeg_id', 'eeg_label_offset_seconds'])
    dif = df['eeg_label_offset_seconds'] - df['eeg_label_offset_seconds'].shift(1)
    df = df.loc[~((dif > 0) & (dif < 50))].reset_index(drop=True) # use only different events

    df_bad_ch = pd.read_csv('data/num_bad_channels.csv')
    df.insert(df.shape[1] - 7, 'num_bad_channels', df_bad_ch['num_bad_channels'])

    cv = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    fold_idx = [idx for idx in cv.split(df, df['expert_consensus'], groups=df['patient_id'])]

    df_train = df.loc[ fold_idx[fold][0] ]
    df_train = df_train.loc[df_train['num_bad_channels'] < 4]
    df_val   = df.loc[ fold_idx[fold][1] ]

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)


def dataloaders_spectrograms_combined(path: str, 
                                      batch_size: int, 
                                      fold: int,):

    df_train, df_val = get_train_val(path, fold)

    bad_eeg_ids = [1457334423, 1593385762, 120145971, 1511903313, 579740230, 588638365, 
                   837428467, 2081405553, 3932380488, 1604371226, 2538961182, 2565199369, 
                   1339041688, 812448735, 408047047, 4046938588, 2924540968, 1119914885, 116770645]
    
    df_train = df_train.loc[~df_train['eeg_id'].isin(bad_eeg_ids)].reset_index(drop=True)
    df_val = df_val.loc[~df_val['eeg_id'].isin(bad_eeg_ids)].reset_index(drop=True)

    dataset_train = SpectrogramsCombinedDataset(df_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = SpectrogramsCombinedDataset(df_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_val


def dataloaders_spectrograms_from_eeg_combined( path: str, 
                                                batch_size: int, 
                                                fold: int,):

    df_train, df_val = get_train_val(path, fold)

    bad_eeg_ids = [1457334423, 1593385762, 120145971, 1511903313, 579740230, 588638365, 
                   837428467, 2081405553, 3932380488, 1604371226, 2538961182, 2565199369, 
                   1339041688, 812448735, 408047047, 4046938588, 2924540968, 1119914885, 116770645]
    
    df_train = df_train.loc[~df_train['eeg_id'].isin(bad_eeg_ids)].reset_index(drop=True)
    df_val = df_val.loc[~df_val['eeg_id'].isin(bad_eeg_ids)].reset_index(drop=True)

    dataset_train = SpectrogramFromEEGCombinedDataset(df_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = SpectrogramFromEEGCombinedDataset(df_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_val



def dataloaders_spectrograms_from_eeg(path: str, 
                                      batch_size: int, 
                                      fold: int,
                                      time: float):

    df_train, df_val = get_train_val(path, fold)

    bad_eeg_ids = [1457334423, 1593385762, 120145971, 1511903313, 579740230, 588638365, 
                   837428467, 2081405553, 3932380488, 1604371226, 2538961182, 2565199369, 
                   1339041688, 812448735, 408047047, 4046938588, 2924540968, 1119914885, 116770645]
    
    df_train = df_train.loc[~df_train['eeg_id'].isin(bad_eeg_ids)].reset_index(drop=True)
    df_val = df_val.loc[~df_val['eeg_id'].isin(bad_eeg_ids)].reset_index(drop=True)

    dataset_train = SpectrogramFromEEGDataset(df_train, time)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = SpectrogramFromEEGDataset(df_val, time)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_val


def dataloaders_regional_avg(path: str, 
                             batch_size: int, 
                             fold: int,):

    df_train, df_val = get_train_val(path, fold)

    bad_eeg_ids = [1457334423, 1593385762, 120145971, 1511903313, 579740230, 588638365, 
                   837428467, 2081405553, 3932380488, 1604371226, 2538961182, 2565199369, 
                   1339041688, 812448735, 408047047, 4046938588, 2924540968, 1119914885, 116770645]
    
    df_train = df_train.loc[~df_train['eeg_id'].isin(bad_eeg_ids)].reset_index(drop=True)
    df_val = df_val.loc[~df_val['eeg_id'].isin(bad_eeg_ids)].reset_index(drop=True)

    dataset_train = EEG_RegionalMean_Dataset(df_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = EEG_RegionalMean_Dataset(df_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_val



def dataloaders_regional_avg_differences(path: str, 
                                         batch_size: int, 
                                         fold: int,):

    df_train, df_val = get_train_val(path, fold)

    bad_eeg_ids = [1457334423, 1593385762, 120145971, 1511903313, 579740230, 588638365, 
                   837428467, 2081405553, 3932380488, 1604371226, 2538961182, 2565199369, 
                   1339041688, 812448735, 408047047, 4046938588, 2924540968, 1119914885, 116770645]
    
    df_train = df_train.loc[~df_train['eeg_id'].isin(bad_eeg_ids)].reset_index(drop=True)
    df_val = df_val.loc[~df_val['eeg_id'].isin(bad_eeg_ids)].reset_index(drop=True)

    dataset_train = EEG_RegionalMean_Difference_Dataset(df_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = EEG_RegionalMean_Difference_Dataset(df_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_val




def dataloaders_melspectrogram_hfreq15(path: str, 
                                       batch_size: int, 
                                       fold: int,):

    df_train, df_val = get_train_val(path, fold)

    dataset_train = Melspectrogram_hfreq15_Dataset(df_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = Melspectrogram_hfreq15_Dataset(df_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_val


def dataloaders_dual_spectrograms_4channels(path: str, 
                                            batch_size: int, 
                                            fold: int,):
    
    df = pd.read_csv(path).sort_values(['eeg_id', 'eeg_label_offset_seconds'])
    dif = df['eeg_label_offset_seconds'] - df['eeg_label_offset_seconds'].shift(1)
    df = df.loc[~((dif > 0) & (dif < 50))].reset_index(drop=True) # use only different events

    df_bad_ch = pd.read_csv('data/num_bad_channels.csv')
    df.insert(df.shape[1] - 7, 'num_bad_channels', df_bad_ch['num_bad_channels'])

    cv = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    fold_idx = [idx for idx in cv.split(df, df['expert_consensus'], groups=df['patient_id'])]

    df_train = df.loc[ fold_idx[fold][0] ]
    df_train = df_train.loc[df_train['num_bad_channels'] < 4]
    df_val   = df.loc[ fold_idx[fold][1] ]

    dataset_train = Dual_Spectrogram_4channels_Dataset(df_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = Dual_Spectrogram_4channels_Dataset(df_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_val



def dataloaders_melspectrogram_hfreq15_4channels(path: str, 
                                                 batch_size: int, 
                                                 fold: int,):

    df = pd.read_csv(path).sort_values(['eeg_id', 'eeg_label_offset_seconds'])
    dif = df['eeg_label_offset_seconds'] - df['eeg_label_offset_seconds'].shift(1)
    df = df.loc[~((dif > 0) & (dif < 50))].reset_index(drop=True) # use only different events

    df_bad_ch = pd.read_csv('data/num_bad_channels.csv')
    df.insert(df.shape[1] - 7, 'num_bad_channels', df_bad_ch['num_bad_channels'])

    cv = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    fold_idx = [idx for idx in cv.split(df, df['expert_consensus'], groups=df['patient_id'])]

    df_train = df.loc[ fold_idx[fold][0] ]
    df_train = df_train.loc[df_train['num_bad_channels'] < 4]
    df_val   = df.loc[ fold_idx[fold][1] ]

    dataset_train = Melspectrogram_hfreq15_4channels_Dataset(df_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = Melspectrogram_hfreq15_4channels_Dataset(df_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_val



def dataloaders_spectrogram_4channels(path: str, 
                                      batch_size: int, 
                                      fold: int):

    df = pd.read_csv(path).sort_values(['eeg_id', 'eeg_label_offset_seconds'])
    dif = df['eeg_label_offset_seconds'] - df['eeg_label_offset_seconds'].shift(1)
    df = df.loc[~((dif > 0) & (dif < 50))].reset_index(drop=True) # use only different events

    cv = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    fold_idx = [idx for idx in cv.split(df, df['expert_consensus'], groups=df['patient_id'])]

    df_train = df.loc[ fold_idx[fold][0] ]
    df_val   = df.loc[ fold_idx[fold][1] ]

    dataset_train = Spectrogram_4channels_Dataset(df_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = Spectrogram_4channels_Dataset(df_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_val



def dataloaders_spectrogram_efficientnet(path: str, 
                                         batch_size: int, 
                                         fold: int):

    df = pd.read_csv(path).sort_values(['eeg_id', 'eeg_label_offset_seconds'])
    dif = df['eeg_label_offset_seconds'] - df['eeg_label_offset_seconds'].shift(1)
    df = df.loc[~((dif > 0) & (dif < 50))].reset_index(drop=True) # use only different events

    cv = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    fold_idx = [idx for idx in cv.split(df, df['expert_consensus'], groups=df['patient_id'])]

    df_train = df.loc[ fold_idx[fold][0] ]
    df_val   = df.loc[ fold_idx[fold][1] ]

    dataset_train = SpectrogramDataset(df_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = SpectrogramDataset(df_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_val


def dataloaders_eeg_8(path: str, 
                      batch_size: int, 
                      fold: int):

    df = pd.read_csv(path).sort_values(['eeg_id', 'eeg_label_offset_seconds'])
    dif = df['eeg_label_offset_seconds'] - df['eeg_label_offset_seconds'].shift(1)
    df = df.loc[~((dif > 0) & (dif < 50))].reset_index(drop=True) # use only different events

    cv = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    fold_idx = [idx for idx in cv.split(df, df['expert_consensus'], groups=df['patient_id'])]

    df_train = df.loc[ fold_idx[fold][0] ]
    df_val   = df.loc[ fold_idx[fold][1] ]

    bad_eeg_ids = [1457334423, 1593385762, 120145971, 1511903313, 579740230, 588638365, 
                   837428467, 2081405553, 3932380488, 1604371226, 2538961182, 2565199369, 
                   1339041688, 812448735, 408047047, 4046938588, 2924540968, 1119914885, 116770645]
    
    df_train = df_train.loc[~df_train['eeg_id'].isin(bad_eeg_ids)].reset_index(drop=True)
    df_val = df_val.loc[~df_val['eeg_id'].isin(bad_eeg_ids)].reset_index(drop=True)

    dataset_train = EEG_Dataset(df_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = EEG_Dataset(df_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_val



def normalize(data):
    return (data - data.mean()) / (data.std() + 1e-10)

def dataloaders_eeg_difference(path: str, 
                               batch_size: int, 
                               fold: int):

    df = pd.read_csv(path).sort_values(['eeg_id', 'eeg_label_offset_seconds'])
    dif = df['eeg_label_offset_seconds'] - df['eeg_label_offset_seconds'].shift(1)
    df = df.loc[~((dif > 0) & (dif < 50))].reset_index(drop=True) # use only different events

    cv = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    fold_idx = [idx for idx in cv.split(df, df['expert_consensus'], groups=df['patient_id'])]

    df_train = df.loc[ fold_idx[fold][0] ]
    df_val   = df.loc[ fold_idx[fold][1] ]

    bad_eeg_ids = [1457334423, 1593385762, 120145971, 1511903313, 579740230, 588638365, 
                   837428467, 2081405553, 3932380488, 1604371226, 2538961182, 2565199369, 
                   1339041688, 812448735, 408047047, 4046938588, 2924540968, 1119914885, 116770645]
    
    df_train = df_train.loc[~df_train['eeg_id'].isin(bad_eeg_ids)].reset_index(drop=True)
    df_val = df_val.loc[~df_val['eeg_id'].isin(bad_eeg_ids)].reset_index(drop=True)

    dataset_train = EEG_Dif_Dataset(df_train, normalize)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = EEG_Dif_Dataset(df_val, normalize)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_val



def dataloaders_spectrogram_idealized(path: str, 
                                      batch_size: int, 
                                      fold: int):

    df = pd.read_csv(path).sort_values(['eeg_id', 'eeg_label_offset_seconds'])
    dif = df['eeg_label_offset_seconds'] - df['eeg_label_offset_seconds'].shift(1)
    df = df.loc[~((dif > 0) & (dif < 50))].reset_index(drop=True) # use only different events

    cv = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    fold_idx = [idx for idx in cv.split(df, df['expert_consensus'], groups=df['patient_id'])]

    df_train = df.loc[ fold_idx[fold][0] ]
    df_val   = df.loc[ fold_idx[fold][1] ]

    # order train so that the first samples were samples with unanimous expert consensus
    df_ideal = pd.concat([df_train.loc[df_train.iloc[:, -col] == 1] for col in range(1,7)], axis=0)

    dataset_train = SpectrogramDataset(df_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = SpectrogramDataset(df_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_val



def dataloaders_spectrogram_idealizedFirst(path: str, 
                                           batch_size: int, 
                                           fold: int):

    df = pd.read_csv(path).sort_values(['eeg_id', 'eeg_label_offset_seconds'])
    dif = df['eeg_label_offset_seconds'] - df['eeg_label_offset_seconds'].shift(1)
    df = df.loc[~((dif > 0) & (dif < 50))].reset_index(drop=True) # use only different events

    cv = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    fold_idx = [idx for idx in cv.split(df, df['expert_consensus'], groups=df['patient_id'])]

    df_train = df.loc[ fold_idx[fold][0] ]
    df_val   = df.loc[ fold_idx[fold][1] ]

    # order train so that the first samples were samples with unanimous expert consensus
    df_ideal = pd.concat([df_train.loc[df_train.iloc[:, -col] == 1] for col in range(1,7)], axis=0)
    df_train = pd.concat([df_ideal.sample(frac=1), df_train.drop(df_ideal.index, axis=0)], axis=0)

    dataset_train = SpectrogramDataset(df_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    dataset_val = SpectrogramDataset(df_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_val
