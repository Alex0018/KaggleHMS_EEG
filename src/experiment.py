import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import gc


from src.dataloaders import dataloaders_spectrogram_efficientnet, dataloaders_spectrogram_idealized, dataloaders_spectrogram_idealizedFirst
from src.training_loop import training_loop
from src.efficientnet_model import build_efficientnet_b0
from src.dataloaders import dataloaders_eeg_difference, dataloaders_eeg_8, dataloaders_spectrograms_combined, dataloaders_spectrogram_4channels
from src.dataloaders import dataloaders_melspectrogram_hfreq15, dataloaders_melspectrogram_hfreq15_4channels
from src.dataloaders import dataloaders_dual_spectrograms_4channels, dataloaders_spectrograms_from_eeg
from src.dataloaders import dataloaders_regional_avg, dataloaders_regional_avg_differences
from src.dataloaders import dataloaders_spectrograms_from_eeg_combined

from src.efficientnet_starter_data import dataloaders_efficientnet_starter
from src.efficientnet_starter_model import EEGEffnetB0
from src.BidirGRU import MultiResidualBiGRU
from src.wavenet import SequentialWaveNet
from src.conv_gru_model import EEGNet
from src.CNN_model import CNN, CNN_MultiChannel, CNN_2
from src.CNN_dual_input_model import CNN_Dual_Input
from src.efficientnet_wrapper_model import EfficientnetWrapper


PROJECT_NAME = 'HMS_EEG_spectrograms'

dataloaders_dict = {'spectrogram_efficientnet': dataloaders_spectrogram_efficientnet,
                    'spectrogram_4channels': dataloaders_spectrogram_4channels,
                    'spectrogram_idealized': dataloaders_spectrogram_idealized,
                    'spectrogram_idealizedFirst': dataloaders_spectrogram_idealizedFirst,
                    'eeg_8_difference': dataloaders_eeg_difference,
                    'eeg_8': dataloaders_eeg_8,
                    'eeg_regional_avg': dataloaders_regional_avg,
                    'eeg_regional_avg_differences':dataloaders_regional_avg_differences,
                    'starter_effnet': dataloaders_efficientnet_starter,
                    'melspectrogram_hfreq15':dataloaders_melspectrogram_hfreq15,
                    'melspectrogram_hfreq15_4channels':dataloaders_melspectrogram_hfreq15_4channels,
                    'spectrograms_combined':dataloaders_spectrograms_combined,
                    'dual_spectrograms_4channels':dataloaders_dual_spectrograms_4channels,
                    'spectrograms_from_eeg':dataloaders_spectrograms_from_eeg,
                    'spectrograms_from_eeg_combined':dataloaders_spectrograms_from_eeg_combined}
model_dict = {'efficientnet_b0': build_efficientnet_b0,
              'starter_effnet': EEGEffnetB0,
              'MultiResidualBiGRU': MultiResidualBiGRU,
              'wavenet': SequentialWaveNet,
              'eeg_conv': EEGNet,
              'cnn': CNN,
              'cnn2': CNN_2,
              'cnn_dual_input': CNN_Dual_Input,
              'cnn_multichannel': CNN_MultiChannel,
              'efficientnet_wrapper':EfficientnetWrapper}
loss_dict = {'KLDivLoss': nn.KLDivLoss}
optim_dict = {'Adam': optim.Adam}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def experiment(config):    

    if config['project_name'] is not None:
        run = wandb.init(project=config['project_name'], config=config, name=config['run_name'])
    else:
        run = None
    
    create_dataloaders = dataloaders_dict.get(config['create_dataloaders_func'])    
    dataloader_train, dataloader_valid = create_dataloaders(**config['data_parameters'])        

    model_class = model_dict.get(config['model'])
    model = model_class(**config['model_parameters']).to(device=config['device'], dtype=torch.float32)
    print('', '-'*100, '\n', '  Number of trainable parameters in model: ', count_parameters(model), '\n', '-'*100)
    
    loss_criterion = loss_dict.get(config['loss'])(reduction="batchmean")
    optimizer = optim_dict.get(config['optimizer'])(model.parameters(), lr=config['learning_rate'])

    
    training_loop(dataloader_train, 
                  dataloader_valid, 
                  model, 
                  optimizer, 
                  loss_criterion, 
                  num_epochs=config['epochs'], 
                  start_epoch=0,
                  device=config['device'],
                  wandb_run=run)

    # run.finish()
    wandb.finish()

    del run, dataloader_train, dataloader_valid, model, loss_criterion, optimizer
    gc.collect()

    torch.cuda.empty_cache()
    





def continue_experiment(run_id, checkpoint_name='checkpoint', version='latest', num_epochs=None, learning_rate=None):
    run = wandb.init(project=PROJECT_NAME, id=run_id, resume='allow')
    config = run.config
    
    create_dataloaders = dataloaders_dict.get(config['create_dataloaders_func'])    
    dataloader_train, dataloader_valid = create_dataloaders(**config['data_parameters'])        

    model_class = model_dict.get(config['model'])
    model = model_class(**config['model_parameters']).to(device=config['device'], dtype=torch.float32)
    print('', '-'*100, '\n', '  Number of trainable parameters in model: ', count_parameters(model), '\n', '-'*100)
    
    loss_criterion = loss_dict.get(config['loss'])(reduction="batchmean")
    optimizer = optim_dict.get(config['optimizer'])(model.parameters(), lr=config['learning_rate'])
    
    last_epoch = 0    

    if wandb.run.resumed:
        
        checkpoint_name = f'{run.id}_{checkpoint_name}'
        artifact = run.use_artifact(checkpoint_name + f':{version}')
        entry = artifact.get_path(checkpoint_name + '.pth')
        
        file = entry.download()
        
        checkpoint = torch.load(file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        
        if learning_rate is not None:
            for g in optimizer.param_groups:
                g['lr'] = learning_rate

        print(f'\nResuming training after epoch {last_epoch}\n')
        print(f'Pevious best loss:  train {checkpoint["loss_train"]:.5f}')
        print(f'                    valid {checkpoint["loss_valid"]:.5f}\n')


    
    training_loop(dataloader_train, 
                  dataloader_valid, 
                  model, 
                  optimizer, 
                  loss_criterion, 
                  num_epochs=config['epochs'] if num_epochs is None else num_epochs, 
                  start_epoch=last_epoch+1,
                  device=config['device'],
                  wandb_run=run,
                  prev_best_loss=checkpoint["loss_valid"], 
                  post_proc=None)

    run.finish()
    wandb.finish()

