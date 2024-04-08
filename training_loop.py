import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import wandb
from src.styles import TXT_ACC, TXT_RESET
import gc

from src.efficientnet_wrapper_model import EfficientnetWrapper

import torch.nn.functional as F

def train_epoch(dataloader, model, optimizer, loss_criterion, device):
    
    model.train()
    loss = 0.0

    for b, batch in enumerate(pbar := tqdm(dataloader)):

        data = [batch[i].to(device=device) for i in range(len(batch) - 1)]
        target = batch[-1].to(device=device)
        
        output = model(*data)

        cur_loss = loss_criterion(F.log_softmax(output, dim=1), target)
        loss += cur_loss.item()

        cur_loss.backward()
        optimizer.step()        
        optimizer.zero_grad()
        
        pbar.set_description(f'  training batch:    batch loss {cur_loss: .5f}     mean loss {loss / (b+1): .5f}')

    return loss / len(dataloader)


def validate_epoch(dataloader, model, loss_criterion, device):
    model.eval()
    loss = 0.0

    # preds = []
    # targets = []

    with torch.no_grad():
        for b, batch in enumerate(pbar := tqdm(dataloader)):
            data = [batch[i].to(device=device) for i in range(len(batch) - 1)]
            target = batch[-1].to(device=device)
            
            output = model(*data)
            cur_loss = loss_criterion(F.log_softmax(output, dim=1), target)
            loss += cur_loss

            # preds.append(F.log_softmax(output, dim=1).cpu().detach().numpy())
            # targets.append(target.cpu().detach().numpy())

            pbar.set_description(f'validation batch:    batch loss {cur_loss: .5f}     mean loss {loss / (b+1): .5f}')
    

    # total_loss = torch.nn.KLDivLoss(reduction='batchmean')(torch.Tensor(np.concatenate(preds, axis=0)), 
    #                                                        torch.Tensor(np.concatenate(targets, axis=0)))
    # print(f'\tvalidation:  loss {total_loss: .5f}')


    return loss / len(dataloader)
 


def log_training(wandb_run, epoch, model, optimizer, loss_train, loss_valid, log_name):
    name = f'{wandb_run.id}_{log_name}'
    path = 'trained_models/' + name + '.pth'
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_train': loss_train, 
                'loss_valid': loss_valid}, 
            path)
                      
    artifact = wandb.Artifact(name=name, type=log_name)
    artifact.add_file(path)            
    wandb_run.log_artifact(artifact)


def training_loop(dataloader_train, 
                  dataloader_valid, 
                  model, 
                  optimizer, 
                  loss_criterion, 
                  num_epochs, 
                  start_epoch,
                  device,
                  wandb_run,
                  prev_best_loss = np.inf,
                  post_proc=None):
    
    losses_train = np.zeros(num_epochs)
    losses_valid = np.zeros(num_epochs)
    
    best_loss = prev_best_loss
    best_epoch = start_epoch - 1

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    
    for epoch in range(num_epochs):  

        if epoch > 1 and isinstance(model, EfficientnetWrapper):
            model.unfreeze_feature_exctractor_weights()
        
        print(f'{TXT_ACC} Epoch {start_epoch+epoch} {TXT_RESET}')
        
        loss_train = train_epoch(dataloader_train, model, optimizer, loss_criterion, device)
        loss_valid = validate_epoch(dataloader_valid, model, loss_criterion, device)
        
        scheduler.step()

        losses_train[epoch] = loss_train
        losses_valid[epoch] = loss_valid        
            
        print(f'\ttrain:       loss {loss_train: .5f}')        
        print(f'\tvalidation:  loss {loss_valid: .5f}')

        if wandb_run is not None:
            wandb_run.log({'loss_train': loss_train, 
                           'loss_valid': loss_valid}) 
        
        if loss_valid < (best_loss - 0.005):
            best_epoch = epoch
            best_loss = loss_valid
            
            if wandb_run is not None:
                log_training(wandb_run, epoch+start_epoch, model, optimizer, loss_train, loss_valid, 'best_model')
            
    if wandb_run is not None:       
        log_training(wandb_run, epoch+start_epoch, model, optimizer, loss_train, loss_valid,  'checkpoint')

    return losses_train, losses_valid