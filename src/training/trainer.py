import os
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import wandb
import src.training.training_utils as ut
import copy 


class BimodalTrainer:
    def __init__(self, 
        brain_encoder: torch.nn.Module, 
        image_encoder: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        loss,
        save_path, 
        filename, 
        epochs: int,
        lr: float,
        min_lr: float=1e-6,
        warmup_epochs: int=0,
        lr_patience=10, 
        es_patience=30,
        return_subject_id=False,
        scheduler="cosine",
        device='cuda:0', **kwargs):

        self.device = device

        self.brain_encoder = brain_encoder.to(device)
        self.image_encoder = image_encoder.to(device) if image_encoder is not None else None
        self.return_subject_id = return_subject_id
        self.precompute_img_emb = kwargs['precompute_img_emb']

        self.loss = loss.to("cuda" if device.startswith("cuda") else "cpu")
        self.optimizer = optimizer

        self.epochs = epochs
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.scheduler = scheduler
        self.initial_epochs = kwargs['initial_epochs'] if 'initial_epochs' in kwargs else 0
        
        self.save_path = save_path
        self.filename = filename
        self.lr_patience = lr_patience
        self.es_patience = es_patience
        self.mixed_precision = True
        self.clip_grad = 1.0
        self.log_every = 0

    def train(self, train_data_loader: DataLoader, val_data_loader: DataLoader):
        scaler = GradScaler(enabled=self.mixed_precision)

        warmup_scheduler = ut.WarmupScheduler(self.optimizer, self.warmup_epochs, self.lr, start_lr=self.min_lr)
        if self.scheduler == "plateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=self.lr_patience)
        elif self.scheduler == "cosine":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=self.min_lr)
        else:
            raise NotImplementedError


        best_model = None
        best_loss = 10000000
        patience = self.es_patience
        print("Training Started...")
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}/{self.epochs}.")

            steps = 0
            loss_epoch = []
            self.brain_encoder.train()
            if self.image_encoder is not None:
                self.image_encoder.eval()
            progress_bar = tqdm(train_data_loader)
            for data, _ in progress_bar:

                self.optimizer.zero_grad()

                if self.return_subject_id:
                    eeg, image = data[0]
                    subject_id = data[1]
                    subject_id = subject_id.to(self.device, non_blocking=True)
                else:
                    eeg, image = data
                eeg, image = eeg.to(self.device, non_blocking=True), image.to(self.device, non_blocking=True)

                with torch.autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu", enabled=self.mixed_precision):
                    if self.return_subject_id:
                        z_i = self.brain_encoder(eeg, subject_id)
                    else:
                        z_i = self.brain_encoder(eeg)
                    if self.image_encoder is not None:
                        z_j = self.image_encoder(image)
                    elif self.precompute_img_emb:
                        z_j = image
                    else:
                        z_j = self.brain_encoder(image)
                
                    z_i = z_i - torch.mean(z_i, dim=-1, keepdim=True)
                    z_i = F.normalize(z_i, p=2, dim=-1)
                    z_j = z_j - torch.mean(z_j, dim=-1, keepdim=True)
                    z_j = F.normalize(z_j, p=2, dim=-1)
                    loss = self.loss(z_i, z_j)

                loss_epoch.append(loss.item())

                scaler.scale(loss).backward()
                if self.clip_grad is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.brain_encoder.parameters(), self.clip_grad)
                scaler.step(self.optimizer)
                scaler.update()

                if self.device == torch.device('cuda:0'):
                    self.lr = self.optimizer.param_groups[0]["lr"]

                steps += 1

            train_loss = np.mean(loss_epoch)

            val_loss = self.evaluate(self.brain_encoder, self.image_encoder, val_data_loader)
            if val_loss < best_loss:
                best_loss = val_loss
                patience = self.es_patience
                best_model = {
                    'epoch': epoch,
                    'model_state_dict': copy.deepcopy(self.brain_encoder.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                    'val_loss': val_loss,
                }
            else:
                patience -= 1

            if patience == 0:
                break

            # Warmup phase
            if epoch < self.warmup_epochs:
                warmup_scheduler.step()  # Adjust learning rate during warmup
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # After warmup, apply ReduceLROnPlateau scheduler
            if epoch >= self.warmup_epochs:
                lr_scheduler.step(val_loss)  # Reduce LR if val_loss does not improve
                current_lr = self.optimizer.param_groups[0]['lr']

            print(f'Epoch: {epoch}')
            print(f'Training Loss: {train_loss}| Validation Loss: {val_loss}')

            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr
            })

            if self.log_every != 0 and epoch % self.log_every == 0:
                torch.save(best_model, os.path.join(self.save_path, self.filename + f"_{epoch}" + ".pth"))

        print("Finished training.")
        print("Creating checkpoint.")

        if best_model is None:
            best_model = {
                'epoch': self.epochs,
                'model_state_dict': copy.deepcopy(self.brain_encoder.state_dict()),
                'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                'val_loss': val_loss,
            }

        print(f"Best Validation Loss = {best_model['val_loss']} (Epoch = {best_model['epoch']})")
        torch.save(best_model, os.path.join(self.save_path, self.filename + f"_{epoch}" + ".pth"))
        print("Finished creating checkpoint.")

        return best_model

    def evaluate(self, brain_encoder, image_encoder, dataloader):

        brain_encoder.eval()
        if image_encoder is not None:
            image_encoder.eval()
        with torch.no_grad():

            loss_epoch = []
            progress_bar = tqdm(dataloader)
            for data, y in progress_bar:
                if self.return_subject_id:
                    data, subject_id = data
                    subject_id = subject_id.to(self.device, non_blocking=True)
                eeg, image = data
                eeg, image = eeg.to(self.device, non_blocking=True), image.to(self.device, non_blocking=True)

                with torch.autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu", enabled=self.mixed_precision):
                    if self.return_subject_id:
                        z_i = brain_encoder(eeg, subject_id)
                    else:
                        z_i = brain_encoder(eeg)
                    if image_encoder is not None:
                        z_j = image_encoder(image)
                    elif self.precompute_img_emb:
                        z_j = image
                    else:
                        z_j = brain_encoder(image)
                    z_i = z_i - torch.mean(z_i, dim=-1, keepdim=True)
                    z_i = F.normalize(z_i, p=2, dim=-1)
                    z_j = z_j - torch.mean(z_j, dim=-1, keepdim=True)
                    z_j = F.normalize(z_j, p=2, dim=-1)

                    loss = self.loss(z_i, z_j)
                loss_epoch.append(loss.item())
                
            mean_loss_epoch = np.mean(loss_epoch)
            
        return mean_loss_epoch
