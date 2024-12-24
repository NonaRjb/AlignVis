import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import inf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import math
import os


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def adjust_learning_rate(optimizer, epoch, **kwargs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < kwargs['warmup_epochs']:
        lr = kwargs['lr'] * epoch / kwargs['warmup_epochs']
    else:
        lr = kwargs['min_lr'] + (kwargs['lr'] - kwargs['min_lr']) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - kwargs['warmup_epochs']) / (kwargs['num_epoch'] - kwargs['warmup_epochs'])))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


@torch.no_grad()
def plot_recon_figures(recon_model, curr_device, dataset, output_path, num_figures=5, config=None, logger=None,
                       model_without_ddp=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    recon_model.eval()
    fig, axs = plt.subplots(num_figures, 3, figsize=(30, 15))
    fig.tight_layout()
    axs[0, 0].set_title('Ground-truth')
    axs[0, 1].set_title('Masked Ground-truth')
    axs[0, 2].set_title('Reconstruction')

    for ax in axs:
        sample = next(iter(dataloader))
        sample = sample.to(curr_device)
        _, pred, mask = recon_model(sample, mask_ratio=config.mask_ratio)
        sample_with_mask = sample.to('cpu').squeeze(0)[0].numpy().reshape(-1, model_without_ddp.patch_size)
        pred = model_without_ddp.unpatchify(pred).to('cpu').squeeze(0)[0].numpy()
        sample = sample.to('cpu').squeeze(0)[0].numpy()
        mask = mask.to('cpu').numpy().reshape(-1)

        cor = np.corrcoef([pred, sample])[0,1]

        x_axis = np.arange(0, sample.shape[-1])
        # groundtruth
        ax[0].plot(x_axis, sample)
        # groundtruth with mask
        s = 0
        for x, m in zip(sample_with_mask, mask):
            if m == 0:
                ax[1].plot(x_axis[s:s+len(x)], x, color='#1f77b4')
            s += len(x)
        # pred
        ax[2].plot(x_axis, pred)
        ax[2].set_ylabel('cor: %.4f'%cor, weight = 'bold')
        ax[2].yaxis.set_label_position("right")

    fig_name = 'reconst-%s'%(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    fig.savefig(os.path.join(output_path, f'{fig_name}.png'))
    if logger is not None:
        logger.log_image('reconst', fig)
    plt.close(fig)


class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, max_lr, start_lr=1e-6):
        """
        Warmup scheduler for the first `warmup_epochs`.
        
        Args:
            optimizer (torch.optim.Optimizer): Optimizer for which to schedule the learning rate.
            warmup_epochs (int): Number of epochs to warm up the learning rate.
            max_lr (float): Maximum learning rate after warmup.
            start_lr (float): Starting learning rate at the beginning of warmup.
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.current_epoch = 0

    def step(self):
        """Update the learning rate for the current epoch."""
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linearly increase the learning rate during warmup
            lr = self.start_lr + (self.max_lr - self.start_lr) * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

class CLIPLoss(torch.nn.Module):

    def __init__(self, temperature: float = 0.04):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor):
        logit_scale = torch.exp(self.logit_scale)
        z_i_logits = (z_i @ z_j.T) * logit_scale
        z_j_logits = z_i_logits.T

        batch_size = z_i.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=z_i.device)

        loss_ij = nn.functional.cross_entropy(z_i_logits, labels)
        loss_ji = nn.functional.cross_entropy(z_j_logits, labels)

        return (loss_ij + loss_ji) / 2


class SoftCLIPLoss(torch.nn.Module):

    def __init__(self, temperature: float = 0.04):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor):
        logit_scale = torch.exp(self.logit_scale)
        clip_clip = (z_j @ z_j.T) * logit_scale
        brain_clip = (z_i @ z_j.T) * logit_scale

        loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
        loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
        loss = (loss1 + loss2)/2
        return loss


def mixco(voxels, beta=0.15, s_thresh=0.5, perm=None, betas=None, select=None):
    if perm is None:
        perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device,dtype=voxels.dtype)
    if betas is None:
        betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device,dtype=voxels.dtype)
    if select is None:
        select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select


def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False, 
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (preds @ targs.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  nn.functional.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = nn.functional.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss


class MultiPosConLoss(nn.Module):
    """Multi-positive contrastive loss, for image-text contrast."""
    def __init__(self, temperature=0.1, w2=1.0):
        """
        Args:
            temperature: temperature for contrastive loss
            w2: weight for the image-text part
        """
        super(MultiPosConLossMM, self).__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
        self.last_local_batch_size = None
        self.v_label_matrix = None
        self.t_label_matrix = None

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor, y):
        device = z_i.device

        logit_scale = torch.exp(self.logit_scale)

        # Compute logits for image-text contrasting
        z_i_logits = (z_i @ z_j.T) * logit_scale
        z_j_logits = z_i_logits.T

        # Compute label matrices for image-text contrastive loss
        

        # Image-text loss
        v_mask = self.v_label_matrix
        p_v = v_mask / v_mask.sum(1, keepdim=True).clamp(min=1.0)
        t_mask = self.t_label_matrix
        p_t = t_mask / t_mask.sum(1, keepdim=True).clamp(min=1.0)
        img_txt_loss = (nn.functional.cross_entropy(p_v, logits_v) + nn.functional.cross_entropy(p_t, logits_t)) / 2

        # Total loss
        loss = img_txt_loss

        return {'loss': loss, 'img_txt_loss': img_txt_loss}
