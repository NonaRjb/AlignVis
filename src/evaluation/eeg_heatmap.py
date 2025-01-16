import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/alignvis")
# sys.path.append("/proj/rep-learning-robotics/users/x_nonra/signal-heatmaps")

import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torchsummary import summary
import os
import re
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# from generate_heatmap import GradCAM, heatmap
from src.evaluation.gradcam import GradCAM, viz_attn, gradCAM
from src.models.brain_encoder import BrainEncoder
from src.train_brain_clip import model_configs
import src.utils as utils


model_name_mapping = {
    'dreamsim_clip_vitb32': 'dreamsim_CLIP_ViT-B32',
    'CLIP_ViT-B32_noalign': 'CLIP_ViT-B32_noalign',
    'dreamsim_synclr_vitb16': 'dreamsim_synclr_vitb16',
    'dreamsim_synclr_vitb16_noalign': 'dreamsim_synclr_vitb16_noalign',
}


def seed_everything(seed_val):
    np.random.seed(seed_val)
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return


def load_checkpoint(directory, brain_encoder, dataset_name, seed_n):
    # Create a regex pattern to match the filename format
    pattern = rf"{re.escape(brain_encoder)}_{re.escape(dataset_name)}_seed{re.escape(str(seed_n))}_\d+\.pth"
    
    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        if re.fullmatch(pattern, filename):
            checkpoint_path = os.path.join(directory, filename)
            print(f"Loading checkpoint: {checkpoint_path}")
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path)['model_state_dict']
            return checkpoint
    
    # Raise an error if no matching checkpoint is found
    raise FileNotFoundError(f"No checkpoint found matching the pattern {pattern} in directory {directory}")


def load_brain_encoder(model_path, backbone_encoder, embedding_size, device='cuda', **kwargs):
    brain_encoder = BrainEncoder(
            embed_dim=embedding_size,
            backbone=backbone_encoder,
            n_channels=kwargs['n_channels'], 
            n_samples=kwargs['n_samples'],
            n_classes=kwargs['n_classes'],
            model_path=None,
            device=device, 
            **model_configs[backbone_encoder]
            )
    brain_encoder = brain_encoder.float()
    brain_encoder.to(device)
    # print(summary(brain_encoder, (1, 63, 250)))
    if model_path:
        checkpoint = load_checkpoint(model_path, backbone_encoder, kwargs['dataset_name'], kwargs['seed'])
        # checkpoint = torch.load(model_path)['model_state_dict']
        brain_encoder.load_state_dict(checkpoint, strict=False)
        brain_encoder.to(device)
    return brain_encoder


def load_data(dataset_name, subject_id, data_path, img_encoder, seed_val=42, split='test', device_type='cuda'):
    data, ds_configs = utils.load_dataset(
        dataset_name=dataset_name, 
        data_path=data_path, 
        sid=subject_id, 
        load_img="embedding", 
        return_subject_id=False,
        split=split,
        select_channels=None, 
        subj_training_ratio=1.0, 
        img_encoder=img_encoder,
        interpolate=None,
        )
    g = torch.Generator().manual_seed(seed_val)
    data_loader = DataLoader(data, batch_size=1, shuffle=False,
                            drop_last=False,
                            pin_memory=True if 'cuda' in device_type else False,
                            generator=g)
    return data_loader, ds_configs


def parse_args():
    parser = argparse.ArgumentParser(description='Generate heatmaps for EEG signals')
    parser.add_argument('--data_path', type=str, default='/data')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='things-eeg-2')
    parser.add_argument('--subject_id', nargs='+', type=int, default=[1])
    parser.add_argument('--img_encoder', type=str, default='dreamsim_clip_vitb32')
    parser.add_argument('--brain_encoder', type=str, default='nice')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    seed_everything(42)

    heatmaps = {}
    inputs = {}
    for subject_id in args.subject_id:
        
        data_loader, ds_configs = load_data(
            dataset_name=args.dataset, 
            subject_id=subject_id, 
            data_path=args.data_path, 
            img_encoder=args.img_encoder, 
            split=args.split,
            device_type=device)

        model_path = os.path.join(
            args.model_path, 
            model_name_mapping[args.img_encoder], 
            f"sub-{subject_id:02}",
            "models")

        brain_encoder = load_brain_encoder(
            model_path,
            args.brain_encoder,
            embedding_size=model_configs[args.img_encoder]['embed_dim'],
            device=device,
            dataset_name=args.dataset,
            seed=args.seed,
            **ds_configs)

        brain_encoder.eval()
        # print(brain_encoder)
        # grad_cam_model = GradCAM(model=brain_encoder, target_layer='brain_backbone.encoder.0.projection.0')
        for i, (data, l) in enumerate(data_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            attn_map = gradCAM(brain_encoder, x, y, "brain_backbone.encoder.0.tsconv.3")    # brain_backbone.encoder.0.projection.0
            attn_map = attn_map.squeeze().detach().cpu().numpy()
            # attn_map = grad_cam_model.generate_heatmap(x, y).squeeze()
            if i not in heatmaps.keys():
                heatmaps[str(i)] = [attn_map]
                inputs[str(i)] = [x.squeeze().detach().cpu().numpy()]
            else:
                heatmaps[str(i)].append(attn_map)
                inputs[str(i)].append(x.squeeze().detach().cpu().numpy())
            if i == 10:
                break
    hm = {key: np.mean(np.array(heatmaps[key]), axis=0) for key in heatmaps.keys()}
    ins = {key: np.mean(np.array(inputs[key]), axis=0) for key in inputs.keys()}
    for i, k in enumerate(hm.keys()):
        viz_attn(ins[k], hm[k], blur=False, ch_names=ds_configs['ch_names'], save_path=os.path.join(args.save_path, f"heatmap_{args.img_encoder}_{i}.png"))
        # plt.imshow(hm[k], cmap="hot")
        # plt.savefig(os.path.join(args.save_path, f"heatmap_{args.img_encoder}_{i}.png"))
    print("Heatmaps generated")

    
