import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/alignvis")

import itertools
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import numpy as np
from datetime import datetime
import wandb
import random
import argparse
import os
import re
import warnings
import pickle

import src.utils as utils
from src.models.brain_encoder import BrainEncoder
from src.models.image_encoder import ImageEncoder
from src.training.trainer import UnimodalTrainer
from sklearn.preprocessing import LabelEncoder
from src import downstream


label_type = {
    'classification': ['color', 'object', 'color_high', 'object_high'],
    'regression': ['gabor', 'hist', 'wavelet']
}

model_configs = {
        'eegnet': {},
        'eegconformer': {},
        'nice': {'emb_size': 40, 'embedding_dim': 1440, 'drop_proj': 0.5},
        'atms': {},
        'lstm': {'lstm_size': 128, 'lstm_layers': 1},
        'resnet1d': {},
        'resnet1d_subj': {},
        'resnet1d_subj_resblk': {},
        'brain-mlp': {},
        'dinov2_vit-l-14_noalign': {'embed_dim': 1024},
        'DINOv2_ViT-L14_noalign': {'embed_dim': 1024},
        'DINO_ViT-B8_noalign': {'embed_dim': 768},
        'DINOv2_ViT-B14_noalign': {'embed_dim': 768},
        'DINO_ViT-B16_noalign': {'embed_dim': 768},
        'CLIP_ViT-L14_noalign': {'embed_dim': 768},
        'CLIP_ViT-B32_noalign': {'embed_dim': 512},
        'OpenCLIP_ViT-L14_laion400m_noalign': {'embed_dim': 768},
        'OpenCLIP_ViT-L14_laion2b_noalign': {'embed_dim': 768},
        'OpenCLIP_ViT-B32_laion400m_noalign': {'embed_dim': 512},
        'dreamsim_clip_vitb32_768': {'embed_dim': 768},
        'dreamsim_clip_vitb32': {'embed_dim': 512},
        'dreamsim_clip_vitb32_noalign': {'embed_dim': 512},
        'dreamsim_open_clip_vitb32': {'embed_dim': 512},
        'dreamsim_open_clip_vitb32_noalign': {'embed_dim': 512},
        'dreamsim_synclr_vitb16': {'embed_dim': 768},
        'dreamsim_synclr_vitb16_noalign': {'embed_dim': 768},
        'dreamsim_ensemble': {'embed_dim': 1792},
        'dreamsim_ensemble_noalign': {'embed_dim': 1792},
        'dreamsim_dino_vitb16': {'embed_dim': 768},
        'dreamsim_dino_vitb16_noalign': {'embed_dim': 768},
        'dreamsim_dinov2_vitb14': {'embed_dim': 768},
        'dreamsim_dinov2_vitb14_noalign': {'embed_dim': 768},
        'gLocal_openclip_vit-l-14_laion2b_s32b_b82k': {'embed_dim': 768},
        'gLocal_openclip_vit-l-14_laion2b_s32b_b82k_noalign': {'embed_dim': 768},
        'gLocal_openclip_vit-l-14_laion400m_e32': {'embed_dim': 768},
        'gLocal_openclip_vit-l-14_laion400m_e32_noalign': {'embed_dim': 768},
        'gLocal_clip_vit-l-14': {'embed_dim': 768},
        'gLocal_clip_vit-l-14_noalign': {'embed_dim': 768},
        'gLocal_dino-vit-base-p8': {'embed_dim': 768},
        'gLocal_dino-vit-base-p8_noalign': {'embed_dim': 768},
        'gLocal_dino-vit-base-p16': {'embed_dim': 768},
        'gLocal_dino-vit-base-p16_noalign': {'embed_dim': 768},
        'gLocal_dinov2-vit-base-p14': {'embed_dim': 768},
        'gLocal_dinov2-vit-base-p14_noalign': {'embed_dim': 768},
        'gLocal_dinov2-vit-large-p14': {'embed_dim': 1024},
        'gLocal_dinov2-vit-large-p14_noalign': {'embed_dim': 1024},
        'gLocal_clip_rn50': {'embed_dim': 1024},
        'gLocal_clip_rn50_noalign': {'embed_dim': 1024},
        'harmonization_vitb16': {'embed_dim': 768},
        'harmonization_vitb16_noalign': {'embed_dim': 768},
        'harmonization_resnet50': {'embed_dim': 2048},
        'harmonization_resnet50_noalign': {'embed_dim': 2048},
        'harmonization_convnext': {'embed_dim': 768},
        'harmonization_convnext_noalign': {'embed_dim': 768},
        'harmonization_levit': {'embed_dim': 384},
        'harmonization_levit_noalign': {'embed_dim': 384},
        'harmonization_vgg16': {'embed_dim': 4096},
        'harmonization_vgg16_noalign': {'embed_dim': 4096},
    }

model_name_mapping = {
    'dreamsim_clip_vitb32': 'dreamsim_CLIP_ViT-B32',
    'CLIP_ViT-B32_noalign': 'CLIP_ViT-B32_noalign',
    'dreamsim_synclr_vitb16': 'dreamsim_synclr_vitb16',
    'dreamsim_synclr_vitb16_noalign': 'dreamsim_synclr_vitb16_noalign',
    'DINO_ViT-B16_noalign': 'DINO_ViT-B16_noalign',
    'dreamsim_dino_vitb16': 'dreamsim_DINO_ViT-B16',
    'DINOv2_ViT-B14_noalign': 'DINOv2_ViT-B14_noalign',
    'dreamsim_dinov2_vitb14': 'dreamsim_DINOv2_ViT-B14',
    'dreamsim_ensemble': 'dreamsim_ensemble',
    'dreamsim_ensemble_noalign': 'dreamsim_ensemble_noalign',
    'OpenCLIP_ViT-B32_laion400m_noalign': "OpenCLIP_ViT-B32_noalign",
    'dreamsim_open_clip_vitb32': "dreamsim_OpenCLIP_ViT-B32",
    'gLocal_clip_vit-l-14': 'glocal_CLIP_ViT-L14',
    'CLIP_ViT-L14_noalign': 'CLIP_ViT-L14_noalign',
    'gLocal_dino-vit-base-p8': 'gLocal_DINO_ViT-B8',
    'DINO_ViT-B8_noalign': 'DINO_ViT-B8_noalign',
    'gLocal_dino-vit-base-p16': 'gLocal_DINO_ViT-B16',
    'DINO_ViT-B16_noalign': 'DINO_ViT-B16_noalign',
    'gLocal_openclip_vit-l-14_laion400m_e32': 'gLocal_OpenCLIP_ViT-L14',
    'OpenCLIP_ViT-L14_laion400m_noalign': 'OpenCLIP_ViT-L14_noalign',
    'gLocal_clip_rn50': 'gLocal_clip_rn50',
    'gLocal_clip_rn50_noalign': 'gLocal_clip_rn50_noalign',
}


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
    if model_path:
        checkpoint = load_checkpoint(model_path, backbone_encoder, kwargs['dataset_name'], kwargs['seed'])
        # checkpoint = torch.load(model_path)['model_state_dict']
        brain_encoder.load_state_dict(checkpoint, strict=False)
        brain_encoder.to(device)
    return brain_encoder


def return_dataloaders(
    dataset_nm, 
    data_pth, 
    sid, 
    batch, 
    num_workers, 
    seed_val, 
    device_type, 
    separate_test=False, 
    **kwargs):

    data, ds_configs = utils.load_dataset(
        dataset_name=dataset_nm, 
        data_path=data_path, 
        sid=sid, 
        subj_training_ratio=1.0,
        split='train',
        **kwargs
        )
    
    print(ds_configs)
    
    g = torch.Generator().manual_seed(seed_val)

    if not separate_test:
        train_data, val_data, test_data = torch.utils.data.random_split(
            data, [0.8, 0.1, 0.1], generator=g)
    else:
        pca = PCA(n_components=0.95)
        data.labels_list = pca.fit_transform(data.labels_list).astype(np.float32)
        train_data, val_data = torch.utils.data.random_split(
            data, [0.9, 0.1], generator=g)
        test_data, _ = utils.load_dataset(
            dataset_name=dataset_nm, 
            data_path=data_path, 
            sid=kwargs['test_subject'], 
            split='test',  
            subj_training_ratio=1.0, 
            **kwargs)
        test_data.labels_list = pca.transform(test_data.labels_list).astype(np.float32)
        print("Number of reg classes: ", np.array(test_data.labels_list).shape[-1])
    train_dl = DataLoader(train_data, batch_size=batch, shuffle=True,
                            drop_last=True,
                            num_workers=num_workers,
                            pin_memory=True if 'cuda' in device_type else False,
                            generator=g)
    val_dl = DataLoader(val_data, batch_size=64, shuffle=False,
                            drop_last=False,
                            num_workers=num_workers,
                            pin_memory=True if 'cuda' in device_type else False,
                            generator=g)
    test_dl = DataLoader(test_data, batch_size=batch, shuffle=False,
                            drop_last=False,
                            pin_memory=True if 'cuda' in device_type else False,
                            generator=g)

    if 'new_labels_type' in kwargs.keys() and kwargs['new_labels_type'] is not None:
        ds_configs['n_classes'] = len(np.unique(np.array(data.labels_list))) if kwargs['new_labels_type'] in label_type['classification'] else np.array(data.labels_list).shape[-1]
        print("Number of classes: ", ds_configs['n_classes'])
    return train_dl, val_dl, test_dl, ds_configs


def seed_everything(seed_val):
    np.random.seed(seed_val)
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('--data_path', type=str, help="Path to the EEG data")
    parser.add_argument('--save_path', type=str, help="Path to save the model")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to the pretrained model")
    parser.add_argument("--new_label_path", type=str, default=None, help="Path to the directory containing the new label files")
    parser.add_argument("--new_label_type", type=str, default="color", help="Type of new labels to use")
    parser.add_argument('--dataset', type=str, default="things-eeg-2")
    parser.add_argument('--subject_id', type=int, nargs='+', default=[0], help="Subject ID(s). Provide one or more subject IDs.")
    parser.add_argument('--test_subject', type=int, default=None)
    parser.add_argument('--interpolate', type=int, default=None, help="Resampling rate for EEG data")
    parser.add_argument('--eeg_enc', type=str, default="resnet1d", help="EEG Encoder")
    parser.add_argument('--img_encoder', type=str, default="dreamsim_clip_vitb32", help="Image Encoder")
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=None)
    parser.add_argument('--net_seq_length', type=int, nargs='+', default=None)
    parser.add_argument('--epoch', type=int, default=1000, help="Number of epochs for pretraining")
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--scheduler', type=str, default="plateau")
    parser.add_argument('--patience', type=int, default=25, help="Patience for the reduce_lr_on_plateau scheduler")
    parser.add_argument('--temperature', type=float, default=0.04)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--separate_test', action="store_true")
    parser.add_argument('-b', '--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment', type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with wandb.init():
        # args = wandb.config
        seed = args.seed
        dataset_name = args.dataset
        subject_id = args.subject_id
        if len(subject_id) == 1:    #TODO to be compatible with Spampinato until I fix it
            subject_id = subject_id[0]
        test_subject = args.test_subject
        brain_enc_name = args.eeg_enc
        batch_size = args.batch
        lr = args.lr
        epochs = args.epoch
        data_path = args.data_path
        save_path = args.save_path
        separate_test_set = args.separate_test

        if args.net_filter_size:
            model_configs['resnet1d']['net_filter_size'] = args.net_filter_size
            model_configs['resnet1d_subj']['net_filter_size'] = args.net_filter_size
            model_configs['resnet1d_subj_resblk']['net_filter_size'] = args.net_filter_size

        if args.net_seq_length:
            model_configs['resnet1d']['net_seq_length'] = args.net_seq_length
            model_configs['resnet1d_subj']['net_seq_length'] = args.net_seq_length
            model_configs['resnet1d_subj_resblk']['net_seq_length'] = args.net_seq_length

        if args.dataset == "things-meg":
            model_configs['nice']['embedding_dim'] = 217360
            model_configs['nice']['emb_size'] = 40

        if args.checkpoint:
            model_configs[brain_enc_name]['subject_ids'] = [str(s) for s in range(1, 11)]
        else:
            model_configs[brain_enc_name]['subject_ids'] = [str(s) for s in subject_id] if isinstance(subject_id, list) else [str(subject_id)]
        
        print("training subjects: ", subject_id)
        print("test subjects: ", test_subject if test_subject is not None else subject_id)

        # constants
        min_lr = 1e-07
        warmup_epochs = args.warmup
        weight_decay=0.1
        
        seed_everything(seed)
        if args.experiment is not None:
            save_path = os.path.join(save_path, args.experiment)
            os.makedirs(save_path, exist_ok=True)
        paths = {"brain_data": data_path, "save_path": save_path}
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("device = ", device)

        print("**********************************************************************************************")
        print(f"Starting a run on {dataset_name} with {brain_enc_name}")

        directory_name = args.new_label_type + "_" + args.img_encoder + "_" + brain_enc_name
        os.makedirs(os.path.join(paths["save_path"], directory_name), exist_ok=True)
        if len(args.subject_id) == 1:
            os.makedirs(os.path.join(paths["save_path"], directory_name, f"sub-{subject_id:02}"), exist_ok=True)
            paths["save_path"] = os.path.join(paths["save_path"], directory_name, f"sub-{subject_id:02}")
        else:
            os.makedirs(os.path.join(paths["save_path"], directory_name, f"sub-{test_subject:02}"), exist_ok=True)
            paths["save_path"] = os.path.join(paths["save_path"], directory_name, f"sub-{test_subject:02}")
        
        print(f"Directory '{directory_name}' created.")
        utils.save_config(args, root_path=paths['save_path'])
        print(vars(args))

        train_data_loader, val_data_loader, test_data_loader, data_configs = return_dataloaders(
            dataset_nm=dataset_name, 
            data_pth=paths['brain_data'], sid=subject_id, 
            test_subject=test_subject if test_subject is not None else subject_id,
            batch=batch_size, 
            num_workers=args.n_workers,
            seed_val=seed, 
            load_img="no-image",
            separate_test=separate_test_set,
            return_subject_id=False,
            img_encoder=None,
            interpolate=args.interpolate,
            window=None,
            new_labels_type=args.new_label_type,
            new_labels_path=args.new_label_path,
            device_type=device)    
        
        # Create label encoder
        if args.new_label_type in label_type['classification']:
            with open(os.path.join(paths['brain_data'], "things_eeg_2", "labels_clip", f'{args.new_label_type}.txt'), "r") as f:
                labels = [line.strip() for line in f.readlines()]
            label_encoder = LabelEncoder()
            label_encoder.fit(labels)  # Fit with all possible labels
        else:
            label_encoder = None
        
        model_path = os.path.join(args.checkpoint, model_name_mapping[args.img_encoder], f"sub-{test_subject:02}", "models")

        brain_encoder = load_brain_encoder(
            model_path,
            brain_enc_name,
            embedding_size=model_configs[args.img_encoder]['embed_dim'],
            dataset_name=args.dataset,
            seed=args.seed,
            device=device,
            **data_configs)
        for param in brain_encoder.parameters():
            param.requires_grad = False 
        model = torch.nn.Sequential(brain_encoder, torch.nn.Linear(model_configs[args.img_encoder]['embed_dim'], data_configs['n_classes']))
        model = model.float()
        model.to(device)

        loss = torch.nn.CrossEntropyLoss() if args.new_label_type in label_type['classification'] else torch.nn.MSELoss()
        optim = torch.optim.AdamW(model.parameters(), lr=min_lr if warmup_epochs>0 else lr, weight_decay=weight_decay)
        trainer = UnimodalTrainer(
            brain_encoder=model, 
            optimizer=optim, 
            loss=loss,
            task="classification" if args.new_label_type in label_type['classification'] else "regression",
            save_path=paths["save_path"], 
            filename=f'{brain_enc_name}_{args.img_encoder}_{dataset_name}_seed{seed}',
            epochs=epochs,
            lr=lr,
            num_classes=data_configs['n_classes'],  # Added parameter for class count
            label_encoder=label_encoder,  # Added label encoder
            min_lr=min_lr,
            warmup_epochs=warmup_epochs,
            lr_patience=args.patience, 
            es_patience=30,
            scheduler=args.scheduler,
            device='cuda:0')
        
        best_classifier = trainer.train(train_data_loader, val_data_loader)
        model.load_state_dict(best_classifier['model_state_dict'])
        model.eval()
        test_loss, test_acc, test_auroc = trainer.evaluate(model, test_data_loader)
        print(f"Best Epoch: {best_classifier['epoch']}")
        if args.new_label_type in label_type['classification']:
            print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}, Test AUROC: {test_auroc}")
        else:
            print(f"Test Loss: {test_loss}")

        if args.new_label_type in label_type['regression']:
            y_test_pred = []
            y_test_true = []
            for x, y in test_data_loader:
                x = x.to(device)
                y_test_pred.extend(model(x).cpu().detach().numpy())
                y_test_true.extend(y.cpu().detach().numpy())
            y_test_pred = np.array(y_test_pred)
            y_test_pred = (y_test_pred - np.mean(y_test_pred, axis=-1, keepdims=True)) / np.std(y_test_pred, axis=-1, keepdims=True)
            y_test_true = np.array(y_test_true)
            y_test_true = (y_test_true - np.mean(y_test_true, axis=-1, keepdims=True)) / np.std(y_test_true, axis=-1, keepdims=True)

            r2 = r2_score(y_test_true, y_test_pred)
            print(f"R2 Score: {r2}")

        
        os.makedirs(os.path.join(paths["save_path"], "performances"), exist_ok=True)
        performances = {
            "train_loss": best_classifier["train_loss"],
            "train_acc": best_classifier["train_acc"],
            "train_auroc": best_classifier["train_auroc"],
            "val_loss": best_classifier["val_loss"],
            "val_acc": best_classifier["val_acc"],
            "val_auroc": best_classifier["val_auroc"],
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_auroc": test_auroc
        }
        with open(os.path.join(paths["save_path"], "performances", f'{seed}.pkl'), "wb") as f:
            pickle.dump(performances, f)
