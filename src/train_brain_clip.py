import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/alignvis")

import itertools
import torch
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import wandb
import random
import argparse
import os
import warnings
import pickle

import src.utils as utils
from src.models.brain_encoder import BrainEncoder
from src.models.image_encoder import ImageEncoder
from src.training.trainer import BimodalTrainer
from src.training.training_utils import CLIPLoss, SoftCLIPLoss
from src import downstream


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
    parser.add_argument('--dataset', type=str, default="things-eeg-2")
    parser.add_argument('--subject_id', type=int, nargs='+', default=[0], help="Subject ID(s). Provide one or more subject IDs.")
    parser.add_argument('--test_subject', type=int, default=None)
    parser.add_argument('--subj_training_ratio', type=float, default=1, help="a ratio between 0 and 1 determining how much of participants training samples to be used")
    parser.add_argument('--channels', type=int, nargs='+', default=None)
    parser.add_argument('--interpolate', type=int, default=None, help="Resampling rate for EEG data")
    parser.add_argument('--window', type=float, nargs=2, default=None, help="Window start and end for EEG data")
    parser.add_argument('--eeg_enc', type=str, default="resnet1d", help="EEG Encoder")
    parser.add_argument('--img_enc', type=str, default="CLIP_IMG", help="Image Encoder")
    parser.add_argument('--img_enc_model', type=str, default=None, help="Image Encoder Model")
    parser.add_argument('--loss', type=str, default="clip-loss", help="Loss function")
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=None)
    parser.add_argument('--net_seq_length', type=int, nargs='+', default=None)
    parser.add_argument('--img', type=str, default="embedding")
    parser.add_argument('--epoch', type=int, default=1000, help="Number of epochs for pretraining")
    parser.add_argument('--finetune_epoch',  type=int, default=50, help="Number of epochs for finetuning (if the downstream task is classification)")
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--scheduler', type=str, default="plateau")
    parser.add_argument('--patience', type=int, default=25, help="Patience for the reduce_lr_on_plateau scheduler")
    parser.add_argument('--temperature', type=float, default=0.04)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--downstream', type=str, default=None)
    parser.add_argument('--separate_test', action="store_true")
    parser.add_argument('--return_subject_id', action="store_true")
    parser.add_argument('--subj_spec_epochs', type=int, default=0)
    parser.add_argument('-b', '--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment', type=str, default=None)
    return parser.parse_args()

def return_dataloaders(
    dataset_nm, 
    data_pth, 
    sid, 
    subj_training_ratio,
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
        subj_training_ratio=subj_training_ratio,
        split='train',
        **kwargs
        )
    
    print(ds_configs)
    
    g = torch.Generator().manual_seed(seed_val)

    if not separate_test:
        train_data, val_data, test_data = torch.utils.data.random_split(
            data, [0.8, 0.1, 0.1], generator=g)
    else:
        train_data, val_data = torch.utils.data.random_split(
            data, [0.9, 0.1], generator=g)
        test_data, _ = utils.load_dataset(
            dataset_name=dataset_nm, 
            data_path=data_path, 
            sid=kwargs['test_subject'], 
            split='test',  
            subj_training_ratio=1.0, 
            **kwargs)
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
    return train_dl, val_dl, test_dl, ds_configs

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
        img_enc_name = args.img_enc
        batch_size = args.batch
        lr = args.lr
        epochs = args.epoch
        finetune_epochs = args.finetune_epoch
        data_path = args.data_path
        save_path = args.save_path
        downstream_task = args.downstream
        separate_test_set = args.separate_test
        channels=args.channels

        if args.subj_training_ratio == 0:
            epochs=0

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

        if separate_test_set and downstream_task == "classification":
            warnings.warn("The test set won't be used to finetune the classifier. seperate_test will be set to False")
            separate_test_set = False
        
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

        start_str = "scratch" if args.checkpoint is None else "pretrained"

        if test_subject is None:
            directory_name = f"{img_enc_name}"
        else:
            directory_name = f"{img_enc_name}"
        
        current_datetime = datetime.now()
        # directory_name += current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
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
            load_img=args.img,
            separate_test=separate_test_set,
            select_channels=channels,
            return_subject_id=args.return_subject_id,
            subj_training_ratio=args.subj_training_ratio if args.subj_training_ratio > 0 else 0.01,
            img_encoder=img_enc_name,
            interpolate=args.interpolate,
            window=args.window,
            device_type=device)   
        
        if args.img == "raw":
            img_encoder = ImageEncoder(
                backbone=img_enc_name,
                embed_dim=None,
                add_ln_layer=False,
                model_name=args.img_enc_model if args.img_enc_model is not None else "openai/clip-vit-base-patch32",
            )
            img_encoder = img_encoder.float()

            embedding_size = img_encoder.embed_dim
        elif args.img == "embedding":
            img_encoder = None
            embedding_size = model_configs[img_enc_name]['embed_dim']
        else:
            raise NotImplementedError
        
        print("eeg embedding size: ", embedding_size)
        brain_encoder = BrainEncoder(
            embed_dim=embedding_size,
            backbone=brain_enc_name,
            n_channels=data_configs["n_channels"],
            n_samples=data_configs["n_samples"],
            n_classes=data_configs["n_classes"],
            model_path=None,
            device=device, 
            **model_configs[brain_enc_name]
            )
        brain_encoder = brain_encoder.float()
        brain_encoder.to(device)
        # for n, p in brain_encoder.named_parameters():
        #     print(n)

        if args.loss == "clip-loss":
            loss = CLIPLoss(temperature=args.temperature)
        elif args.loss == "soft-clip":
            loss = SoftCLIPLoss(temperature=args.temperature)
        else:
            loss = CLIPLoss(temperature=args.temperature)

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint)['model_state_dict']
            brain_encoder.load_state_dict(checkpoint, strict=False)
            brain_encoder.to(device)
            
        if epochs > 0:
    
            optim = torch.optim.AdamW(brain_encoder.parameters(), 
                                    lr=min_lr if warmup_epochs>0 else lr, weight_decay=weight_decay)

            trainer = BimodalTrainer(
                brain_encoder=brain_encoder,
                image_encoder=img_encoder,
                optimizer=optim, 
                loss=loss, 
                epochs=epochs, 
                warmup_epochs=warmup_epochs,
                scheduler=args.scheduler,
                lr=lr, min_lr=min_lr,  
                mixed_precision=True,
                num_classes=data_configs["n_classes"],
                return_subject_id=args.return_subject_id,
                lr_patience=args.patience,
                es_patience=30,
                save_path=paths["save_path"], 
                filename=f'{brain_enc_name}_{dataset_name}_seed{seed}', 
                initial_epochs=args.subj_spec_epochs,
                precompute_img_emb=True if args.img == "embedding" else False,
                device=device
                )
            best_brain_encoder = trainer.train(train_data_loader, val_data_loader)
            brain_encoder.load_state_dict(best_brain_encoder['model_state_dict']) # TODO What if we also train the image encoder (embedding layer)
            # test_loss = trainer.evaluate(brain_encoder, img_encoder, test_data_loader)
            # print(f"Test Loss: {test_loss}")

        test_subject = test_subject if test_subject is not None else subject_id


        print(f"Performing the Downstream Task for S{test_subject if test_subject is not None else subject_id} (tr={args.subj_training_ratio})")
        if downstream_task == "classification":
            train_data_loader, val_data_loader, test_data_loader, data_configs = return_dataloaders(
                dataset_nm=dataset_name, 
                data_pth=paths['brain_data'], sid=subject_id, 
                test_subject=test_subject if test_subject is not None else subject_id,
                batch=batch_size, 
                num_workers=args.n_workers,
                seed_val=seed, 
                load_img=args.img,
                separate_test=separate_test_set,
                return_subject_id=args.return_subject_id,
                select_channels=channels,
                subj_training_ratio=args.subj_training_ratio,
                img_encoder=img_enc_name,
                interpolate=args.interpolate,
                window=args.window,
                device_type=device)
            loaders = {'train': train_data_loader, 'val': val_data_loader, 'test': test_data_loader} 
            raise NotImplementedError
            # test_loss, test_acc = downstream.classification(
            #     loaders=loaders,
            #     brain_enc_name=brain_enc_name, 
            #     dataset_name=dataset_name, n_channels=data_configs['n_channels'], n_samples=data_configs['n_samples'], n_classes=n_classes, 
            #     finetune_epochs=finetune_epochs, warmup_epochs=20, lr=lr, min_lr=min_lr, weight_decay=weight_decay,
            #     save_path=paths['save_path'],
            #     pretrained_encoder=brain_encoder, return_subject_id=args.return_subject_id, model_configs=model_configs, device=device
            # )
        elif downstream_task == "retrieval":
            _, _, test_data_loader, data_configs = return_dataloaders(
                dataset_nm=dataset_name, 
                data_pth=paths['brain_data'], sid=subject_id, 
                test_subject=test_subject if test_subject is not None else subject_id,
                batch=batch_size, 
                num_workers=args.n_workers,
                seed_val=seed, 
                load_img=args.img,
                separate_test=separate_test_set,
                return_subject_id=args.return_subject_id,
                select_channels=channels,
                subj_training_ratio=args.subj_training_ratio if args.subj_training_ratio > 0 else 0.01,
                img_encoder=img_enc_name,
                interpolate=args.interpolate,
                window=args.window,
                device_type=device)
            top1_acc, top3_acc, top5_acc = downstream.retrieval(brain_encoder, img_encoder, test_data_loader, return_subject_id=args.return_subject_id, device=device)
            topk_scores = {
                'top1': top1_acc,
                'top3': top3_acc,
                'top5': top5_acc
            }
            os.makedirs(os.path.join(paths["save_path"], "performances"), exist_ok=True)
            with open(os.path.join(paths["save_path"], "performances", f"topk_performances_{seed}.pkl"), 'wb') as f:
                pickle.dump(topk_scores, f)
        else:
            print("No Downstream Task Selected. We Are Done!")