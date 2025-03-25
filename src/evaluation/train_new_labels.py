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
from src.models.brain_encoder import BrainClassifier
from src.models.image_encoder import ImageEncoder
from src.training.trainer import UnimodalTrainer
from src.training.training_utils import CLIPLoss, SoftCLIPLoss
from sklearn.preprocessing import LabelEncoder
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
    }


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
    if 'new_labels_type' in kwargs.keys() and kwargs['new_labels_type'] is not None:
        ds_configs['n_classes'] = len(np.unique(np.array(data.labels_list)))
        print("Number of classes: ", ds_configs['n_classes'])
    
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
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=None)
    parser.add_argument('--net_seq_length', type=int, nargs='+', default=None)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=1000, help="Number of epochs for pretraining")
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--scheduler', type=str, default="plateau")
    parser.add_argument('--patience', type=int, default=25, help="Patience for the reduce_lr_on_plateau scheduler")
    parser.add_argument('--temperature', type=float, default=0.04)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--downstream', type=str, default=None)
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
        downstream_task = args.downstream
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

        directory_name = args.new_label_type + "_" + brain_enc_name
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
        with open(os.path.join(paths['brain_data'], "things_eeg_2", "labels_clip", f'{args.new_label_type}.txt'), "r") as f:
            labels = [line.strip() for line in f.readlines()]
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)  # Fit with all possible labels
        
        brain_classifier = BrainClassifier(
            model_name=brain_enc_name, 
            n_channels=data_configs['n_channels'],
            n_samples=data_configs['n_samples'],
            n_classes=data_configs['n_classes'], 
            embed_dim = args.embed_dim,
            device=device,
            **model_configs[brain_enc_name]
            )
        brain_classifier = brain_classifier.float()
        brain_classifier.to(device) 

        loss = torch.nn.CrossEntropyLoss()
        optim = torch.optim.AdamW(brain_classifier.parameters(), lr=min_lr if warmup_epochs>0 else lr, weight_decay=weight_decay)
        trainer = UnimodalTrainer(
            brain_encoder=brain_classifier, 
            optimizer=optim, 
            loss=loss,
            save_path=paths["save_path"], 
            filename=f'{brain_enc_name}_{dataset_name}_seed{seed}',
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
        brain_classifier.load_state_dict(best_classifier['model_state_dict'])
        brain_classifier.eval()
        test_loss, test_acc, test_auroc = trainer.evaluate(brain_classifier, test_data_loader)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}, Test AUROC: {test_auroc}")

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



        
    