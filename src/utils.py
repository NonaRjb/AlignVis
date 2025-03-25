import torch
import numpy as np
from tqdm import tqdm
import json
import os

from src.dataset import ThingsEEG2Processed, ThingsEEG2, ThingsMEG, SpampinatoDataset


def save_config(loaded_config, root_path, filename='config_run.json'):
    with open(os.path.join(root_path, filename), 'w') as file:
        json.dump(vars(loaded_config), file)

def load_dataset(dataset_name, data_path, **kwargs):
    
    if dataset_name == "spampinato": #TODO Still not ready to use
        data_configs = {
            "t_l": 0.02,
            "t_h": 0.46,
            "fs": 1000,
            "n_samples": 440,
            "n_channels": 128,
            "n_classes": 40,
        }
        dataset = SpampinatoDataset(
            data_path=data_path,
            subject_id=kwargs['sid'],
            load_img=kwargs['load_img']
        )
    elif dataset_name == "things-eeg-preprocessed": #TODO Still not ready to use
        fs = kwargs['interpolate'] if 'interpolate' in kwargs.keys() and kwargs['interpolate'] is not None else 100
        new_labels_type = kwargs['new_labels_type'] if 'new_labels_type' in kwargs.keys() else None
        new_labels_path = kwargs['new_labels_path'] if 'new_labels_path' in kwargs.keys() else None
        n_samples = fs if 'window' not in kwargs.keys() or kwargs['window'] is None else int((kwargs['window'][1]-kwargs['window'][0])*fs)
        data_configs = {
            "t_l": -0.2,
            "t_h": 0.8,
            "fs": kwargs['interpolate'] if 'interpolate' in kwargs.keys() else 100,  # I have changed this from 100 to 128 in the Dataset description
            "n_samples": n_samples,
            "n_channels": 17 if 'select_channels' not in kwargs.keys()  or kwargs['select_channels'] is None else len(kwargs['select_channels']),
            "n_classes": 1654,
        }
        test = kwargs['test'] if 'test' in kwargs.keys() else False
        print("TEST = ", test)
        dataset = ThingsEEG2Processed(
            data_path=data_path,
            subject_id=kwargs['sid'],
            load_img=kwargs['load_img'],
            return_subject_id=kwargs['return_subject_id'],
            split=kwargs['split'],
            select_channels=kwargs['select_channels'] if 'select_channels' in kwargs.keys() else None,
            training_ratio=kwargs['subj_training_ratio'],
            img_encoder=kwargs['img_encoder'],
            interpolate=kwargs['interpolate'] if 'interpolate' in kwargs.keys() else None,
            window=kwargs['window'] if 'window' in kwargs.keys() else None,
            new_labels_type=new_labels_type,
            new_labels_path=new_labels_path,
        )
    elif dataset_name == "things-eeg-2":
        fs = kwargs['interpolate'] if 'interpolate' in kwargs.keys() and kwargs['interpolate'] is not None else 251
        new_labels_type = kwargs['new_labels_type'] if 'new_labels_type' in kwargs.keys() else None
        new_labels_path = kwargs['new_labels_path'] if 'new_labels_path' in kwargs.keys() else None
        n_samples = fs if 'window' not in kwargs.keys() or kwargs['window'] is None else int((kwargs['window'][1]-kwargs['window'][0])*fs)
        data_configs = {
            "t_l": 0.0,
            "t_h": 1.0,
            "fs": kwargs['interpolate'] if 'interpolate' in kwargs.keys() else 250, 
            "n_samples": n_samples,
            "n_channels": 63 if 'select_channels' not in kwargs.keys() or kwargs['select_channels'] is None else len(kwargs['select_channels']),
            "n_classes": 1654,
        }
        test = kwargs['test'] if 'test' in kwargs.keys() else False
        print("TEST = ", test)
        dataset = ThingsEEG2(
            data_path=data_path,
            subject_id=kwargs['sid'],
            load_img=kwargs['load_img'],
            return_subject_id=kwargs['return_subject_id'],
            split=kwargs['split'],
            select_channels=kwargs['select_channels'] if 'select_channels' in kwargs.keys() else None,
            training_ratio=kwargs['subj_training_ratio'],
            img_encoder=kwargs['img_encoder'],
            interpolate=kwargs['interpolate'] if 'interpolate' in kwargs.keys() else None,
            window=kwargs['window'] if 'window' in kwargs.keys() else None,
            new_labels_type=new_labels_type,
            new_labels_path=new_labels_path,
        )
        data_configs['ch_names'] = dataset.channel_names
    elif dataset_name == "things-meg":
        data_configs = {
            "t_l": 0.0,
            "t_h": 1.0,
            "fs": kwargs['interpolate'] if 'interpolate' in kwargs.keys() else 200,
            "n_samples": kwargs['interpolate'] if 'interpolate' in kwargs.keys() and kwargs['interpolate'] is not None else 201,
            "n_channels": 271,
            "n_classes": 1654,
        }
        dataset = ThingsMEG(
            data_path=data_path,
            subject_id=kwargs['sid'],
            load_img=kwargs['load_img'], # embedding or raw
            return_subject_id=kwargs['return_subject_id'],
            split=kwargs['split'],
            select_channels=kwargs['select_channels'] if 'select_channels' in kwargs.keys() else None,
            training_ratio=kwargs['subj_training_ratio'],
            img_encoder=kwargs['img_encoder'],
            interpolate=kwargs['interpolate'] if 'interpolate' in kwargs.keys() else None,
        )
    else: 
        raise NotImplementedError
    return dataset, data_configs

def get_embeddings(model, data_loader, modality="eeg", return_subject_id=False, save=False, save_path=None, device='cuda'):
    
    # progress_bar = tqdm(data_loader)
    embeddings = None
    labels = None
    if model is not None:
        model.eval()
    with torch.no_grad():
        for i, (data, y) in enumerate(data_loader):
            if return_subject_id:
                subject_id = data[1]
                data = data[0]
            if modality == "eeg":
                try:
                    x, _ = data
                except:
                    x = data
                # x, _ = data
            else:
                _, x = data
            x = x.to(device)
            y = y.to(device)
            if model is not None:
                e = model(x)
            else:
                e = x
            e = e - torch.mean(e, dim=-1, keepdim=True)
            e = torch.nn.functional.normalize(e, p=2, dim=-1)
            if embeddings is None:
                embeddings = e.detach().cpu().numpy()
                labels = y.detach().cpu().numpy()
            else:
                embeddings = np.concatenate((embeddings, e.detach().cpu().numpy()), axis=0)
                labels = np.concatenate((labels, y.detach().cpu().numpy()), axis=0)
            # embeddings.append(e.detach().cpu().numpy())
            # labels.append(y.detach().cpu().numpy())
    if save:
        print("Saving the Embeddings")
        if save_path:
            np.save(save_path, embeddings)
        else:
            np.save("./embeddings.npy", embeddings)
    print(embeddings.shape)
    print(labels)
    return embeddings, labels

    