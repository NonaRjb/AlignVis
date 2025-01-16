import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/alignvis")

from torch.utils.data import Dataset
from PIL import Image
from scipy.interpolate import interp1d
import numpy as np
import torch
import pickle
import os

from src.dataset.data_utils import _transform


class ThingsEEG2(Dataset):
    def __init__(
        self,
        data_path,
        subject_id,
        load_img="embedding",
        return_subject_id=False,
        split='train',
        select_channels=None,
        training_ratio=1.0,
        img_encoder=None,
        interpolate=None,
        window=None,
        download=False,
        ):

        self.data_path = os.path.join(data_path, "things_eeg_2")
        os.makedirs(self.data_path, exist_ok=True)
        self.load_img = load_img
        self.img_transform = _transform(224)
        self.return_subject_id = return_subject_id
        self.split = split
        self.select_channels = select_channels
        self.training_ratio = training_ratio
        self.img_encoder = img_encoder
        self.interpolate = interpolate
        self.window = window
        self.channel_names = None
        self.fs = self.interpolate if self.interpolate is not None else 250

        if isinstance(subject_id, int):
            subject_id = [subject_id]
        
        # img_data['train_img_concepts'] defines the classes 
        self.img_parent_dir  = os.path.join(self.data_path, 'images' if load_img == "raw" else 'image_embeddings')
        self.img_metadata = np.load(os.path.join(self.img_parent_dir, 'image_metadata.npy'),
	            allow_pickle=True).item()
        self.img_concepts = self.img_metadata['test_img_concepts'] if self.split == 'test' else self.img_metadata['train_img_concepts']
        self.img_files = self.img_metadata['test_img_files'] if self.split == 'test' else self.img_metadata['train_img_files']

        self.eeg_data_list = []
        self.labels_list = []
        self.subj_list = []
        
        for sid in subject_id:
        
            eeg_parent_dir = os.path.join(self.data_path, 'Preprocessed_data_250Hz', 'sub-'+"{:02d}".format(sid))
            eeg_data = np.load(os.path.join(eeg_parent_dir,
                    'preprocessed_eeg_training.npy' if self.split == "train" else 'preprocessed_eeg_test.npy'), allow_pickle=True, mmap_mode='r')
            subject_eeg_data = eeg_data['preprocessed_eeg_data']
            print(eeg_data['ch_names'])
            if self.channel_names is None:
                self.channel_names = eeg_data['ch_names']
            if select_channels:
                subject_eeg_data = subject_eeg_data[:, :, select_channels, :]

            tmp_labels = self.img_concepts
            labels = [int(l.split("_")[0])-1 for l in tmp_labels]

            self.eeg_data_list.append(subject_eeg_data)
            self.subj_list.extend([str(sid)]*len(labels))
            self.labels_list.extend(labels)
        
        # Concatenate all subjects' EEG data
        self.eeg_data = np.concatenate(self.eeg_data_list, axis=0)
        self.selected_indices = None

        if split == 'train' and training_ratio < 1.0:
            self._sample_data()
        print("len(self.eeg_data) = ", len(self.eeg_data))

    def _sample_data(self):
        """Randomly samples the training data based on the given training_ratio."""
        num_samples = len(self.eeg_data)
        num_samples_to_use = int(self.training_ratio * num_samples)
        
        # Randomly select indices to keep
        self.selected_indices = np.random.choice(np.arange(num_samples), num_samples_to_use, replace=False)
        self.selected_indices = np.sort(self.selected_indices)
        
        # Subset the data and labels based on the selected indices
        self.eeg_data = self.eeg_data[self.selected_indices]
        self.subj_list = [self.subj_list[i] for i in self.selected_indices]
        self.labels_list = [self.labels_list[i] for i in self.selected_indices]
    
    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, item):
        
        eeg = self.eeg_data[item]
        eeg = eeg.copy()
        if eeg.shape[0] > 1:
            eeg = np.expand_dims(eeg, axis=0)

        if self.interpolate is not None:
            #  set time length to 256
            x1 = np.linspace(0, 1, eeg.shape[-1])
            x2 = np.linspace(0, 1, self.interpolate)
            f = interp1d(x1, eeg, axis=-1)
            eeg = f(x2)
        
        if self.window is not None:
            n0 = int(self.window[0]*self.fs)
            n1 = int(self.window[1]*self.fs)
            eeg = eeg[..., n0:n1]
            x1 = np.linspace(0, 1, eeg.shape[-1])
            x2 = np.linspace(0, 1, 250)
            f = interp1d(x1, eeg, axis=-1)
            eeg = f(x2)

        eeg = (eeg - np.mean(eeg, axis=-1, keepdims=True)) / np.linalg.norm(eeg, axis=-1, keepdims=True)

        img_item = self.selected_indices[item] if self.selected_indices is not None else item
        img_idx = img_item % len(self.img_concepts)
            
        if self.load_img == "embedding":
            img_file = os.path.join(self.img_parent_dir, 'training_images' if self.split == 'train' else 'test_images', 
                self.img_concepts[img_idx], self.img_files[img_idx].replace(".jpg", f"_{self.img_encoder}.npy")) # dreamsim_clip_vitb32
            pair = np.load(img_file)
            sample = (torch.from_numpy(np.mean(eeg, axis=1)).to(torch.float), torch.from_numpy(pair.squeeze()).to(torch.float))
        elif self.load_img == "raw":
            img_file = os.path.join(self.img_parent_dir, 'training_images' if self.split == 'train' else 'test_images', 
                self.img_concepts[img_idx], self.img_files[img_idx]) 
            pair = Image.open(img_file).convert('RGB')
            sample = (torch.from_numpy(np.mean(eeg, axis=1)).to(torch.float), (self.img_transform(pair).to(torch.float)))
        else:
            sample = torch.from_numpy(np.mean(eeg, axis=1)).to(torch.float)
            
        label = self.labels_list[item]
        assert int(self.img_concepts[img_idx].split("_")[0])-1 == label
        # img_file = self.image_files[self.indices[item]].copy()
        if self.return_subject_id:
            return (sample, self.subj_list[item]), label
        else:
            return sample, label
