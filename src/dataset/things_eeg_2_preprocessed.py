import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/alignvis")

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import pickle
import os

from src.dataset.data_utils import _transform, unzip_nested_files


class ThingsEEG2_preprocessed(Dataset):
    def __init__(
        self,
        data_path,
        subject_id,
        load_img=False,
        load_img_embedding=False,
        test=False,
        select_channels=None,
        training_ratio=1.0,
        download=False,
        return_subject_id=False,
        img_encoder=None
        ):

        self.load_img = load_img
        self.return_subject_id = return_subject_id
        self.load_img_embedding = load_img_embedding
        self.img_encoder = img_encoder
        self.img_transform = _transform(224)
        self.test = test
        self.training_ratio = training_ratio
        data_path = os.path.join(data_path, "things_eeg_2")
        os.makedirs(data_path, exist_ok=True)

        if download:
            # download raw EEG
            # os.makedirs(os.path.join(data_path, "raw_eeg"), exist_ok=True)
            # os.system(f"wget https://files.osf.io/v1/resources/crxs4/providers/googledrive/?zip= -O temp_things2_raw.zip")
            # os.system(f"export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE")
            # os.system(f"unzip temp_things2_raw.zip -d {os.path.join(data_path, 'raw_eeg')}")
            # os.system(f"rm temp_things2_raw.zip")
            # download preprocessed EEG
            os.makedirs(os.path.join(data_path, "preprocessed_eeg"), exist_ok=True)
            os.system(f"wget https://files.de-1.osf.io/v1/resources/anp5v/providers/osfstorage/?zip= -O temp_things2_preprocessed.zip")
            os.system(f"export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE")
            os.system(f"unzip temp_things2_preprocessed.zip -d {os.path.join(data_path, 'preprocessed_eeg')}")
            os.system(f"rm temp_things2_preprocessed.zip")
            # download images
            os.makedirs(os.path.join(data_path, "images"), exist_ok=True)
            os.system(f"wget https://files.de-1.osf.io/v1/resources/y63gw/providers/osfstorage/?zip= -O temp_images.zip")
            os.system(f"export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE")
            os.system(f"unzip temp_images.zip -d {os.path.join(data_path, 'images')}")
            os.system(f"rm temp_images.zip")

            unzip_nested_files(data_path)

        if isinstance(subject_id, int):
            subject_id = [subject_id]
        
        # img_data['train_img_concepts'] defines the classes 
        self.img_parent_dir  = os.path.join(data_path, 'images')
        self.img_metadata = np.load(os.path.join(self.img_parent_dir, 'image_metadata.npy'),
	            allow_pickle=True).item()
        self.img_concepts = self.img_metadata['test_img_concepts'] if self.test else self.img_metadata['train_img_concepts']
        self.img_files = self.img_metadata['test_img_files'] if self.test else self.img_metadata['train_img_files']

        self.eeg_data_list = []
        self.labels_list = []
        self.subj_list = []
        
        for sid in subject_id:
        
            eeg_parent_dir = os.path.join(data_path, 'preprocessed_eeg', 'sub-'+"{:02d}".format(sid))
            eeg_data = np.load(os.path.join(eeg_parent_dir,
                    'preprocessed_eeg_training.npy' if not self.test else 'preprocessed_eeg_test.npy'), allow_pickle=True).item()
            subject_eeg_data = eeg_data['preprocessed_eeg_data']
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
        if not self.test:
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
        if torch.is_tensor(item):
            item = item.tolist()
        eeg = self.eeg_data[item]
        eeg = eeg.copy()
        if eeg.shape[0] > 1:
            eeg = np.expand_dims(eeg, axis=0)

        # set time length to 128 # TODO: change this to scipy resample
        x1 = np.linspace(0, 1, eeg.shape[-1])
        x2 = np.linspace(0, 1, 128)
        f = interp1d(x1, eeg, axis=-1)
        eeg = f(x2)
        # eeg /= np.linalg.norm(eeg, axis=-1, keepdims=True)
        # TODO EEGChannelNet only works with even number of channels but that's not the only issue
        if self.load_img:
            
            img_item = self.selected_indices[item] if self.selected_indices is not None else item
            img_idx = img_item % len(self.img_concepts)

            if self.load_img_embedding:
                img_file = os.path.join(self.img_parent_dir, 'training_images' if not self.test else 'test_images', 
                    self.img_concepts[img_idx], self.img_files[img_idx].replace(".jpg", f"_{self.img_encoder}.npy")) # dreamsim_clip_vitb32
                pair = np.load(img_file)
                sample = (torch.from_numpy(np.mean(eeg[:, :, :, :], axis=1)).to(torch.float), torch.from_numpy(pair.squeeze()).to(torch.float))
            else:
                img_file = os.path.join(self.img_parent_dir, 'training_images' if not self.test else 'test_images', 
                        self.img_concepts[img_idx], self.img_files[img_idx]) 
                pair = Image.open(img_file).convert('RGB')
                sample = (torch.from_numpy(np.mean(eeg[:, :, :, :], axis=1)).to(torch.float), (self.img_transform(pair).to(torch.float)))
            label = self.labels_list[item]
            assert int(self.img_concepts[img_idx].split("_")[0])-1 == label
        else:
            sample = torch.from_numpy(np.mean(eeg[:, :, :, :], axis=1)).to(torch.float)
            label = self.labels_list[item]
        # img_file = self.image_files[self.indices[item]].copy()
        if self.return_subject_id:
            return (sample, self.subj_list[item]), label
        else:
            return sample, label
