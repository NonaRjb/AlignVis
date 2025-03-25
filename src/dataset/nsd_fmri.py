import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/alignvis")

from torch.utils.data import Dataset
from PIL import Image
import scipy.io as spio
import numpy as np
import h5py
import torch
import pickle
import json
import os

from src.dataset.data_utils import _transform


class NSD(Dataset):
    def __init__(
        self,
        data_path,
        subject_id,
        load_img="embedding",
        return_subject_id=False,
        split='train',
        img_encoder=None,
    ):

        self.data_path = os.path.join(data_path, "NSD_fmri")
        self.subject_id = subject_id

        self.load_img = load_img
        self.img_transform = _transform(224)
        self.return_subject_id = return_subject_id
        self.split = split
        self.img_encoder = img_encoder

        if self.split == "train":
            self.fmri = np.load(os.path.join(save_path, "nsd_train_fmri_sub" + "{:02d}".format(self.subject_id) + ".npy"), memmap_mode='r')
            self.images = np.load(os.path.join(save_path ,"nsd_train_stim_sub" + "{:02d}".format(subject_id) + ".npy"), memmap_mode='r')
        else:
            self.fmri = np.load(os.path.join(save_path, "nsd_test_fmri_sub" + "{:02d}".format(self.subject_id) + ".npy"), memmap_mode='r')
            self.images = np.load(os.path.join(save_path ,"nsd_test_stim_sub" + "{:02d}".format(subject_id) + ".npy"), memmap_mode='r')


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        fmri = torch.from_numpy(self.fmri[idx]).to(torch.float)

        if self.load_img == "raw":
            img = self.images[idx]
            img = Image.fromarray(img)
            img = self.img_transform(img).to(torch.float)
            sample = (fmri, img)
        elif self.load_img == "embedding":
            raise NotImplementedError
        else:
            sample = fmri

        label = torch.tensor([0])

        if self.return_subject_id:
            return (sample, self.subject_id), label
        else:
            return sample, label
        




