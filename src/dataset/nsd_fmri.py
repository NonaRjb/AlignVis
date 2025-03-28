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

        self.fmri = np.load(os.path.join(self.data_path, "processed_data", "sub" + "{:02d}".format(self.subject_id), f"nsd_{self.split}_fmri_sub" + "{:02d}".format(self.subject_id) + ".npy"), mmap_mode='r')
        if self.load_img == "embedding":
            self.images = np.load(os.path.join(self.data_path , "image_embeddings", "sub" + "{:02d}".format(self.subject_id), f"{self.split}_{self.img_encoder}.npy"), mmap_mode='r')
        else:
            self.images = np.load(os.path.join(self.data_path , "images", "sub" + "{:02d}".format(self.subject_id), f"{self.split}.npy"), mmap_mode='r')


    def __len__(self):
        return len(self.fmri)

    def __getitem__(self, idx):
        fmri = torch.from_numpy(self.fmri[idx].copy()).to(torch.float)

        if self.load_img == "raw":
            img = self.images[idx].copy().astype(np.uint8)
            img = Image.fromarray(img)
            img = self.img_transform(img).to(torch.float)
            sample = (fmri, img)
        elif self.load_img == "embedding":
            img = torch.from_numpy(self.images[idx].copy()).to(torch.float)
            sample = (fmri, img)
        else:
            sample = fmri

        label = torch.tensor([0])

        if self.return_subject_id:
            return (sample, self.subject_id), label
        else:
            return sample, label
        




