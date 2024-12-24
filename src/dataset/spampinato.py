import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/alignvis")

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import pickle
import os

from src.dataset.data_utils import _transform, unzip_nested_files

class SpampinatoDataset:
    
    # Constructor
    def __init__(
        self, 
        data_path,
        subject_id,
        time_low=0.02,
        time_high=0.46,
        fs=1000,
        img_size=224,
        load_img=False,
        download=False
        ):

        self.fs = fs
        self.time_low = int((time_low - 0.02) * self.fs)
        self.time_high = int((time_high - 0.02) * self.fs)
        self.subject_id = subject_id

        self.image_parent_dir = "/proj/common-datasets/ImageNet/train/"
        self.img_preprocess = _transform(img_size)
        self.load_img = load_img

        data_path = os.path.join(data_path, "spampinato_et_al")
        os.makedirs(data_path, exist_ok=True)

        if download:

            os.system(f"wget https://files.de-1.osf.io/v1/resources/temjc/providers/osfstorage/66e8279aaee31b2bbb1c3e62/?zip= -O temp_spampinato.zip")
            os.system(f"export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE")
            os.system(f"unzip temp_spampinato.zip -d {data_path}")
            os.system(f"rm temp_spampinato.zip")

            unzip_nested_files(data_path)

        # Load EEG signals
        eeg_signals_path = os.path.join(data_path, "eeg_55_95_std.pth")
        image_data_path="/proj/common-datasets/ImageNet/train/"
        loaded = torch.load(eeg_signals_path)
        if subject_id!=0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==subject_id]
        else:
            self.data=loaded['dataset']      
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        
        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float()
        eeg = eeg[:, self.time_low:self.time_high]
        eeg = eeg.unsqueeze(axis=0)
        # Get image
        if self.load_img:
            image_synset = self.images[self.data[i]["image"]].split("_")[0]
            train_img_file = os.path.join(self.image_parent_dir, image_synset, self.images[self.data[i]["image"]]+".JPEG")
            pair = Image.open(train_img_file).convert('RGB')
            sample = (eeg.to(torch.float), (self.img_preprocess(pair).to(torch.float)))
            label = self.data[i]["label"]
        else:
            sample = eeg.to(torch.float)
            label = self.data[i]["label"]
        # Return
        return sample, label