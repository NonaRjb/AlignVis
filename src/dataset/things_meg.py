import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/alignvis")

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import pickle
import os

from src.dataset.data_utils import _transform


class ThingsMEG(Dataset):
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
        download=False
        ):
        self.data_path = os.path.join(data_path, "things_meg")
        os.makedirs(self.data_path, exist_ok=True)
        self.load_img = load_img
        self.img_transform = _transform(224)
        self.return_subject_id = return_subject_id
        self.split = split
        self.select_channels = select_channels
        self.training_ratio = training_ratio
        self.img_encoder = img_encoder
        self.interpolate = interpolate

        if download:
            os.makedirs(os.path.join(self.data_path, "preprocessed_meg"), exist_ok=True)
            pass

        if isinstance(subject_id, int):
            subject_id = [subject_id]

        self.img_parent_dir = os.path.join(self.data_path, "images") if load_img == "raw" else os.path.join(self.data_path, "image_embeddings")
        data_list = []
        image_list = []
        label_list = []
        subject_list = []
        for sid in subject_id:
            with open(os.path.join(self.data_path, "preprocessed_meg", f"{split}_sub{sid}.pkl"), "rb") as f:
                data = pickle.load(f)
            meg_data_subj = data["meg_data"]
            labels_subj = data["meg_labels"]
            image_list_subj = data["meg_images"]

            if self.select_channels is not None:
                meg_data_subj = meg_data_subj[:, self.select_channels, :]
            # labels miss some values
            unique_labels = np.sort(np.unique(np.array(labels_subj)))
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            mapped_labels_subj = [label_mapping[label] for label in labels_subj]
            
            # TODO: average test data repetitions
            if split == "test":
                # Find unique images
                unique_images = np.unique(image_list_subj)

                # Initialize lists to store results
                averaged_data = []
                averaged_images = []
                averaged_labels = []

                for img in unique_images:
                    # Find indices corresponding to the current image
                    indices = np.where(image_list_subj == img)[0]
                    
                    # Compute the average of the MEG data for this image
                    avg_data = meg_data_subj[indices].mean(axis=0)  # Average across trials
                    
                    # Append to results
                    averaged_data.append(avg_data)
                    averaged_images.append(img)
                    averaged_labels.append(mapped_labels_subj[indices[0]])  # Assuming label is the same for all trials of the same image

                # Convert results to numpy arrays
                meg_data_subj = np.array(averaged_data)  # Shape: (#unique_images, #channels, #time)
                image_list_subj = averaged_images  # Shape: (#unique_images,)
                mapped_labels_subj = averaged_labels  # Shape: (#unique_images,)

                print(f"Number of unique images in test data: {len(image_list_subj)}")
                print("test meg_data_subj.shape = ", meg_data_subj.shape)
                print("test mapped_labels_subj = ", mapped_labels_subj)

            subjects_subj = [sid] * len(mapped_labels_subj)
            data_list.append(meg_data_subj)
            image_list.extend(image_list_subj)
            label_list.extend(mapped_labels_subj)
            subject_list.extend(subjects_subj)
        
        self.data = np.concatenate(data_list, axis=0)
        self.labels = np.array(label_list)
        self.image_list = image_list
        self.subjects = np.array(subject_list)
        print(self.data.shape)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        meg = self.data[idx]
        meg = meg.copy()
        label = self.labels[idx]
        
        # normalize
        meg = (meg - np.mean(meg, axis=-1, keepdims=True)) / (np.linalg.norm(meg, axis=-1, keepdims=True) + 1e-08)

        meg = np.expand_dims(meg, axis=0)

        if self.interpolate is not None:
            x1 = np.linspace(0, 1, meg.shape[-1])
            x2 = np.linspace(0, 1, 256)
            f = interp1d(x1, meg, axis=-1)
            meg = f(x2)

        if self.load_img == "embedding":
            img_file = self.image_list[idx].split("/")[-1] 
            img_path = os.path.join(self.img_parent_dir, self.image_list[idx].replace(img_file, f"{img_file.split('.')[0]}_{self.img_encoder}.npy"))
            img = np.load(img_path)
            sample = (torch.from_numpy(meg).to(torch.float), torch.from_numpy(img.squeeze()).to(torch.float))
        elif self.load_img == "raw":
            img_path = os.path.join(self.img_parent_dir, self.image_list[idx])
            img = Image.open(img_path).convert("RGB")
            sample = (torch.from_numpy(meg).to(torch.float), self.img_transform(img).to(torch.float))
        else:
            sample = torch.from_numpy(meg).to(torch.float)
        if self.return_subject_id:
            return (sample, self.subjects[idx]), label
        return sample, label


if __name__ == "__main__":
    data_path = "/proj/rep-learning-robotics/users/x_nonra/alignvis/data/"
    img_enc_name = "dreamsim_clip_vitb32"
    dataset = ThingsMEG(data_path, subject_id=[1], load_img="embedding", return_subject_id=False, img_encoder=img_enc_name)
    print(len(dataset))
    data_sample = dataset[0]
    eeg, img = data_sample[0]
    l = data_sample[1]

    print(eeg.shape, img.shape, l)
