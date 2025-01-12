import rsatoolbox
import numpy as np
import os


def load_things_eeg_2(data_path, subject_id, img_encoder, split='train', normalize=True):

    data_path = os.path.join(data_path, 'things_eeg_2')

    img_parent_dir  = os.path.join(data_path, 'images')
    img_metadata = np.load(os.path.join(img_parent_dir, 'image_metadata.npy'),
	        allow_pickle=True).item()
    img_concepts = img_metadata['test_img_concepts'] if split == 'test' else img_metadata['train_img_concepts']
    img_files = img_metadata['test_img_files'] if split == 'test' else img_metadata['train_img_files']
        
    eeg_parent_dir = os.path.join(data_path, 'Preprocessed_data_250Hz', 'sub-'+"{:02d}".format(subject_id))
    eeg_data = np.load(os.path.join(eeg_parent_dir,
            'preprocessed_eeg_training.npy' if split == 'train' else 'preprocessed_eeg_test.npy'), allow_pickle=True)
    subject_eeg_data = eeg_data['preprocessed_eeg_data']
    channel_names = eeg_data['ch_names']
    times = eeg_data['times']
    tmp_labels = img_concepts
    labels = [int(l.split("_")[0])-1 for l in tmp_labels]

    # load images
    images = []
    for i in len(subject_eeg_data):
        img_file = os.path.join(img_parent_dir, 'training_images' if self.split == 'train' else 'test_images', 
                img_concepts[i], img_files[i].replace(".jpg", f"_{img_encoder}.npy"))
        img = np.load(img_file)
        images.append(img)
    images = np.array(images)
    
    # if average_trials:
    #     labels, inverse_indices = np.unique(np.array(labels), return_inverse=True)
    #     subject_eeg_data = np.array([subject_eeg_data[inverse_indices == i].mean(axis=0) for i in range(len(labels))])  # Compute averaged trials per label
        
    # Average across repetitions
    subject_eeg_data = np.mean(subject_eeg_data, axis=1).squeeze()
    print(f"EEG data shape = {subject_eeg_data.shape}")

    if normalize:
        subject_eeg_data = (subject_eeg_data - np.mean(subject_eeg_data, axis=-1, keepdims=True)) / np.linalg.norm(subject_eeg_data, axis=-1, keepdims=True)
    # reshape to vector
    subject_eeg_data = 

    return subject_eeg_data, images, labels, channel_names, times