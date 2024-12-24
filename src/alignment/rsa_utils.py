import rsatoolbox
import numpy as np
import os


def load_things_eeg_2(data_path, subject_ids, split='train', normalize=True):

    data_path = os.path.join(data_path, 'things_eeg_2')

    img_parent_dir  = os.path.join(data_path, 'images')
    img_metadata = np.load(os.path.join(img_parent_dir, 'image_metadata.npy'),
	        allow_pickle=True).item()
    img_concepts = img_metadata['test_img_concepts'] if split == 'test' else img_metadata['train_img_concepts']
    img_files = img_metadata['test_img_files'] if split == 'test' else img_metadata['train_img_files']

    
    channel_names = None
    times = None
    eeg_data_list = []
    subj_list = []
    labels_list = []
    for sid in subject_ids:
        
        eeg_parent_dir = os.path.join(data_path, 'Preprocessed_data_250Hz', 'sub-'+"{:02d}".format(sid))
        eeg_data = np.load(os.path.join(eeg_parent_dir,
                'preprocessed_eeg_training.npy' if split == 'train' else 'preprocessed_eeg_test.npy'), allow_pickle=True)
        subject_eeg_data = eeg_data['preprocessed_eeg_data']
        if channel_names is None:
            channel_names = eeg_data['ch_names']
        if times is None:
            times = eeg_data['times']
        # if select_channels:
        #     subject_eeg_data = subject_eeg_data[:, :, select_channels, :]
        tmp_labels = img_concepts
        labels = [int(l.split("_")[0])-1 for l in tmp_labels]
        
        unique_labels, inverse_indices = np.unique(np.array(labels), return_inverse=True)
        # Compute averaged trials per label
        averaged_data = np.array([subject_eeg_data[inverse_indices == i].mean(axis=0) for i in range(len(unique_labels))])
        
        eeg_data_list.append(averaged_data)
        # subj_list.extend([str(sid)]*len(labels))
        labels_list.extend(list(unique_labels))
        
    all_subject_data = np.concatenate(eeg_data_list, axis=0)
    unique_labels, inverse_indices = np.unique(np.array(labels_list), return_inverse=True)
    averaged_data = np.array([all_subject_data[inverse_indices == i].mean(axis=0) for i in range(len(unique_labels))])
    # Concatenate all subjects' EEG data
    # eeg_data = np.concatenate(eeg_data_list, axis=0)
    # Average across repetitions
    eeg_data = np.mean(averaged_data, axis=1).squeeze()
    print(f"EEG data shape = {eeg_data.shape}")

    if normalize:
        eeg_data = eeg_data / np.linalg.norm(eeg_data, axis=-1, keepdims=True)

    return eeg_data, labels_list, channel_names, times