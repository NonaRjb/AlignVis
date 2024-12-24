import rsatoolbox
import argparse
import os

from rsa_utils import load_things_eeg_2


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Image Embeddings")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--save_path", type=str, default=None, help="Path to the output directory")
    return parser.parse_args()

def load_brain_data(dataset, datapath, subject_ids, **kwargs):
    
    if dataset == 'things-eeg-2':
        return load_things_eeg_2(datapath, subject_ids, kwargs['split'])
    else:
        raise NotImplementedError(f"Dataset {dataset} not recognized")


def create_temporal_dataset(measurements, channel_names, stimulus, t):
    data = rsatoolbox.data.TemporalDataset(
        measurements,
        channel_descriptors={'names': channel_names},
        obs_descriptors={'stimulus': stimulus},
        time_descriptors={'time': t}
    )


if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    save_path = args.save_path

    # Load brain data
    brain_data, labels, ch_names, times = load_brain_data('things-eeg-2', data_path, subject_ids=[1, 2, 3], split='train')
    print(f"Brain data shape = {brain_data.shape}")

    rsa_dataset = create_temporal_dataset(brain_data, ch_names, labels, times)
    rdm_brain = rsatoolbox.rdm.calc_rdm(rsa_dataset, method='correlation')

    # # Load image data
    # image_data, image_labels = load_image_data('flickr30k', data_path, split='train')
    # print(f"Image data shape = {image_data.shape}")

    # # Compute RSA
    # rsa = rsatoolbox.rsa.RSA()
    # rsa.fit(brain_data, image_data)
    # rsa.save(os.path.join(save_path, 'rsa_results.pkl'))