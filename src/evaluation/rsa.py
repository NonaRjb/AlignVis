import matplotlib.pyplot as plt
from rsatoolbox.rdm import get_categorical_rdm
from rsatoolbox.rdm import compare
import rsatoolbox
import numpy as np
import argparse
import json
import os

from eval_utils import load_things_eeg_2


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Image Embeddings")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--save_path", type=str, default=None, help="Path to the output directory")
    parser.add_argument("--new_label_path", type=str, default=None, help="Path to the directory containing the new label files")
    parser.add_argument("--new_label_type", type=str, default="color", help="Type of new labels to use")
    parser.add_argument("--split", type=str, default='train', help="Dataset split to use (train or test)")
    parser.add_argument("--subject_id", type=int, default=1, help="Subject ID to use")
    return parser.parse_args()

def load_brain_data(dataset, datapath, subject_id, **kwargs):
    
    if dataset == 'things-eeg-2':
        return load_things_eeg_2(datapath, subject_id, split=kwargs['split'], labels_new=kwargs['new_labels'])
    else:
        raise NotImplementedError(f"Dataset {dataset} not recognized")


def create_temporal_dataset(measurements, channel_names, stimulus, t, new_labels=None):
    if new_labels is not None:
        print("Creating temporal dataset with new labels")
        data = rsatoolbox.data.TemporalDataset(
            measurements,
            channel_descriptors={'names': channel_names},
            obs_descriptors={'stimulus': stimulus, 'new_labels': new_labels},
            time_descriptors={'time': t}
        )
        data.sort_by('new_labels')
    else:
        data = rsatoolbox.data.TemporalDataset(
            measurements,
            channel_descriptors={'names': channel_names},
            obs_descriptors={'stimulus': stimulus},
            time_descriptors={'time': t}
        )
        data.sort_by('stimulus')
    return data


def create_dataset(measurements, channel_names, stimulus, new_labels=None):
    if new_labels is not None:
        data = rsatoolbox.data.Dataset(
            measurements.mean(axis=-1),
            channel_descriptors={'names': channel_names},
            obs_descriptors={'stimulus': stimulus, 'new_labels': new_labels}
        )
        data.sort_by('new_labels')
    else:
        data = rsatoolbox.data.Dataset(
            measurements.mean(axis=-1),
            channel_descriptors={'names': channel_names},
            obs_descriptors={'stimulus': stimulus}
        )
        data.sort_by('stimulus')
    return data


if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    save_path = args.save_path

    if args.new_label_path is not None:
        with open(os.path.join(args.new_label_path, f'gpt_4o_mini_{args.new_label_type}_{args.split}.json'), "r") as f:
            new_labels = json.load(f)
    else:
        new_labels = None

    # Load brain data
    brain_data, _, labels, ch_names, times, other_labels = load_brain_data('things-eeg-2', data_path, subject_id=args.subject_id, split=args.split, new_labels=new_labels)
    print(f"Brain data shape = {brain_data.shape}")
    print(f"times shape = {times.shape}")

    rsa_dataset = create_temporal_dataset(brain_data, ch_names, labels, times, new_labels=other_labels)

    bins = np.reshape(times, [-1, 25])
    rsa_dataset_binned = rsa_dataset.bin_time('time', bins)
    rdm_brain = rsatoolbox.rdm.calc_rdm_movie(rsa_dataset_binned, method='correlation', descriptor='stimulus')

    # add formated time as rdm_descriptor
    rdm_brain.rdm_descriptors['time_formatted'] = ['%0.0f ms' % (np.round(x*1000,2)) for x in rdm_brain.rdm_descriptors['time']]

    fig, _, _ = rsatoolbox.vis.show_rdm(rdm_brain,
                                        # pattern_descriptor='new_labels',
                                        rdm_descriptor='time_formatted',
                                        show_colorbar='panel',
                                        cmap='Blues',
                                        vmin=0.,
                                        n_column=5,
                                        n_row=2,
                                        )
    
    # Save the figure
    save_filename = os.path.join(save_path, "rdm_brain.png")
    fig.savefig(save_filename, dpi=300, bbox_inches='tight')

    # Close the figure to free memory
    plt.close(fig)

    rsa_dataset_averaged = create_dataset(brain_data, ch_names, labels, new_labels=other_labels)
    rdm_brain_averaged = rsatoolbox.rdm.calc_rdm(rsa_dataset_averaged, method='correlation', descriptor='stimulus')
    fig2, _, _ = rsatoolbox.vis.show_rdm(rdm_brain_averaged,
                                        #   pattern_descriptor='new_labels',
                                        show_colorbar='panel',
                                        cmap='Blues',
                                        vmin=0.,
                                        )
    save_filename = os.path.join(save_path, "rdm_brain_averaged_time.png")
    fig2.savefig(save_filename, dpi=300, bbox_inches='tight')

    # compare with model rdms
    def get_one_vs_all_rdm(labels, target_label):
        """
        Create a one-vs-all categorical RDM.
        Labels matching `target_label` are assigned condition 1, all others 0.
        """
        binary_labels = (labels == target_label).astype(int)
        return get_categorical_rdm(binary_labels)


    unique_labels, numeric_labels = np.unique(rsa_dataset_averaged.obs_descriptors['new_labels'], return_inverse=True)
    print(f"Unique labels = {unique_labels}")
    print(f"Numeric labels = {numeric_labels}")
    target = 2
    # rdms_model_color = get_categorical_rdm(numeric_labels)
    rdms_model_color = get_one_vs_all_rdm(numeric_labels, target)
    model_names = ['color']

    model_rdms = rdms_model_color
    model_rdms.rdm_descriptors['model_names'] = model_names
    model_rdms.pattern_descriptors['cond_names'] = labels

    fig3, _, _ = rsatoolbox.vis.show_rdm(model_rdms, rdm_descriptor='model_names', pattern_descriptor = 'cond_names')
    fig3.savefig(os.path.join(save_path, "rdm_model_color.png"), dpi=300, bbox_inches='tight')

    r = []
    for mod in model_rdms:
        r.append(compare(mod, rdm_brain, method='corr'))

    fig4 = plt.figure()
    for i, r_ in enumerate(r):
        plt.plot(rdm_brain.rdm_descriptors['time'], r_.squeeze(), label=model_names[i])

    plt.xlabel('time')
    plt.ylabel('model-data cosine similarity')
    plt.legend()
    plt.savefig(os.path.join(save_path, "model_data_similarity.png"), dpi=300, bbox_inches='tight')
