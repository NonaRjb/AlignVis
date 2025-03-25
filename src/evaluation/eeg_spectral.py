import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/alignvis")

import matplotlib.pyplot as plt
import numpy as np
import argparse
import mne
import os

from src.evaluation.eval_utils import load_things_eeg_2


def parse_args():
    parser = argparse.ArgumentParser(description="Spectral Analysis of EEG Data")
    parser.add_argument("--data_path", type=str, default="/proj/rep-learning-robotics/users/x_nonra/alignvis/data", help="Path to the data directory")
    parser.add_argument("--save_path", type=str, default="/proj/rep-learning-robotics/users/x_nonra/data/visualization/spectral", help="Path to the output directory")
    parser.add_argument("--subject_id", type=int, default=1, help="Subject ID")
    parser.add_argument("--split", type=str, default="train", help="Data split")

    return parser.parse_args()


def create_epochsarray(brain_array, channel_names, times, labels):

    n_trials, n_channels, n_times = brain_array.shape
    
    event_ids = np.array(labels)
    events = np.column_stack((np.arange(n_trials), np.zeros(n_trials, dtype=int), event_ids))

    info = mne.create_info(ch_names=channel_names, sfreq=250, ch_types='eeg')
    epochs = mne.EpochsArray(brain_array, info, tmin=0, events=events)

    return epochs


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "power"), exist_ok=True)
    
    brain_data, _, labels, ch_names, times = load_things_eeg_2(args.data_path, subject_id=args.subject_id, img_encoder='dreamsim_clip_vitb32', split=args.split)
    epochs = create_epochsarray(brain_data, ch_names, times, labels)

    print(epochs)
    
    # frequencies = np.arange(1, 70, 1)
    # power = epochs.compute_tfr("morlet", n_cycles=frequencies/2.0, return_itc=False, freqs=frequencies, decim=3, average=True, n_jobs=-1)

    spectrum = epochs.compute_psd(method='welch', fmin=2.0, fmax=40.0, tmax=1.0, n_jobs=-1)
    # average across epochs first
    mean_spectrum = spectrum.average(method="median")
    psds, freqs = mean_spectrum.get_data(return_freqs=True)
    # then convert to dB and take mean & standard deviation across channels
    psds = 10 * np.log10(psds)
    psds_mean = psds.mean(axis=0)
    psds_std = psds.std(axis=0)/np.sqrt(psds.shape[0])

    _, ax = plt.subplots()
    ax.plot(freqs, psds_mean, color="k")
    ax.fill_between(
        freqs,
        psds_mean - psds_std,
        psds_mean + psds_std,
        color="k",
        alpha=0.5,
        edgecolor="none",
    )
    ax.set(
        title="Multitaper PSD (gradiometers)",
        xlabel="Frequency (Hz)",
        ylabel="Power Spectral Density (dB)",
    )

    plt.savefig(os.path.join(args.save_path, f"power_{args.subject_id}.png"))
    np.save(os.path.join(args.save_path, "power", f"power_{args.subject_id}.npy"), psds)

    print("Done!")
    