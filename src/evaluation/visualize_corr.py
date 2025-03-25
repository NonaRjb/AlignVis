import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/alignvis")

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from src.evaluation.stat_test import ttest

def load_data(data_path):
    correlations = []
    noise_ceiling_lower = []
    noise_ceiling_upper = []
    for file in os.listdir(data_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(data_path, file), allow_pickle=True).item()
            correlations.append(data['correlation'])
            noise_ceiling_lower.append(data['noise_ceiling_lower'])
            noise_ceiling_upper.append(data['noise_ceiling_upper'])
    correlations = np.array(correlations).mean(axis=0)
    noise_ceiling_lower = np.array(noise_ceiling_lower).mean(axis=0)
    noise_ceiling_upper = np.array(noise_ceiling_upper).mean(axis=0)
    return correlations, noise_ceiling_lower, noise_ceiling_upper


def plot_encoding_results(correlations1, correlations2, noise_ceiling_lower, noise_ceiling_upper, significance_diff, save_path=None):
    """
    Plots the encoding model's correlation results over time.

    Parameters:
    - correlations: numpy array of shape (iterations, channels, time_points)
                    containing model correlations with real data.
    - noise_ceiling_lower: numpy array of shape (time_points,)
                           containing the lower bound of the noise ceiling.
    - noise_ceiling_upper: numpy array of shape (time_points,)
                           containing the upper bound of the noise ceiling.
    """
    # Compute mean and standard error of the correlations across iterations and channels
    mean_corr1 = np.mean(correlations1, axis=(0, 1))
    std_err_corr1 = np.std(correlations1, axis=(0, 1)) / np.sqrt(correlations1.shape[0] * correlations1.shape[1])

    mean_corr2 = np.mean(correlations2, axis=(0, 1))
    std_err_corr2 = np.std(correlations2, axis=(0, 1)) / np.sqrt(correlations2.shape[0] * correlations2.shape[1])

    # Generate the time vector
    time_vector = np.arange(-0.1, 0.61, 1/100)[:mean_corr1.shape[0]]

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(time_vector, mean_corr1, label="Model_1 Correlation", color="blue")
    plt.fill_between(time_vector, mean_corr1 - std_err_corr1, mean_corr1 + std_err_corr1, color="blue", alpha=0.2)

    plt.plot(time_vector, mean_corr2, label="Model_2 Correlation", color="orange")
    plt.fill_between(time_vector, mean_corr2 - std_err_corr2, mean_corr2 + std_err_corr2, color="orange", alpha=0.2)

    # Plot noise ceiling
    plt.plot(time_vector, noise_ceiling_lower, label="Noise Ceiling (Lower)", linestyle="dashed", color="red")
    # plt.plot(time_vector, noise_ceiling_upper, label="Noise Ceiling (Upper)", linestyle="dashed", color="green")

    sig_timepoints = time_vector[significance_diff]  # Get time points where significance_diff is True
    plt.scatter(sig_timepoints, np.full_like(sig_timepoints, 0.01), marker='o', color='black', label="Significant Time Points")

    # Formatting
    plt.axvline(0, color='black', linestyle='--', linewidth=1)  # Mark stimulus onset at 0s
    plt.xlabel("Time (s)")
    plt.ylabel("Correlation")
    plt.title("Encoding Model Performance Over Time")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize encoding model results")
    parser.add_argument("--data_path", type=str, help="Path to the .npy file containing the encoding model results")
    parser.add_argument("--save_path", type=str, help="Path to save the plot")
    parser.add_argument("--subject_id", type=int, help="Subject ID")
    parser.add_argument("--dnn1", type=str, help="DNN-1 model name")
    parser.add_argument("--dnn2", type=str, help="DNN-2 model name")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    corrs1, noise_ceils_lower, noise_ceils_upper = load_data(os.path.join(args.data_path, "sub-"+format(args.subject_id,'02'), "dnn-"+args.dnn1))
    corrs2, _, _ = load_data(os.path.join(args.data_path, "sub-"+format(args.subject_id,'02'), "dnn-"+args.dnn2))
    noise_ceil_lower = np.mean(noise_ceils_lower, axis=(0, 1))
    noise_ceil_upper = np.mean(noise_ceils_upper, axis=(0, 1)) 

    _, _, significance = ttest(corrs1, corrs2, noise_ceil_lower)
    plot_encoding_results(corrs1, corrs2, noise_ceil_lower, noise_ceil_upper, significance, save_path=os.path.join(args.save_path, "sub"+format(args.subject_id,'02')+"_dnn-"+args.dnn1+".png"))   
