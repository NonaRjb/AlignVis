import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from scipy.stats import ttest_1samp, ttest_rel
from statsmodels.stats.multitest import multipletests
import argparse
import os


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
def bootsrtap_ci(correlation, diff_noise_ceiling):
    # CI matrices of shape: (Layer, Time)
    ci_lower = {}
    ci_upper = {}
    ci_lower_diff_noise_ceiling = {}
    ci_upper_diff_noise_ceiling = {}

    # Calculate the CIs independently at each time point
    for layer in correlation.keys():
        # CI matrices of shape: (Time)
        ci_lower[layer] = np.zeros((correlation[layer].shape[2]))
        ci_upper[layer] = np.zeros((correlation[layer].shape[2]))
        ci_lower_diff_noise_ceiling[layer] = np.zeros((
            diff_noise_ceiling[layer].shape[2]))
        ci_upper_diff_noise_ceiling[layer] = np.zeros((
            diff_noise_ceiling[layer].shape[2]))
        for t in tqdm(range(correlation[layer].shape[2])):
            sample_dist = np.zeros(args.n_iter)
            sample_dist_diff = np.zeros(args.n_iter)
            for i in range(args.n_iter):
                # Calculate the sample distribution of the correlation values
                # averaged across channels
                sample_dist[i] = np.mean(resample(np.mean(
                    correlation[layer][:,:,t], 1)))
                sample_dist_diff[i] = np.mean(resample(np.mean(
                    diff_noise_ceiling[layer][:,:,t], 1)))
            # Calculate the 95% confidence intervals
            ci_lower[layer][t] = np.percentile(sample_dist, 2.5)
            ci_upper[layer][t] = np.percentile(sample_dist, 97.5)
            ci_lower_diff_noise_ceiling[layer][t] = np.percentile(
                sample_dist_diff, 2.5)
            ci_upper_diff_noise_ceiling[layer][t] = np.percentile(
                sample_dist_diff, 97.5)


# =============================================================================
# One-sample t-tests for significance & multiple comparisons correction
# =============================================================================
def ttest(correlations_1, correlations_2, noise_ceilings):

    print(np.std(np.mean(correlations_1, axis=1), axis=(0)))
    
    diff_correlations = (correlations_1 - correlations_2)
    
    # p-values matrices of shape: (Time)
    p_values_1 = np.ones((correlations_1.shape[2]))
    p_values_2 = np.ones((correlations_2.shape[2]))
    p_values_diff = np.ones((diff_correlations.shape[2]))
    
    for t in range(correlations_1.shape[2]):
        # Fisher transform the correlation values and perform the t-tests
        fisher_values_1 = np.arctanh(np.mean(correlations_1[:,:,t], 1))
        fisher_values_2 = np.arctanh(np.mean(correlations_2[:,:,t], 1))

        # fisher_values_diff = np.arctanh(np.mean(diff_correlations[:,:,t], 1).squeeze())
        # fisher_values_diff = fisher_values_1 - fisher_values_2
        p_values_1[t] = ttest_1samp(fisher_values_1, 0, alternative='greater')[1]
        p_values_2[t] = ttest_1samp(fisher_values_2, 0, alternative='greater')[1]
        p_values_diff[t] = ttest_rel(fisher_values_1, fisher_values_2, 0, alternative='greater')[1]
        
    # Correct for multiple comparisons
    significance_1, p_vals_corrected_1, _, _ = multipletests(p_values_1, 0.05, 'bonferroni')
    significance_2, p_vals_corrected_2, _, _ = multipletests(p_values_2, 0.05, 'bonferroni')
    significance_diff, p_vals_corrected_diff, _, _ = multipletests(p_values_diff, 0.05, 'bonferroni')

    return significance_1, significance_2, significance_diff


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
    significance_1, significance_2, significance_diff = ttest(corrs1, corrs2, noise_ceils_lower)
    

