"""Fit a linear regression to predict EEG data using the DNN feature maps as
predictors. The linear regression is trained using varying amounts (25%, 50%,
75%, 100%) of image conditions and EEG repetitions of the training images EEG
data (Y) and feature maps (X). The learned weights are used to synthesize the
EEG data of the test images. The synthetic EEG test data is then correlated
with the biological EEG test data.

Parameters
----------
sub : int
	Used subject.
dnn : str
	Used DNN network.
pretrained : bool
	If True use the pretrained network feature maps, if False use the randomly
	initialized network feature maps.
layers : str
	If 'all', the EEG data will be predicted using the feature maps downsampled
	through PCA applied across all DNN layers. If 'single', the EEG data will be
	independently predicted using the PCA-downsampled feature maps of each DNN
	layer independently. If 'appended', the EEG data will be predicted using the
	PCA-downsampled feature maps of each DNN layer appended onto each other.
n_components : int
	Number of DNN feature maps PCA components retained.
n_img_cond : int
	Number of used image conditions.
n_eeg_rep : int
	Number of used EEG repetitions.
n_iter : int
	Number of analysis iterations.
project_dir : str
	Directory of the project folder.


Returns
-------
Saves the predicted test EEG data.

"""

import argparse
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from end_to_end_encoding_utils import load_images_reg
from end_to_end_encoding_utils import load_eeg_data_reg
from end_to_end_encoding_utils import perform_regression, perform_ridge_regression
from end_to_end_encoding_utils import correlation_analysis
from end_to_end_encoding_utils import save_data


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--dnn', default='dreamsim_clip_vitb32', type=str)
parser.add_argument('--dataset', default='things-eeg-preprocessed', type=str)
parser.add_argument('--n_components', default=0.99, type=float)
parser.add_argument('--n_img_cond', default=16540, type=int)
parser.add_argument('--n_eeg_rep', default=4, type=int)
parser.add_argument('--n_iter', default=100, type=int)
parser.add_argument('--t_start', default=-0.1, type=float)
parser.add_argument('--t_end', default=0.6, type=float)
parser.add_argument('--project_dir', default='./', type=str)
parser.add_argument('--experiment', default='encoding_models', type=str)
parser.add_argument('--save_dir', default='./', type=str)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

print('>>> Training data amount <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
# seed = 20200220
np.random.seed(args.seed)


# =============================================================================
# Used image conditions and EEG repetitions combinations
# =============================================================================
tot_img_conditions = 16540
tot_eeg_repetitions = 4

# Loop across iterations
# correlation = np.zeros(args.n_iter)
# noise_ceiling = np.zeros(args.n_iter)
# for i in tqdm(range(args.n_iter)):
# Randomly select the image conditions and repetitions
cond_idx = np.sort(resample(np.arange(0, tot_img_conditions),
	replace=False, n_samples=args.n_img_cond))
rep_idx = np.sort(resample(np.arange(0, tot_eeg_repetitions),
	replace=False, n_samples=args.n_eeg_rep))


# =============================================================================
# Load the DNN feature maps
# =============================================================================
X_train, X_test = load_images_reg(args, cond_idx)


# =============================================================================
# Load the EEG data
# =============================================================================
y_train, y_test = load_eeg_data_reg(args, cond_idx, rep_idx, start_t=args.t_start, end_t=args.t_end, dataset=args.dataset)


# =============================================================================
# Train a linear regression to predict the EEG data
# =============================================================================
# y_test_pred = perform_regression(X_train, X_test, y_train)
y_test_pred = perform_ridge_regression(X_train, X_test, y_train)

# =============================================================================
# Test the encoding prediction accuracy through a correlation
# =============================================================================
corr_res, noise_ceil_lower, noise_ceil_upper = correlation_analysis(args, y_test_pred, y_test)
# Results matrices of shape: (Iterations)
# if i == 0:
# 	correlation = np.zeros(args.n_iter)
# 	noise_ceiling = np.zeros(args.n_iter)
# # Store the results
# correlation[i] = corr_res
# noise_ceiling[i] = noise_ceil
correlation = corr_res
noise_ceiling_lower = noise_ceil_lower 
noise_ceiling_upper = noise_ceil_upper

# Average the results across iterations
# correlation = np.mean(correlation)
# noise_ceiling = np.mean(noise_ceiling)


# =============================================================================
# Save the correlation results
# =============================================================================
save_data(args, correlation, noise_ceiling_lower, noise_ceiling_upper)