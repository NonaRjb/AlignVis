"""Train DNN models end-to-end to predict the EEG responses to arbitrary
images.

Parameters
----------
sub : int
	Used subject.
modeled_time_points : str
	Whether each DNN should model 'single' or 'all' EEG time points.
dnn : str
	DNN model used.
pretrained : bool
	If True use a pretrained network, if False use a randomly initialized one.
epochs : int
	Number of training epochs.
lr : float
	Learning rate.
weight_decay : float
	Weight decay coefficient.
batch_size : int
	Batch size for weight update.
save_trained_models : bool
	If True, the weights of the trained models are saved.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import random
import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.utils import resample
from copy import deepcopy

from end_to_end_encoding_utils import load_images_nn
from end_to_end_encoding_utils import load_eeg_data_nn
from end_to_end_encoding_utils import create_dataloader


model_configs = {
        'eegnet': {},
        'eegconformer': {},
        'nice': {'emb_size': 40, 'embedding_dim': 1440, 'drop_proj': 0.5},
        'atms': {},
        'lstm': {'lstm_size': 128, 'lstm_layers': 1},
        'resnet1d': {},
        'resnet1d_subj': {},
        'resnet1d_subj_resblk': {},
        'brain-mlp': {},
        'dinov2_vit-l-14_noalign': {'embed_dim': 1024},
        'DINOv2_ViT-L14_noalign': {'embed_dim': 1024},
        'DINO_ViT-B8_noalign': {'embed_dim': 768},
        'DINOv2_ViT-B14_noalign': {'embed_dim': 768},
        'DINO_ViT-B16_noalign': {'embed_dim': 768},
        'CLIP_ViT-L14_noalign': {'embed_dim': 768},
        'CLIP_ViT-B32_noalign': {'embed_dim': 512},
        'OpenCLIP_ViT-L14_laion400m_noalign': {'embed_dim': 768},
        'OpenCLIP_ViT-L14_laion2b_noalign': {'embed_dim': 768},
        'OpenCLIP_ViT-B32_laion400m_noalign': {'embed_dim': 512},
        'dreamsim_clip_vitb32_768': {'embed_dim': 768},
        'dreamsim_clip_vitb32': {'embed_dim': 512},
        'dreamsim_clip_vitb32_noalign': {'embed_dim': 512},
        'dreamsim_open_clip_vitb32': {'embed_dim': 512},
        'dreamsim_open_clip_vitb32_noalign': {'embed_dim': 512},
        'dreamsim_synclr_vitb16': {'embed_dim': 768},
        'dreamsim_synclr_vitb16_noalign': {'embed_dim': 768},
        'dreamsim_ensemble': {'embed_dim': 1792},
        'dreamsim_ensemble_noalign': {'embed_dim': 1792},
        'dreamsim_dino_vitb16': {'embed_dim': 768},
        'dreamsim_dino_vitb16_noalign': {'embed_dim': 768},
        'dreamsim_dinov2_vitb14': {'embed_dim': 768},
        'dreamsim_dinov2_vitb14_noalign': {'embed_dim': 768},
        'gLocal_openclip_vit-l-14_laion2b_s32b_b82k': {'embed_dim': 768},
        'gLocal_openclip_vit-l-14_laion2b_s32b_b82k_noalign': {'embed_dim': 768},
        'gLocal_openclip_vit-l-14_laion400m_e32': {'embed_dim': 768},
        'gLocal_openclip_vit-l-14_laion400m_e32_noalign': {'embed_dim': 768},
        'gLocal_clip_vit-l-14': {'embed_dim': 768},
        'gLocal_clip_vit-l-14_noalign': {'embed_dim': 768},
        'gLocal_dino-vit-base-p8': {'embed_dim': 768},
        'gLocal_dino-vit-base-p8_noalign': {'embed_dim': 768},
        'gLocal_dino-vit-base-p16': {'embed_dim': 768},
        'gLocal_dino-vit-base-p16_noalign': {'embed_dim': 768},
        'gLocal_dinov2-vit-base-p14': {'embed_dim': 768},
        'gLocal_dinov2-vit-base-p14_noalign': {'embed_dim': 768},
        'gLocal_dinov2-vit-large-p14': {'embed_dim': 1024},
        'gLocal_dinov2-vit-large-p14_noalign': {'embed_dim': 1024},
        'gLocal_clip_rn50': {'embed_dim': 1024},
        'gLocal_clip_rn50_noalign': {'embed_dim': 1024},
        'harmonization_vitb16': {'embed_dim': 768},
        'harmonization_vitb16_noalign': {'embed_dim': 768},
        'harmonization_resnet50': {'embed_dim': 2048},
        'harmonization_resnet50_noalign': {'embed_dim': 2048},
        'harmonization_convnext': {'embed_dim': 768},
        'harmonization_convnext_noalign': {'embed_dim': 768},
        'harmonization_levit': {'embed_dim': 384},
        'harmonization_levit_noalign': {'embed_dim': 384},
        'harmonization_vgg16': {'embed_dim': 4096},
        'harmonization_vgg16_noalign': {'embed_dim': 4096},
    }

parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default=1)
parser.add_argument('--modeled_time_points', type=str, default='all')
parser.add_argument('--dnn', type=str, default='alexnet')
parser.add_argument('--img_type', type=str, default='embedding')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--save_trained_models', type=bool, default=False)
parser.add_argument('--project_dir', default='./', type=str)
parser.add_argument('--experiment', default='encoding_models', type=str)
parser.add_argument('--save_dir', default='./', type=str)
args = parser.parse_args()

print('>>> End-to-end encoding <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Set random seeds to make results reproducible and GPU
# =============================================================================
# Random seeds
seed = 20200220
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
# Generator object for DataLoader random batching
g_cpu = torch.Generator()
g_cpu.manual_seed(seed)

# Checking for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Load the images (X) and the EEG data (y)
# =============================================================================
# Randomly select 100 training image concepts to use´for validation. For each
# image concept, all 10 exemplars are used for validation, resulting in:
# (100 image conceps × 10 exemplars per concept = 1,000 validation images).

train_img_concepts = np.arange(1654)
img_per_concept = 10
val_concepts = np.sort(resample(train_img_concepts, replace=False,
	n_samples=100))
idx_val = np.zeros((len(train_img_concepts)*img_per_concept), dtype=bool)
for i in val_concepts:
	idx_val[i*img_per_concept:i*img_per_concept+img_per_concept] = True

# Load the images
X_train, X_val, X_test = load_images_nn(args, idx_val)

# Load the EEG data
y_train, y_val, y_test, ch_names, times = load_eeg_data_nn(args, idx_val)


# =============================================================================
# Models loop
# =============================================================================
# Loop across the EEG temporal dimension if the encoding models are trained
# independently on each time point.
# If the encoding models are predicting all time points at once, then the loop
# will only run for one iteration.

# Define the amount of models which will be trained and the amount of their EEG
# output features
if args.modeled_time_points == 'single':
	num_models = y_test.shape[2]
	out_features = y_test.shape[1]
elif args.modeled_time_points == 'all':
	num_models = 1
	out_features = y_test.shape[1] * y_test.shape[2]

# Synthetic EEG data matrix of shape:
# (Test image conditions × EEG channels × EEG time points)
synthetic_data = np.zeros((y_test.shape))
best_epochs = np.zeros((num_models))

# Loop across independent models
for m in range(num_models):
	print(f'\nModel: [{m+1}/{num_models}]')


# =============================================================================
# Create PyTorch-compatible Dataloaders
# =============================================================================
	train_dl, val_dl, test_dl = create_dataloader(args, m, g_cpu, X_train,
		X_val, X_test, y_train, y_val, y_test)


# =============================================================================
# Load the DNN model and change the last layer to the amount of EEG features
# =============================================================================
	# # AlexNet
	# if args.dnn == 'alexnet':
	# 	model = torchvision.models.alexnet(pretrained=args.pretrained)
	# 	model.classifier[6] = nn.Linear(in_features=4096,
	# 		out_features=out_features)
	# # ResNet-50
	# if args.dnn == 'resnet50':
	# 	model = torchvision.models.resnet50(pretrained=args.pretrained)
	# 	model.fc = nn.Linear(in_features=2048, out_features=out_features)
	if args.img_type == 'embedding':
		img_feature_size = model_configs[args.dnn]['embed_dim']
		model = nn.Linear(in_features=img_feature_size, out_features=out_features)
	else:
		raise NotImplementedError('Only image embeddings are supported.')
	model.to(device)


# =============================================================================
# TensorBoard, loss function and optimizer
# =============================================================================
	# TensorBoard
	save_dir = os.path.join(args.save_dir, args.experiment, 'sub-'+
		format(args.sub,'02'),
		'dnn-'+args.dnn, 'modeled_time_points-'+args.modeled_time_points)
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)
	writer = SummaryWriter(save_dir)

	# Loss function
	loss_fn = torch.nn.MSELoss().to(device)

	# Optimizer
	optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr,
		weight_decay=args.weight_decay)

	# Gradient scaler
	scaler = torch.cuda.amp.GradScaler()

	# Benchmark multiple convolution algorithms and select the fastest
	torch.backends.cudnn.benchmark = True


# =============================================================================
# Training and validation loops
# =============================================================================
	def train_loop(train_dl, model, loss_fn, optimizer):
		tot_loss = 0
		model.train()
		for X, y in train_dl:
			# Prediction and loss
			X = X.to(device)
			y = y.to(device)
			with torch.autocast(device_type='cuda', dtype=torch.float16):
				y_pred = model(X).squeeze()
				loss = loss_fn(y_pred, y)
			# Backpropagation
			optimizer.zero_grad(set_to_none=True)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
			# Total loss
			tot_loss += loss.item() * len(X)
		tot_loss /= len(train_dl.dataset)
		return tot_loss

	def val_loop(val_dl, model, loss_fn):
		tot_loss = 0
		model.eval()
		with torch.no_grad():
			for X, y in val_dl:
				# Prediction and loss
				X = X.to(device)
				y = y.to(device)
				with torch.autocast(device_type='cuda', dtype=torch.float16):
					y_pred = model(X).squeeze()
					loss = loss_fn(y_pred, y)
				tot_loss += loss.item() * len(X)
			tot_loss /= len(val_dl.dataset)
		return tot_loss


# =============================================================================
# Train the model and log the training statistics to TensorBoard
# =============================================================================
	best_val_loss = 10000

	for e in tqdm(range(args.epochs)):
		# Train the model and train loss
		train_loss = train_loop(train_dl, model, loss_fn, optimizer)
		# Validation loss
		val_loss = val_loop(val_dl, model, loss_fn)
		# Add the training stats to TensorBoard
		writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, e+1)
		writer.flush()
		# Retain the best epoch model according to the validation loss
		if val_loss < best_val_loss:
			best_model = deepcopy(model)
			best_epochs[m] = e + 1
			best_val_loss = val_loss

	# Delete the model from GPU memory
	del model
	if device == 'cuda':
		torch.cuda.empty_cache()


# =============================================================================
# Synthesize the EEG test data using the best model
# =============================================================================
	best_model.to(device)
	best_model.eval()
	with torch.no_grad():
		for X, y in test_dl:
			X = X.to(device)
			y = y.to(device)
			if args.modeled_time_points == 'single':
				synthetic_data[:,:,m] = best_model(X).detach().cpu().numpy()
			elif args.modeled_time_points == 'all':
				synthetic_data = np.reshape(
					best_model(X).detach().cpu().numpy(),
					(synthetic_data.shape))


# =============================================================================
# Save the best model
# =============================================================================
	if args.save_trained_models == True:
		file_name = 'model_state_dict.pt'
		best_model = best_model.to('cpu')
		torch.save({
			'args': args,
			'best_model': best_model.state_dict(),
			'out_features': out_features,
			'epoch': best_epochs[m]
			}, os.path.join(save_dir, file_name))

	# Delete the model from GPU memory
	del best_model
	if device == 'cuda':
		torch.cuda.empty_cache()


# =============================================================================
# Save the modeling results
# =============================================================================
# Put the synthesized data into a dictionary for compatibility with the
# linearizing encoding synthetic data
synthetic_data_dict = {}
if args.modeled_time_points == 'single':
	synthetic_data_dict['single_time_points'] = synthetic_data
if args.modeled_time_points == 'all':
	synthetic_data_dict['all_time_points'] = synthetic_data

modeling_results = {
	'synthetic_data': synthetic_data_dict,
	'ch_names': ch_names,
	'times': times,
	'best_epochs': best_epochs
}

file_name = 'synthetic_eeg_test'
np.save(os.path.join(save_dir, file_name), modeling_results)