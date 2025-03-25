from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader	


def load_images_nn(args, idx_val):
	"""Load and preprocess the training, validation and test images.	
	Parameters
	----------
	args : Namespace
		Input arguments.
	idx_val : bool
		Indices of the validation images.	
	Returns
	-------
	X_train : list of tensor
		Training images.
	X_val : list of tensor
		Validation images.
	X_test : list of tensor
		Test images.	
	"""	
	### Define the image preprocesing ###
	preprocess = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])	
	### Load and preprocess the training and validation images ###
	img_parent_dir = os.path.join(args.project_dir, 'images' if args.img_type == "raw" else 'image_embeddings')
	img_metadata = np.load(os.path.join(img_parent_dir, 'image_metadata.npy'), allow_pickle=True).item()	
	training_img_concepts = img_metadata['train_img_concepts']
	training_img_files = img_metadata['train_img_files']	
	image_list = []
	for i in range(len(training_img_files)):
	    if args.img_type == "raw":
	        image_list.append(os.path.join(img_parent_dir, 'training_images', 
	            training_img_concepts[i], training_img_files[i]))
	    else:
	        image_list.append(os.path.join(img_parent_dir, 'training_images', 
	            training_img_concepts[i], training_img_files[i].replace(".jpg", f"_{args.dnn}.npy")))
	image_list.sort()
	X_train = []
	X_val = []
	for i, image in enumerate(tqdm(image_list)):
		if args.img_type == "raw":
		    img = Image.open(image).convert('RGB')
		    img = preprocess(img)
		else:
		    img = np.load(image)
		if idx_val[i] == True:
			X_val.append(img)
		else:
			X_train.append(img)	
	### Load and preprocess the test images ###
	test_img_concepts = img_metadata['test_img_concepts']
	test_img_files = img_metadata['test_img_files']
	image_list = []
	for i in range(len(test_img_files)):
		if args.img_type == "raw":
		    image_list.append(os.path.join(img_parent_dir, 'test_images', 
		        test_img_concepts[i], test_img_files[i]))
		else:
		    image_list.append(os.path.join(img_parent_dir, 'test_images', 
		        test_img_concepts[i], test_img_files[i].replace(".jpg", f"_{args.dnn}.npy")))
	image_list.sort()
	X_test = []
	for image in tqdm(image_list):
		if args.img_type == "raw":
		    img = Image.open(image).convert('RGB')
		    img = preprocess(img)
		else:
		    img = np.load(image)
		X_test.append(img)	
	### Output ###
	return X_train, X_val, X_test


def load_images_reg(args, cond_idx, apply_pca=False):
	"""Load the DNN feature maps of the training and test images.

	Parameters
	----------
	args : Namespace
		Input arguments.
	cond_idx : int
		Indices of the used image conditions.

	Returns
	-------
	X_train : dict of float
		Training images feature maps.
	X_test : dict of float
		Test images feature maps.
	"""
	### Define the image preprocesing ###
	preprocess = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])	
	### Load and preprocess the training and validation images ###
	img_parent_dir = os.path.join(args.project_dir, 'image_embeddings')
	img_metadata = np.load(os.path.join(img_parent_dir, 'image_metadata.npy'), allow_pickle=True).item()	
	training_img_concepts = img_metadata['train_img_concepts']
	training_img_files = img_metadata['train_img_files']	
	image_list = []
	for i in range(len(training_img_files)):
	    image_list.append(os.path.join(img_parent_dir, 'training_images', 
	        training_img_concepts[i], training_img_files[i].replace(".jpg", f"_{args.dnn}.npy")))
	image_list.sort()
	X_train = []
	for i, image in enumerate(tqdm(image_list)):
		img = np.load(image)
		X_train.append(img.squeeze())	
	X_train = np.array(X_train)[cond_idx, ...]
	
	### Load and preprocess the test images ###
	test_img_concepts = img_metadata['test_img_concepts']
	test_img_files = img_metadata['test_img_files']
	image_list = []
	for i in range(len(test_img_files)):
		image_list.append(os.path.join(img_parent_dir, 'test_images', 
		        test_img_concepts[i], test_img_files[i].replace(".jpg", f"_{args.dnn}.npy")))
	image_list.sort()
	X_test = []
	for image in tqdm(image_list):
		img = np.load(image)
		X_test.append(img.squeeze())	
	X_test = np.array(X_test)
	if apply_pca:
		pca = PCA(n_components=0.99)
		X_train = pca.fit_transform(X_train)
		X_test = pca.transform(X_test)

	### Output ###
	return X_train, X_test


def load_eeg_data_nn(args, idx_val):
	"""Load the EEG training and test data.	
	Parameters
	----------
	args : Namespace
		Input arguments.
	idx_val : bool
		Indices of the validation images.	
	Returns
	-------
	y_train : tensor
		Training EEG data.
	y_val : tensor
		Validation EEG data.
	y_test : tensor
		Test EEG data.
	ch_names : list of str
		EEG channel names.
	times : float
		EEG time points.	
	"""		
	### Load the EEG training data ###
	data_dir = os.path.join('preprocessed_eeg', 'sub-'+
		format(args.sub,'02'))
	training_file = 'preprocessed_eeg_training.npy'
	data = np.load(os.path.join(args.project_dir, data_dir, training_file),
		allow_pickle=True).item()
	y_train = data['preprocessed_eeg_data']
	ch_names = data['ch_names']
	times = data['times']
	# Average across repetitions
	y_train = np.mean(y_train, 1)
	# Extract the validation data
	y_val = y_train[idx_val]
	y_train = np.delete(y_train, idx_val, 0)
	# Convert to float32 and tensor (for DNN training with Pytorch)
	y_train = torch.tensor(np.float32(y_train))
	y_val = torch.tensor(np.float32(y_val))	
	### Load the EEG test data ###
	test_file = 'preprocessed_eeg_test.npy'
	data = np.load(os.path.join(args.project_dir, data_dir, test_file),
		allow_pickle=True).item()
	y_test = data['preprocessed_eeg_data']
	# Average across repetitions
	y_test = np.mean(y_test, 1)
	# Convert to float32 and tensor (for DNN training with Pytorch)
	y_test = torch.tensor(np.float32(y_test))	
	### Output ###
	return y_train, y_val, y_test, ch_names, times


def load_eeg_data_reg(args, cond_idx, rep_idx, start_t=-0.1, end_t=0.6, dataset="things-eeg-preprocessed"):
	"""Load the EEG training and test data, and select only the time point
	interval 60-500ms for the subsequent analyses.

	Parameters
	----------
	args : Namespace
		Input arguments.
	cond_idx : int
		Indices of the used image conditions.
	rep_idx : int
		Indices of the used EEG repetitions.

	Returns
	-------
	y_train : float
		Training EEG data.
	y_test : float
		Test EEG data.

	"""

	import os
	import numpy as np

	### Load the EEG training data ###
	if dataset == "things-eeg-preprocessed":
		data_dir = os.path.join('preprocessed_eeg', 'sub-'+
			format(args.sub,'02'))
	else:
		data_dir = os.path.join('Preprocessed_data_250Hz', 'sub-'+
			format(args.sub,'02'))
	training_file = 'preprocessed_eeg_training.npy'
	if dataset == "things-eeg-preprocessed":
		data = np.load(os.path.join(args.project_dir, data_dir, training_file),
			allow_pickle=True).item()
	else:
		data = np.load(os.path.join(args.project_dir, data_dir, training_file),
			allow_pickle=True)
	y_train = data['preprocessed_eeg_data']
	times = np.round(data['times'], 2)
	# Select the desired amount of training categories
	y_train = y_train[cond_idx]
	# Average across the selected amount of training repetitions
	y_train = np.mean(y_train[:,rep_idx], 1)
	# Select the time points between 60-500ms
	times_start = np.where(times == start_t)[0][0]
	times_end = np.where(times == end_t)[0][0]
	y_train = y_train[:,:,times_start:times_end]

	### Load the EEG test data ###
	test_file = 'preprocessed_eeg_test.npy'
	if dataset == "things-eeg-preprocessed":
		data = np.load(os.path.join(args.project_dir, data_dir, test_file),
			allow_pickle=True).item()
	else:
		data = np.load(os.path.join(args.project_dir, data_dir, test_file),
			allow_pickle=True)
	y_test = data['preprocessed_eeg_data']
	# Select the time points between 60-500ms
	y_test = y_test[:,:,:,times_start:times_end]

	### Output ###
	return y_train, y_test


def create_dataloader(args, time_point, g_cpu, X_train, X_val, X_test, y_train,
	y_val, y_test):
	"""Put the training, validation and test data into a PyTorch-compatible
	Dataloader format.	
	Parameters
	----------
	args : Namespace
		Input arguments.
	time_point : int
		Modeled EEG time point.
	g_cpu : torch.Generator
		Generator object for DataLoader random batching.
	X_train : list of tensor
		Training images.
	X_val : list of tensor
		Validation images.
	X_test : list of tensor
		Test images.
	y_train : float
		Training EEG data.
	y_val : float
		Validation EEG data.
	y_test : float
		Test EEG data.	
	Returns
	----------
	train_dl : Dataloader
		Training Dataloader.
	val_dl : Dataloader
		Validation Dataloader.
	test_dl : Dataloader
		Test Dataloader.	
	"""	
	### Dataset class ###
	class EegDataset(Dataset):
		def __init__(self, X, y, modeled_time_points, time, transform=None,
			target_transform=None):
			self.modeled_time_points = modeled_time_points
			self.time = time
			self.X = X
			if self.modeled_time_points == 'single':
				self.y = y[:,:,self.time]
			elif self.modeled_time_points == 'all':
				self.y = torch.reshape(y, (y.shape[0],-1))
			self.transform = transform
			self.target_transform = target_transform	
		def __len__(self):
			return len(self.y)	
		def __getitem__(self, idx):
			image = self.X[idx]
			target = self.y[idx]
			if self.transform:
				image = self.transform(image)
			if self.target_transform:
				target = self.target_transform(target)
			return image, target	
	### Convert the data to PyTorch's Dataset format ###
	train_ds = EegDataset(X_train, y_train, args.modeled_time_points,
		time_point)
	val_ds = EegDataset(X_val, y_val, args.modeled_time_points, time_point)
	test_ds = EegDataset(X_test, y_test, args.modeled_time_points, time_point)	
	### Convert the Datasets to PyTorch's Dataloader format ###
	train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
		generator=g_cpu)
	val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
	test_dl = DataLoader(test_ds, batch_size=test_ds.__len__(), shuffle=False)	
	### Output ###
	return train_dl, val_dl, test_dl


def perform_regression(X_train, X_test, y_train):
	"""Train a linear regression on the training images DNN feature maps (X) and
	training EEG data (Y), and use the trained weights to synthesize the EEG
	responses to the test images.

	Parameters
	----------
	X_train : dict of float
		Training images feature maps.
	X_test : dict of float
		Test images feature maps.
	y_train : float
		Training EEG data.

	Returns
	-------
	y_test_pred : dict of float
		Predicted test EEG data.

	"""

	import numpy as np
	from ols import OLS_pytorch

	### Fit the regression at each time-point and channel ###
	eeg_shape = y_train.shape
	y_train = np.reshape(y_train, (y_train.shape[0],-1))
	reg = OLS_pytorch(use_gpu=False)
	reg.fit(X_train, y_train.T)
	y_test_pred = np.reshape(reg.predict(X_test), (-1,eeg_shape[1],eeg_shape[2]))

	### Output ###
	return y_test_pred


def perform_ridge_regression(X_train, X_test, y_train):
	_, channels, time_points = y_train.shape
	trials = X_test.shape[0]
	X_train = np.reshape(X_train, (X_train.shape[0], -1))
	X_test = np.reshape(X_test, (X_test.shape[0], -1))
	pca = PCA(n_components=0.99)
	X_train = pca.fit_transform(X_train)
	X_test = pca.transform(X_test)
	y_train = np.reshape(y_train, (y_train.shape[0], -1))
	regressor = RidgeCV(alphas=[1e-2, 1e-1, 1, 10], cv=10)
	regressor = regressor.fit(X_train, y_train)
	y_test_pred = regressor.predict(X_test)
	y_test_pred = np.reshape(y_test_pred, (trials, channels, time_points))

	return y_test_pred


def correlation_analysis(args, y_test_pred, y_test):
	"""Evaluate the encoding models prediction accuracy by correlating the
	synthetic EEG test data with biological test data.

	Parameters
	----------
	args : Namespace
		Input arguments.
	y_test_pred : dict of float
		Predicted test EEG data.
	y_test : float
		Test EEG data.

	Returns
	-------
	correlation : dict of float
		Correlation results.
	noise_ceiling : float
		Noise ceiling results.

	"""

	import numpy as np
	from sklearn.utils import resample
	from scipy.stats import pearsonr as corr

	### Perform the correlation ###
	# Results matrices of shape:
	# (Iterations ×  EEG channels × EEG time points)
	correlation = np.zeros((args.n_iter,y_test.shape[2],
		y_test.shape[3]))
	noise_ceiling_lower = np.zeros((args.n_iter,y_test.shape[2],y_test.shape[3]))
	noise_ceiling_upper = np.zeros((args.n_iter,y_test.shape[2],y_test.shape[3]))
	for i in range(args.n_iter):
		# Random data repetitions index
		shuffle_idx = resample(np.arange(0, y_test.shape[1]), replace=False,
			n_samples=int(y_test.shape[1]/2))
		# Average across one half of the biological data repetitions
		bio_data_avg_half_1 = np.mean(np.delete(y_test, shuffle_idx, 1), 1)
		# Average across the other half of the biological data repetitions for
		# the noise ceiling calculation
		bio_data_avg_half_2 = np.mean(y_test[:,shuffle_idx,:,:], 1)
		# Compute the correlation and noise ceiling
		for t in range(y_test.shape[3]):
			for c in range(y_test.shape[2]):
				correlation[i,c,t] = corr(y_test_pred[:,c,t],
					bio_data_avg_half_1[:,c,t])[0]
				noise_ceiling_lower[i,c,t] = corr(bio_data_avg_half_2[:,c,t],
					bio_data_avg_half_1[:,c,t])[0]
				noise_ceiling_upper[i,c,t] = corr(np.mean(y_test, 1)[:,c,t],
					bio_data_avg_half_1[:,c,t])[0]
	# Average the results across iterations, EEG channels and time points
	# correlation = np.mean(correlation)
	# noise_ceiling = np.mean(noise_ceiling)

	### Output ###
	return correlation, noise_ceiling_lower, noise_ceiling_upper


def save_data(args, correlation, noise_ceiling_lower, noise_ceiling_upper):
	"""Save the results.

	Parameters
	----------
	args : Namespace
		Input arguments.
	correlation : dict of float
		Correlation results.
	noise_ceiling : float
		Noise ceiling results.

	"""

	import numpy as np
	import os

	### Store the results into a dictionary ###
	results_dict = {
		'correlation': correlation,
		'noise_ceiling_lower': noise_ceiling_lower,
		'noise_ceiling_upper': noise_ceiling_upper
	}

	### Save the results ###
	# Save directories
	save_dir = os.path.join(args.save_dir, args.experiment, 'sub-'+
		format(args.sub,'02'), 'dnn-'+args.dnn)
	file_name = 'img_cond-'+\
		format(args.n_img_cond,'06')+'_n_eeg_rep-'+format(args.n_eeg_rep,'02') + f"_seed-{args.seed}" + '.npy'
	# Create the directory if not existing and save the data
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	np.save(os.path.join(save_dir, file_name), results_dict)