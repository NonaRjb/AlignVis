import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/alignvis")

from torch.utils.data import Dataset
from PIL import Image
import scipy.io as spio
import numpy as np
import argparse
import h5py
import torch
import pickle
import json
import os

from src.dataset.data_utils import _transform


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare NSD dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--sub", type=int, required=True, help="Subject ID")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    data_path = os.path.join(args.data_path, "NSD_fmri")
    subject_id = args.sub

    save_path = os.path.join(data_path, 'processed_data', 'subj' + "{:02d}".format(subject_id))
    os.makedirs(save_path, exist_ok=True)

    dataset = h5py.File(os.path.join(data_path, "nsdgeneral", "subj" + "{:02d}".format(subject_id) + "_nsdgeneral.hdf5"), 'r')
    # images = dataset['images']
    voxels = dataset['voxels']
    f_stim = h5py.File(os.path.join(data_path, "images", "nsd_stimuli.hdf5"), 'r')
    stim = f_stim['imgBrick'][:]
    stim_order_f = os.path.join(data_path, "nsd_expdesign.mat")
    stim_order = loadmat(stim_order_f)
    ## Selecting ids for training and test data
    sig_train = {}
    sig_test = {}
    num_trials = 37*750
    for idx in range(num_trials):
        ''' nsdId as in design csv files'''
        nsdId = stim_order['subjectim'][subject_id-1, stim_order['masterordering'][idx] - 1] - 1
        if stim_order['masterordering'][idx]>1000:
            if nsdId not in sig_train:
                sig_train[nsdId] = []
            sig_train[nsdId].append(idx)
        else:
            if nsdId not in sig_test:
                sig_test[nsdId] = []
            sig_test[nsdId].append(idx)
    train_im_idx = list(sig_train.keys())
    test_im_idx = list(sig_test.keys())
    num_train, num_test = len(train_im_idx), len(test_im_idx)
    vox_dim, im_dim, im_c = voxels.shape[-1], 425, 3
    fmri_array = np.zeros((num_train,vox_dim))
    stim_array = np.zeros((num_train,im_dim,im_dim,im_c))
    for i,idx in enumerate(train_im_idx):
        stim_array[i] = stim[idx]
        fmri_array[i] = voxels[sorted(sig_train[idx])].mean(0)
        print(i)
    
    np.save(os.path.join(save_path, "nsd_train_fmri_sub" + "{:02d}".format(subject_id) + ".npy"), fmri_array)
    np.save(os.path.join(save_path ,"nsd_train_stim_sub" + "{:02d}".format(subject_id) + ".npy"), stim_array)
    print("Training data is saved.")

    fmri_array = np.zeros((num_test,vox_dim))
    stim_array = np.zeros((num_test,im_dim,im_dim,im_c))
    for i,idx in enumerate(test_im_idx):
        stim_array[i] = stim[idx]
        fmri_array[i] = voxels[sorted(sig_test[idx])].mean(0)
        print(i)
    np.save(os.path.join(save_path, "nsd_test_fmri_sub" + "{:02d}".format(subject_id) + ".npy"), fmri_array)
    np.save(os.path.join(save_path ,"nsd_test_stim_sub" + "{:02d}".format(subject_id) + ".npy"), stim_array)
    print("Test data is saved.")