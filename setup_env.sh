#!/bin/bash
set -e

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda."
    exit 1
fi

# Create a new conda environment with Python 3.10
conda create -n alignvis -y python=3.10

# Initialize conda for the current shell session
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate alignvis

# Install pip using conda
conda install -y pip

# Install PyTorch and related packages from pytorch and nvidia channels
conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# Install transformers from conda-forge
conda install -y -c conda-forge transformers==4.40.2

# Install scipy
conda install -y scipy

# Install scikit-learn from conda-forge
conda install -y -c conda-forge scikit-learn

# Install torchmetrics from conda-forge
conda install -y -c conda-forge torchmetrics

# Install matplotlib
conda install -y matplotlib

# Install wandb from conda-forge
conda install -y -c conda-forge wandb==0.16.6

# Install pandas from the anaconda channel
conda install -y -c anaconda pandas

# Install seaborn
conda install -y seaborn

# Install einops from conda-forge
conda install -y -c conda-forge einops

# Install timm from conda-forge
conda install -y -c conda-forge timm

# Install tensorboardx from conda-forge
conda install -y -c conda-forge tensorboardx

# Install rsatoolbox from conda-forge
conda install -y -c conda-forge rsatoolbox

# Install additional packages via pip
pip install mne
pip install mat73
pip install pyhealth
pip install braindecode
pip install moabb
pip install reformer-pytorch

echo "Conda environment 'eeg2image' setup complete."
