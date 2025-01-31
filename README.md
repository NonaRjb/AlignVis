# Human-Aligned Image Models Improve Visual Decoding from the Brain

<div align="center">
  <img src="https://github.com/NonaRjb/AlignVis/blob/main/alignment_method.png?raw=true" alt="alt text" width="800"/>
</div>

## Summary

Decoding visual images from brain activity has significant potential for advancing brain-computer interaction and enhancing the understanding of human perception. Recent approaches align the representation spaces of images and brain activity to enable visual decoding. In this paper, we introduce the use of human-aligned image encoders to map brain signals to images. We hypothesize that these models more effectively capture perceptual attributes associated with the rapid visual stimuli presentations commonly used in visual brain data recording experiments. Our empirical results support this hypothesis, demonstrating that this simple modification improves image retrieval accuracy by up to 21\% compared to state-of-the-art methods. Comprehensive experiments confirm consistent performance improvements across diverse EEG architectures, image encoders, alignment methods, participants, and brain imaging modalities.

## Setup
We have provided the `requirements.txt` file containing all packages required to run the code. To setup the environment, please run the following:

```bash
conda create -n alignvis
conda activate alignvis
conda install pip
pip install -r requirements.txt
export PYTHONPATH="$PYTHONPATH:$(realpath ./AlignVis)"
```
## Dataset
We used publicly available EEG and MEG datasets. 
### EEG 
**Option 1**

1. Download raw EEG data from *Things EEG2*'s repository ([link](https://osf.io/3jk45/)). 
2. Run the following command for every subject.
   ```bash
   python src/dataset/things_eeg_preprocessing.py --sub <subject_id> --n_ses 4 --sfreq 250 --mvnn_dim epochs --project_dir <path/to/the/data/directory>
   ```
**Option 2**

In progress...

### MEG
1. Download preprocessed MEG data from *Things MEG*'s repository ([link](https://openneuro.org/datasets/ds004212)) (It's under derivatives directory).
2. Use things_meg_preprocess.ipynb to epoch the data (segment trials) and exclude test classes from the training.
## Usage
### Training Brain Encoders
The main Python script to train the brain encoder using frozen image encoders and contrastive learning is provided by `src/train_brain_clip.py`.
### Testing the Performance with Pretrained Models
### Grad-CAM analysis
### Visualizing Top-K Retrieved Images

