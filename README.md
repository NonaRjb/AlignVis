# Human-Aligned Image Models Improve Visual Decoding from the Brain
### Official repository for the paper: [Human-Aligned Image Models Improve Visual Decoding from the Brain](https://openreview.net/pdf?id=i6uxIAAMje) (ICML 2025)
<div align="center">
  <img src="https://github.com/NonaRjb/AlignVis/blob/main/alignment_method_cr.png?raw=true" alt="alt text" width="800"/>
</div>

## Summary

Decoding visual images from brain activity has significant potential for advancing brain-computer interaction and enhancing the understanding of human perception. Recent approaches align the representation spaces of images and brain activity to enable visual decoding. In this paper, we introduce the use of human-aligned image encoders to map brain signals to images. We hypothesize that these models more effectively capture perceptual attributes associated with the rapid visual stimuli presentations commonly used in visual brain data recording experiments. Our empirical results support this hypothesis, demonstrating that this simple modification improves image retrieval accuracy by up to 21\% compared to state-of-the-art methods. Comprehensive experiments confirm consistent performance improvements across diverse EEG architectures, image encoders, alignment methods, participants, and brain imaging modalities.

## Setup
We have provided the `requirements.txt` file containing all packages required to run the code. To set up the environment, please run the following:

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
### fMRI
1. Download NSD image stimuli from [here](https://natural-scenes-dataset.s3.amazonaws.com/index.html#nsddata_stimuli/stimuli/nsd/) and place it under *NSD_fmri/images* folder
2. Download nsdgeneral files for the desired subjects from [here](https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main) and put them under *NSD_fmri/nsdgeneral* folder
3. Download *nsd_expdesign.mat* from [here](https://natural-scenes-dataset.s3.amazonaws.com/index.html#nsddata/experiments/nsd/) and place it under NSD_fmri folder. 
3. Run the following command for every subject. 
  ```bash
  python src/dataset/prepare_nsd.py --data_path path/to/NSD_fMRI/directory --sub subject_id
  ```

## Usage
### Training Brain Encoders
The main Python script to train the brain encoder using frozen image encoders and contrastive learning is provided by `src/train_brain_clip.py`.
### Testing the Performance with Pretrained Models
When you train the model with `train_brain_clip.py`, you can perform the image retrieval downstream task on the test set by setting `--downstream retrieval`. 
### Grad-CAM analysis
To generate the gradient maps of a trained brain encoder, you can run:
```bash
python src/evaluation/eeg_heatmap.py --data_path <path/to/the/data/directory> --save_path <path/to/the/output/directory> --model_path <path/to/the/checkpoints/directory> --img_encoder <image_encoder_name> --brain_encoder <brain_encoder_name> --subject_id <subject_id> --split <data_split> --seed <seed_number>
```
### Visualizing Top-K Retrieved Images
```bash
python src/evaluation/visualize_topk_images.py --data_path <path/to/the/data/directory> --save_path <path/to/the/output/directory> --model_path "$model_path" --img_encoder_aligned <human_aligned_image_encoder_name> --img_encoder_noalign <unaligned_image_encoder_name> --brain_encoder <brain_encoder_name> --subject_id <subject_id> --split <data_split> --seed <seed_number>
```
## Citation
```
@inproceedings{
rajabi2025humanaligned,
title={Human-Aligned Image Models Improve Visual Decoding from the Brain},
author={Nona Rajabi and Antonio H. Ribeiro and Miguel Vasco and Farzaneh Taleb and M{\r{a}}rten Bj{\"o}rkman and Danica Kragic},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=i6uxIAAMje}
}
``
