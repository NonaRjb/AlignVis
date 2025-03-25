import cv2
import numpy as np
import pywt
import os
import argparse
import itertools
import matplotlib.pyplot as plt

def compute_gabor_features(image, ksize=15, sigma=4.0, theta_values=[0, np.pi/4, np.pi/2, 3*np.pi/4], lambd=10.0, gamma=0.5, filename=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = cv2.resize(gray, (32, 32))
    responses = []
    for theta in theta_values:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        responses.append(filtered.flatten())
    return np.array(responses).flatten()

def compute_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def compute_wavelet_features(image, wavelet='haar', level=3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = cv2.resize(gray, (32, 32))
    coeffs = pywt.wavedec2(gray, wavelet, level=level)
    coeffs_flat = [coeffs[0]/np.abs(coeffs[0]).max()]
    coeffs_flat.extend(list(itertools.chain(*coeffs[1:])))
    coeffs_flat = [coeffs_flat[i].flatten() for i in range(len(coeffs_flat))]
    feature_vector = np.hstack(coeffs_flat)
    return feature_vector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, help="Full path to the root directory of the dataset")
    parser.add_argument("--output_path", type=str, help="Output JSON file")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (train or test)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_path = args.root_path
    output_path = args.output_path
    split = args.split
    img_parent_dir = os.path.join(root_path, "things_eeg_2", "images")
    img_metadata = np.load(os.path.join(img_parent_dir, 'image_metadata.npy'), allow_pickle=True).item()
    img_concepts = img_metadata['test_img_concepts'] if split == 'test' else img_metadata['train_img_concepts']
    img_files = img_metadata['test_img_files'] if split == 'test' else img_metadata['train_img_files']

    output_path = os.path.join(output_path, "things_eeg_2", "img_features", 'training_images' if split == 'train' else 'test_images')
    os.makedirs(output_path, exist_ok=True)

    n_images = 16540 if split == "train" else 200
    
    for item in range(n_images):
        img_file = os.path.join(img_parent_dir, 'training_images' if split == 'train' else 'test_images', img_concepts[item], img_files[item])
        image = cv2.imread(img_file)
        
        if image is None:
            print(f"Warning: Unable to load image {img_file}")
            continue
        
        gabor_features = compute_gabor_features(image, filename=img_files[item])
        color_hist_features = compute_color_histogram(image)
        wavelet_features = compute_wavelet_features(image, level=2)
        
        base_name = img_files[item].split('.')[0]
        base_path = os.path.join(output_path, img_concepts[item])
        os.makedirs(base_path, exist_ok=True)
        gabor_path = os.path.join(base_path, base_name + "_gabor.npy")
        hist_path = os.path.join(base_path, base_name + "_hist.npy")
        wavelet_path = os.path.join(base_path, base_name + "_wavelet.npy")
        
        np.save(gabor_path, gabor_features)
        np.save(hist_path, color_hist_features)
        np.save(wavelet_path, wavelet_features)
        
        print(f"Saved features for {img_file}")
