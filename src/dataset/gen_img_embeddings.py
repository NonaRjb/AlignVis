import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/alignvis")
sys.path.append("/proj/rep-learning-robotics/users/x_nonra/dreamsim")

import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import pickle
import argparse

from dreamsim import dreamsim, PerceptualModel
from dreamsim.config import dreamsim_args

def get_embeddings(encoder, name, images, root):
  embeddings = []
  for img in tqdm(images):
    img = preprocess(img).to(device)
    embeddings.append(encoder.embed(img).detach().cpu())
  with open(os.path.join(root, f"{name}_embeds.pkl"), "wb") as f:
    pickle.dump(embeddings, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Image Embeddings")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--save_path", type=str, default=None, help="Path to the output directory")
    parser.add_argument("--model_type", type=str, default="clip_vitb32", help="Type of model to use")
    parser.add_argument("--human_aligned", action="store_true", help="Use human aligned model")
    parser.add_argument("--model_name", type=str, default="dreamsim", help="Name of the method")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = dreamsim(pretrained=True, device=device, dreamsim_type=args.model_type, normalize_embeds=False)
    
    if not args.human_aligned:
        model_list = dreamsim_args['model_config'][args.model_type]['model_type'].split(",")
        model = PerceptualModel(**dreamsim_args['model_config'][args.model_type], device=device, load_dir="./models",
                                normalize_embeds=False, baseline=True)

    data_path = args.data_path
    save_path = data_path if args.save_path is None else args.save_path
    train_dir = 'images_meg'
    os.makedirs(os.path.join(save_path, train_dir), exist_ok=True)
    train_img_parent_dir = os.path.join(data_path, train_dir)
    train_concepts = os.listdir(train_img_parent_dir)
    train_concepts.sort()
    train_embeddings = []
    for concept in train_concepts:
        train_img_files = os.listdir(os.path.join(train_img_parent_dir, concept))
        train_img_files.sort()
        for i, item in enumerate(train_img_files):
            img_file = os.path.join(train_img_parent_dir, concept, item)
            img = Image.open(img_file).convert('RGB')
            img = preprocess(img).to(device)

            with torch.no_grad():
                e = model.embed(img).detach().cpu().numpy()

            # train_embeddings.append(e)
            os.makedirs(os.path.join(save_path, train_dir, concept), exist_ok=True)
            if args.human_aligned:
                np.save(os.path.join(save_path, train_dir, concept, item.replace(".jpg", f"_{args.model_name}_{args.model_type}.npy")), e)
            else:
                np.save(os.path.join(save_path, train_dir, concept, item.replace(".jpg", f"_{args.model_name}_{args.model_type}_noalign.npy")), e)
            if i % 1000 == 0:
                print(f"{item} items out of 16540 done")
                print(f"e.shape = {e.shape}")
    
    print("Start Embedidng Test Images")
    test_dir = 'images_test_meg'
    os.makedirs(os.path.join(save_path, test_dir), exist_ok=True)
    test_img_parent_dir = os.path.join(data_path, test_dir)
    test_img_files = os.listdir(test_img_parent_dir)
    test_img_files.sort()
    test_embeddings = []
    for item in test_img_files:
        img_file = os.path.join(test_img_parent_dir, item)
        img = Image.open(img_file).convert('RGB')
        img = preprocess(img).to(device)

        with torch.no_grad():
            e = model.embed(img).detach().cpu().numpy()

        # test_embeddings.append(e)
        if args.human_aligned:
            np.save(os.path.join(save_path, test_dir, item.replace(".jpg", f"_{args.model_name}_{args.model_type}.npy")), e)
        else:
            np.save(os.path.join(save_path, test_dir, item.replace(".jpg", f"_{args.model_name}_{args.model_type}_noalign.npy")), e)
    