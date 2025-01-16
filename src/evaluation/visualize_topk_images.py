import sys
sys.path.append("/proj/rep-learning-robotics/users/x_nonra/alignvis")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import re

import src.utils as utils
from src.utils import get_embeddings
from src.models.brain_encoder import BrainEncoder
from src.train_brain_clip import model_configs


model_name_mapping = {
    'dreamsim_clip_vitb32': 'dreamsim_CLIP_ViT-B32',
    'CLIP_ViT-B32_noalign': 'CLIP_ViT-B32_noalign',
    'dreamsim_synclr_vitb16': 'dreamsim_synclr_vitb16',
    'dreamsim_synclr_vitb16_noalign': 'dreamsim_synclr_vitb16_noalign',
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize top-k images")
    parser.add_argument(
        "--k", type=int, default=5, help="Number of top-k images to visualize"
    )
    parser.add_argument('--data_path', type=str, default='/data')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--subject_id', type=int, default=1)
    parser.add_argument('--brain_encoder', type=str, default='nice')
    parser.add_argument('--img_encoder_aligned', type=str, default='dreamsim_clip_vitb32')
    parser.add_argument('--img_encoder_noalign', type=str, default='clip_vitb32')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='things-eeg-2')
    return parser.parse_args()


def load_checkpoint(directory, brain_encoder, dataset_name, seed_n):
    # Create a regex pattern to match the filename format
    pattern = rf"{re.escape(brain_encoder)}_{re.escape(dataset_name)}_seed{re.escape(str(seed_n))}_\d+\.pth"
    
    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        if re.fullmatch(pattern, filename):
            checkpoint_path = os.path.join(directory, filename)
            print(f"Loading checkpoint: {checkpoint_path}")
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path)['model_state_dict']
            return checkpoint
    
    # Raise an error if no matching checkpoint is found
    raise FileNotFoundError(f"No checkpoint found matching the pattern {pattern} in directory {directory}")


def load_brain_encoder(model_path, backbone_encoder, embedding_size, device='cuda', **kwargs):
    brain_encoder = BrainEncoder(
            embed_dim=embedding_size,
            backbone=backbone_encoder,
            n_channels=kwargs['n_channels'], 
            n_samples=kwargs['n_samples'],
            n_classes=kwargs['n_classes'],
            model_path=None,
            device=device, 
            **model_configs[backbone_encoder]
            )
    brain_encoder = brain_encoder.float()
    brain_encoder.to(device)
    if model_path:
        checkpoint = load_checkpoint(model_path, backbone_encoder, kwargs['dataset_name'], kwargs['seed'])
        # checkpoint = torch.load(model_path)['model_state_dict']
        brain_encoder.load_state_dict(checkpoint, strict=False)
        brain_encoder.to(device)
    return brain_encoder


def load_data(dataset_name, subject_id, data_path, img_encoder, seed_val=42, split='test', device_type='cuda'):
    data, ds_configs = utils.load_dataset(
        dataset_name=dataset_name, 
        data_path=data_path, 
        sid=subject_id, 
        load_img="embedding", 
        return_subject_id=False,
        split=split,
        select_channels=None, 
        subj_training_ratio=1.0, 
        img_encoder=img_encoder,
        interpolate=None,
        )
    g = torch.Generator().manual_seed(seed_val)
    data_loader = DataLoader(data, batch_size=1, shuffle=False,
                            drop_last=False,
                            pin_memory=True if 'cuda' in device_type else False,
                            generator=g)
    return data_loader, ds_configs


def retrieve_images(eeg_encoder, img_encoder, data_loader, device="cuda:0", return_subject_id=False, **kwargs):

    eeg_encoder.eval()

    if img_encoder is not None:
        img_encoder.eval()

    img_embeddings, _ = get_embeddings(img_encoder, data_loader, modality="img", return_subject_id=return_subject_id, device=device)
    img_embeddings = torch.from_numpy(img_embeddings).to(device)

    total = 0
    top1 = 0
    top3 = 0
    top5 = 0

    top5_labels = []
    top5_sim_values = []

    with torch.no_grad():
        for i, (data, y) in enumerate(data_loader):
            if return_subject_id:
                subject_id = data[1]
                subject_id = subject_id.to(device, non_blocking=True)
                data = data[0]
            eeg, img = data
            eeg = eeg.to(device, non_blocking=True)
            img = img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if img_encoder is not None:
                img_embeddings_batch = img_encoder(img)
            else:
                img_embeddings_batch = img
            img_embeddings_batch = img_embeddings_batch - torch.mean(img_embeddings_batch, dim=-1, keepdim=True)
            img_embeddings_batch = F.normalize(img_embeddings_batch, p=2, dim=-1)
            sim_img = (img_embeddings_batch @ img_embeddings.t())
            _, tt_label = sim_img.topk(1)
            if return_subject_id:
                eeg_embeddings = eeg_encoder(eeg, subject_id)
            else:
                eeg_embeddings = eeg_encoder(eeg)
            eeg_embeddings = eeg_embeddings - torch.mean(eeg_embeddings, dim=-1, keepdim=True)
            eeg_embeddings = F.normalize(eeg_embeddings, p=2, dim=-1)

            similarity = (eeg_embeddings @ img_embeddings.t())
            similarity_values, indices = similarity.topk(5)
            tt_label = tt_label.view(-1, 1)

            total += y.size(0)
            top1 += (tt_label == indices[:, :1]).sum().item()
            top3 += (tt_label == indices[:, :3]).sum().item()
            top5 += (tt_label == indices).sum().item()

            top5_labels.extend(indices[:, :5].tolist())
            top5_sim_values.extend(similarity_values[:, :5].tolist())

        top1_acc = float(top1) / float(total)
        top3_acc = float(top3) / float(total)
        top5_acc = float(top5) / float(total)

    print('The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc, top3_acc, top5_acc))

    return top1_acc, top3_acc, top5_acc, top5_labels, top5_sim_values


def visualize_things_eeg_images(query_label, retrieved_labels_a, retrieved_labels_b, retrieved_vals_a, retrieved_vals_b, data_path, save_path=None):
    """
    Visualizes the query image and retrieved images from two models in a single plot for comparison.

    Parameters:
        query_label (int): The label of the query image.
        retrieved_labels_a (list of int): The labels of the retrieved images from model A.
        retrieved_labels_b (list of int): The labels of the retrieved images from model B.
        data_path (str): Path to the dataset containing image folders.
        save_path (str, optional): Path to save the generated visualization. Default is None.
    """
    def get_image_path(label):
        # Find the folder corresponding to the label by splitting at the first underscore
        for folder in os.listdir(data_path):
            folder_id = int(folder.split('_', 1)[0]) - 1
            if folder_id == label:
                folder_path = os.path.join(data_path, folder)
                images = os.listdir(folder_path)
                if images:
                    return os.path.join(folder_path, images[0])
        return None

    # Get the query image path
    query_image_path = get_image_path(query_label)
    if query_image_path is None:
        raise FileNotFoundError(f"Query image for label {query_label} not found.")

    # Get the retrieved images paths for model A
    retrieved_image_paths_a = []
    for label in retrieved_labels_a:
        image_path = get_image_path(label)
        if image_path is not None:
            retrieved_image_paths_a.append(image_path)
        else:
            print(f"Warning: Retrieved image for label {label} not found (Model A).")

    # Get the retrieved images paths for model B
    retrieved_image_paths_b = []
    for label in retrieved_labels_b:
        image_path = get_image_path(label)
        if image_path is not None:
            retrieved_image_paths_b.append(image_path)
        else:
            print(f"Warning: Retrieved image for label {label} not found (Model B).")

    # Load the query image
    query_image = Image.open(query_image_path)

    # Load the retrieved images for model A and model B
    retrieved_images_a = [Image.open(path) for path in retrieved_image_paths_a]
    retrieved_images_b = [Image.open(path) for path in retrieved_image_paths_b]

    # Set up the plot
    num_retrieved = max(len(retrieved_images_a), len(retrieved_images_b))
    total_columns = num_retrieved + 1

    fig, axes = plt.subplots(2, total_columns, figsize=(5 * total_columns, 12), gridspec_kw={"width_ratios": [1] + [1] * num_retrieved})

    # Merge the first column to display the query image in the middle
    query_ax = fig.add_subplot(1, total_columns, 1)
    query_ax.imshow(query_image)
    query_ax.axis("off")
    query_ax.set_title("Query Image", fontsize=16)

    # Disable the original first column axes
    axes[0, 0].remove()
    axes[1, 0].remove()

    # Plot the retrieved images for model A
    for i in range(num_retrieved):
        if i < len(retrieved_images_a):
            axes[0, i + 1].imshow(retrieved_images_a[i])
            axes[0, i + 1].axis("off")
            axes[0, i + 1].set_title(f"Dreamsim - {retrieved_vals_a[i]:.2f}", fontsize=16)
        else:
            axes[0, i + 1].axis("off")

    # Plot the retrieved images for model B
    for i in range(num_retrieved):
        if i < len(retrieved_images_b):
            axes[1, i + 1].imshow(retrieved_images_b[i])
            axes[1, i + 1].axis("off")
            axes[1, i + 1].set_title(f"Base Model - {retrieved_vals_b[i]:.2f}", fontsize=16)
        else:
            axes[1, i + 1].axis("off")

    # Adjust spacing
    plt.tight_layout()

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    plt.close()
    # Show the plot
    # plt.show()


if __name__ == "__main__":
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    save_dir = f"topk_{args.img_encoder_aligned}_s{args.subject_id}_seed{args.seed}"
    save_path = os.path.join(args.save_path, save_dir)
    os.makedirs(save_path, exist_ok=True)

    model_path_aligned = os.path.join(args.model_path, model_name_mapping[args.img_encoder_aligned], f"sub-{args.subject_id:02}", "models")
    model_path_noalign = os.path.join(args.model_path, model_name_mapping[args.img_encoder_noalign], f"sub-{args.subject_id:02}", "models")

    data_loader_aligned, ds_configs = load_data(
        dataset_name=args.dataset, 
        subject_id=args.subject_id, 
        data_path=args.data_path, 
        img_encoder=args.img_encoder_aligned, 
        split=args.split,
        device_type=device)
    
    data_loader_noalign, _ = load_data(
        dataset_name=args.dataset, 
        subject_id=args.subject_id, 
        data_path=args.data_path, 
        img_encoder=args.img_encoder_noalign, 
        split=args.split,
        device_type=device)

    brain_encoder_aligned = load_brain_encoder(
        model_path_aligned,
        args.brain_encoder,
        embedding_size=model_configs[args.img_encoder_aligned]['embed_dim'],
        dataset_name=args.dataset,
        seed=args.seed,
        device=device,
        **ds_configs)

    brain_encoder_noalign = load_brain_encoder(
        model_path_noalign,
        args.brain_encoder,
        embedding_size=model_configs[args.img_encoder_noalign]['embed_dim'],
        dataset_name=args.dataset,
        seed=args.seed,
        device=device,
        **ds_configs)

    print("Aligned model")
    _, _, _, top5_aligned, top5_simvals_aligned = retrieve_images(
        brain_encoder_aligned, 
        None, 
        data_loader_aligned, 
        device=device)

    print("No align model")
    _, _, _, top5_noalign, top5_simvals_noalign = retrieve_images(
        brain_encoder_noalign, 
        None, 
        data_loader_noalign, 
        device=device)

    
    for query, (retrieved_aligned, retrieved_noalign) in enumerate(zip(top5_aligned, top5_noalign)):
        query_label = query
        visualize_things_eeg_images(
            query_label, 
            retrieved_aligned, 
            retrieved_noalign,
            top5_simvals_aligned[query],
            top5_simvals_noalign[query],
            os.path.join(args.data_path, "things_eeg_2", 'images', 'test_images'), 
            save_path=os.path.join(save_path, f"query_{query}_retrieved.png"))
    
    print("Done")






    

    

