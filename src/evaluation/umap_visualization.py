import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/alignvis")

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import umap
import numpy as np
import pandas as pd
import torch
import argparse
import os

from src.models.brain_encoder import BrainEncoder
from src.train_brain_clip import model_configs


def load_category_mapping(csv_path):
    """
    Load the category mapping from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        dict: Mapping from image category labels to one-hot encoded vectors.
    """
    df = pd.read_csv(csv_path)
    category_names = df.columns[1:-1].tolist()
    # Exclude first and last columns
    one_hot_columns = df.iloc[:, 1:-1]
    # Filter rows that are one-hot encoded
    is_one_hot = (one_hot_columns.sum(axis=1) == 1) & (one_hot_columns.isin([0, 1]).all(axis=1))

    # Filter rows where the relevant columns are not one-hot
    df = df[is_one_hot]

    # Create a mapping of labels to one-hot encodings
    label_to_one_hot = {
        row['concept']: row.iloc[1:-1].values.flatten() for _, row in df.iterrows()
    }

    return label_to_one_hot, category_names


def load_brain_encoder(model_path, backbone_encoder, embedding_size, device='cuda'):
    brain_encoder = BrainEncoder(
            embed_dim=embedding_size,
            backbone=backbone_encoder,
            n_channels=63,  # things_eeg_2 dataset
            n_samples=251,
            n_classes=1654,
            model_path=None,
            device=device, 
            **model_configs[backbone_encoder]
            )
    brain_encoder = brain_encoder.float()
    brain_encoder.to(device)
    if model_path:
        checkpoint = torch.load(model_path)['model_state_dict']
        brain_encoder.load_state_dict(checkpoint, strict=False)
        brain_encoder.to(device)
    return brain_encoder



def filter_and_update_labels(labels, mapping):
    """
    Filter labels based on a mapping and update them to one-hot encodings.

    Args:
        labels (list): List of image category labels.
        mapping (dict): Mapping from labels to one-hot encodings.

    Returns:
        tuple: Updated list of one-hot encoded labels and indices of matching labels.
    """
    filtered_labels = []
    matching_indices = []
    for i, label in enumerate(labels):
        if label in mapping:
            filtered_labels.append(mapping[label])
            matching_indices.append(i)
    return filtered_labels, matching_indices


def load_things_eeg_2(data_path, subject_id, img_encoder_aligned, img_encoder_noalign, category_mapping, split='train', normalize=True):
    """
    Load and preprocess the EEG and image data, filter by category mapping.

    Args:
        data_path (str): Path to the dataset.
        subject_id (int): Subject ID.
        img_encoder (str): Image encoder identifier.
        category_mapping (dict): Mapping from labels to one-hot encodings.
        split (str): Data split ('train' or 'test').
        normalize (bool): Whether to normalize EEG data.

    Returns:
        tuple: Filtered EEG data, image embeddings, one-hot encoded labels, and metadata.
    """
    data_path = os.path.join(data_path, 'things_eeg_2')

    img_parent_dir = os.path.join(data_path, 'image_embeddings')
    img_metadata = np.load(os.path.join(img_parent_dir, 'image_metadata.npy'),
                           allow_pickle=True).item()
    img_concepts = img_metadata['test_img_concepts'] if split == 'test' else img_metadata['train_img_concepts']
    img_files = img_metadata['test_img_files'] if split == 'test' else img_metadata['train_img_files']

    eeg_parent_dir = os.path.join(data_path, 'Preprocessed_data_250Hz', f'sub-{subject_id:02d}')
    eeg_data = np.load(os.path.join(eeg_parent_dir,
                                    'preprocessed_eeg_training.npy' if split == 'train' else 'preprocessed_eeg_test.npy'),
                       allow_pickle=True)
    subject_eeg_data = eeg_data['preprocessed_eeg_data']
    tmp_labels = img_concepts
    labels_int = [int(l.split("_")[0]) - 1 for l in tmp_labels]
    labels_str = [l.split("_", 1)[1] for l in tmp_labels]

    # Apply label filtering
    filtered_labels, matching_indices = filter_and_update_labels(labels_str, category_mapping)

    # Filter EEG and image data
    subject_eeg_data = np.mean(subject_eeg_data, axis=1).squeeze()  # Average across repetitions
    subject_eeg_data = subject_eeg_data[matching_indices]
    filtered_images_aligned = []
    filtered_images_noalign = []
    for i in matching_indices:
        img_file = os.path.join(img_parent_dir, 'training_images' if split == 'train' else 'test_images',
                                img_concepts[i], img_files[i].replace(".jpg", f"_{img_encoder_aligned}.npy"))
        img = np.load(img_file)
        filtered_images_aligned.append(img)
        img_file = os.path.join(img_parent_dir, 'training_images' if split == 'train' else 'test_images',
                                img_concepts[i], img_files[i].replace(".jpg", f"_{img_encoder_noalign}.npy"))
        img = np.load(img_file)
        filtered_images_noalign.append(img)
    filtered_images_aligned = np.array(filtered_images_aligned).squeeze()
    filtered_images_noalign = np.array(filtered_images_noalign).squeeze()

    subject_eeg_data = np.expand_dims(subject_eeg_data, axis=1)

    print(f"Filtered EEG data shape = {subject_eeg_data.shape}")
    print(f"Filtered image data shape = {filtered_images_aligned.shape}")
    print(f"Filtered labels count = {len(filtered_labels)}")

    if normalize:
        subject_eeg_data = (subject_eeg_data - np.mean(subject_eeg_data, axis=-1, keepdims=True)) / np.linalg.norm(
            subject_eeg_data, axis=-1, keepdims=True)

    return subject_eeg_data, filtered_images_aligned, filtered_images_noalign, filtered_labels, matching_indices


def get_embeddings(model, data, save=False, save_path=None, device='cuda'):
    
    embeddings = None
    if model is not None:
        model.eval()
    with torch.no_grad():
        for i, x in enumerate(data):
            x = torch.from_numpy(x).unsqueeze(dim=0).float()
            x = x.to(device)
            if model is not None:
                e = model(x)
            else:
                e = x
            e = e - torch.mean(e, dim=-1, keepdim=True)
            e = torch.nn.functional.normalize(e, p=2, dim=-1)
            if embeddings is None:
                embeddings = e.detach().cpu().numpy()
            else:
                embeddings = np.concatenate((embeddings, e.detach().cpu().numpy()), axis=0)
    if save:
        print("Saving the Embeddings")
        if save_path:
            np.save(save_path, embeddings)
        else:
            np.save("./embeddings.npy", embeddings)
    print(embeddings.shape)
    return embeddings


def visualize_embeddings(embeddings, labels_one_hot, category_names, title="UMAP Visualization", save=False, save_path=None):
    """
    Visualize embeddings using UMAP, assigning category names to unique colors.

    Args:
        embeddings (np.ndarray): Array of embeddings (shape: [n_samples, n_features]).
        labels_one_hot (list): List of one-hot encoded labels.
        category_names (list): List of category names corresponding to the one-hot labels.
        title (str): Title for the plot.
        save (bool): Whether to save the plot.
        save_path (str): Path to save the plot.

    Returns:
        None
    """
    
    # Convert one-hot labels to integer labels
    labels = np.argmax(labels_one_hot, axis=1)

    # Reduce dimensions with UMAP
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.25, metric='euclidean', random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    # Assign a unique color to each category
    unique_labels = np.unique(labels)
    cmap = cm.get_cmap('tab20', len(unique_labels))
    color_mapping = {label: cmap(i) for i, label in enumerate(unique_labels)}

    # Plot embeddings with category-specific colors
    plt.figure(figsize=(12, 10))
    for label in unique_labels:
        indices = labels == label
        plt.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            color=color_mapping[label],
            label=category_names[label],
            s=40,
            alpha=0.8
        )

    # Add legend for category names
    plt.legend(title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)

    # Save the plot if needed
    if save:
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.savefig("./umap_visualization_with_categories.png", bbox_inches="tight")
    plt.show()


def visualize_model_comparison(embeddings_a, embeddings_b, labels, category_names, title="Model Comparison with UMAP", save=False, save_path=None):
    """
    Compare embeddings from two models using UMAP.

    Args:
        embeddings_a (np.ndarray): Embeddings from Model A (shape: [n_samples, n_features]).
        embeddings_b (np.ndarray): Embeddings from Model B (shape: [n_samples, n_features]).
        labels (list): Labels for the data (e.g., one-hot encoded or categorical).
        category_names (list): Category names corresponding to labels.
        title (str): Title for the plot.
        save (bool): Whether to save the plot.
        save_path (str): Path to save the plot.

    Returns:
        None
    """
    # Normalize embeddings
    embeddings_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    embeddings_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    # Combine embeddings and assign model labels
    combined_embeddings = np.vstack([embeddings_a, embeddings_b])
    model_labels = np.array(["Model A"] * len(embeddings_a) + ["Model B"] * len(embeddings_b))

    # Reduce dimensions with UMAP
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='euclidean', random_state=42)
    embeddings_2d = reducer.fit_transform(combined_embeddings)

    # Separate embeddings for visualization
    embeddings_a_2d = embeddings_2d[:len(embeddings_a)]
    embeddings_b_2d = embeddings_2d[len(embeddings_a):]

    # Plot embeddings
    plt.figure(figsize=(12, 10))
    
    # Plot Model A
    scatter_a = plt.scatter(embeddings_a_2d[:, 0], embeddings_a_2d[:, 1],
                            c=np.argmax(labels, axis=1), cmap='tab20', marker='o', label="Aligned", alpha=0.8)

    # Plot Model B
    scatter_b = plt.scatter(embeddings_b_2d[:, 0], embeddings_b_2d[:, 1],
                            c=np.argmax(labels, axis=1), cmap='tab20', marker='^', label="Not-Aligned", alpha=0.8)

    # Add category legend
    handles_a, _ = scatter_a.legend_elements(prop="colors", alpha=0.8)
    category_legend = plt.legend(handles_a, category_names, title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add model legend
    plt.legend(["Aligned", "Non-aligned"], title="Models", loc="lower right", fontsize='small')
    plt.gca().add_artist(category_legend)

    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)

    if save:
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.savefig("./umap_model_comparison.png", bbox_inches="tight")
    plt.show()



def parse_args():
    parser = argparse.ArgumentParser(description='EEG Embeddings')
    parser.add_argument('--data_path', type=str, default='/data')
    parser.add_argument('--model_path_aligned', type=str, default=None)
    parser.add_argument('--model_path_noalign', type=str, default=None)
    parser.add_argument(
        '--category_mapping', 
        type=str, 
        default="/proj/rep-learning-robotics/users/x_nonra/eeg_asif_img/data/single_category_16.csv"
        )
    parser.add_argument('--subject_id', type=int, default=1)
    parser.add_argument('--brain_encoder', type=str, default='nice')
    parser.add_argument('--img_encoder_aligned', type=str, default='dreamsim_clip_vitb32')
    parser.add_argument('--img_encoder_noalign', type=str, default='clip_vitb32')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save_path', type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_size = model_configs[args.img_encoder_aligned]["embed_dim"]
    brain_encoder_aligned = load_brain_encoder(args.model_path_aligned, args.brain_encoder, embedding_size, device=device)
    brain_encoder_noalign = load_brain_encoder(args.model_path_noalign, args.brain_encoder, embedding_size, device=device)
    # Load category mapping
    category_mapping, category_names = load_category_mapping(args.category_mapping)
    
    brain_data, img_data_aligned, img_data_noalign, one_hot_labels, indices = load_things_eeg_2(
        data_path=args.data_path,
        subject_id=args.subject_id,
        img_encoder_aligned=args.img_encoder_aligned,
        img_encoder_noalign=args.img_encoder_noalign,
        category_mapping=category_mapping,
        split=args.split,
        normalize=args.normalize
    )

    brain_embeddings_aligned = get_embeddings(
        brain_encoder_aligned, 
        brain_data, 
        save=False, 
        save_path=None, 
        device=device)
    brain_embeddings_noalign = get_embeddings(
        brain_encoder_noalign, 
        brain_data, 
        save=False, 
        save_path=None, 
        device=device)

    figname = f"umap_visualization_{args.brain_encoder}_{args.img_encoder_aligned}_s{args.subject_id}_{args.split}.png"
    # visualize_embeddings(
    #     brain_embeddings, 
    #     one_hot_labels,
    #     category_names, 
    #     title="UMAP Visualization", 
    #     save=True, 
    #     save_path=os.path.join(args.save_path, figname))

    visualize_model_comparison(
        brain_embeddings_aligned, 
        img_data_aligned, 
        one_hot_labels, 
        category_names, 
        title="Model Comparison with UMAP", 
        save=True, 
        save_path=os.path.join(args.save_path, f"umap_model_comparison_{args.brain_encoder}_{args.img_encoder_aligned}_s{args.subject_id}_{args.split}.png"))

    visualize_model_comparison(
        brain_embeddings_noalign, 
        img_data_noalign, 
        one_hot_labels, 
        category_names, 
        title="Model Comparison with UMAP", 
        save=True, 
        save_path=os.path.join(args.save_path, f"umap_model_comparison_{args.brain_encoder}_{args.img_encoder_noalign}_s{args.subject_id}_{args.split}.png"))
    
    visualize_model_comparison(
        brain_embeddings_aligned, 
        brain_embeddings_noalign, 
        one_hot_labels, 
        category_names, 
        title="Model Comparison with UMAP", 
        save=True, 
        save_path=os.path.join(args.save_path, f"umap_model_comparison_{args.brain_encoder}_aligned_noalign_s{args.subject_id}_{args.split}.png"))
    
    visualize_model_comparison(
        img_data_aligned, 
        img_data_noalign, 
        one_hot_labels, 
        category_names, 
        title="Model Comparison with UMAP", 
        save=True, 
        save_path=os.path.join(args.save_path, f"umap_model_comparison_{args.img_encoder_aligned}_{args.img_encoder_noalign}_s{args.subject_id}_{args.split}.png"))


