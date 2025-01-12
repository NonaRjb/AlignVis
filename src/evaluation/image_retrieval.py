import numpy as np
import argparse
import os


def load_all_embeddings(root_path, dataset, img_encoder, split="train"):
    if dataset == "things-eeg-2":
        img_parent_dir = os.path.join(root_path, "things_eeg_2", "image_embeddings")
        img_metadata = np.load(os.path.join(img_parent_dir, 'image_metadata.npy'), allow_pickle=True).item()
        img_concepts = img_metadata['test_img_concepts'] if split == 'test' else img_metadata['train_img_concepts']
        img_files = img_metadata['test_img_files'] if split == 'test' else img_metadata['train_img_files']
        n_images = 16540 if split == "train" else 200
        labels = [int(l.split("_")[0])-1 for l in img_concepts]
        images = []
        for item in range(n_images):
            img_file = os.path.join(img_parent_dir, 'training_images' if split == 'train' else 'test_images', 
                            img_concepts[item], img_files[item].replace(".jpg", f"_{img_encoder}.npy"))
            img = np.load(img_file)
            images.append(img.squeeze())
        images = np.array(images)
    elif dataset == "things-meg":
        labels = [] # TODO: Add MEG labels
        img_parent_dir = os.path.join(root_path, "things_meg", "images")
        split_dir = "images_meg" if split == "train" else "images_test_meg"
        if split == "train":
            train_img_parent_dir = os.path.join(img_parent_dir, split_dir)
            train_concepts = os.listdir(train_img_parent_dir)
            train_concepts.sort()
            images = []
            for n, concept in enumerate(train_concepts):
                train_img_files = os.listdir(os.path.join(train_img_parent_dir, concept))
                train_img_files.sort()
                for i, item in enumerate(train_img_files):
                    img_file = os.path.join(img_parent_dir.replace("images", "image_embeddings"), split_dir, concept, item.replace(".jpg", f"_{img_encoder}.npy"))
                    img = np.load(img_file)
                    images.append(img)
        else:
            test_img_parent_dir = os.path.join(img_parent_dir, split_dir)
            test_img_files = os.listdir(test_img_parent_dir)
            test_img_files.sort()
            images = []
            for item in test_img_files:
                img_file = os.path.join(img_parent_dir.replace("images", "image_embeddings"), split_dir, item.replace(".jpg", f"_{img_encoder}.npy"))
                img = np.load(img_file)
                images.append(img)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")

    images = np.array(images)
    
    return images, labels


def compute_distance_matrix(images):
    n_images = images.shape[0]
    # norm_images = np.expand_dims(np.linalg.norm(images, axis=1), axis=1) @ np.expand_dims(np.linalg.norm(images, axis=1), axis=0)
    images = images / np.linalg.norm(images, axis=1, keepdims=True)
    similarity_matrix = (images @ images.T)
    return 1 - similarity_matrix


def retrieve_top_k_and_compute_recall(distance_matrix, labels, k):
    """
    For each query image, retrieves the top k images with the lowest distance,
    checks if the labels match the query image's label, and computes recall@k.

    Parameters:
        distance_matrix (numpy.ndarray): NxN matrix of distances between images.
        labels (list): List of labels corresponding to the images.
        k (int): Number of top images to retrieve.

    Returns:
        float: Recall@k value.
    """
    n_images = distance_matrix.shape[0]
    total_recall = 0
    recall_class = {}
    for i in range(n_images):
        distances = distance_matrix[i]
        sorted_indices = np.argsort(distances)  # Ascending order
        sorted_indices = sorted_indices[sorted_indices != i]  # Exclude self
        top_k_indices = sorted_indices[:k]
        top_k_labels = [labels[idx] for idx in top_k_indices]

        query_label = labels[i]
        relevant_count = top_k_labels.count(query_label)

        total_relevant = labels.count(query_label)-1 # Exclude self

        recall = relevant_count / min(k, total_relevant) if total_relevant > 0 else 0
        total_recall += recall
        if str(query_label) not in recall_class.keys():
            recall_class[str(query_label)] = [recall] 
        else:
            recall_class[str(query_label)].append(recall)

    # Average recall across all query images
    average_recall_at_k = total_recall / n_images
    return average_recall_at_k, recall_class


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image retrieval performance.")
    parser.add_argument("--root_path", type=str, help="Path to the root directory of the dataset.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset.")
    parser.add_argument("--img_encoder", type=str, help="Name of the image encoder.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--k", type=int, default=10, help="Number of top images to retrieve.")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    images, labels = load_all_embeddings(args.root_path, args.dataset, args.img_encoder, args.split)
    distance_matrix = compute_distance_matrix(images)
    recall_at_k, recall_at_k_per_class = retrieve_top_k_and_compute_recall(distance_matrix, labels, args.k)
    recall_at_k_per_class = {key: np.mean(val) for key, val in recall_at_k_per_class.items()}
    print(f"Recall@{args.k} = {recall_at_k:.4f}")