from PIL import Image
import requests
import argparse
import torch
import numpy as np
import json
import os

from transformers import CLIPProcessor, CLIPModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Full path to the root directory of the dataset")
    parser.add_argument("--save_path", type=str, help="Full path to the directory to save the output files")
    parser.add_argument("--label_path", type=str, help="Full path to the directory containing the label files")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (train or test)")
    parser.add_argument("--label_type", type=str, default="color")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    root_path = args.data_path
    split = args.split
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    with open(os.path.join(args.label_path, f'{args.label_type}.txt'), "r") as f:
        labels = [line.strip() for line in f.readlines()]
    # labels_new = [f"the object in the image is {label}" for label in labels]
    

    img_parent_dir = os.path.join(root_path, "things_eeg_2", "images")
    img_metadata = np.load(os.path.join(img_parent_dir, 'image_metadata.npy'), allow_pickle=True).item()
    img_concepts = img_metadata['test_img_concepts'] if split == 'test' else img_metadata['train_img_concepts']
    img_files = img_metadata['test_img_files'] if split == 'test' else img_metadata['train_img_files']
    n_images = 16540 if split == "train" else 200

    image_labels = {}
    for item in range(n_images):
        img_file = os.path.join(img_parent_dir, 'training_images' if split == 'train' else 'test_images', 
                        img_concepts[item], img_files[item])
                    
        category_name_parts = img_files[item].split("_")[:-1]  # Exclude the last part
        category_name = "_".join(category_name_parts)  # Rejoin to preserve multi-word names
        if args.label_type == "object_high":
            # Process category name with predefined classes using CLIP's text encoder
            image = Image.open(img_file)
            # Process predefined classes for image and category separately
            text_inputs = processor(text=labels, return_tensors="pt", padding=True)
            category_inputs = processor(text=[category_name], return_tensors="pt", padding=True)
            image_inputs = processor(images=image, return_tensors="pt", padding=True)

            # Encode text, category, and image separately
            text_features = model.get_text_features(**text_inputs)
            category_features = model.get_text_features(**category_inputs)
            image_features = model.get_image_features(**image_inputs)

            # Compute cosine similarity between image and predefined classes
            probs_image = torch.nn.functional.cosine_similarity(image_features, text_features)
            
            # Compute cosine similarity between category name and predefined classes
            probs_text = torch.nn.functional.cosine_similarity(category_features, text_features)

            # Normalize probabilities
            probs_image = probs_image.softmax(dim=0)
            probs_text = probs_text.softmax(dim=0)
            combined_probs = (probs_image + probs_text) / 2
            best_label = labels[combined_probs.argmax().item()]

            # inputs = processor(text=[category_name] + predefined_classes, return_tensors="pt", padding=True).to(device)
            # outputs = model.get_text_features(**inputs)
            # # Compute cosine similarity between category name and predefined classes
            # category_embedding = outputs[0]  # First element is the category name embedding
            # class_embeddings = outputs[1:]   # Remaining elements are the predefined classes

            # similarities = torch.nn.functional.cosine_similarity(category_embedding.unsqueeze(0), class_embeddings)
            # best_label = labels[similarities.argmax().item()]
        else:
        
            image = Image.open(img_file)
            labels_new = [f"this is a {label} {category_name}" for label in labels]
            inputs = processor(text=labels_new, images=image, return_tensors="pt", padding=True)

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            best_label = labels[probs.argmax().item()]
            
        image_labels[img_file] = best_label

        print(f"Labeled {img_file}: {best_label}")

    with open(os.path.join(args.save_path, f"clip_vitb32_{args.label_type}_{split}.json"), "w") as f:
        json.dump(image_labels, f, indent=4)
    
    print("Done")