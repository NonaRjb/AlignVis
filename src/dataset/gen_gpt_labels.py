import numpy as np
import openai
import os
from PIL import Image
import base64
import json
import argparse
import httpx
import re

# Set your OpenAI API key
OPENAI_API_KEY = "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"

client = openai.OpenAI(api_key=OPENAI_API_KEY)
# client = openai.OpenAI(
#     base_url="https://aalto-openai-apigw.azure-api.net/v1/openai/gpt4-turbo/",
#     api_key=False,
#     default_headers = {
#         "Ocp-Apim-Subscription-Key": 'b8c5a710c27f4dc283b5708ab5b3df37',
#     },
    
# )

# Define labels
# LABELS = [
#     "pink", "red", "brown", "orange", "gold", "yellow", "green", "blue",
#     "purple", "gray", "white", "silver", "black", "colorful",
# ]

# Function to encode image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to classify an image
def label_image(image_path, new_labels, label_type):
    image_base64 = encode_image(image_path)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an image classifier. Label the image using predefined labels. Think of the most relevant label for the image describing its overall appearance."},
            {"role": "user", "content": [
                {"type": "text", "text": f"Label the {label_type.split('_')[0]} of the object in this image using the following predefined labels: {', '.join(new_labels)}. Respond only with the relevant labels."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}
        ],
        max_tokens=100
    )
    
    labels = response.choices[0].message.content
    # Try parsing as JSON list (preferred method)
    try:
        labels_list = json.loads(labels)
        if isinstance(labels_list, list):  # Ensure it is a valid list
            return labels_list
    except json.JSONDecodeError:
        pass  # Fall back to regex-based parsing

    # Fallback: Use regex to extract comma-separated labels
    labels_list = re.split(r',\s*', labels)  # Splits "gray, blue" into ["gray", "blue"]
    
    return [label.strip() for label in labels_list if label.strip() in new_labels]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, help="Full path to the root directory of the dataset")
    parser.add_argument("--output_path", type=str, help="Output JSON file")
    parser.add_argument("--label_path", type=str, help="Full path to the directory containing the label files")
    parser.add_argument("--label_type", type=str, default="color", help="Type of labels to use")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (train or test)")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path
    output_path = args.output_path
    split = args.split

    with open(os.path.join(args.label_path, f'{args.label_type}.txt'), "r") as f:
        LABELS = [line.strip() for line in f.readlines()]

    img_parent_dir = os.path.join(root_path, "things_eeg_2", "images")
    img_metadata = np.load(os.path.join(img_parent_dir, 'image_metadata.npy'), allow_pickle=True).item()
    img_concepts = img_metadata['test_img_concepts'] if split == 'test' else img_metadata['train_img_concepts']
    img_files = img_metadata['test_img_files'] if split == 'test' else img_metadata['train_img_files']
    n_images = 16540 if split == "train" else 200
    image_labels = {}
    for item in range(n_images):
        img_file = os.path.join(img_parent_dir, 'training_images' if split == 'train' else 'test_images', 
                        img_concepts[item], img_files[item])
        labels = label_image(img_file, LABELS, args.label_type)
        image_labels[img_file] = labels
        print(f"Labeled {img_file}: {labels}")

    with open(os.path.join(output_path, f"gpt_4o_mini_{args.label_type}_{args.split}.json"), "w") as f:
        json.dump(image_labels, f, indent=4)

    print(f"Labels saved to ", os.path.join(output_path, f"gpt_4o_mini_{args.label_type}_{args.split}.json"))
