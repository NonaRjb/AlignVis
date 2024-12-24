import os
import shutil

# Paths to the original directory and the new directory for .npy files
original_dir = "/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_2/images"
new_dir = "/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_2/image_embeddings"

# Traverse the original directory
for root, dirs, files in os.walk(original_dir):
    # Determine the relative path to replicate the structure
    relative_path = os.path.relpath(root, original_dir)
    new_subdir = os.path.join(new_dir, relative_path)
    
    # Create the corresponding directory in the new location
    os.makedirs(new_subdir, exist_ok=True)
    
    # Iterate through the files in the current directory
    for file in files:
        file_path = os.path.join(root, file)
        if file.endswith(".npy"):
            # Move the .npy file to the new directory
            shutil.move(file_path, os.path.join(new_subdir, file))
        elif file.endswith(".jpg"):
            # Leave .jpg files in place (do nothing)
            continue
