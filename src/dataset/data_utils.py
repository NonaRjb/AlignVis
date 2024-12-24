import torchvision.transforms as T

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return T.Compose([
        T.Resize(n_px, interpolation=BICUBIC),
        T.CenterCrop(n_px),
        _convert_image_to_rgb,
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

# Function to unzip any zip file in a directory
def unzip_nested_files(root_dir):
    # Iterate over all files and directories within the root directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Iterate over each file in the directory
        for filename in filenames:
            # Check if the file is a zip file
            if filename.endswith('.zip'):
                zip_file_path = os.path.join(dirpath, filename)
                # Unzip the file into the same directory
                print(f"Unzipping {zip_file_path}...")
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    # Extract all the contents to the directory where the zip file is located
                    zip_ref.extractall(dirpath)
                print(f"Unzipped {filename} in {dirpath}")
                os.remove(zip_file_path)


        # After unzipping, continue checking subdirectories
        for sub_dir in dirnames:
            # Join to get the full path of the subdirectory
            sub_dir_path = os.path.join(dirpath, sub_dir)
            # Recursively call the function on the subdirectory
            unzip_nested_files(sub_dir_path)