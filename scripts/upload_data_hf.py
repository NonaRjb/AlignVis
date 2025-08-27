from huggingface_hub import upload_folder, upload_large_folder

# First: `huggingface-cli login` in a shell, or pass token=...
repo_id = "nonarjb/alignvis"  # must already exist or set create_pr=... below

upload_large_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_2",
    # path_in_repo="./things_eeg_2",           # or "data" if you want a subdir
    # commit_message="upload things_eeg_2 dataset",
    max_workers=1,          # ↓ key: avoid 429
    ignore_patterns=["*.tmp", "*.DS_Store"]  # add your own ignore rules
)
