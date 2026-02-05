from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="siri-public/pi-0.5-dataset-lerobot",
    repo_type="dataset",
    local_dir="/Users/Pi05Model/pi0.5-dataset-lerobot",
    local_dir_use_symlinks=False,
    resume_download=True,
)