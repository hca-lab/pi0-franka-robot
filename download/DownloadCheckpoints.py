from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="siri-public/pi-0.5-finetuned",
    revision="main",  # or an actual branch/tag from list_repo_refs
    local_dir="/Users/Pi05Model",
    allow_patterns=["pi0.5_two_task_310/29999/**"],
)