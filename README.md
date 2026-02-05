# Finetune pi0.5 with Real World Dataset

This code is to finetune the pi0.5 pipeline in the real world experiment, which built based on [pi0.5](https://github.com/Physical-Intelligence/openpi.git).
<!-- We have the VLA runner site and Franka controller site, VLA site recevies the image and promt into VLA model and send the predicted action to the Franka site. -->


# Installation
Our installation includes two parts.


- **VLA Runner Site**

Create an environment, clone the repo and install the required packages.

```bash
# Create environment
conda create --name openpi python=3.11

# Create environment
git clone https://github.com/hca-lab/pi0-franka-robot.git
cd openpi_franka
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```


- **Franka Controller Site**: 

Install franky

```bash
# Install the franky
pip install franky-control
```
Please move the folder 'FrankyControl' to the Franka controller site.


<!-- # Checkpoints Download
Please download the pi0.5 release checkpoints.

```bash
# run download
cd openpi_franka
python download.py
```

Here we use pi0_fast_droid. More checkpoints can be checked from [pi0](https://github.com/Physical-Intelligence/openpi?tab=readme-ov-file#fine-tuned-models). -->



# Dataset Download
Please download the real-world Franka robot dataset for two grasping tasks.

```bash
# run DownloadDataset
cd download
python DownloadDataset.py
```

# Finetune with Own Dataset
Please check and modify the policy config

```bash
# check config
cd src/openpi/training/config.py
# full finetune, use "pi05_full_finetune"
# lora finetune, use "pi05_lora_finetune"
# repo_id should be Pi0.5 Lerobot dataset path.
```

Then finetune Pi0.5 model by running

```bash
# full fine-tuning
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_full_finetune --exp-name=my_experiment –overwrite 
```

or 
```bash
# lora fine-tuning
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_lora_finetune --exp-name=my_experiment –overwrite 
```



# Model Checkpoints

We provide multiple pi0.5 model checkpoints and can be used for real-world testing.


All checkpoins are available [here](https://huggingface.co/siri-public/pi-0.5-finetuned/tree/main/pi0.5_two_task_310).

```bash
# download checkpoints
cd download
python DownloadCheckpoints.py
```



# Real-World Evaluation

Please run the VLA site and then run the Franka site.

- **Run VLA Site**


```bash
# Execute the policy trained via full fine-tuning
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config= pi05_full_finetune\
    --policy.dir=checkpoints/29999  # model checkpoint path
```

or

```bash
# Execute the policy trained via lora fine-tuning
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config= pi05_lora_finetune\
    --policy.dir=checkpoints/29999 # model checkpoint path
```

- **Run Franka Site**

```bash
# Control Franka robot
python FrankaInference_pi05.py 
```
Please change the SERVER_IP, BASE_CAMERA_SN, WRIST_CAMERA_SN based on your setting. We use two RealSense cameras for wrist and front views. Please ensure that the image inputs to the VLA model are correctly assigned.

