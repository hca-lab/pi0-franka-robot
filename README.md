# Pi0 Real World Experiment

This code is to validate the pi0 pipeline in the real world experiment, which built based on [pi0](https://github.com/Physical-Intelligence/openpi.git).
We have the VLA runner site and Franka controller site, VLA site recevies the image and promt into VLA model and send the predicted action to the Franka site.


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
pip install uv
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


# Checkpoints Download
Please download the pi0 release checkpoints.

```bash
# run download
cd openpi_franka
python download.py
```

Here we use pi0_fast_droid. More checkpoints can be checked from [pi0](https://github.com/Physical-Intelligence/openpi?tab=readme-ov-file#fine-tuned-models).


# Real-World Experiment

Please run the Franka site and then run the VLA site.

- **Run Franka Site**


```bash
# run test_pi.py file
python test_pi.py
```

```bash
# To quit the control, press 'CTR'+'C'.
```

Please change the IP address based on your setting and adjust the scale parameter on action to avoid large movement.

- **Run VLA Site**

```bash
# run inference_pi.py file
python inference_pi.py 
```
Please change the IP address based on your setting and the checkpoints path. We use two RealSense cameras for wrist and front views. Please ensure that the image inputs to the VLA model are correctly assigned.

```bash
# To quit the control, press 'q'.
```


