# OccWorld Deployment Guide for Lambda Cloud

Deploy and train OccWorld on Lambda Labs GPU Cloud for Tokyo Gazebo dataset fine-tuning.

> **Note**: This project includes a standalone `train.py` that works out-of-the-box.
> For full OccWorld features (VQVAE, advanced architectures), see the OccWorld Integration section.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Lambda Cloud Setup](#2-lambda-cloud-setup)
3. [Instance Selection](#3-instance-selection)
4. [Environment Setup](#4-environment-setup)
5. [Data Transfer](#5-data-transfer)
6. [Training Execution](#6-training-execution)
7. [Monitoring & Checkpoints](#7-monitoring--checkpoints)
8. [Cost Optimization](#8-cost-optimization)
9. [Automated Deployment Script](#9-automated-deployment-script)

---

## 1. Overview

### Why Lambda Cloud?

| Feature | Benefit |
|---------|---------|
| GPU Availability | A100, H100, B200 GPUs |
| Pricing | $1.29-$4.99/hr per GPU |
| Billing | Per-minute billing |
| Storage | Persistent storage available |
| No Egress Fees | Free data download |
| Pre-installed | CUDA, PyTorch, common ML libs |

### Training Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1x A100 40GB | 1x A100 80GB or H100 |
| VRAM | 40GB | 80GB |
| RAM | 64GB | 128GB+ |
| Storage | 200GB | 500GB+ |
| Est. Time | ~24-48 hrs | ~12-24 hrs |

---

## 2. Lambda Cloud Setup

### 2.1 Create Account

1. Go to [Lambda Cloud](https://cloud.lambda.ai/)
2. Sign up with email or Google/GitHub
3. Add payment method (credit card required)
4. Verify account

### 2.2 Generate SSH Keys

**Option A: Use existing key**
```bash
# Display your public key
cat ~/.ssh/id_rsa.pub
```

**Option B: Generate new key**
```bash
# Generate new SSH key pair
ssh-keygen -t ed25519 -C "lambda-occworld" -f ~/.ssh/lambda_key

# Display public key
cat ~/.ssh/lambda_key.pub
```

### 2.3 Add SSH Key to Lambda

1. Go to [Lambda Cloud Console](https://cloud.lambda.ai/ssh-keys)
2. Click "Add SSH Key"
3. Paste your public key
4. Give it a name (e.g., "occworld-training")

### 2.4 Configure Firewall (Optional)

Default: Only port 22 (SSH) is open.

For TensorBoard access:
1. Go to [Firewall Settings](https://cloud.lambda.ai/firewall)
2. Add rule: TCP port 6006 (TensorBoard)
3. Add rule: TCP port 8888 (Jupyter, if needed)

---

## 3. Instance Selection

### 3.1 Recommended Instances

| Instance | GPU | VRAM | Price/hr | Best For |
|----------|-----|------|----------|----------|
| 1x A100 40GB | A100 | 40GB | $1.29 | Budget training |
| 1x A100 80GB | A100 | 80GB | $1.79 | **Recommended** |
| 1x H100 SXM | H100 | 80GB | $3.29 | Fast training |
| 8x A100 80GB | 8x A100 | 640GB | $14.32 | Multi-GPU training |

### 3.2 Launch Instance

1. Go to [Lambda Cloud Dashboard](https://cloud.lambda.ai/instances)
2. Click "Launch Instance"
3. Select GPU type: **1x A100 (80 GB SXM)**
4. Select region (closest to you)
5. Select your SSH key
6. Click "Launch"

**Wait 3-5 minutes for instance to boot.**

### 3.3 Connect to Instance

```bash
# Get SSH command from Lambda dashboard, or:
ssh -i ~/.ssh/lambda_key ubuntu@<INSTANCE_IP>

# Example:
ssh ubuntu@192.0.2.1
```

---

## 4. Environment Setup

### 4.1 Quick Setup Script

SSH into your instance and run:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/VeryLargeWeebModel.git
cd VeryLargeWeebModel

# Run the Lambda setup script
chmod +x scripts/lambda_setup.sh
./scripts/lambda_setup.sh
```

### 4.2 Manual Setup (Alternative)

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    htop \
    tmux \
    nvtop

# Create conda environment
conda create -n occworld python=3.8 -y
conda activate occworld

# Install PyTorch (Lambda has CUDA pre-installed)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install MMDetection3D stack
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
pip install mmdet==2.28.0
pip install mmsegmentation==0.30.0

# Install OccWorld dependencies
pip install \
    nuscenes-devkit \
    torchpack \
    tqdm \
    open3d \
    scipy \
    opencv-python \
    pillow \
    trimesh \
    tensorboard

# Clone OccWorld
git clone https://github.com/wzzheng/OccWorld.git ~/OccWorld
cd ~/OccWorld
pip install -e .
```

### 4.3 Verify GPU Setup

```bash
# Check GPU
nvidia-smi

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Expected output:
# CUDA: True, Device: NVIDIA A100-SXM4-80GB
```

---

## 5. Data Transfer

### 5.1 Download Data on Instance

```bash
cd ~/VeryLargeWeebModel

# Download PLATEAU models and pretrained weights
./scripts/download_and_prepare_data.sh --all

# Or skip large PLATEAU download if not needed for training
./scripts/download_and_prepare_data.sh --models --skip-plateau
```

### 5.2 Manual Model Downloads

Tsinghua Cloud requires browser download. Use your local machine:

```bash
# On your LOCAL machine, download from:
# https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/

# Then upload to Lambda instance:
scp -i ~/.ssh/lambda_key ~/Downloads/vqvae_epoch_125.pth \
    ubuntu@<INSTANCE_IP>:~/VeryLargeWeebModel/pretrained/vqvae/epoch_125.pth

scp -i ~/.ssh/lambda_key ~/Downloads/occworld_latest.pth \
    ubuntu@<INSTANCE_IP>:~/VeryLargeWeebModel/pretrained/occworld/latest.pth
```

### 5.3 Upload Existing Dataset

If you have pre-recorded Tokyo Gazebo data:

```bash
# From your LOCAL machine
rsync -avz --progress \
    -e "ssh -i ~/.ssh/lambda_key" \
    ./data/tokyo_gazebo/ \
    ubuntu@<INSTANCE_IP>:~/VeryLargeWeebModel/data/tokyo_gazebo/
```

### 5.4 Use Persistent Storage (Recommended)

Lambda offers persistent filesystems to preserve data between instances:

1. Go to [Lambda Storage](https://cloud.lambda.ai/file-systems)
2. Create new filesystem (e.g., "occworld-data", 500GB)
3. Attach to instance when launching
4. Data persists at `/home/ubuntu/persistent/`

```bash
# Move data to persistent storage
mv ~/VeryLargeWeebModel/data /home/ubuntu/persistent/
ln -s /home/ubuntu/persistent/data ~/VeryLargeWeebModel/data
```

---

## 6. Training Execution

### 6.1 Start Training in tmux

Always use tmux to prevent training loss if SSH disconnects:

```bash
# Start tmux session
tmux new -s training

# Activate environment
conda activate occworld
cd ~/VeryLargeWeebModel
```

### 6.2 Train VQVAE (If Starting Fresh)

```bash
# Train VQVAE first (required for OccWorld)
cd ~/OccWorld

python train.py \
    --py-config config/train_vqvae.py \
    --work-dir /home/ubuntu/persistent/checkpoints/vqvae

# Takes ~12-24 hours on A100
```

### 6.3 Fine-tune OccWorld on Tokyo Dataset

```bash
cd ~/VeryLargeWeebModel

# Fine-tune with our config
python train.py \
    --py-config config/finetune_tokyo.py \
    --work-dir /home/ubuntu/persistent/checkpoints/occworld_tokyo \
    --resume-from pretrained/occworld/latest.pth

# Or with custom settings
python train.py \
    --py-config config/finetune_tokyo.py \
    --work-dir /home/ubuntu/persistent/checkpoints/occworld_tokyo \
    --gpu-ids 0 \
    --seed 42
```

### 6.4 Multi-GPU Training (8x A100)

```bash
# Distributed training on 8 GPUs
torchpack dist-run -np 8 python train.py \
    --py-config config/finetune_tokyo.py \
    --work-dir /home/ubuntu/persistent/checkpoints/occworld_tokyo
```

### 6.5 Detach from tmux

```bash
# Detach: Ctrl+B, then D
# Reattach later:
tmux attach -t training
```

---

## 7. Monitoring & Checkpoints

### 7.1 Monitor Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or use nvtop (more detailed)
nvtop

# View training logs
tail -f /home/ubuntu/persistent/checkpoints/occworld_tokyo/training.log
```

### 7.2 TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir /home/ubuntu/persistent/checkpoints/ --port 6006 --bind_all

# Access from your browser:
# http://<INSTANCE_IP>:6006
```

**SSH Tunnel (if firewall blocks 6006):**
```bash
# On your LOCAL machine
ssh -i ~/.ssh/lambda_key -L 6006:localhost:6006 ubuntu@<INSTANCE_IP>

# Then open: http://localhost:6006
```

### 7.3 Download Checkpoints

```bash
# On your LOCAL machine
scp -i ~/.ssh/lambda_key -r \
    ubuntu@<INSTANCE_IP>:/home/ubuntu/persistent/checkpoints/occworld_tokyo/epoch_*.pth \
    ./checkpoints/
```

### 7.4 Automatic Checkpoint Backup

Add to your training script or run separately:

```bash
# Backup checkpoints to local machine every hour
while true; do
    rsync -avz --progress \
        -e "ssh -i ~/.ssh/lambda_key" \
        ubuntu@<INSTANCE_IP>:/home/ubuntu/persistent/checkpoints/ \
        ./lambda_backups/
    sleep 3600
done
```

---

## 8. Cost Optimization

### 8.1 Estimated Costs

| Task | Instance | Time | Cost |
|------|----------|------|------|
| VQVAE Training | 1x A100 80GB | ~20 hrs | ~$36 |
| OccWorld Fine-tune | 1x A100 80GB | ~24 hrs | ~$43 |
| Full Pipeline | 1x A100 80GB | ~48 hrs | ~$86 |
| Fast Fine-tune | 1x H100 | ~12 hrs | ~$40 |

### 8.2 Cost-Saving Tips

1. **Use Persistent Storage**: Don't re-download data ($0.20/GB/mo is cheap)

2. **Spot/Preemptible Instances**: Not available on Lambda, but check availability

3. **Right-size Instance**:
   - A100 40GB for small experiments
   - A100 80GB for full training
   - H100 only if time-critical

4. **Stop When Idle**: Lambda bills per minute - terminate immediately when done

5. **Use Checkpoints**: Save frequently, resume if interrupted

6. **Off-peak Hours**: Better availability (not necessarily cheaper)

### 8.3 Terminate Instance

**Important: Stop billing immediately when done!**

```bash
# Save any unsaved work first!

# Option 1: Lambda Dashboard
# Go to Instances > Select > Terminate

# Option 2: Lambda CLI (if installed)
lambda instance terminate <INSTANCE_ID>
```

---

## 9. Automated Deployment Script

### 9.1 Lambda Setup Script

Create this on your Lambda instance:

```bash
#!/bin/bash
# scripts/lambda_setup.sh
# Automated setup script for Lambda Cloud instances

set -e

echo "=========================================="
echo "OccWorld Lambda Cloud Setup"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }

# Check if on Lambda instance
if [ ! -f /etc/lambda-stack-version ]; then
    echo "Warning: This doesn't appear to be a Lambda instance"
fi

# Install system packages
log "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq tmux htop nvtop tree

# Setup conda environment
log "Setting up conda environment..."
if ! conda env list | grep -q "occworld"; then
    conda create -n occworld python=3.8 -y
fi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate occworld

# Install PyTorch
log "Installing PyTorch..."
pip install -q torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install ML dependencies
log "Installing ML dependencies..."
pip install -q \
    mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
pip install -q mmdet==2.28.0 mmsegmentation==0.30.0

# Install OccWorld dependencies
log "Installing OccWorld dependencies..."
pip install -q \
    nuscenes-devkit torchpack tqdm open3d scipy \
    opencv-python pillow trimesh tensorboard

# Clone OccWorld if not exists
if [ ! -d ~/OccWorld ]; then
    log "Cloning OccWorld..."
    git clone https://github.com/wzzheng/OccWorld.git ~/OccWorld
    cd ~/OccWorld && pip install -q -e .
fi

# Setup persistent storage link
if [ -d /home/ubuntu/persistent ]; then
    log "Setting up persistent storage..."
    mkdir -p /home/ubuntu/persistent/checkpoints
    mkdir -p /home/ubuntu/persistent/data
fi

# Create convenience aliases
cat >> ~/.bashrc << 'EOF'

# OccWorld aliases
alias occworld='conda activate occworld'
alias tb='tensorboard --logdir /home/ubuntu/persistent/checkpoints --port 6006 --bind_all'
alias gpu='watch -n 1 nvidia-smi'
alias train='cd ~/VeryLargeWeebModel && conda activate occworld'
EOF

success "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. source ~/.bashrc"
echo "  2. conda activate occworld"
echo "  3. ./scripts/download_and_prepare_data.sh --models"
echo "  4. Upload pretrained models from Tsinghua Cloud"
echo "  5. Start training with: python train.py --py-config config/finetune_tokyo.py"
echo ""
```

### 9.2 One-Line Deploy

From your local machine:

```bash
# Clone, setup, and start training in one command
ssh -i ~/.ssh/lambda_key ubuntu@<INSTANCE_IP> 'bash -s' << 'EOF'
cd ~
git clone https://github.com/YOUR_USERNAME/VeryLargeWeebModel.git
cd VeryLargeWeebModel
chmod +x scripts/lambda_setup.sh
./scripts/lambda_setup.sh
EOF
```

### 9.3 Full Automation Script

Save locally as `deploy_to_lambda.sh`:

```bash
#!/bin/bash
# deploy_to_lambda.sh - Deploy OccWorld training to Lambda Cloud

INSTANCE_IP="$1"
SSH_KEY="${2:-~/.ssh/lambda_key}"

if [ -z "$INSTANCE_IP" ]; then
    echo "Usage: $0 <INSTANCE_IP> [SSH_KEY_PATH]"
    exit 1
fi

SSH_CMD="ssh -i $SSH_KEY ubuntu@$INSTANCE_IP"
SCP_CMD="scp -i $SSH_KEY"

echo "Deploying to Lambda instance: $INSTANCE_IP"

# Upload project
echo "Uploading project..."
rsync -avz --progress \
    -e "ssh -i $SSH_KEY" \
    --exclude '.git' \
    --exclude 'data/plateau/raw/*' \
    --exclude '__pycache__' \
    ./ ubuntu@$INSTANCE_IP:~/VeryLargeWeebModel/

# Run setup
echo "Running setup..."
$SSH_CMD 'cd ~/VeryLargeWeebModel && chmod +x scripts/lambda_setup.sh && ./scripts/lambda_setup.sh'

# Upload pretrained models (if they exist locally)
if [ -f "pretrained/vqvae/epoch_125.pth" ]; then
    echo "Uploading VQVAE checkpoint..."
    $SCP_CMD pretrained/vqvae/epoch_125.pth \
        ubuntu@$INSTANCE_IP:~/VeryLargeWeebModel/pretrained/vqvae/
fi

if [ -f "pretrained/occworld/latest.pth" ]; then
    echo "Uploading OccWorld checkpoint..."
    $SCP_CMD pretrained/occworld/latest.pth \
        ubuntu@$INSTANCE_IP:~/VeryLargeWeebModel/pretrained/occworld/
fi

echo ""
echo "Deployment complete!"
echo ""
echo "To start training:"
echo "  $SSH_CMD"
echo "  tmux new -s training"
echo "  conda activate occworld"
echo "  cd ~/VeryLargeWeebModel"
echo "  python train.py --py-config config/finetune_tokyo.py --work-dir ~/persistent/checkpoints/occworld_tokyo"
```

---

## Quick Reference

### SSH Commands

```bash
# Connect
ssh -i ~/.ssh/lambda_key ubuntu@<IP>

# Copy file to instance
scp -i ~/.ssh/lambda_key file.txt ubuntu@<IP>:~/

# Copy directory from instance
scp -i ~/.ssh/lambda_key -r ubuntu@<IP>:~/checkpoints ./

# SSH tunnel for TensorBoard
ssh -i ~/.ssh/lambda_key -L 6006:localhost:6006 ubuntu@<IP>
```

### Training Commands

```bash
# Activate environment
conda activate occworld

# Start training (in tmux!)
tmux new -s train
python train.py --py-config config/finetune_tokyo.py --work-dir ~/persistent/checkpoints/

# Detach: Ctrl+B, D
# Reattach: tmux attach -t train
```

### Monitoring Commands

```bash
# GPU usage
nvidia-smi
nvtop

# Training logs
tail -f ~/persistent/checkpoints/occworld_tokyo/*.log

# TensorBoard
tensorboard --logdir ~/persistent/checkpoints --port 6006 --bind_all
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| SSH connection refused | Wait 3-5 min for instance boot, check IP |
| CUDA out of memory | Reduce batch_size to 1, enable gradient checkpointing |
| Training interrupted | Use tmux, resume from checkpoint |
| Slow data loading | Use persistent storage, increase num_workers |
| Instance terminated | Data lost unless using persistent storage |
| Port 6006 blocked | Use SSH tunnel instead |

---

## References

- [Lambda Cloud Docs](https://docs.lambda.ai/public-cloud/on-demand/)
- [Lambda Getting Started](https://lambdalabs.com/blog/getting-started-with-lambda-cloud-gpu-instances)
- [Lambda Pricing](https://lambda.ai/pricing)
- [OccWorld GitHub](https://github.com/wzzheng/OccWorld)
