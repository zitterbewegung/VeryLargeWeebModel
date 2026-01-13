# OccWorld Deployment Guide for Vast.ai

Deploy and train OccWorld on Vast.ai GPU marketplace - the cheapest A100 option.

**Cost Estimate:** ~$15 for 24-hour training (A100 @ $0.63/hr)

---

## Quick Start

```bash
# 1. Rent A100 on Vast.ai (~$0.63/hr)
# 2. SSH in and run:
git clone https://github.com/YOUR_USERNAME/VeryLargeWeebModel.git
cd VeryLargeWeebModel
./scripts/vastai_setup.sh
python train.py --config config/finetune_tokyo.py --work-dir /workspace/checkpoints
```

---

## Table of Contents

1. [Create Vast.ai Account](#1-create-vastai-account)
2. [Find Cheap A100](#2-find-cheap-a100)
3. [Launch Instance](#3-launch-instance)
4. [Setup Environment](#4-setup-environment)
5. [Start Training](#5-start-training)
6. [Download Results](#6-download-results)

---

## 1. Create Vast.ai Account

1. Go to **https://vast.ai**
2. Click **Sign Up**
3. Verify email
4. Go to **Billing** → Add credits ($20 is plenty)

---

## 2. Find Cheap A100

1. Go to **Console** → **Search**
2. Set filters:
   - **GPU Type**: A100 (or A100 PCIe, A100 SXM)
   - **VRAM**: ≥40 GB
   - **Storage**: ≥100 GB
   - **Reliability**: ≥95%

3. Sort by **$/hr (lowest first)**
4. Look for ~$0.60-0.80/hr deals

### What to Look For

| Criteria | Recommended |
|----------|-------------|
| GPU | A100 40GB or 80GB |
| Price | <$1.00/hr |
| Reliability | >95% |
| Internet Speed | >200 Mbps |
| Storage | >100 GB |

---

## 3. Launch Instance

1. Click **RENT** on your chosen instance
2. Select template: **PyTorch 2.0** (or latest)
3. Set disk space: **100 GB** (minimum)
4. Click **RENT**
5. Wait 1-2 minutes for it to start

### Connect via SSH

Once running, you'll see an SSH command like:
```bash
ssh -p 12345 root@ssh.vast.ai
```

Or use the **Jupyter** button for web interface.

---

## 4. Setup Environment

### Option A: One-Line Setup

```bash
# SSH into your instance, then:
git clone https://github.com/YOUR_USERNAME/VeryLargeWeebModel.git
cd VeryLargeWeebModel
chmod +x scripts/vastai_setup.sh
./scripts/vastai_setup.sh
```

### Option B: Manual Setup

```bash
# Update system
apt-get update && apt-get install -y tmux htop

# Clone project
cd /workspace
git clone https://github.com/YOUR_USERNAME/VeryLargeWeebModel.git
cd VeryLargeWeebModel

# Install dependencies (PyTorch usually pre-installed)
pip install tqdm scipy opencv-python pillow tensorboard open3d

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 5. Start Training

### Always Use screen!

Vast.ai connections can drop. Use screen to keep training alive:

```bash
# Start screen session
screen -S training

# Navigate to project
cd /workspace/VeryLargeWeebModel

# Start training
python train.py \
    --config config/finetune_tokyo.py \
    --work-dir /workspace/checkpoints

# Detach: Ctrl+A, then D
# Reattach later: screen -r training
# List sessions: screen -ls
```

### Monitor Training

```bash
# GPU usage
nvidia-smi -l 1

# Or watch
watch -n 1 nvidia-smi

# Training logs (in another terminal)
tail -f /workspace/checkpoints/occworld_tokyo/*.log
```

### TensorBoard

```bash
tensorboard --logdir /workspace/checkpoints --port 6006 --bind_all
```

Access via the **Open Ports** button in Vast.ai console, or use the Jupyter interface.

---

## 6. Download Results

### Option A: SCP (from your local machine)

```bash
# Get your instance SSH details from Vast.ai console
scp -P 12345 root@ssh.vast.ai:/workspace/checkpoints/occworld_tokyo/best.pth ./
```

### Option B: Vast.ai Cloud Sync

1. Go to instance → **Cloud Sync**
2. Configure S3, Google Cloud, or other storage
3. Sync your checkpoints folder

### Option C: Upload to HuggingFace

```bash
pip install huggingface_hub
huggingface-cli login
python -c "
from huggingface_hub import upload_file
upload_file('/workspace/checkpoints/occworld_tokyo/best.pth',
            'your-username/occworld-tokyo', 'best.pth')
"
```

---

## 7. Stop Instance

**Important: You're charged while instance is running!**

1. Download your checkpoints first
2. Go to Vast.ai Console → **Instances**
3. Click **Destroy** on your instance

---

## Cost Breakdown

| Component | Cost |
|-----------|------|
| A100 @ $0.63/hr × 24hrs | ~$15.12 |
| Storage (100GB included) | $0 |
| Data transfer | ~$0 |
| **Total** | **~$15** |

Compare to Lambda Labs: ~$43 (same training)

**You save: ~$28 (65% cheaper!)**

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Instance won't start | Try different instance, check credits |
| SSH connection refused | Wait 2 min, instance still booting |
| CUDA out of memory | Reduce batch_size to 1 |
| Training interrupted | Use screen, reattach with `screen -r training` |
| Slow download | Use Cloud Sync instead of SCP |
| Instance disappeared | Host rebooted - always save checkpoints frequently |

### Vast.ai Specific Notes

1. **Instances can be preempted** - Save checkpoints every epoch
2. **Reliability score matters** - Don't pick <90% to save $0.05
3. **Check internet speed** - Slow download = wasted GPU time
4. **Use /workspace** - Persistent storage location

---

## Resume Training

If interrupted, resume from checkpoint:

```bash
python train.py \
    --config config/finetune_tokyo.py \
    --work-dir /workspace/checkpoints \
    --resume
```

---

## Comparison: Vast.ai vs Lambda

| Feature | Vast.ai | Lambda |
|---------|---------|--------|
| A100 Price | ~$0.63/hr | $1.79/hr |
| 24hr Training | ~$15 | ~$43 |
| Reliability | Variable (check score) | Consistent |
| Setup | Manual | Pre-configured |
| Support | Community | Professional |
| Best For | Budget training | Production/reliability |

---

## Quick Reference

```bash
# SSH in
ssh -p PORT root@ssh.vast.ai

# Setup
git clone https://github.com/USER/VeryLargeWeebModel.git
cd VeryLargeWeebModel && ./scripts/vastai_setup.sh

# Train (in screen!)
screen -S training
python train.py --config config/finetune_tokyo.py --work-dir /workspace/checkpoints
# Detach: Ctrl+A, then D

# Monitor
nvidia-smi -l 1
screen -r training  # Reattach

# Download (from local)
scp -P PORT root@ssh.vast.ai:/workspace/checkpoints/best.pth ./

# DESTROY INSTANCE WHEN DONE!
```
