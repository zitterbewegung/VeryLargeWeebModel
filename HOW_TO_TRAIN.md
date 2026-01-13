# How to Train OccWorld on Tokyo Data

## Quick Start (Vast.ai - Cheapest)

**Cost: ~$15 for full training**

```bash
# 1. Rent A100 on vast.ai (~$0.63/hr)
#    - Go to vast.ai, search for A100, pick cheapest with >95% reliability

# 2. SSH into your instance
ssh -p PORT root@ssh.vast.ai

# 3. Clone and setup
git clone https://github.com/YOUR_USERNAME/VeryLargeWeebModel.git
cd VeryLargeWeebModel
./scripts/vastai_setup.sh

# 4. Download data
./scripts/download_and_prepare_data.sh --all

# 5. Train (use screen to keep alive)
screen -S training
python train.py --config config/finetune_tokyo.py --work-dir /workspace/checkpoints
# Detach: Ctrl+A, then D

# 6. Download results (from local machine)
scp -P PORT root@ssh.vast.ai:/workspace/checkpoints/best.pth ./

# 7. DESTROY INSTANCE (stop billing!)
```

---

## Training Options

### Option 1: Vast.ai (Cheapest)
```bash
# ~$15 for 24hr training
ssh -p PORT root@ssh.vast.ai
git clone https://github.com/YOUR_USERNAME/VeryLargeWeebModel.git
cd VeryLargeWeebModel
./scripts/vastai_setup.sh
python train.py --config config/finetune_tokyo.py --work-dir /workspace/checkpoints
```

### Option 2: Lambda Cloud (Easiest)
```bash
# ~$43 for 24hr training
ssh ubuntu@YOUR_IP
git clone https://github.com/YOUR_USERNAME/VeryLargeWeebModel.git
cd VeryLargeWeebModel
./scripts/lambda_setup.sh
python train.py --config config/finetune_tokyo.py --work-dir ~/checkpoints
```

### Option 3: Local Mac (Free but Slow)
```bash
# M1/M2/M3 Mac with 32GB+ RAM
git clone https://github.com/YOUR_USERNAME/VeryLargeWeebModel.git
cd VeryLargeWeebModel
pip install torch torchvision tqdm scipy opencv-python
python train.py --config config/finetune_tokyo.py --work-dir ./out --batch-size 1
```

---

## Training Commands

### Basic Training
```bash
python train.py --config config/finetune_tokyo.py --work-dir ./checkpoints
```

### With Custom Settings
```bash
python train.py \
    --config config/finetune_tokyo.py \
    --work-dir ./checkpoints \
    --batch-size 1 \
    --lr 0.0001 \
    --epochs 50
```

### Resume from Checkpoint
```bash
python train.py \
    --config config/finetune_tokyo.py \
    --work-dir ./checkpoints \
    --resume
```

### Resume from Specific Checkpoint
```bash
python train.py \
    --config config/finetune_tokyo.py \
    --work-dir ./checkpoints \
    --resume-from ./checkpoints/epoch_25.pth
```

---

## Monitor Training

### GPU Usage
```bash
nvidia-smi -l 1
# or
watch -n 1 nvidia-smi
```

### TensorBoard
```bash
tensorboard --logdir ./checkpoints --port 6006
# Open http://localhost:6006
```

### Training Logs
```bash
tail -f ./checkpoints/*.log
```

---

## Screen Commands (Keep Training Alive)

```bash
screen -S training      # Start new session
# Ctrl+A, then D        # Detach
screen -r training      # Reattach
screen -ls              # List sessions
```

---

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Val Loss | 0.45 | 0.28 |
| mIoU | 0.32 | 0.47 |

### Training Time
| Hardware | Time | Cost |
|----------|------|------|
| A100 80GB | ~24 hrs | ~$15-43 |
| RTX 4090 | ~36 hrs | ~$18 |
| RTX 3090 | ~48 hrs | ~$12 |
| Mac M3 Max | ~5-7 days | $0 |

---

## Checkpoints

Saved to `--work-dir`:
```
checkpoints/
├── epoch_5.pth
├── epoch_10.pth
├── ...
├── epoch_50.pth
└── best.pth          # Best validation loss
```

### Download Best Model
```bash
# From cloud to local
scp user@host:~/checkpoints/best.pth ./my_model.pth
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Use `--batch-size 1` |
| Training too slow | Use A100 instead of consumer GPU |
| Connection dropped | Use `screen`, reattach with `screen -r` |
| NaN loss | Lower learning rate: `--lr 0.00001` |
| Import errors | Run setup script again |

---

## Cost Summary

| Provider | GPU | $/hr | 24hr Cost |
|----------|-----|------|-----------|
| Vast.ai | A100 | $0.63 | **$15** |
| Vast.ai | RTX 3090 | $0.25 | **$6** |
| RunPod | A100 | $1.19 | **$29** |
| Lambda | A100 | $1.79 | **$43** |

---

## Data Attribution

This project uses Tokyo PLATEAU 3D city data:

> 3D city model data provided by Ministry of Land, Infrastructure, Transport and Tourism (MLIT), Japan - Project PLATEAU. Licensed under CC BY 4.0.
