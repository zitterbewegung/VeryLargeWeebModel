# Training Guide

Complete guide for training AerialWorld (OccWorld on Tokyo PLATEAU data).

## Prerequisites

- Python 3.8+
- PyTorch 2.0+ with CUDA
- GPU with 10GB+ VRAM (24GB+ recommended)

```bash
pip install torch torchvision tqdm scipy opencv-python
pip install wandb  # optional, for experiment tracking
```

## Quick Start

### Using the CLI (recommended)

```bash
python scripts/vlwm_cli.py setup        # Install dependencies, detect GPU
python scripts/vlwm_cli.py download     # Download data + pretrained models
python scripts/vlwm_cli.py train        # Auto-configures batch size, precision
```

### Manual

```bash
# 1. Download pretrained model
mkdir -p pretrained/occworld
# Download latest.pth from project releases

# 2. Prepare data
python scripts/plateau_to_occworld.py \
  --input data/plateau/meshes/obj \
  --output data/tokyo_gazebo \
  --frames 500 --sessions 10

# 3. Train
python train.py \
  --config config/finetune_tokyo.py \
  --work-dir ./checkpoints \
  --batch-size 3 \
  --epochs 50
```

## Cloud Deployment

| Provider | GPU | Approx. Cost | Setup |
|----------|-----|-------------|-------|
| Vast.ai | A100 40GB | ~$8-15 | [docs/vastai_deployment.md](vastai_deployment.md) |
| Lambda Cloud | A100 | ~$20-43 | [docs/lambda_cloud_deployment.md](lambda_cloud_deployment.md) |
| RunPod | A100 | ~$15-30 | Use `vlwm_cli.py setup` |

```bash
# SSH into cloud instance, then:
git clone https://github.com/zitterbewegung/VeryLargeWeebModel.git
cd VeryLargeWeebModel

# Automated setup (detects Vast.ai/Lambda/RunPod automatically)
python scripts/vlwm_cli.py setup
python scripts/vlwm_cli.py download --all
screen -S training
python scripts/vlwm_cli.py train
# Detach: Ctrl+A, then D
```

## train.py Arguments

```
python train.py [OPTIONS]

Required:
  --config FILE              Config file (e.g., config/finetune_tokyo.py)

Optional:
  --work-dir DIR             Output directory (default: ./checkpoints)
  --batch-size N             Batch size per GPU
  --lr RATE                  Learning rate (default: 1e-4)
  --epochs N                 Max epochs (default: 50)
  --resume                   Resume from latest checkpoint in work-dir
  --resume-from FILE         Resume from specific checkpoint file
  --from-scratch             Train without pretrained weights
  --gpu-ids IDS              GPU IDs, comma-separated (default: 0)
  --seed N                   Random seed (default: 42)
  --eval-only                Run evaluation only
  --wandb                    Enable Weights & Biases logging
  --wandb-project NAME       W&B project name (default: occworld-tokyo)
  --wandb-run-name NAME      W&B run name
  --wandb-tags TAG [TAG]     W&B tags
  --amp                      Enable mixed precision training
```

## Configuration Reference

Primary config: `config/finetune_tokyo.py`

```python
# Voxel grid
point_cloud_range = [-40.0, -40.0, -2.0, 40.0, 40.0, 150.0]
voxel_size = [0.4, 0.4, 1.25]       # → 200×200×121 grid

# Temporal
history_frames = 4                    # Past frames as input
future_frames = 6                     # Future frames to predict

# Training
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
lr_config = dict(policy='CosineAnnealing', min_lr=1e-6, warmup='linear', warmup_iters=100)
max_epochs = 50
grad_clip = dict(max_norm=35, norm_type=2)

# Model
model = dict(
    type='TransVQVAE',
    freeze_vae=True,            # VQVAE frozen during fine-tuning
    freeze_transformer=False,   # Transformer is trainable
    freeze_pose=False,          # Pose module is trainable
)

# Data
data = dict(
    samples_per_gpu=3,          # Batch size (adjust for GPU VRAM)
    workers_per_gpu=8,
)
```

### Available Configs

| Config | Dataset | Use Case |
|--------|---------|----------|
| `finetune_tokyo.py` | Tokyo PLATEAU + Gazebo | Primary aerial training |
| `finetune_nuscenes.py` | nuScenes | Ground vehicle baseline |
| `finetune_uavscenes.py` | UAVScenes | Real aerial data |
| `finetune_6dof.py` | Any | 6DoF pose prediction model |
| `finetune_nuscenes_6dof.py` | nuScenes | 6DoF ground vehicle |
| `test_local.py` | Dummy data | Local testing (no GPU needed) |

## Hardware Requirements

| GPU | VRAM | Batch Size | ~Training Time (50 epochs) |
|-----|------|------------|---------------------------|
| A100 80GB | 80GB | 6-8 | ~8 hours |
| A100 40GB | 40GB | 3-4 | ~12 hours |
| RTX 4090 | 24GB | 2-3 | ~20 hours |
| RTX 3090 | 24GB | 1-2 | ~24 hours |
| RTX 3080 | 10GB | 1 | ~48 hours |

### VRAM Usage (approximate)

| Batch Size | VRAM |
|------------|------|
| 1 | ~13GB |
| 2 | ~22GB |
| 3 | ~32GB |
| 4 | ~40GB |
| 8 | ~70GB |

## Data Format

```
data/tokyo_gazebo/
├── drone_20240101_120000/          # Session directory
│   ├── occupancy/                  # Voxelized 3D grids
│   │   └── 000000_occupancy.npz   # Shape: (200, 200, 121), binary
│   ├── poses/                      # Vehicle poses
│   │   └── 000000.json            # {position, orientation, velocity}
│   ├── lidar/                      # Point clouds
│   │   └── 000000_LIDAR.npy       # Shape: (N, 4) - x,y,z,intensity
│   └── images/                     # Camera images
│       └── 000000_CAM_FRONT.jpg   # Shape: (900, 1600, 3)
└── rover_*/                        # Rover sessions (same structure)
```

### Pose Format (13D)

```
[x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
 position   quaternion      linear vel    angular vel
```

## Loss Function

The model uses a composite loss designed for extreme class imbalance (~0.83% occupied voxels):

| Component | Weight | Purpose |
|-----------|--------|---------|
| Focal Loss | alpha=0.99, gamma=2 | Focus on rare occupied voxels |
| Dice Loss | 1.0 | Set overlap, robust to imbalance |
| Mean-Matching | 10.0 | Prevents all-zero collapse |

**Why not standard BCE?** With 99.17% empty voxels, the model learns to predict all zeros (achieving 99% accuracy by doing nothing). Mean-matching forces `pred.mean() ≈ target.mean() ≈ 0.008`.

## Monitoring

```bash
# GPU usage
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir ./checkpoints --port 6006

# Weights & Biases (if enabled)
# Dashboard at wandb.ai

# Reattach to training session
screen -r training
```

### Healthy Training Signs

- Loss stabilizes around 0.1-0.5 (not dropping to 0.0001)
- `Pred mean` stays near 0.008-0.02 (matching target occupancy)
- `Pred max` remains above 0.5 (confident positive predictions)
- Validation loss decreases over epochs

### Warning Signs

- Loss drops to 0.0001 within first epoch → all-zero collapse
- `Pred mean` approaching 0 → model not predicting occupied voxels
- NaN in loss → learning rate too high or numerical instability

## Output Files

```
checkpoints/
├── epoch_5.pth          # Periodic checkpoints (every 5 epochs)
├── epoch_10.pth
├── ...
├── epoch_50.pth         # Final checkpoint
├── best.pth             # Best validation loss
├── logs/                # TensorBoard logs
│   └── events.out.*
└── train_*.log          # Training logs
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce `--batch-size` to 1 |
| Training too slow | Use A100 or larger GPU; increase batch size |
| Loss stuck at 0.0001 | Check mean-matching is enabled; lower lr to 5e-5 |
| NaN loss | Lower `--lr` to 1e-5; check data for corruption |
| Connection dropped | Use `screen -S training` before starting |
| Import errors | Run `python scripts/vlwm_cli.py setup` |
| Empty occupancy grids | Check data generation; verify point cloud range |
| Resume not working | Check `ls checkpoints/*.pth`; use `--resume-from` with explicit path |

### Common Commands

```bash
# Quick test (2 epochs with dummy data)
python train.py --config config/test_local.py --work-dir ./test_out --epochs 2

# Resume after interruption
python train.py --config config/finetune_tokyo.py --work-dir ./checkpoints --resume

# Resume from specific checkpoint
python train.py --config config/finetune_tokyo.py --resume-from ./checkpoints/epoch_25.pth

# Evaluation only
python train.py --config config/finetune_tokyo.py --work-dir ./checkpoints --eval-only

# Download results from cloud
scp -P PORT user@host:~/checkpoints/best.pth ./my_model.pth
```

## W&B Integration

```bash
pip install wandb && wandb login

python train.py \
  --config config/finetune_tokyo.py \
  --wandb \
  --wandb-project occworld-tokyo \
  --wandb-run-name experiment-v1
```

Logged metrics: `train/loss`, `pred/mean`, `pred/min`, `pred/max`, `pred/occupancy_rate`, `epoch/val_loss`, `epoch/lr`.
