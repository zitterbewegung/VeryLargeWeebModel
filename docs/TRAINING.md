# OccWorld Tokyo Training Guide

## Overview

Fine-tune OccWorld (3D occupancy world model) on Tokyo PLATEAU urban data for autonomous drone/rover navigation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OccWorld Model                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  VQVAE (FROZEN during fine-tuning)                                  │   │
│  │  ├── Encoder2D: base_channel=64                                     │   │
│  │  ├── Decoder2D: base_channel=64                                     │   │
│  │  └── VectorQuantizer: 512 embeddings, dim=64, commitment=0.25       │   │
│  │                                                                     │   │
│  │  Purpose: Encode 3D occupancy grids into discrete latent tokens     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Transformer (TRAINABLE)                                            │   │
│  │  ├── num_layers: 2                                                  │   │
│  │  ├── temporal_attn_layers: 6                                        │   │
│  │  ├── embed_dim: 64                                                  │   │
│  │  ├── num_heads: 8                                                   │   │
│  │  └── dropout: 0.1                                                   │   │
│  │                                                                     │   │
│  │  Purpose: Predict future occupancy from past observations           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Pose Encoder/Decoder (TRAINABLE)                                   │   │
│  │  ├── input_dim: 13 (x,y,z + quat(4) + lin_vel(3) + ang_vel(3))     │   │
│  │  ├── hidden_dim: 256                                                │   │
│  │  ├── output_dim: 64                                                 │   │
│  │  └── num_layers: 2                                                  │   │
│  │                                                                     │   │
│  │  Purpose: Encode/decode vehicle pose for trajectory prediction      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Format

```
data/tokyo_gazebo/
├── drone_20240101_120000/          # Session directory
│   ├── occupancy/                  # Voxelized 3D grids
│   │   └── 000000_occupancy.npz    # Shape: (200, 200, 121)
│   ├── poses/                      # Vehicle poses
│   │   └── 000000.json             # {position, orientation, velocity}
│   ├── lidar/                      # Point clouds
│   │   └── 000000_LIDAR.npy        # Shape: (N, 4) - x,y,z,intensity
│   └── images/                     # Camera images
│       └── 000000_CAM_FRONT.jpg    # Shape: (900, 1600, 3)
└── rover_*/                        # Rover sessions (same structure)
```

### Occupancy Grid Dimensions

| Parameter | Value | Description |
|-----------|-------|-------------|
| X range | -40m to +40m | 80m total width |
| Y range | -40m to +40m | 80m total depth |
| Z range | -2m to +150m | 152m height (for drones) |
| Voxel size | 0.4m × 0.4m × 1.25m | Resolution |
| Grid size | 200 × 200 × 121 | Voxel count |

## Training Arguments

### run_training.sh

```bash
./scripts/run_training.sh [OPTIONS]

Options:
  --config FILE       Config file (default: config/finetune_tokyo.py)
  --work-dir DIR      Output directory (default: ./checkpoints)
  --epochs N          Number of epochs (default: 50)
  --batch-size N      Batch size per GPU (default: 1)
  --lr RATE           Learning rate (default: 1e-4)
  --quick             Quick test mode (2 epochs)
  --resume            Resume from latest checkpoint
  --eval              Evaluation only
  --tensorboard       Start tensorboard only
  --generate-data     Generate training data before training
  --data-frames N     Frames per session for data gen (default: 200)
  --data-sessions N   Number of sessions for data gen (default: 5)
```

### train.py

```bash
python train.py [OPTIONS]

Options:
  --config, --py-config FILE   Config file path
  --work-dir DIR               Output directory
  --resume                     Resume from latest checkpoint
  --resume-from FILE           Resume from specific checkpoint
  --from-scratch               Train without pretrained weights
  --gpu-ids IDS                GPU IDs (comma-separated)
  --seed N                     Random seed (default: 42)
  --batch-size N               Override batch size
  --lr RATE                    Override learning rate
  --epochs N                   Override max epochs
  --eval-only                  Run evaluation only
```

## Configuration Reference

### config/finetune_tokyo.py

```python
# Dataset
data_root = 'data/tokyo_gazebo/'
point_cloud_range = [-40.0, -40.0, -2.0, 40.0, 40.0, 150.0]
voxel_size = [0.4, 0.4, 1.25]
history_frames = 4      # Past frames for context
future_frames = 6       # Future frames to predict

# Data loading
data = dict(
    samples_per_gpu=1,  # Batch size (increase for faster training)
    workers_per_gpu=4,  # Data loading workers
)

# Model
model = dict(
    type='TransVQVAE',
    freeze_vae=True,           # Keep VQVAE frozen
    freeze_transformer=False,  # Fine-tune transformer
    freeze_pose=False,         # Fine-tune pose modules
)

# Training
optimizer = dict(
    type='AdamW',
    lr=1e-4,            # Learning rate (lower for fine-tuning)
    weight_decay=0.01,
)

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=100,
)

max_epochs = 50
evaluation = dict(interval=5)  # Validate every 5 epochs

# Mixed precision
fp16 = dict(enabled=True)
```

## Hardware Requirements

| GPU | VRAM | Batch Size | Est. Training Time |
|-----|------|------------|-------------------|
| RTX 3080 | 10GB | 1 | ~48 hours |
| RTX 3090 | 24GB | 2 | ~24 hours |
| RTX 4090 | 24GB | 2-3 | ~16-20 hours |
| A100 40GB | 40GB | 4 | ~10-12 hours |
| A100 80GB | 80GB | 8 | ~6-8 hours |

### VRAM Usage (approximate)

| Batch Size | VRAM Used |
|------------|-----------|
| 1 | ~13GB |
| 2 | ~22GB |
| 3 | ~32GB |
| 4 | ~40GB |
| 8 | ~70GB |

## Quick Start

### 1. Download pretrained model

```bash
mkdir -p pretrained/occworld
curl -L -o pretrained/occworld/latest.pth \
  'https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/files/?p=/latest.pth&dl=1'
```

### 2. Generate training data

```bash
python scripts/plateau_to_occworld.py \
  --input data/plateau/meshes/obj \
  --output data/tokyo_gazebo \
  --frames 500 --sessions 10
```

### 3. Run training

```bash
# Full training
./scripts/run_training.sh --batch-size 3

# Or step by step
python train.py \
  --py-config config/finetune_tokyo.py \
  --work-dir checkpoints \
  --batch-size 3 \
  --epochs 50
```

### 4. Resume after interruption

```bash
./scripts/run_training.sh --resume --batch-size 3
```

### 5. Monitor with TensorBoard

```bash
tensorboard --logdir checkpoints --port 6007
```

## Output Files

After training:

```
checkpoints/
├── epoch_5.pth         # Checkpoint at epoch 5
├── epoch_10.pth        # Checkpoint at epoch 10
├── ...
├── epoch_50.pth        # Final checkpoint
├── best.pth            # Best validation loss
├── logs/               # TensorBoard logs
│   └── events.out.*
└── train_*.log         # Training logs
```

## Expected Metrics

| Metric | Good Value | Description |
|--------|------------|-------------|
| Train Loss | < 0.01 | Reconstruction loss |
| Val Loss | < 0.05 | Validation loss |
| mIoU | > 0.3 | Mean intersection over union |
| VPQ | > 0.2 | Video panoptic quality |

## Troubleshooting

### Out of memory

Reduce batch size:
```bash
./scripts/run_training.sh --resume --batch-size 1
```

### Training too slow

Increase batch size (if VRAM allows):
```bash
./scripts/run_training.sh --resume --batch-size 4
```

### Loss not decreasing

- Check data quality: `ls -la data/tokyo_gazebo/*/occupancy/`
- Try lower learning rate: `--lr 5e-5`
- Check for NaN: Look for "nan" in training output

### Resume not working

```bash
# Check for checkpoints
ls -la checkpoints/checkpoints/

# Resume from specific checkpoint
python train.py --py-config config/finetune_tokyo.py --resume-from checkpoints/checkpoints/epoch_10.pth
```
