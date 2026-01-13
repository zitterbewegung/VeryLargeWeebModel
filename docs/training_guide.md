# OccWorld Training Guide for Tokyo Gazebo Dataset

This guide covers downloading pretrained models, preparing datasets, and training/fine-tuning OccWorld on the Tokyo Gazebo simulation dataset.

---

## Quick Start

```bash
# Download everything (PLATEAU models, pretrained weights, setup directories)
./scripts/download_and_prepare_data.sh --all

# Or download specific components
./scripts/download_and_prepare_data.sh --plateau      # Tokyo 3D city models only
./scripts/download_and_prepare_data.sh --models       # Pretrained models only
./scripts/download_and_prepare_data.sh --skip-plateau # Skip large PLATEAU download
```

---

## Project Architecture

This project provides a **simulation and data generation framework** for training OccWorld models on Tokyo urban environments. It can work in two modes:

### Standalone Mode (Recommended for Quick Start)
```
VeryLargeWeebModel/
├── train.py                  # Training script (included)
├── config/finetune_tokyo.py  # Configuration
├── dataset/                  # Custom Gazebo dataset loader
└── data/tokyo_gazebo/        # Generated simulation data
```

Run training directly:
```bash
python train.py --config config/finetune_tokyo.py --work-dir out/occworld_tokyo
```

### OccWorld Integration Mode (Full Features)
For full OccWorld functionality (VQVAE, advanced architectures):
```
~/OccWorld/                   # Clone from github.com/wzzheng/OccWorld
├── train.py                  # OccWorld's training script
├── models/                   # Full model implementations
└── config/                   # OccWorld configs

~/VeryLargeWeebModel/         # This project
├── config/finetune_tokyo.py  # Use with OccWorld
├── dataset/                  # Dataset loader (import in OccWorld)
└── data/tokyo_gazebo/        # Training data
```

Run with OccWorld:
```bash
cd ~/OccWorld
python train.py --py-config ~/VeryLargeWeebModel/config/finetune_tokyo.py
```

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Model Downloads](#2-model-downloads)
3. [Dataset Preparation](#3-dataset-preparation)
4. [Training from Scratch](#4-training-from-scratch)
5. [Fine-tuning on Tokyo Dataset](#5-fine-tuning-on-tokyo-dataset)
6. [Evaluation](#6-evaluation)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Prerequisites

### 1.1 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3080 (10GB) | RTX 4090 (24GB) |
| RAM | 32GB | 64GB |
| Storage | 500GB SSD | 1TB NVMe |
| CUDA | 11.3+ | 11.8+ |

### 1.2 Software Dependencies

```bash
# Create conda environment
conda create -n occworld python=3.8.0 -y
conda activate occworld

# Install PyTorch (CUDA 11.8)
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install MMDetection3D dependencies
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install mmdet==2.20.0
pip install mmsegmentation==0.20.0

# Install additional dependencies
pip install \
    nuscenes-devkit \
    torchpack \
    tqdm \
    open3d \
    scipy \
    opencv-python \
    pillow==8.4.0

# Clone and install OccWorld
git clone https://github.com/wzzheng/OccWorld.git
cd OccWorld
pip install -e .
```

---

## 2. Model Downloads

### 2.1 OccWorld Pretrained Models

Download from Tsinghua Cloud:

```bash
# Create directories
mkdir -p pretrained/occworld
mkdir -p pretrained/vqvae

# Download pretrained VQVAE (required for training OccWorld)
wget -O pretrained/vqvae/epoch_125.pth \
    "https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/files/?p=%2Fvqvae_epoch_125.pth&dl=1"

# Download pretrained OccWorld
wget -O pretrained/occworld/latest.pth \
    "https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/files/?p=%2Foccworld_latest.pth&dl=1"
```

**Direct Download Links:**
- **Pretrained Models**: https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/
- **Pickle Files**: https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/

### 2.2 BEVFusion Pretrained Models

```bash
# Clone BEVFusion
git clone https://github.com/mit-han-lab/bevfusion.git
cd bevfusion

# Download pretrained models
./tools/download_pretrained.sh

# Models will be saved to ./pretrained/
# - bevfusion-det.pth (Camera+LiDAR detection)
# - bevfusion-seg.pth (Camera+LiDAR segmentation)
```

**Manual Downloads (if script fails):**

| Model | Task | Performance | Download |
|-------|------|-------------|----------|
| BEVFusion (C+L) | Detection | mAP: 68.52, NDS: 71.38 | [Dropbox](https://www.dropbox.com/s/bevfusion-det.pth) |
| BEVFusion (C+L) | Segmentation | mIoU: 62.95 | [Dropbox](https://www.dropbox.com/s/bevfusion-seg.pth) |

### 2.3 Occ3D Ground Truth

```bash
# Download Occ3D semantic occupancy ground truth
git clone https://github.com/Tsinghua-MARS-Lab/Occ3D.git
cd Occ3D

# Follow their instructions to download gts.tar.gz
# Extract to OccWorld/data/gts/
```

---

## 3. Dataset Preparation

### 3.1 nuScenes Dataset (Original OccWorld)

```bash
# Directory structure
OccWorld/data/
├── nuscenes/
│   ├── lidarseg/
│   ├── maps/
│   ├── samples/
│   ├── sweeps/
│   └── v1.0-trainval/
├── gts/                    # From Occ3D
├── nuscenes_infos_train_temporal_v3_scene.pkl
└── nuscenes_infos_val_temporal_v3_scene.pkl
```

**Download nuScenes:**
1. Register at https://www.nuscenes.org/
2. Download Full dataset (v1.0)
3. Download lidarseg add-on

**Download pickle files:**
```bash
wget -O data/nuscenes_infos_train_temporal_v3_scene.pkl \
    "https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/files/?p=%2Fnuscenes_infos_train_temporal_v3_scene.pkl&dl=1"

wget -O data/nuscenes_infos_val_temporal_v3_scene.pkl \
    "https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/files/?p=%2Fnuscenes_infos_val_temporal_v3_scene.pkl&dl=1"
```

### 3.2 Tokyo Gazebo Dataset (This Project)

Generate training data using the simulation stack:

```bash
# 1. Install simulation stack
./scripts/install_ardupilot_gazebo_stack.sh
source ~/.occworld_env.sh

# 2. Launch simulation with recording
./scripts/launch_occworld_simulation.sh \
    --drones 1 \
    --rovers 1 \
    --record \
    --headless \
    --fast

# 3. Run data collection missions (in separate terminal)
python3 scripts/data_collection_mission.py \
    --vehicle drone \
    --pattern survey \
    --size 100 \
    --spacing 15 \
    --altitude 30

# 4. Generate occupancy ground truth
python3 scripts/generate_occupancy_gt.py \
    --input data/drone_*/  \
    --output data/tokyo_processed/
```

**Expected Tokyo Dataset Structure:**
```
data/tokyo_gazebo/
├── drone_20240101_120000/
│   ├── images/
│   │   ├── 000001_CAM_FRONT.jpg
│   │   ├── 000001_CAM_FRONT_LEFT.jpg
│   │   ├── 000001_CAM_FRONT_RIGHT.jpg
│   │   ├── 000001_CAM_BACK.jpg
│   │   ├── 000001_CAM_BACK_LEFT.jpg
│   │   └── 000001_CAM_BACK_RIGHT.jpg
│   ├── lidar/
│   │   └── 000001_LIDAR.npy
│   ├── poses/
│   │   └── 000001.json
│   └── occupancy/
│       └── 000001_occupancy.npz
└── rover_20240101_130000/
    └── ...
```

---

## 4. Training from Scratch

### 4.1 Train VQVAE First

The VQVAE learns to encode occupancy grids into discrete tokens:

```bash
# Train VQVAE on nuScenes
python train.py \
    --py-config config/train_vqvae.py \
    --work-dir out/vqvae

# Training takes ~24 hours on RTX 4090
# Checkpoint saved every epoch to out/vqvae/
```

**VQVAE Config Highlights:**
```python
# config/train_vqvae.py
model = dict(
    type='TransVQVAE',
    encoder=dict(base_channel=64),
    decoder=dict(base_channel=64),
    vq=dict(num_embeddings=512, embedding_dim=64),
)

optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.01)
max_epochs = 200
batch_size = 1
```

### 4.2 Train OccWorld

After VQVAE training, train the world model:

```bash
# Update config with VQVAE checkpoint path
# Edit config/train_occworld.py:
#   vqvae_ckpt = 'out/vqvae/epoch_125.pth'

python train.py \
    --py-config config/train_occworld.py \
    --work-dir out/occworld
```

**OccWorld Config Highlights:**
```python
# config/train_occworld.py
model = dict(
    type='OccWorld',
    vqvae_ckpt='out/vqvae/epoch_125.pth',
    transformer=dict(
        num_layers=2,
        temporal_attn_layers=6,
    ),
    pose_encoder=dict(num_layers=2),
    freeze_vae=True,  # Keep VQVAE frozen
)

optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.01)
max_epochs = 200
warmup_iters = 50
grad_clip = dict(max_norm=35)
```

---

## 5. Fine-tuning on Tokyo Dataset

### 5.1 Create Tokyo Dataset Config

Create `config/tokyo_dataset.py`:

```python
"""Tokyo Gazebo Dataset Configuration for OccWorld Fine-tuning."""

# Dataset settings
data_root = 'data/tokyo_gazebo/'
agent_type = 'both'  # 'drone', 'rover', or 'both'

# Extended occupancy range for aerial vehicles
point_cloud_range = [-40, -40, -2, 40, 40, 150]  # Extended Z for drones
voxel_size = [0.4, 0.4, 1.25]  # Coarser Z voxels for extended range

# Temporal settings
history_frames = 4
future_frames = 6
frame_skip = 1

# Data splits
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Dataloader settings
batch_size = 1  # Increase if GPU memory allows
num_workers = 4
```

### 5.2 Create Fine-tuning Config

Create `config/finetune_tokyo.py`:

```python
"""OccWorld Fine-tuning Configuration for Tokyo Gazebo Dataset."""

_base_ = ['./train_occworld.py']

# ============ Dataset Override ============
data_root = 'data/tokyo_gazebo/'

# Use the custom Gazebo dataset loader
dataset_type = 'GazeboOccWorldDataset'

# Extended point cloud range for drone altitude
point_cloud_range = [-40, -40, -2, 40, 40, 150]
voxel_size = [0.4, 0.4, 1.25]

# Override data config
data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        config=dict(
            history_frames=4,
            future_frames=6,
            agent_type='both',
            split='train',
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
        ),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        config=dict(
            history_frames=4,
            future_frames=6,
            agent_type='both',
            split='val',
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
        ),
    ),
)

# ============ Model Override ============
# Load pretrained OccWorld weights
load_from = 'pretrained/occworld/latest.pth'

# Optionally unfreeze more layers for fine-tuning
model = dict(
    freeze_vae=True,           # Keep VQVAE frozen
    freeze_transformer=False,  # Fine-tune transformer
    freeze_pose=False,         # Fine-tune pose modules
)

# ============ Training Override ============
# Lower learning rate for fine-tuning
optimizer = dict(
    type='AdamW',
    lr=1e-4,           # Reduced from 1e-3
    weight_decay=0.01,
)

# Fewer epochs for fine-tuning
max_epochs = 50

# Learning rate schedule
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1,
)

# Gradient clipping (important for stability)
grad_clip = dict(max_norm=35)

# Logging
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)

# Checkpointing
checkpoint_config = dict(
    interval=5,
    max_keep_ckpts=3,
)
```

### 5.3 Integrate Gazebo Dataset Loader

Register the custom dataset in `OccWorld/datasets/__init__.py`:

```python
# Add this import
import sys
sys.path.insert(0, '/path/to/VeryLargeWeebModel')
from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset, DatasetConfig

# Register dataset
DATASETS.register_module()(GazeboOccWorldDataset)
```

### 5.4 Run Fine-tuning

```bash
# Fine-tune OccWorld on Tokyo Gazebo data
python train.py \
    --py-config config/finetune_tokyo.py \
    --work-dir out/occworld_tokyo \
    --resume-from pretrained/occworld/latest.pth

# Monitor training
tensorboard --logdir out/occworld_tokyo/
```

### 5.5 Fine-tuning Strategies

#### Strategy A: Full Fine-tuning
```python
# Unfreeze all components (higher risk of forgetting)
model = dict(
    freeze_vae=False,
    freeze_transformer=False,
    freeze_pose=False,
)
optimizer = dict(lr=5e-5)  # Very low LR
```

#### Strategy B: Head-only Fine-tuning
```python
# Only fine-tune prediction heads (fastest, most stable)
model = dict(
    freeze_vae=True,
    freeze_transformer=True,
    freeze_pose=False,  # Only pose module trainable
)
optimizer = dict(lr=1e-4)
```

#### Strategy C: Progressive Unfreezing
```python
# Start frozen, gradually unfreeze
# Epoch 1-10: Only pose module
# Epoch 11-30: + Transformer
# Epoch 31-50: + VQVAE decoder

# Implement via custom training hook
```

### 5.6 Domain Adaptation Tips

1. **Data Augmentation**: Add noise, weather effects
   ```python
   transform = dict(
       RandomFlip=dict(prob=0.5),
       RandomRotate=dict(angle=[-5, 5]),
       PointCloudNoise=dict(std=0.02),
   )
   ```

2. **Mixed Training**: Combine nuScenes and Tokyo data
   ```python
   data = dict(
       train=dict(
           type='ConcatDataset',
           datasets=[
               dict(type='NuScenesDataset', ...),
               dict(type='GazeboOccWorldDataset', ...),
           ],
           sample_weights=[0.7, 0.3],  # 70% nuScenes, 30% Tokyo
       ),
   )
   ```

3. **Agent-specific Models**: Train separate models for drone/rover
   ```python
   # For drone-only model
   data = dict(train=dict(config=dict(agent_type='drone')))

   # For rover-only model
   data = dict(train=dict(config=dict(agent_type='rover')))
   ```

---

## 6. Evaluation

### 6.1 Evaluate on Tokyo Dataset

```bash
python eval_metric_stp3.py \
    --py-config config/finetune_tokyo.py \
    --work-dir out/occworld_tokyo \
    --checkpoint out/occworld_tokyo/epoch_50.pth
```

### 6.2 Metrics

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Intersection over Union for occupancy |
| **VPQ** | Video Panoptic Quality for temporal consistency |
| **ADE** | Average Displacement Error for trajectory prediction |
| **FDE** | Final Displacement Error at prediction horizon |

### 6.3 Visualization

```bash
# Generate visualization videos
python visualize_demo.py \
    --py-config config/finetune_tokyo.py \
    --work-dir out/occworld_tokyo \
    --checkpoint out/occworld_tokyo/epoch_50.pth \
    --output-dir visualizations/
```

---

## 7. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA OOM | Batch size too large | Reduce `batch_size` to 1, enable gradient checkpointing |
| NaN loss | Learning rate too high | Reduce `lr` by 10x, check data normalization |
| Poor convergence | Domain gap | Use mixed training, more aggressive augmentation |
| Slow training | Dataloader bottleneck | Increase `num_workers`, use SSD storage |
| Missing cameras | Incomplete recording | Check `_validate_sequence()` in dataset |

### Memory Optimization

```python
# Enable gradient checkpointing
model = dict(
    use_checkpoint=True,
)

# Use mixed precision
fp16 = dict(loss_scale='dynamic')

# Reduce voxel resolution
voxel_size = [0.8, 0.8, 2.5]  # Coarser voxels = less memory
```

### Debugging Data Pipeline

```bash
# Test dataset loading
python -c "
from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset, DatasetConfig
config = DatasetConfig(history_frames=4, future_frames=6)
ds = GazeboOccWorldDataset('data/tokyo_gazebo/', config)
print(f'Dataset size: {len(ds)}')
sample = ds[0]
print(f'History occupancy shape: {sample[\"history_occupancy\"].shape}')
"
```

---

## Quick Reference

### Download Commands

```bash
# OccWorld models
wget "https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/..."

# Pickle files
wget "https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/..."

# BEVFusion
cd bevfusion && ./tools/download_pretrained.sh
```

### Training Commands

```bash
# VQVAE (from scratch)
python train.py --py-config config/train_vqvae.py --work-dir out/vqvae

# OccWorld (from scratch)
python train.py --py-config config/train_occworld.py --work-dir out/occworld

# Fine-tune on Tokyo
python train.py --py-config config/finetune_tokyo.py --work-dir out/occworld_tokyo
```

### Key Files

| File | Purpose |
|------|---------|
| `dataset/gazebo_occworld_dataset.py` | Tokyo Gazebo data loader |
| `config/finetune_tokyo.py` | Fine-tuning configuration |
| `scripts/data_collection_mission.py` | Automated data collection |
| `scripts/generate_occupancy_gt.py` | Occupancy ground truth generation |

---

## Data Attribution

### Tokyo PLATEAU Dataset

This project uses 3D city model data from **Project PLATEAU**, provided by the Ministry of Land, Infrastructure, Transport and Tourism (MLIT), Japan.

- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Commercial use allowed)
- **Source:** https://www.geospatial.jp/ckan/dataset/plateau-tokyo23ku
- **Project Website:** https://www.mlit.go.jp/plateau/

**Required Attribution:**
> 3D city model data provided by Ministry of Land, Infrastructure, Transport and Tourism (MLIT), Japan - Project PLATEAU. Licensed under CC BY 4.0.

---

## References

- [OccWorld Paper (ECCV 2024)](https://arxiv.org/abs/2311.16038)
- [OccWorld GitHub](https://github.com/wzzheng/OccWorld)
- [BEVFusion GitHub](https://github.com/mit-han-lab/bevfusion)
- [Occ3D Project](https://github.com/Tsinghua-MARS-Lab/Occ3D)
- [nuScenes Dataset](https://www.nuscenes.org/)
- [Project PLATEAU](https://www.mlit.go.jp/plateau/)
