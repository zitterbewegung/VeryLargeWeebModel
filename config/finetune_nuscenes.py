"""
OccWorld Fine-tuning Configuration for nuScenes Dataset

This config uses the nuScenes mini dataset for training.

Usage:
    python train.py --config config/finetune_nuscenes.py --work-dir /workspace/checkpoints
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# =============================================================================
# Dataset Configuration
# =============================================================================

# Dataset type (auto-detected, but can be explicit)
dataset_type = 'nuscenes'

# Dataset paths
data_root = os.path.join(PROJECT_ROOT, 'data/nuscenes')

# nuScenes standard point cloud range
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
voxel_size = [0.4, 0.4, 0.4]

# Calculate grid dimensions
grid_size = [
    int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),  # 200
    int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),  # 200
    int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),  # 16
]

# Temporal configuration
history_frames = 4
future_frames = 6

# Dataset config dict
dataset_config = dict(
    version='v1.0-mini',  # or 'v1.0-trainval' for full dataset
    history_frames=history_frames,
    future_frames=future_frames,
    point_cloud_range=tuple(point_cloud_range),
    voxel_size=tuple(voxel_size),
    grid_size=tuple(grid_size),
)

# =============================================================================
# Model Configuration
# =============================================================================

model = dict(
    embed_dim=64,
    num_heads=8,
    num_layers=2,
)

# =============================================================================
# Training Configuration
# =============================================================================

# Training hyperparameters
max_epochs = 100
batch_size = 2
learning_rate = 1e-4
weight_decay = 0.01

# Optimizer
optimizer = dict(
    type='AdamW',
    lr=learning_rate,
    weight_decay=weight_decay,
    betas=(0.9, 0.999),
)

# Learning rate schedule
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1,
)

# =============================================================================
# Logging and Checkpointing
# =============================================================================

checkpoint_interval = 10
work_dir = os.path.join(PROJECT_ROOT, 'out/occworld_nuscenes')

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)

# =============================================================================
# Runtime
# =============================================================================

seed = 42
cudnn_benchmark = True
