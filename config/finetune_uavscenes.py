"""
UAVScenes 6DoF Training Configuration

Train 6DoF world models using REAL aerial UAV data from UAVScenes.

UAVScenes provides:
- Multi-modal UAV data (LiDAR + Camera)
- Ground-truth 6DoF poses from actual flight
- 120k labeled semantic pairs
- 4 diverse scenes (urban + nature)

This is REAL aerial data, unlike nuScenes with augmentation.

Usage:
    python train.py --config config/finetune_uavscenes.py --work-dir out/uavscenes_6dof

Dataset requirements:
    - Download from: https://github.com/sijieaaa/UAVScenes
    - Install: pip install pyquaternion open3d
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================================
# Dataset Configuration
# ============================================================================

# Path to UAVScenes data
data_root = os.environ.get('UAVSCENES_ROOT', os.path.join(PROJECT_ROOT, 'data/uavscenes'))

# Dataset type flag for train.py
dataset_type = 'uavscenes'

# Scene selection
# Available: AMtown, AMvalley, HKairport, HKisland
# Note: Dataset gracefully handles missing scenes (warns and continues)
# Start with AMtown (most commonly downloaded), add more as they become available
uavscenes_scenes = ['AMtown']
# For full training with all scenes:
# uavscenes_scenes = ['AMtown', 'AMvalley', 'HKairport', 'HKisland']

# Interval: 5 = keyframes (available on HuggingFace), 1 = full data (not on HF)
uavscenes_interval = 5

# Aerial-friendly occupancy range (larger Z range for flight altitude)
point_cloud_range = [-40.0, -40.0, -10.0, 40.0, 40.0, 50.0]
voxel_size = [0.4, 0.4, 0.5]

# Calculate grid dimensions
grid_size = [
    int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),  # 200
    int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),  # 200
    int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),  # 120
]

# Temporal configuration
history_frames = 4
future_frames = 6
frame_skip = 1

# Dataset config
dataset_config = dict(
    scenes=uavscenes_scenes,
    interval=uavscenes_interval,
    history_frames=history_frames,
    future_frames=future_frames,
    frame_skip=frame_skip,
    point_cloud_range=tuple(point_cloud_range),
    voxel_size=tuple(voxel_size),
    ego_frame=True,
    fallback_to_lidar_center=True,
    min_in_range_ratio=0.01,
    pose_dim=13,
    val_ratio=0.1,
    test_ratio=0.1,
)

# Dataloader config
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)


# ============================================================================
# Model Configuration (6DoF)
# ============================================================================

model = dict(
    type='OccWorld6DoF',

    # Grid dimensions (aerial range)
    grid_size=tuple(grid_size),

    # Temporal
    history_frames=history_frames,
    future_frames=future_frames,

    # Encoder architecture
    encoder_channels=(64, 128, 256),

    # Temporal modeling
    use_transformer=False,
    num_transformer_layers=4,
    num_heads=8,
    transformer_dim=256,
    dropout=0.1,

    # 6DoF settings
    pose_dim=13,
    uncertainty_dim=6,
    place_embedding_dim=256,

    # Enable all 6DoF heads
    enable_uncertainty=True,
    enable_relocalization=True,
    enable_place_recognition=True,
)


# ============================================================================
# Loss Configuration
# ============================================================================

loss = dict(
    # Occupancy prediction
    occ_weight=1.0,
    pos_weight=10.0,

    # 6DoF pose prediction
    pose_weight=0.5,

    # Uncertainty estimation
    uncertainty_weight=0.1,

    # Relocalization
    reloc_weight=0.2,

    # Place recognition
    place_weight=0.1,
)


# ============================================================================
# Training Configuration
# ============================================================================

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
)

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1,
    by_epoch=True,
)

max_epochs = 50
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

evaluation = dict(
    interval=5,
    metric=['mIoU', 'pose_error', 'uncertainty_calibration'],
    save_best='total_loss',
)


# ============================================================================
# Logging and Checkpointing
# ============================================================================

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)

checkpoint_config = dict(
    interval=5,
    max_keep_ckpts=3,
    save_last=True,
)

work_dir = os.path.join(PROJECT_ROOT, 'out/uavscenes_6dof')


# ============================================================================
# Runtime Configuration
# ============================================================================

dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
seed = 42
deterministic = False
cudnn_benchmark = True

bf16 = dict(enabled=True)
fp16 = dict(enabled=False)


# ============================================================================
# Research Comparison: nuScenes vs UAVScenes
# ============================================================================
#
# | Aspect              | nuScenes + 6DoF Aug    | UAVScenes              |
# |---------------------|------------------------|------------------------|
# | Platform            | Ground vehicle         | UAV (aerial)           |
# | 6DoF Source         | Simulated via rotation | Real flight poses      |
# | Pitch/Roll          | Augmented ±30°/±45°    | Natural flight motion  |
# | Altitude variation  | Simulated              | Real altitude changes  |
# | Domain              | Urban driving          | Urban + nature         |
# | Best for            | Cross-domain learning  | Direct aerial training |
#
# Recommended training strategy:
# 1. Pre-train on nuScenes with 6DoF augmentation (more data)
# 2. Fine-tune on UAVScenes (domain adaptation)
# 3. Evaluate on UAVScenes test set
#
# ============================================================================
