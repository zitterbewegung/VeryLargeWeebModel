"""
OccWorld 6DoF Fine-tuning Configuration

Enhanced configuration for training OccWorld with 6DoF pose prediction,
uncertainty estimation, relocalization, and place recognition.

Usage:
    python train_6dof.py --config config/finetune_6dof.py --work-dir out/6dof

Features enabled:
    - Future occupancy prediction
    - 6DoF pose prediction with uncertainty
    - Global relocalization
    - Place recognition embeddings
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================================
# Dataset Configuration
# ============================================================================

data_root = os.path.join(PROJECT_ROOT, 'data/tokyo_gazebo/')

# Extended occupancy range for aerial vehicles
point_cloud_range = [-40.0, -40.0, -2.0, 40.0, 40.0, 150.0]
voxel_size = [0.4, 0.4, 1.25]

# Calculate grid dimensions
grid_size = [
    int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),  # 200
    int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),  # 200
    int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),  # 121
]

# Temporal configuration
history_frames = 4
future_frames = 6
frame_skip = 1

# Dataset config
dataset_config = dict(
    history_frames=history_frames,
    future_frames=future_frames,
    frame_skip=frame_skip,
    agent_type='both',
    val_ratio=0.1,
    test_ratio=0.1,
    exclude_dummy_sessions=True,
    image_size=(900, 1600),
    normalize_images=True,
    max_points=100000,
    point_cloud_range=tuple(point_cloud_range),
    voxel_size=tuple(voxel_size),
)

# Dataloader config
data = dict(
    samples_per_gpu=4,  # Reduced for 6DoF model (more memory)
    workers_per_gpu=4,
)


# ============================================================================
# Model Configuration (6DoF Enhanced)
# ============================================================================

model = dict(
    type='OccWorld6DoF',

    # Grid dimensions
    grid_size=tuple(grid_size),

    # Temporal
    history_frames=history_frames,
    future_frames=future_frames,

    # Encoder architecture
    encoder_channels=(64, 128, 256),

    # Temporal modeling
    use_transformer=False,  # Set True for transformer (more params, better for large data)
    num_transformer_layers=4,
    num_heads=8,
    transformer_dim=256,
    dropout=0.1,

    # 6DoF settings
    pose_dim=13,  # x,y,z, quat(4), linear_vel(3), angular_vel(3)
    uncertainty_dim=6,  # Position (3) + Orientation (3) covariance
    place_embedding_dim=256,

    # Enable/disable heads
    enable_uncertainty=True,
    enable_relocalization=True,
    enable_place_recognition=True,
)


# ============================================================================
# Loss Configuration
# ============================================================================

loss = dict(
    # Occupancy prediction (primary task)
    occ_weight=1.0,
    pos_weight=10.0,  # BCE weight for occupied voxels

    # 6DoF pose prediction
    pose_weight=0.5,

    # Uncertainty estimation (regularizes pose prediction)
    uncertainty_weight=0.1,

    # Relocalization (global pose correction)
    reloc_weight=0.2,

    # Place recognition (for loop closure)
    place_weight=0.1,
)


# ============================================================================
# Training Configuration
# ============================================================================

# Optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

# Gradient clipping
optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
)

# Learning rate schedule
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1,
    by_epoch=True,
)

# Training epochs
max_epochs = 50
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

# Validation
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

work_dir = os.path.join(PROJECT_ROOT, 'out/occworld_6dof')


# ============================================================================
# Runtime Configuration
# ============================================================================

dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
seed = 42
deterministic = False
cudnn_benchmark = True

# Mixed precision (recommended for 6DoF model)
bf16 = dict(enabled=True)
fp16 = dict(enabled=False)


# ============================================================================
# Data Augmentation
# ============================================================================

train_pipeline = [
    dict(type='LoadMultiViewImages'),
    dict(type='LoadLiDARPoints'),
    dict(type='LoadOccupancy'),

    # Spatial augmentations
    dict(type='RandomFlip3D', flip_ratio=0.5),
    dict(type='RandomRotate', angle=(-30, 30), prob=0.7),
    dict(type='RandomScale', scale_range=(0.95, 1.05), prob=0.3),

    # Point cloud augmentations
    dict(type='PointCloudNoise', std=0.03, prob=0.3),

    # Pose augmentations (important for 6DoF training)
    dict(type='PoseNoise', position_std=0.1, orientation_std=0.02, prob=0.3),

    # Format
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['history_images', 'history_lidar', 'history_occupancy',
                                'future_occupancy', 'history_poses', 'future_poses']),
]

val_pipeline = [
    dict(type='LoadMultiViewImages'),
    dict(type='LoadLiDARPoints'),
    dict(type='LoadOccupancy'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['history_images', 'history_lidar', 'history_occupancy',
                                'future_occupancy', 'history_poses', 'future_poses']),
]


# ============================================================================
# Expected Model Size
# ============================================================================
#
# Component              Parameters
# -----------------------------------
# spatial_encoder        ~1.1M
# pose_encoder           ~50K
# fusion                 ~130K
# temporal_encoder       ~1.0M
# spatial_decoder        ~2.6M
# pose_decoder           ~50K
# future_pose_rnn        ~200K
# uncertainty_head       ~20K
# relocalization_head    ~100K
# place_recognition_head ~130K
# -----------------------------------
# TOTAL                  ~5.4M params
# Model size (FP32)      ~22 MB
# Model size (FP16)      ~11 MB
# Inference VRAM         ~3-5 GB
# ============================================================================
