"""
OccWorld Fine-tuning Configuration for Tokyo Gazebo Dataset

This config extends the base OccWorld training config to fine-tune
on simulated Tokyo urban environment data from Gazebo.

Usage:
    python train.py --py-config config/finetune_tokyo.py --work-dir out/occworld_tokyo

Requirements:
    - Pretrained OccWorld checkpoint in pretrained/occworld/
    - Tokyo Gazebo data in data/tokyo_gazebo/
    - GazeboOccWorldDataset registered in datasets
"""

import os
import sys

# Add project root to path for custom dataset
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================================
# Dataset Configuration
# ============================================================================

# Data paths
data_root = os.path.join(PROJECT_ROOT, 'data/tokyo_gazebo/')

# Extended occupancy range for aerial vehicles (drones fly up to 150m)
point_cloud_range = [-40.0, -40.0, -2.0, 40.0, 40.0, 150.0]
voxel_size = [0.4, 0.4, 1.25]  # Coarser Z voxels for extended altitude range

# Calculate grid dimensions
grid_size = [
    int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),  # 200
    int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),  # 200
    int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),  # 121
]

# Temporal configuration
history_frames = 4    # Past frames for context
future_frames = 6     # Future frames to predict
frame_skip = 1        # Use every frame (increase for temporal diversity)

# Dataset config dict for GazeboOccWorldDataset
dataset_config = dict(
    history_frames=history_frames,
    future_frames=future_frames,
    frame_skip=frame_skip,
    agent_type='both',  # 'drone', 'rover', or 'both'
    val_ratio=0.1,
    test_ratio=0.1,
    image_size=(900, 1600),
    normalize_images=True,
    max_points=100000,
    point_cloud_range=tuple(point_cloud_range),
    voxel_size=tuple(voxel_size),
)

# Data pipeline
data = dict(
    samples_per_gpu=1,  # Batch size per GPU
    workers_per_gpu=4,
    train=dict(
        type='GazeboOccWorldDataset',
        data_root=data_root,
        config=dict(**dataset_config, split='train'),
    ),
    val=dict(
        type='GazeboOccWorldDataset',
        data_root=data_root,
        config=dict(**dataset_config, split='val'),
    ),
    test=dict(
        type='GazeboOccWorldDataset',
        data_root=data_root,
        config=dict(**dataset_config, split='test'),
    ),
)


# ============================================================================
# Model Configuration
# ============================================================================

# Pretrained model paths
vqvae_checkpoint = os.path.join(PROJECT_ROOT, 'pretrained/vqvae/epoch_125.pth')
occworld_checkpoint = os.path.join(PROJECT_ROOT, 'pretrained/occworld/latest.pth')

# Load from pretrained OccWorld for fine-tuning
load_from = occworld_checkpoint

# Model architecture (must match pretrained)
model = dict(
    type='TransVQVAE',

    # VQVAE components (frozen during fine-tuning)
    encoder_2d=dict(
        type='VAEEncoder2D',
        base_channel=64,
    ),
    decoder_2d=dict(
        type='VAEDecoder2D',
        base_channel=64,
    ),
    vq=dict(
        type='VectorQuantizer',
        num_embeddings=512,
        embedding_dim=64,
        commitment_cost=0.25,
    ),

    # Transformer for temporal modeling
    transformer=dict(
        type='OccWorldTransformer',
        num_layers=2,
        temporal_attn_layers=6,
        embed_dim=64,
        num_heads=8,
        dropout=0.1,
    ),

    # Pose encoder/decoder
    pose_encoder=dict(
        type='PoseEncoder',
        input_dim=13,  # x,y,z, quat(4), linear_vel(3), angular_vel(3)
        hidden_dim=256,
        output_dim=64,
        num_layers=2,
    ),
    pose_decoder=dict(
        type='PoseDecoder',
        input_dim=64,
        hidden_dim=256,
        output_dim=13,
        num_layers=2,
    ),

    # Freezing config for fine-tuning
    freeze_vae=True,           # Keep VQVAE frozen (learned representations)
    freeze_transformer=False,  # Fine-tune transformer
    freeze_pose=False,         # Fine-tune pose modules

    # Use pretrained VQVAE
    vqvae_ckpt=vqvae_checkpoint,
)


# ============================================================================
# Training Configuration
# ============================================================================

# Optimizer (lower LR for fine-tuning)
optimizer = dict(
    type='AdamW',
    lr=1e-4,           # Reduced from 1e-3 for fine-tuning
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

# Optimizer wrapper for gradient clipping
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

# Training epochs (fewer for fine-tuning)
max_epochs = 50
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

# Validation
evaluation = dict(
    interval=5,      # Validate every 5 epochs
    metric=['mIoU', 'VPQ'],
    save_best='mIoU',
)


# ============================================================================
# Loss Configuration
# ============================================================================

loss = dict(
    # Occupancy prediction loss
    occupancy_loss=dict(
        type='CrossEntropyLoss',
        weight=1.0,
        ignore_index=0,  # Ignore empty voxels
    ),
    # Pose prediction loss
    pose_loss=dict(
        type='SmoothL1Loss',
        weight=0.1,
        beta=1.0,
    ),
    # Flow field loss (optional)
    flow_loss=dict(
        type='L1Loss',
        weight=0.05,
    ),
)


# ============================================================================
# Logging and Checkpointing
# ============================================================================

# Logging
log_config = dict(
    interval=10,  # Log every 10 iterations
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)

# Checkpointing
checkpoint_config = dict(
    interval=5,        # Save every 5 epochs
    max_keep_ckpts=3,  # Keep last 3 checkpoints
    save_last=True,
)

# Work directory (override via command line)
work_dir = os.path.join(PROJECT_ROOT, 'out/occworld_tokyo')


# ============================================================================
# Runtime Configuration
# ============================================================================

# Distributed training
dist_params = dict(backend='nccl')

# Logging level
log_level = 'INFO'

# Resume from checkpoint (set via command line)
resume_from = None

# Random seed for reproducibility
seed = 42
deterministic = False

# CUDA settings
cudnn_benchmark = True

# Mixed precision training (saves memory)
fp16 = dict(
    loss_scale='dynamic',
    enabled=True,
)


# ============================================================================
# Data Augmentation (Optional)
# ============================================================================

train_pipeline = [
    dict(type='LoadMultiViewImages'),
    dict(type='LoadLiDARPoints'),
    dict(type='LoadOccupancy'),
    # Augmentations
    dict(type='RandomFlip3D', flip_ratio=0.5),
    dict(type='RandomRotate', angle=(-5, 5), prob=0.5),
    dict(type='PointCloudNoise', std=0.02, prob=0.3),
    dict(type='ImageNormalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Format
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['history_images', 'history_lidar', 'history_occupancy',
                                'future_occupancy', 'history_poses', 'future_poses']),
]

val_pipeline = [
    dict(type='LoadMultiViewImages'),
    dict(type='LoadLiDARPoints'),
    dict(type='LoadOccupancy'),
    dict(type='ImageNormalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['history_images', 'history_lidar', 'history_occupancy',
                                'future_occupancy', 'history_poses', 'future_poses']),
]


# ============================================================================
# Agent-Specific Variants
# ============================================================================

# Uncomment one of these to train agent-specific models:

# --- Drone-only model ---
# dataset_config['agent_type'] = 'drone'
# point_cloud_range = [-40.0, -40.0, -2.0, 40.0, 40.0, 150.0]  # Full altitude

# --- Rover-only model ---
# dataset_config['agent_type'] = 'rover'
# point_cloud_range = [-40.0, -40.0, -2.0, 40.0, 40.0, 10.0]  # Ground level only
# voxel_size = [0.4, 0.4, 0.4]  # Finer Z resolution for ground
