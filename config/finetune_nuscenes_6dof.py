"""
nuScenes 6DoF Training Configuration

Train 6DoF world models using nuScenes data with geometric augmentation
to simulate aerial viewpoints from ground-level driving data.

Research contribution:
- Domain adaptation: Ground vehicles → Aerial drones
- Geometric augmentation: Pitch/roll rotation simulates aerial viewpoints
- Cross-domain evaluation potential

Usage:
    python train.py --config config/finetune_nuscenes_6dof.py --work-dir out/nuscenes_6dof

Dataset requirements:
    - nuScenes v1.0-mini or v1.0-trainval
    - Download from: https://www.nuscenes.org/download
    - Install: pip install nuscenes-devkit pyquaternion
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================================
# Dataset Configuration
# ============================================================================

# Path to nuScenes data
data_root = os.environ.get('NUSCENES_ROOT', os.path.join(PROJECT_ROOT, 'data/nuscenes'))

# nuScenes version: 'v1.0-mini' for testing, 'v1.0-trainval' for full training
nuscenes_version = 'v1.0-mini'

# Dataset type flag for train.py
dataset_type = 'nuscenes_6dof'

# Standard nuScenes occupancy range (smaller Z range than PLATEAU/aerial)
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
frame_skip = 1

# ============================================================================
# 6DoF Augmentation Configuration (KEY RESEARCH CONTRIBUTION)
# ============================================================================

augmentation_6dof = dict(
    enabled=True,

    # Pitch augmentation: Simulate drone looking up/down
    # Ground vehicles: ~0° pitch
    # Drones in forward flight: typically ±30° pitch
    max_pitch_deg=30.0,

    # Roll augmentation: Simulate drone banking
    # Ground vehicles: ~0° roll
    # Drones in aggressive maneuvers: can be ±45° or more
    max_roll_deg=45.0,

    # Yaw augmentation: Additional rotation diversity
    # Ground vehicles already have full yaw range, but add more
    max_yaw_deg=180.0,

    # Altitude shift: Simulate different flying heights
    # Positive = higher altitude view, negative = lower
    altitude_shift_range=(-2.0, 10.0),  # meters

    # Apply same augmentation to entire temporal window
    # This maintains temporal consistency (important for motion prediction)
    consistent_augmentation=True,

    # Probability of applying augmentation (for ablation: set to 0.0)
    augmentation_prob=0.8,
)

# Dataset config for NuScenes6DoFDataset
dataset_config = dict(
    version=nuscenes_version,
    history_frames=history_frames,
    future_frames=future_frames,
    frame_skip=frame_skip,
    point_cloud_range=tuple(point_cloud_range),
    voxel_size=tuple(voxel_size),
    pose_dim=13,

    # 6DoF augmentation settings
    augment_6dof=augmentation_6dof['enabled'],
    max_pitch_deg=augmentation_6dof['max_pitch_deg'],
    max_roll_deg=augmentation_6dof['max_roll_deg'],
    max_yaw_deg=augmentation_6dof['max_yaw_deg'],
    altitude_shift_range=augmentation_6dof['altitude_shift_range'],
    consistent_augmentation=augmentation_6dof['consistent_augmentation'],
    augmentation_prob=augmentation_6dof['augmentation_prob'],
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

    # Grid dimensions (from nuScenes range)
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

work_dir = os.path.join(PROJECT_ROOT, 'out/nuscenes_6dof')


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
# Ablation Study Configurations
# ============================================================================
#
# To run ablation studies, create copies of this config with variations:
#
# 1. No augmentation (baseline):
#    augmentation_6dof['enabled'] = False
#
# 2. Pitch only:
#    augmentation_6dof['max_roll_deg'] = 0.0
#    augmentation_6dof['max_yaw_deg'] = 0.0
#
# 3. Roll only:
#    augmentation_6dof['max_pitch_deg'] = 0.0
#    augmentation_6dof['max_yaw_deg'] = 0.0
#
# 4. Small augmentation:
#    augmentation_6dof['max_pitch_deg'] = 10.0
#    augmentation_6dof['max_roll_deg'] = 15.0
#
# 5. Large augmentation:
#    augmentation_6dof['max_pitch_deg'] = 45.0
#    augmentation_6dof['max_roll_deg'] = 60.0
#
# ============================================================================
