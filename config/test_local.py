"""
Small local test configuration for UAVScenes.
Uses reduced grid size for fast CPU testing.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Dataset
data_root = os.path.join(PROJECT_ROOT, 'data/uavscenes')
dataset_type = 'uavscenes'

# Only use AMtown (the only scene we have downloaded)
uavscenes_scenes = ['AMtown']
uavscenes_interval = 1

# SMALL grid for fast local testing (50x50x30 vs 200x200x120)
point_cloud_range = [-20.0, -20.0, -5.0, 20.0, 20.0, 25.0]
voxel_size = [0.8, 0.8, 1.0]  # Larger voxels = smaller grid

grid_size = [
    int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),  # 50
    int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),  # 50
    int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),  # 30
]

# Fewer temporal frames
history_frames = 2
future_frames = 2
frame_skip = 10  # Skip frames for faster iteration

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

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
)

# Smaller model
model = dict(
    type='OccWorld6DoF',
    grid_size=tuple(grid_size),
    history_frames=history_frames,
    future_frames=future_frames,
    encoder_channels=(32, 64, 128),  # Smaller channels
    use_transformer=False,
    num_transformer_layers=2,
    num_heads=4,
    transformer_dim=128,
    dropout=0.1,
    pose_dim=13,
    uncertainty_dim=6,
    place_embedding_dim=128,
    enable_uncertainty=True,
    enable_relocalization=True,
    enable_place_recognition=True,
)

loss = dict(
    occ_weight=1.0,
    pos_weight=10.0,
    pose_weight=0.5,
    uncertainty_weight=0.1,
    reloc_weight=0.2,
    place_weight=0.1,
)

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
    warmup_iters=10,
    warmup_ratio=0.1,
    by_epoch=True,
)

max_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

evaluation = dict(
    interval=1,
    metric=['mIoU', 'pose_error'],
    save_best='total_loss',
)

log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
    ],
)

checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=2,
    save_last=True,
)

work_dir = os.path.join(PROJECT_ROOT, 'out/test_local')

seed = 42
deterministic = False
cudnn_benchmark = False
