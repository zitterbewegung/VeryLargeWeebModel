"""
Eval-only configuration for Tokyo Gazebo UAV environment.

Use with:
    python train.py --config config/eval_tokyo_gazebo.py --eval-only

Notes:
- This config is tuned to evaluate all sequences as validation data.
- Set exclude_dummy_sessions=True once real Gazebo sessions exist.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Dataset
dataset_type = 'gazebo'

data_root = os.path.join(PROJECT_ROOT, 'data/tokyo_gazebo')

# Aerial-friendly range (matches finetune_tokyo)
point_cloud_range = [-40.0, -40.0, -2.0, 40.0, 40.0, 150.0]
voxel_size = [0.4, 0.4, 1.25]

grid_size = [
    int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),  # 200
    int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),  # 200
    int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),  # 121
]

history_frames = 4
future_frames = 6
frame_skip = 1

# Treat all sequences as validation
# exclude_dummy_sessions=False so dummy data can be evaluated until real data exists.
dataset_config = dict(
    history_frames=history_frames,
    future_frames=future_frames,
    frame_skip=frame_skip,
    agent_type='both',
    val_ratio=1.0,
    test_ratio=0.0,
    exclude_dummy_sessions=False,
    point_cloud_range=tuple(point_cloud_range),
    voxel_size=tuple(voxel_size),
)

# Runtime
max_epochs = 1
work_dir = os.path.join(PROJECT_ROOT, 'out/eval_tokyo_gazebo')

seed = 42
cudnn_benchmark = True
