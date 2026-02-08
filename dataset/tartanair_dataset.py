#!/usr/bin/env python3
"""
TartanAir Dataset Loader for 6DoF World Model Training

TartanAir is a challenging synthetic dataset from CMU AirLab providing:
- Stereo RGB images
- Depth images (as .npy files)
- Semantic segmentation
- Perfect 6-DoF ground truth poses
- 30 diverse photo-realistic environments
- Various weather/lighting conditions

Dataset structure:
    data_root/
    ├── abandonedfactory/
    │   ├── Easy/
    │   │   ├── P000/
    │   │   │   ├── depth_left/000000_left_depth.npy
    │   │   │   ├── image_left/000000_left.png
    │   │   │   ├── seg_left/000000_left_seg.npy
    │   │   │   └── pose_left.txt
    │   │   ├── P001/
    │   │   └── ...
    │   └── Hard/
    │       └── ...
    ├── amusement/
    └── ...

Download from: https://theairlab.org/tartanair-dataset/

Usage:
    from dataset.tartanair_dataset import TartanAirDataset, TartanAirConfig

    config = TartanAirConfig(environments=['abandonedfactory', 'amusement'])
    dataset = TartanAirDataset('data/tartanair', config)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import cv2

cv2.setNumThreads(0)  # Avoid OpenCV threading issues


@dataclass
class TartanAirConfig:
    """Configuration for TartanAir dataset."""

    # Environment selection (None = all available)
    environments: Optional[List[str]] = None

    # Difficulty level: 'Easy', 'Hard', or 'both'
    difficulty: str = 'both'

    # Trajectory selection (None = all)
    trajectories: Optional[List[str]] = None  # e.g., ['P000', 'P001']

    # Temporal settings
    history_frames: int = 4
    future_frames: int = 6
    frame_skip: int = 1

    # Split
    split: str = 'train'
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Depth to occupancy conversion
    point_cloud_range: Tuple[float, ...] = (-40.0, -40.0, -10.0, 40.0, 40.0, 50.0)
    voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.5)

    # Depth settings
    max_depth: float = 100.0  # meters
    depth_scale: float = 1.0  # TartanAir depth is in meters

    # Image settings
    image_size: Tuple[int, int] = (480, 640)  # TartanAir default

    # Camera intrinsics (TartanAir default)
    fx: float = 320.0
    fy: float = 320.0
    cx: float = 320.0
    cy: float = 240.0

    # Pose format: 13D = position(3) + quaternion(4) + linear_vel(3) + angular_vel(3)
    pose_dim: int = 13

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """Calculate grid size from range and voxel size."""
        pc_range = np.array(self.point_cloud_range)
        voxel_sz = np.array(self.voxel_size)
        return tuple(((pc_range[3:] - pc_range[:3]) / voxel_sz).astype(int))


class TartanAirDataset(Dataset):
    """
    TartanAir dataset for 6DoF aerial world model training.

    Converts depth images to point clouds/occupancy grids for OccWorld training.
    """

    def __init__(self, data_root: str, config: TartanAirConfig, transform=None):
        self.data_root = Path(data_root)
        self.config = config
        self.transform = transform

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"TartanAir data not found at: {data_root}\n"
                f"Download from: https://theairlab.org/tartanair-dataset/"
            )

        # Build camera intrinsic matrix
        self.K = np.array([
            [config.fx, 0, config.cx],
            [0, config.fy, config.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Build sample index
        self.samples = self._build_sample_index()

        # Apply split
        self._apply_split()

        print(f"TartanAir {config.split}: {len(self.samples)} samples")
        print(f"  Grid size: {config.grid_size}")

    def _build_sample_index(self) -> List[Dict]:
        """Build index of all valid temporal windows."""
        samples = []

        # Find all environments
        if self.config.environments:
            envs = [e for e in self.config.environments if (self.data_root / e).exists()]
        else:
            envs = [d.name for d in self.data_root.iterdir() if d.is_dir() and not d.name.startswith('.')]

        if not envs:
            print(f"Warning: No environments found in {self.data_root}")
            return samples

        print(f"Found {len(envs)} environments: {envs[:5]}{'...' if len(envs) > 5 else ''}")

        for env in envs:
            env_path = self.data_root / env

            # Get difficulty levels
            if self.config.difficulty == 'both':
                difficulties = ['Easy', 'Hard']
            else:
                difficulties = [self.config.difficulty]

            for diff in difficulties:
                diff_path = env_path / diff
                if not diff_path.exists():
                    continue

                # Find trajectories
                if self.config.trajectories:
                    trajs = [t for t in self.config.trajectories if (diff_path / t).exists()]
                else:
                    trajs = sorted([d.name for d in diff_path.iterdir() if d.is_dir() and d.name.startswith('P')])

                for traj in trajs:
                    traj_path = diff_path / traj

                    # Load poses
                    pose_file = traj_path / 'pose_left.txt'
                    if not pose_file.exists():
                        continue

                    poses = self._load_poses(pose_file)
                    if len(poses) == 0:
                        continue

                    # Find depth files
                    depth_dir = traj_path / 'depth_left'
                    if not depth_dir.exists():
                        continue

                    depth_files = sorted(depth_dir.glob('*.npy'))
                    if not depth_files:
                        continue

                    # Build frame info
                    frame_info = []
                    for idx, depth_file in enumerate(depth_files):
                        if idx >= len(poses):
                            break
                        frame_num = int(depth_file.stem.split('_')[0])
                        frame_info.append({
                            'idx': idx,
                            'frame_num': frame_num,
                            'depth_path': depth_file,
                            'image_path': traj_path / 'image_left' / f'{frame_num:06d}_left.png',
                            'seg_path': traj_path / 'seg_left' / f'{frame_num:06d}_left_seg.npy',
                            'pose': poses[idx],
                        })

                    # Apply frame skip
                    if self.config.frame_skip > 1:
                        frame_info = frame_info[::self.config.frame_skip]

                    # Create sliding windows
                    total_frames = self.config.history_frames + self.config.future_frames
                    for i in range(len(frame_info) - total_frames + 1):
                        window = frame_info[i:i + total_frames]
                        samples.append({
                            'env': env,
                            'difficulty': diff,
                            'trajectory': traj,
                            'frames': window,
                            'history_frames': window[:self.config.history_frames],
                            'future_frames': window[self.config.history_frames:],
                        })

        return samples

    def _load_poses(self, pose_file: Path) -> np.ndarray:
        """
        Load poses from TartanAir pose file.

        Format: tx ty tz qx qy qz qw (NED frame)
        """
        try:
            poses = np.loadtxt(str(pose_file), dtype=np.float32)
            if poses.ndim == 1:
                poses = poses.reshape(1, -1)
            return poses
        except Exception as e:
            print(f"Warning: Failed to load poses from {pose_file}: {e}")
            return np.array([])

    def _apply_split(self):
        """Apply train/val/test split."""
        np.random.seed(42)

        n_samples = len(self.samples)
        indices = np.random.permutation(n_samples)

        n_val = int(n_samples * self.config.val_ratio)
        n_test = int(n_samples * self.config.test_ratio)
        n_train = n_samples - n_val - n_test

        if self.config.split == 'train':
            keep_indices = indices[:n_train]
        elif self.config.split == 'val':
            keep_indices = indices[n_train:n_train + n_val]
        else:
            keep_indices = indices[n_train + n_val:]

        self.samples = [self.samples[i] for i in keep_indices]

    def _depth_to_pointcloud(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth image to point cloud using camera intrinsics."""
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Backproject to 3D
        z = depth
        x = (u - self.config.cx) * z / self.config.fx
        y = (v - self.config.cy) * z / self.config.fy

        # Stack and filter invalid points
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # Filter by max depth and invalid values
        valid = (z.flatten() > 0) & (z.flatten() < self.config.max_depth)
        points = points[valid]

        return points.astype(np.float32)

    def _pointcloud_to_occupancy(self, points: np.ndarray) -> np.ndarray:
        """Convert point cloud to occupancy grid."""
        pc_range = np.array(self.config.point_cloud_range)
        voxel_size = np.array(self.config.voxel_size)
        grid_size = self.config.grid_size

        # Filter points within range
        mask = (
            (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) &
            (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]) &
            (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
        )
        points = points[mask]

        if len(points) == 0:
            return np.zeros(grid_size, dtype=np.float32)

        # Voxelize
        voxel_coords_float = (points - pc_range[:3]) / voxel_size
        voxel_coords_float = np.clip(voxel_coords_float, 0, np.array(grid_size) - 1)
        voxel_coords = voxel_coords_float.astype(np.int32)

        # Create occupancy grid
        occupancy = np.zeros(grid_size, dtype=np.float32)
        occupancy[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1.0

        return occupancy

    def _pose_to_tensor(self, pose: np.ndarray) -> np.ndarray:
        """
        Convert TartanAir pose to 13D tensor.

        Input: [tx, ty, tz, qx, qy, qz, qw]
        Output: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        """
        if len(pose) >= 7:
            # Reorder quaternion: TartanAir uses xyzw, we use wxyz
            position = pose[:3]
            quat = np.array([pose[6], pose[3], pose[4], pose[5]])  # wxyz
            # Velocities not provided, set to zero
            linear_vel = np.zeros(3, dtype=np.float32)
            angular_vel = np.zeros(3, dtype=np.float32)

            return np.concatenate([position, quat, linear_vel, angular_vel]).astype(np.float32)
        else:
            return np.zeros(13, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        history_occupancy = []
        history_poses = []
        future_occupancy = []
        future_poses = []

        # Process history frames
        for frame in sample['history_frames']:
            # Load depth and convert to occupancy
            depth = np.load(frame['depth_path'])
            points = self._depth_to_pointcloud(depth)
            occ = self._pointcloud_to_occupancy(points)
            history_occupancy.append(occ)

            # Process pose
            pose = self._pose_to_tensor(frame['pose'])
            history_poses.append(pose)

        # Process future frames
        for frame in sample['future_frames']:
            depth = np.load(frame['depth_path'])
            points = self._depth_to_pointcloud(depth)
            occ = self._pointcloud_to_occupancy(points)
            future_occupancy.append(occ)

            pose = self._pose_to_tensor(frame['pose'])
            future_poses.append(pose)

        result = {
            'history_occupancy': torch.from_numpy(np.stack(history_occupancy)),
            'history_poses': torch.from_numpy(np.stack(history_poses)),
            'future_occupancy': torch.from_numpy(np.stack(future_occupancy)),
            'future_poses': torch.from_numpy(np.stack(future_poses)),
            'agent_type': 1,  # Aerial
            'env': sample['env'],
            'trajectory': sample['trajectory'],
            'domain_tag': 'sim',
        }

        if self.transform is not None:
            result = self.transform(result)

        return result


def create_tartanair_dataloader(
    data_root: str,
    config: TartanAirConfig,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for TartanAir dataset."""
    dataset = TartanAirDataset(data_root, config)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


if __name__ == '__main__':
    # Test the dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/tartanair')
    parser.add_argument('--env', default=None, help='Specific environment')
    args = parser.parse_args()

    config = TartanAirConfig(
        environments=[args.env] if args.env else None,
        split='train',
    )

    try:
        dataset = TartanAirDataset(args.data_root, config)
        print(f"\nDataset size: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample keys: {sample.keys()}")
            print(f"History occupancy shape: {sample['history_occupancy'].shape}")
            print(f"History poses shape: {sample['history_poses'].shape}")
            print(f"Future occupancy shape: {sample['future_occupancy'].shape}")
            print(f"Environment: {sample['env']}")
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
