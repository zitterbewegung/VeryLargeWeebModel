#!/usr/bin/env python3
"""
Mid-Air Dataset Loader for 6DoF World Model Training

Mid-Air is a synthetic drone dataset from University of Liege providing:
- RGB stereo + downward camera images
- Depth maps
- Semantic segmentation
- Perfect 6-DoF ground truth poses (100Hz)
- IMU and GPS data
- 54 trajectories, 420k+ frames
- Multiple climate conditions

Dataset structure:
    data_root/
    ├── Kite_test/
    │   ├── cloudy/
    │   │   ├── color_left/
    │   │   │   └── trajectory_0001/
    │   │   │       ├── 000000.JPEG
    │   │   │       └── ...
    │   │   ├── depth/
    │   │   │   └── trajectory_0001/
    │   │   │       ├── 000000.PNG
    │   │   │       └── ...
    │   │   ├── segmentation/
    │   │   └── sensor_records.hdf5
    │   ├── foggy/
    │   └── sunny/
    ├── Kite_training/
    └── ...

Download from: https://midair.ulg.ac.be/

Usage:
    from dataset.midair_dataset import MidAirDataset, MidAirConfig

    config = MidAirConfig(climates=['sunny', 'cloudy'])
    dataset = MidAirDataset('data/midair', config)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import cv2

cv2.setNumThreads(0)

# Try to import h5py for HDF5 support
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None


@dataclass
class MidAirConfig:
    """Configuration for Mid-Air dataset."""

    # Environment selection
    environments: Optional[List[str]] = None  # e.g., ['Kite_test', 'Kite_training']

    # Climate conditions: 'sunny', 'cloudy', 'foggy', 'sunset', or list
    climates: Optional[List[str]] = None  # None = all available

    # Trajectory selection (None = all)
    trajectories: Optional[List[str]] = None

    # Temporal settings
    history_frames: int = 4
    future_frames: int = 6
    frame_skip: int = 2  # Mid-Air is 25Hz, skip frames for diversity

    # Split
    split: str = 'train'
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Depth to occupancy conversion
    point_cloud_range: Tuple[float, ...] = (-40.0, -40.0, -10.0, 40.0, 40.0, 50.0)
    voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.5)

    # Depth settings
    max_depth: float = 100.0
    # Mid-Air depth is stored as 16-bit PNG, needs scaling
    depth_scale: float = 256.0  # depth_meters = pixel_value / depth_scale

    # Image settings
    image_size: Tuple[int, int] = (1024, 1024)  # Mid-Air default

    # Camera intrinsics (Mid-Air default - 90 degree FOV)
    fx: float = 512.0
    fy: float = 512.0
    cx: float = 512.0
    cy: float = 512.0

    # Pose format
    pose_dim: int = 13

    # Use HDF5 for poses (faster) or interpolate from sparse GPS
    use_hdf5_poses: bool = True

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """Calculate grid size from range and voxel size."""
        pc_range = np.array(self.point_cloud_range)
        voxel_sz = np.array(self.voxel_size)
        return tuple(((pc_range[3:] - pc_range[:3]) / voxel_sz).astype(int))


class MidAirDataset(Dataset):
    """
    Mid-Air dataset for 6DoF aerial world model training.

    Converts depth images to occupancy grids for OccWorld training.
    """

    def __init__(self, data_root: str, config: MidAirConfig, transform=None):
        self.data_root = Path(data_root)
        self.config = config
        self.transform = transform

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Mid-Air data not found at: {data_root}\n"
                f"Download from: https://midair.ulg.ac.be/"
            )

        if config.use_hdf5_poses and not HAS_H5PY:
            print("Warning: h5py not installed, falling back to image-only mode")
            print("  Install with: pip install h5py")

        # Build camera intrinsic matrix
        self.K = np.array([
            [config.fx, 0, config.cx],
            [0, config.fy, config.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Cache for HDF5 file handles
        self._hdf5_cache: Dict[str, h5py.File] = {}

        # Build sample index
        self.samples = self._build_sample_index()

        # Apply split
        self._apply_split()

        print(f"Mid-Air {config.split}: {len(self.samples)} samples")
        print(f"  Grid size: {config.grid_size}")

    def __del__(self):
        """Close cached HDF5 file handles."""
        for f in self._hdf5_cache.values():
            try:
                f.close()
            except Exception:
                pass
        self._hdf5_cache.clear()

    def _build_sample_index(self) -> List[Dict]:
        """Build index of all valid temporal windows."""
        samples = []

        # Find all environments
        if self.config.environments:
            envs = [e for e in self.config.environments if (self.data_root / e).exists()]
        else:
            envs = [d.name for d in self.data_root.iterdir()
                    if d.is_dir() and not d.name.startswith('.')]

        if not envs:
            print(f"Warning: No environments found in {self.data_root}")
            return samples

        print(f"Found {len(envs)} environments: {envs}")

        for env in envs:
            env_path = self.data_root / env

            # Find climate conditions
            if self.config.climates:
                climates = [c for c in self.config.climates if (env_path / c).exists()]
            else:
                climates = [d.name for d in env_path.iterdir()
                           if d.is_dir() and not d.name.startswith('.')]

            for climate in climates:
                climate_path = env_path / climate

                # Load HDF5 sensor records if available
                hdf5_path = climate_path / 'sensor_records.hdf5'
                poses_dict = {}
                if hdf5_path.exists() and HAS_H5PY and self.config.use_hdf5_poses:
                    poses_dict = self._load_hdf5_poses(hdf5_path)

                # Find depth directory
                depth_base = climate_path / 'depth'
                if not depth_base.exists():
                    continue

                # Find trajectories
                if self.config.trajectories:
                    trajs = [t for t in self.config.trajectories if (depth_base / t).exists()]
                else:
                    trajs = sorted([d.name for d in depth_base.iterdir()
                                   if d.is_dir() and d.name.startswith('trajectory')])

                for traj in trajs:
                    traj_depth_path = depth_base / traj

                    # Find depth files
                    depth_files = sorted(traj_depth_path.glob('*.PNG')) or \
                                  sorted(traj_depth_path.glob('*.png'))
                    if not depth_files:
                        continue

                    # Get poses for this trajectory
                    traj_poses = poses_dict.get(traj, None)

                    # Build frame info
                    frame_info = []
                    for idx, depth_file in enumerate(depth_files):
                        frame_num = int(depth_file.stem)

                        # Get pose if available
                        pose = None
                        if traj_poses is not None and idx < len(traj_poses):
                            pose = traj_poses[idx]

                        frame_info.append({
                            'idx': idx,
                            'frame_num': frame_num,
                            'depth_path': depth_file,
                            'image_path': climate_path / 'color_left' / traj / f'{frame_num:06d}.JPEG',
                            'seg_path': climate_path / 'segmentation' / traj / f'{frame_num:06d}.PNG',
                            'pose': pose,
                            'env': env,
                            'climate': climate,
                            'trajectory': traj,
                        })

                    # Apply frame skip
                    if self.config.frame_skip > 1:
                        frame_info = frame_info[::self.config.frame_skip]

                    # Create sliding windows
                    total_frames = self.config.history_frames + self.config.future_frames
                    for i in range(len(frame_info) - total_frames + 1):
                        window = frame_info[i:i + total_frames]

                        # Skip if poses unavailable and required
                        if self.config.use_hdf5_poses and any(f['pose'] is None for f in window):
                            continue

                        samples.append({
                            'env': env,
                            'climate': climate,
                            'trajectory': traj,
                            'frames': window,
                            'history_frames': window[:self.config.history_frames],
                            'future_frames': window[self.config.history_frames:],
                        })

        return samples

    def _load_hdf5_poses(self, hdf5_path: Path) -> Dict[str, np.ndarray]:
        """Load poses from HDF5 sensor records file."""
        poses_dict = {}

        try:
            with h5py.File(str(hdf5_path), 'r') as f:
                # Iterate through trajectories
                for traj_name in f.keys():
                    if not traj_name.startswith('trajectory'):
                        continue

                    traj_group = f[traj_name]

                    # Get groundtruth poses
                    if 'groundtruth' in traj_group:
                        gt = traj_group['groundtruth']

                        # Position: [N, 3]
                        if 'position' in gt:
                            positions = np.array(gt['position'])
                        else:
                            continue

                        # Attitude (quaternion): [N, 4] as [w, x, y, z]
                        if 'attitude' in gt:
                            attitudes = np.array(gt['attitude'])
                        else:
                            attitudes = np.zeros((len(positions), 4))
                            attitudes[:, 0] = 1.0  # Identity quaternion

                        # Velocity (optional)
                        if 'velocity' in gt:
                            velocities = np.array(gt['velocity'])
                        else:
                            velocities = np.zeros((len(positions), 3))

                        # Groundtruth is at 100Hz, camera is at 25Hz
                        # Subsample to match camera frames (every 4th sample)
                        step = 4
                        positions = positions[::step]
                        attitudes = attitudes[::step]
                        velocities = velocities[::step]

                        # Build pose array [N, 13]
                        poses = np.zeros((len(positions), 13), dtype=np.float32)
                        poses[:, :3] = positions
                        poses[:, 3:7] = attitudes  # Already wxyz
                        poses[:, 7:10] = velocities
                        # Angular velocity not provided, leave as zeros

                        poses_dict[traj_name] = poses

        except Exception as e:
            print(f"Warning: Failed to load HDF5 poses from {hdf5_path}: {e}")

        return poses_dict

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

    def _load_depth(self, depth_path: Path) -> np.ndarray:
        """Load depth image and convert to meters."""
        # Mid-Air stores depth as 16-bit PNG
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        if depth is None:
            return np.zeros(self.config.image_size, dtype=np.float32)

        # Convert to float32 meters
        depth = depth.astype(np.float32) / self.config.depth_scale

        return depth

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

        if len(points) == 0:
            return np.zeros(grid_size, dtype=np.float32)

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

    def _pose_to_tensor(self, pose: Optional[np.ndarray]) -> np.ndarray:
        """Convert pose to 13D tensor."""
        if pose is not None and len(pose) >= 7:
            return pose[:13].astype(np.float32)
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
            depth = self._load_depth(frame['depth_path'])
            points = self._depth_to_pointcloud(depth)
            occ = self._pointcloud_to_occupancy(points)
            history_occupancy.append(occ)

            pose = self._pose_to_tensor(frame['pose'])
            history_poses.append(pose)

        # Process future frames
        for frame in sample['future_frames']:
            depth = self._load_depth(frame['depth_path'])
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
            'climate': sample['climate'],
            'trajectory': sample['trajectory'],
            'domain_tag': 'sim',
        }

        if self.transform is not None:
            result = self.transform(result)

        return result


def create_midair_dataloader(
    data_root: str,
    config: MidAirConfig,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for Mid-Air dataset."""
    dataset = MidAirDataset(data_root, config)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/midair')
    parser.add_argument('--climate', default=None, help='Specific climate')
    args = parser.parse_args()

    config = MidAirConfig(
        climates=[args.climate] if args.climate else None,
        split='train',
    )

    try:
        dataset = MidAirDataset(args.data_root, config)
        print(f"\nDataset size: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample keys: {sample.keys()}")
            print(f"History occupancy shape: {sample['history_occupancy'].shape}")
            print(f"History poses shape: {sample['history_poses'].shape}")
            print(f"Environment: {sample['env']}")
            print(f"Climate: {sample['climate']}")
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
