#!/usr/bin/env python3
"""
nuScenes dataset adapter for OccWorld training.

Loads nuScenes data in format compatible with our training pipeline.
"""
import os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    HAS_NUSCENES = True
except ImportError:
    HAS_NUSCENES = False
    print("Warning: nuscenes-devkit not installed. Run: pip install nuscenes-devkit")


@dataclass
class NuScenesConfig:
    """Configuration for nuScenes dataset."""
    version: str = 'v1.0-mini'
    history_frames: int = 4
    future_frames: int = 6
    point_cloud_range: Tuple = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)
    voxel_size: Tuple = (0.4, 0.4, 0.4)
    grid_size: Tuple = (200, 200, 16)
    split: str = 'train'
    max_sweeps: int = 10
    use_occ3d: bool = True
    occ3d_path: Optional[str] = None
    estimate_velocity: bool = False  # Estimate velocity via finite differences


class NuScenesOccWorldDataset(Dataset):
    """nuScenes dataset for OccWorld training."""

    def __init__(self, data_root: str, config: NuScenesConfig):
        if not HAS_NUSCENES:
            raise ImportError("nuscenes-devkit required. Install with: pip install nuscenes-devkit")

        self.data_root = data_root
        self.config = config

        # Calculate grid size from range and voxel size
        pc_range = np.array(config.point_cloud_range)
        voxel_size = np.array(config.voxel_size)
        self.grid_size = tuple(((pc_range[3:] - pc_range[:3]) / voxel_size).astype(int))

        # Initialize nuScenes
        self.nusc = NuScenes(
            version=config.version,
            dataroot=data_root,
            verbose=True
        )

        # Get scene splits
        splits = create_splits_scenes()
        if config.version == 'v1.0-mini':
            # Mini has different split
            scene_names = [s['name'] for s in self.nusc.scene]
            if config.split == 'train':
                self.scene_names = scene_names[:8]
            else:
                self.scene_names = scene_names[8:]
        else:
            self.scene_names = splits.get(config.split, [])

        # Warn about zero velocity channels
        if not config.estimate_velocity:
            warnings.warn(
                "NuScenesOccWorldDataset: velocity channels (indices 7-12) are zero-padded. "
                "46% of the 13D pose input is uninformative. Set estimate_velocity=True "
                "in NuScenesConfig to enable finite-difference velocity estimation, or use "
                "NuScenes6DoFDataset which computes velocities by default.",
                stacklevel=2,
            )

        # Build sample index
        self.samples = self._build_sample_index()
        print(f"NuScenes {config.split}: {len(self.samples)} samples from {len(self.scene_names)} scenes")

    def _build_sample_index(self) -> List[Dict]:
        """Build index of valid samples with enough history/future frames."""
        samples = []

        for scene in self.nusc.scene:
            if scene['name'] not in self.scene_names:
                continue

            # Get all samples in scene
            sample_tokens = []
            sample_token = scene['first_sample_token']
            while sample_token:
                sample_tokens.append(sample_token)
                sample = self.nusc.get('sample', sample_token)
                sample_token = sample['next']

            # Create valid windows
            total_needed = self.config.history_frames + self.config.future_frames
            for i in range(len(sample_tokens) - total_needed + 1):
                samples.append({
                    'scene_name': scene['name'],
                    'sample_tokens': sample_tokens[i:i + total_needed],
                    'history_tokens': sample_tokens[i:i + self.config.history_frames],
                    'future_tokens': sample_tokens[i + self.config.history_frames:i + total_needed],
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_lidar(self, sample_token: str) -> np.ndarray:
        """Load LiDAR points for a sample."""
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_path = os.path.join(self.data_root, lidar_data['filename'])

        # Load point cloud (x, y, z, intensity, ring)
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        return points[:, :4]  # x, y, z, intensity

    def _load_ego_pose(self, sample_token: str) -> np.ndarray:
        """Load ego pose for a sample."""
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)

        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])

        # Extract position and rotation
        pos = np.array(ego_pose['translation'])  # x, y, z
        rot = np.array(ego_pose['rotation'])     # qw, qx, qy, qz

        # Combine into 13D pose for compatibility with model pose encoder:
        # [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        # nuScenes quaternion is already (qw, qx, qy, qz) — keep as-is
        # WARNING: nuScenes basic loader provides no velocity data.
        # This zeros-out the velocity channels (linear_vel + angular_vel),
        # which weakens motion forecasting quality. For velocity estimation,
        # use NuScenes6DoFDataset instead (computes velocity from consecutive poses).
        vel = np.zeros(6)
        return np.concatenate([pos, rot, vel])

    def _points_to_occupancy(self, points: np.ndarray) -> np.ndarray:
        """Convert point cloud to occupancy grid."""
        pc_range = np.array(self.config.point_cloud_range)
        voxel_size = np.array(self.config.voxel_size)
        grid_size = np.array(self.grid_size)

        # Filter points in range
        mask = (
            (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) &
            (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]) &
            (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
        )
        points = points[mask]

        # Convert to voxel indices
        voxel_coords_float = (points[:, :3] - pc_range[:3]) / voxel_size
        voxel_coords_float = np.clip(voxel_coords_float, 0, grid_size - 1)
        voxel_coords = voxel_coords_float.astype(np.int32)

        # Create occupancy grid
        occupancy = np.zeros(grid_size, dtype=np.uint8)
        occupancy[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1

        return occupancy

    def _estimate_velocity(self, pose_curr: np.ndarray, pose_prev: np.ndarray,
                            token_curr: str, token_prev: str) -> np.ndarray:
        """Estimate velocity via finite differences between consecutive poses.

        Args:
            pose_curr: Current 13D pose [pos(3) + quat(4) + vel(6)]
            pose_prev: Previous 13D pose
            token_curr: Current sample token (for timestamp lookup)
            token_prev: Previous sample token

        Returns:
            Updated pose with estimated velocity channels filled in.
        """
        # Get timestamps for dt calculation
        sample_curr = self.nusc.get('sample', token_curr)
        sample_prev = self.nusc.get('sample', token_prev)
        lidar_curr = self.nusc.get('sample_data', sample_curr['data']['LIDAR_TOP'])
        lidar_prev = self.nusc.get('sample_data', sample_prev['data']['LIDAR_TOP'])
        dt = (lidar_curr['timestamp'] - lidar_prev['timestamp']) / 1e6  # seconds
        dt = max(dt, 0.01)  # avoid division by zero

        result = pose_curr.copy()
        # Linear velocity
        result[7:10] = (pose_curr[:3] - pose_prev[:3]) / dt
        # Angular velocity (simple finite difference on quaternion)
        # For more accurate angular velocity, use quaternion log map
        result[10:13] = 0.0  # placeholder — quaternion diff is complex without pyquaternion
        return result

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]
        all_tokens = sample_info['history_tokens'] + sample_info['future_tokens']

        # Load all poses first
        all_poses = [self._load_ego_pose(token) for token in all_tokens]

        # Optionally estimate velocities via finite differences
        if self.config.estimate_velocity:
            for i in range(1, len(all_poses)):
                all_poses[i] = self._estimate_velocity(
                    all_poses[i], all_poses[i - 1],
                    all_tokens[i], all_tokens[i - 1],
                )

        n_hist = len(sample_info['history_tokens'])

        # Load history
        history_occ = []
        for token in sample_info['history_tokens']:
            points = self._load_lidar(token)
            occ = self._points_to_occupancy(points)
            history_occ.append(occ)

        # Load future
        future_occ = []
        for token in sample_info['future_tokens']:
            points = self._load_lidar(token)
            occ = self._points_to_occupancy(points)
            future_occ.append(occ)

        history_poses = all_poses[:n_hist]
        future_poses = all_poses[n_hist:]

        return {
            'history_occupancy': torch.from_numpy(np.stack(history_occ)).float(),
            'future_occupancy': torch.from_numpy(np.stack(future_occ)).float(),
            'history_poses': torch.from_numpy(np.stack(history_poses)).float(),
            'future_poses': torch.from_numpy(np.stack(future_poses)).float(),
            'scene_name': sample_info['scene_name'],
            'domain_tag': 'real',
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    return {
        'history_occupancy': torch.stack([b['history_occupancy'] for b in batch]),
        'future_occupancy': torch.stack([b['future_occupancy'] for b in batch]),
        'history_poses': torch.stack([b['history_poses'] for b in batch]),
        'future_poses': torch.stack([b['future_poses'] for b in batch]),
    }


if __name__ == '__main__':
    # Test the dataset
    import sys

    data_root = sys.argv[1] if len(sys.argv) > 1 else 'data/nuscenes'

    config = NuScenesConfig(
        version='v1.0-mini',
        split='train',
    )

    dataset = NuScenesOccWorldDataset(data_root, config)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"History occupancy: {sample['history_occupancy'].shape}")
        print(f"Future occupancy: {sample['future_occupancy'].shape}")
        print(f"History poses: {sample['history_poses'].shape}")
