#!/usr/bin/env python3
"""
nuScenes 6DoF Dataset with Geometric Augmentation

Research contribution: Training aerial 6DoF world models from ground-level driving data
via principled geometric transformations.

Key Features:
1. Full 13D pose format (position, quaternion, linear_vel, angular_vel)
2. Gravity-aware 6DoF augmentations (pitch, roll, altitude)
3. Consistent transformations across temporal windows
4. Point cloud transformation before voxelization (not occupancy rotation)

Theory:
- Ground vehicles have limited pitch/roll (±2°), full yaw
- Aerial drones have full 6DoF (pitch ±90°, roll ±180°, yaw ±180°)
- By augmenting ground data with rotations, we simulate aerial viewpoints
- This enables cross-domain transfer: Ground → Aerial

Usage:
    from dataset.nuscenes_6dof_dataset import NuScenes6DoFDataset, NuScenes6DoFConfig

    config = NuScenes6DoFConfig(
        augment_6dof=True,
        max_pitch_deg=30.0,
        max_roll_deg=45.0,
    )
    dataset = NuScenes6DoFDataset('data/nuscenes', config)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
    HAS_NUSCENES = True
except ImportError:
    HAS_NUSCENES = False
    Quaternion = None


@dataclass
class NuScenes6DoFConfig:
    """Configuration for nuScenes 6DoF dataset."""

    # Data version
    version: str = 'v1.0-mini'  # 'v1.0-mini', 'v1.0-trainval', 'v1.0-test'
    split: str = 'train'

    # Temporal settings
    history_frames: int = 4
    future_frames: int = 6
    frame_skip: int = 1  # Use every N-th frame

    # Occupancy grid settings
    point_cloud_range: Tuple[float, ...] = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)
    voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4)

    # Pose format: 13D = position(3) + quaternion(4) + linear_vel(3) + angular_vel(3)
    pose_dim: int = 13

    # ========== 6DoF Augmentation Settings ==========
    # Enable/disable 6DoF augmentation (for ablation studies)
    augment_6dof: bool = True

    # Pitch augmentation (looking up/down)
    # Drones typically pitch ±30° for forward flight
    max_pitch_deg: float = 30.0

    # Roll augmentation (banking)
    # Drones can roll ±45° for aggressive maneuvers
    max_roll_deg: float = 45.0

    # Yaw augmentation (already present in driving, but add more)
    max_yaw_deg: float = 180.0

    # Altitude simulation: shift Z coordinate
    altitude_shift_range: Tuple[float, float] = (-2.0, 10.0)  # meters

    # Apply same augmentation to entire sequence (temporal consistency)
    consistent_augmentation: bool = True

    # Probability of applying augmentation
    augmentation_prob: float = 0.8

    # ========== Other Settings ==========
    max_points: int = 100000
    use_intensity: bool = True
    cache_data: bool = False

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """Calculate grid size from range and voxel size."""
        pc_range = np.array(self.point_cloud_range)
        voxel_sz = np.array(self.voxel_size)
        return tuple(((pc_range[3:] - pc_range[:3]) / voxel_sz).astype(int))


class Transform6DoF:
    """
    6DoF geometric transformations for point clouds and poses.

    Applies gravity-aware rotations that simulate aerial viewpoints
    while maintaining physical plausibility.
    """

    def __init__(self, config: NuScenes6DoFConfig):
        self.config = config
        self._current_rotation = None
        self._current_translation = None

    def sample_transformation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a random 6DoF transformation.

        Returns:
            rotation_matrix: [3, 3] rotation matrix
            translation: [3] translation vector
        """
        if not self.config.augment_6dof or np.random.random() > self.config.augmentation_prob:
            # No augmentation
            self._current_rotation = np.eye(3)
            self._current_translation = np.zeros(3)
            return self._current_rotation, self._current_translation

        # Sample rotation angles
        pitch = np.random.uniform(
            -self.config.max_pitch_deg,
            self.config.max_pitch_deg
        ) * np.pi / 180.0

        roll = np.random.uniform(
            -self.config.max_roll_deg,
            self.config.max_roll_deg
        ) * np.pi / 180.0

        yaw = np.random.uniform(
            -self.config.max_yaw_deg,
            self.config.max_yaw_deg
        ) * np.pi / 180.0

        # Build rotation matrix (ZYX Euler: yaw, pitch, roll)
        # This order means: first roll, then pitch, then yaw
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        self._current_rotation = Rz @ Ry @ Rx

        # Sample altitude shift
        altitude_shift = np.random.uniform(
            self.config.altitude_shift_range[0],
            self.config.altitude_shift_range[1]
        )
        self._current_translation = np.array([0, 0, altitude_shift])

        return self._current_rotation, self._current_translation

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform point cloud with current transformation.

        Args:
            points: [N, 3+] point cloud (xyz + optional features)

        Returns:
            transformed_points: [N, 3+] transformed points
        """
        if self._current_rotation is None:
            self.sample_transformation()

        # Transform XYZ
        xyz = points[:, :3]
        xyz_transformed = (self._current_rotation @ xyz.T).T + self._current_translation

        # Keep other features unchanged
        if points.shape[1] > 3:
            return np.concatenate([xyz_transformed, points[:, 3:]], axis=1)
        return xyz_transformed

    def transform_pose(self, pose: np.ndarray) -> np.ndarray:
        """
        Transform a 13D pose with current transformation.

        Args:
            pose: [13] = position(3) + quaternion(4) + linear_vel(3) + angular_vel(3)

        Returns:
            transformed_pose: [13] transformed pose
        """
        if self._current_rotation is None:
            self.sample_transformation()

        # Extract components
        position = pose[:3]
        quaternion = pose[3:7]  # qw, qx, qy, qz
        linear_vel = pose[7:10]
        angular_vel = pose[10:13]

        # Transform position
        new_position = self._current_rotation @ position + self._current_translation

        # Transform quaternion
        if Quaternion is not None:
            q_orig = Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
            q_aug = Quaternion(matrix=self._current_rotation)
            q_new = q_aug * q_orig
            new_quaternion = np.array([q_new.w, q_new.x, q_new.y, q_new.z])
        else:
            new_quaternion = quaternion  # Fallback: no rotation

        # Transform velocities (rotate only, no translation)
        new_linear_vel = self._current_rotation @ linear_vel
        new_angular_vel = self._current_rotation @ angular_vel

        return np.concatenate([new_position, new_quaternion, new_linear_vel, new_angular_vel])

    def reset(self):
        """Reset transformation (call at start of each sequence if using consistent aug)."""
        self._current_rotation = None
        self._current_translation = None


class NuScenes6DoFDataset(Dataset):
    """
    nuScenes dataset with 6DoF augmentation for aerial world model training.

    Key features:
    - Computes 13D poses from nuScenes ego poses
    - Applies 6DoF augmentations to simulate aerial viewpoints
    - Transforms point clouds before voxelization (not occupancy)
    """

    def __init__(self, data_root: str, config: NuScenes6DoFConfig):
        if not HAS_NUSCENES:
            raise ImportError(
                "nuscenes-devkit required.\n"
                "Install with: pip install nuscenes-devkit pyquaternion"
            )

        self.data_root = data_root
        self.config = config
        self.transformer = Transform6DoF(config)

        # Initialize nuScenes API
        print(f"Loading nuScenes {config.version}...")
        self.nusc = NuScenes(
            version=config.version,
            dataroot=data_root,
            verbose=True
        )

        # Get scene splits
        self.scene_names = self._get_split_scenes()

        # Build sample index
        self.samples = self._build_sample_index()
        print(f"NuScenes 6DoF {config.split}: {len(self.samples)} samples from {len(self.scene_names)} scenes")
        print(f"6DoF augmentation: {'ON' if config.augment_6dof else 'OFF'}")
        if config.augment_6dof:
            print(f"  Pitch: ±{config.max_pitch_deg}°, Roll: ±{config.max_roll_deg}°")

    def _get_split_scenes(self) -> List[str]:
        """Get scene names for the configured split."""
        if self.config.version == 'v1.0-mini':
            # Mini dataset has limited scenes
            scene_names = [s['name'] for s in self.nusc.scene]
            if self.config.split == 'train':
                return scene_names[:8]
            else:
                return scene_names[8:]
        else:
            splits = create_splits_scenes()
            return splits.get(self.config.split, [])

    def _build_sample_index(self) -> List[Dict]:
        """Build index of valid temporal windows."""
        samples = []

        for scene in self.nusc.scene:
            if scene['name'] not in self.scene_names:
                continue

            # Collect all sample tokens in scene
            sample_tokens = []
            sample_token = scene['first_sample_token']
            while sample_token:
                sample_tokens.append(sample_token)
                sample = self.nusc.get('sample', sample_token)
                sample_token = sample['next']

            # Apply frame skip
            if self.config.frame_skip > 1:
                sample_tokens = sample_tokens[::self.config.frame_skip]

            # Create sliding windows
            total_frames = self.config.history_frames + self.config.future_frames
            for i in range(len(sample_tokens) - total_frames + 1):
                window = sample_tokens[i:i + total_frames]
                samples.append({
                    'scene_name': scene['name'],
                    'sample_tokens': window,
                    'history_tokens': window[:self.config.history_frames],
                    'future_tokens': window[self.config.history_frames:],
                })

        return samples

    def _load_lidar(self, sample_token: str) -> np.ndarray:
        """Load LiDAR points for a sample."""
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_path = os.path.join(self.data_root, lidar_data['filename'])

        # Load point cloud (x, y, z, intensity, ring)
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)

        if self.config.use_intensity:
            return points[:, :4]  # x, y, z, intensity
        return points[:, :3]  # x, y, z only

    def _load_ego_pose_13d(self, sample_token: str, prev_token: Optional[str] = None) -> np.ndarray:
        """
        Load ego pose and compute velocities for 13D format.

        Returns:
            pose: [13] = position(3) + quaternion(4) + linear_vel(3) + angular_vel(3)
        """
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)

        # Get ego pose
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])

        position = np.array(ego_pose['translation'])  # x, y, z
        rotation = np.array(ego_pose['rotation'])     # qw, qx, qy, qz

        # Compute velocities from previous frame
        if prev_token is not None:
            prev_sample = self.nusc.get('sample', prev_token)
            prev_lidar_token = prev_sample['data']['LIDAR_TOP']
            prev_lidar_data = self.nusc.get('sample_data', prev_lidar_token)
            prev_ego_pose = self.nusc.get('ego_pose', prev_lidar_data['ego_pose_token'])

            prev_position = np.array(prev_ego_pose['translation'])
            prev_rotation = np.array(prev_ego_pose['rotation'])

            # Time delta (nuScenes is ~2Hz for keyframes)
            dt = (lidar_data['timestamp'] - prev_lidar_data['timestamp']) / 1e6  # microseconds to seconds
            dt = max(dt, 0.01)  # Avoid division by zero

            # Linear velocity
            linear_vel = (position - prev_position) / dt

            # Angular velocity (from quaternion difference)
            if Quaternion is not None:
                q_curr = Quaternion(rotation[0], rotation[1], rotation[2], rotation[3])
                q_prev = Quaternion(prev_rotation[0], prev_rotation[1], prev_rotation[2], prev_rotation[3])
                q_diff = q_curr * q_prev.inverse
                # Convert to axis-angle
                angle = 2 * np.arccos(np.clip(q_diff.w, -1, 1))
                sin_half = np.sin(angle / 2)
                if sin_half > 1e-6:
                    axis = np.array([q_diff.x, q_diff.y, q_diff.z]) / sin_half
                    angular_vel = axis * angle / dt
                else:
                    angular_vel = np.zeros(3)
            else:
                angular_vel = np.zeros(3)
        else:
            linear_vel = np.zeros(3)
            angular_vel = np.zeros(3)

        return np.concatenate([position, rotation, linear_vel, angular_vel])

    def _points_to_occupancy(self, points: np.ndarray) -> np.ndarray:
        """Convert point cloud to occupancy grid."""
        pc_range = np.array(self.config.point_cloud_range)
        voxel_size = np.array(self.config.voxel_size)
        grid_size = np.array(self.config.grid_size)

        # Filter points in range
        xyz = points[:, :3]
        mask = (
            (xyz[:, 0] >= pc_range[0]) & (xyz[:, 0] < pc_range[3]) &
            (xyz[:, 1] >= pc_range[1]) & (xyz[:, 1] < pc_range[4]) &
            (xyz[:, 2] >= pc_range[2]) & (xyz[:, 2] < pc_range[5])
        )
        xyz = xyz[mask]

        if len(xyz) == 0:
            return np.zeros(grid_size, dtype=np.uint8)

        # Convert to voxel indices (clip before int conversion to prevent overflow)
        voxel_coords_float = (xyz - pc_range[:3]) / voxel_size
        voxel_coords_float = np.clip(voxel_coords_float, 0, np.array(grid_size) - 1)
        voxel_coords = voxel_coords_float.astype(np.int32)

        # Create occupancy grid
        occupancy = np.zeros(grid_size, dtype=np.uint8)
        occupancy[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1

        return occupancy

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]

        # Reset transformer for new sequence (if using consistent augmentation)
        if self.config.consistent_augmentation:
            self.transformer.reset()
            self.transformer.sample_transformation()

        # Load history frames
        history_occ = []
        history_poses = []
        prev_token = None

        for token in sample_info['history_tokens']:
            # Load LiDAR
            points = self._load_lidar(token)

            # Transform points (6DoF augmentation applied here)
            if not self.config.consistent_augmentation:
                self.transformer.sample_transformation()
            points = self.transformer.transform_points(points)

            # Voxelize
            occ = self._points_to_occupancy(points)

            # Load and transform pose
            pose = self._load_ego_pose_13d(token, prev_token)
            pose = self.transformer.transform_pose(pose)

            history_occ.append(occ)
            history_poses.append(pose)
            prev_token = token

        # Load future frames
        future_occ = []
        future_poses = []

        for token in sample_info['future_tokens']:
            points = self._load_lidar(token)

            if not self.config.consistent_augmentation:
                self.transformer.sample_transformation()
            points = self.transformer.transform_points(points)

            occ = self._points_to_occupancy(points)
            pose = self._load_ego_pose_13d(token, prev_token)
            pose = self.transformer.transform_pose(pose)

            future_occ.append(occ)
            future_poses.append(pose)
            prev_token = token

        return {
            'history_occupancy': torch.from_numpy(np.stack(history_occ)).float(),
            'future_occupancy': torch.from_numpy(np.stack(future_occ)).float(),
            'history_poses': torch.from_numpy(np.stack(history_poses)).float(),
            'future_poses': torch.from_numpy(np.stack(future_poses)).float(),
            'agent_type': torch.tensor(0),  # 0 = ground (but augmented toward aerial)
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
        'agent_type': torch.stack([b['agent_type'] for b in batch]),
    }


def create_dataloader(
    data_root: str,
    config: NuScenes6DoFConfig,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader for nuScenes 6DoF dataset."""
    dataset = NuScenes6DoFDataset(data_root, config)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )


# =============================================================================
# Testing and Validation
# =============================================================================

if __name__ == '__main__':
    import sys

    print("=" * 70)
    print("nuScenes 6DoF Dataset - Research Module")
    print("=" * 70)
    print()
    print("This module enables training aerial 6DoF world models from ground data")
    print("via principled geometric augmentation.")
    print()

    # Check dependencies
    print("Dependencies:")
    print(f"  nuscenes-devkit: {'OK' if HAS_NUSCENES else 'MISSING'}")
    print(f"  pyquaternion: {'OK' if Quaternion is not None else 'MISSING'}")

    if not HAS_NUSCENES:
        print("\nInstall with: pip install nuscenes-devkit pyquaternion")
        sys.exit(1)

    # Test with mini dataset
    data_root = sys.argv[1] if len(sys.argv) > 1 else 'data/nuscenes'

    if not os.path.exists(data_root):
        print(f"\nnuScenes data not found at: {data_root}")
        print("Download from: https://www.nuscenes.org/download")
        sys.exit(1)

    print(f"\nTesting with data from: {data_root}")

    # Test without augmentation
    print("\n--- Test 1: Without 6DoF augmentation ---")
    config_no_aug = NuScenes6DoFConfig(
        version='v1.0-mini',
        split='train',
        augment_6dof=False,
    )
    dataset_no_aug = NuScenes6DoFDataset(data_root, config_no_aug)

    if len(dataset_no_aug) > 0:
        sample = dataset_no_aug[0]
        print(f"  History occupancy: {sample['history_occupancy'].shape}")
        print(f"  Future occupancy: {sample['future_occupancy'].shape}")
        print(f"  History poses: {sample['history_poses'].shape}")
        print(f"  Future poses: {sample['future_poses'].shape}")
        print(f"  Pose range: [{sample['history_poses'].min():.2f}, {sample['history_poses'].max():.2f}]")

    # Test with augmentation
    print("\n--- Test 2: With 6DoF augmentation ---")
    config_aug = NuScenes6DoFConfig(
        version='v1.0-mini',
        split='train',
        augment_6dof=True,
        max_pitch_deg=30.0,
        max_roll_deg=45.0,
    )
    dataset_aug = NuScenes6DoFDataset(data_root, config_aug)

    if len(dataset_aug) > 0:
        sample_aug = dataset_aug[0]
        print(f"  History occupancy: {sample_aug['history_occupancy'].shape}")
        print(f"  Augmented pose range: [{sample_aug['history_poses'].min():.2f}, {sample_aug['history_poses'].max():.2f}]")

        # Compare occupancy sparsity
        occ_orig = sample['history_occupancy'].sum()
        occ_aug = sample_aug['history_occupancy'].sum()
        print(f"  Original occupied voxels: {occ_orig}")
        print(f"  Augmented occupied voxels: {occ_aug}")

    print("\n" + "=" * 70)
    print("Tests completed!")
    print("=" * 70)
