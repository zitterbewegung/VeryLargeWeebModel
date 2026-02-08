#!/usr/bin/env python3
"""
3D Occupancy Data Augmentation Transforms for Training

Implements geometric augmentations for 3D occupancy grids and corresponding 6DoF poses.
Designed for training world models with spatial data augmentation while maintaining
physical consistency across temporal sequences.

Transform classes:
    - RandomFlip3D: Random flips along X or Y axis
    - RandomRotate90: Random 90° rotations around Z axis
    - OccupancyDropout: Random voxel dropout for sensor noise simulation
    - Compose: Standard composition of transforms

Usage:
    from dataset.transforms import get_train_transforms

    transforms = get_train_transforms(flip_p=0.5, rotate_p=0.5, dropout_p=0.1)
    dataset = MyDataset(data_root, config, transform=transforms)
"""

import random
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import torch


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions in [w, x, y, z] format.

    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]

    Returns:
        Result quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z], dtype=np.float32)


class RandomFlip3D:
    """
    Randomly flip occupancy grids and corresponding poses along X or Y axis.

    Flipping operations:
    - X-axis flip: Flips along X (dim 0), negates x position and vx velocity
    - Y-axis flip: Flips along Y (dim 1), negates y position and vy velocity

    For simplicity, quaternions are kept unchanged as rotation symmetry is preserved
    in most aerial navigation scenarios.

    Args:
        p: Probability of applying flip for each axis (default: 0.5)
        axes: Tuple of axes to consider for flipping ('x', 'y')
    """

    def __init__(self, p: float = 0.5, axes: Tuple[str, ...] = ('x', 'y')):
        self.p = p
        self.axes = axes

    def __call__(self, sample: Dict) -> Dict:
        """
        Apply random flips to sample.

        Args:
            sample: Dictionary with keys:
                - history_occupancy: Tensor [T_h, X, Y, Z]
                - future_occupancy: Tensor [T_f, X, Y, Z]
                - history_poses: Tensor [T_h, 13]
                - future_poses: Tensor [T_f, 13]

        Returns:
            Transformed sample dictionary
        """
        # Check which flips to apply
        flip_x = 'x' in self.axes and random.random() < self.p
        flip_y = 'y' in self.axes and random.random() < self.p

        if not flip_x and not flip_y:
            return sample

        result = {}

        for key, value in sample.items():
            # Handle occupancy tensors
            if key in ('history_occupancy', 'future_occupancy'):
                if isinstance(value, torch.Tensor):
                    occ = value.clone()
                    # Occupancy shape: [T, X, Y, Z]
                    # Spatial dimensions start at index 1
                    if flip_x:
                        occ = torch.flip(occ, dims=[1])  # Flip X dimension
                    if flip_y:
                        occ = torch.flip(occ, dims=[2])  # Flip Y dimension
                    result[key] = occ
                else:
                    result[key] = value

            # Handle pose tensors
            elif key in ('history_poses', 'future_poses'):
                if isinstance(value, torch.Tensor):
                    poses = value.clone()
                    # Pose format: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
                    # Indices: [0, 1, 2, 3,  4,  5,  6,  7,  8,  9,  10, 11, 12]

                    if flip_x:
                        poses[..., 0] = -poses[..., 0]   # Negate x position
                        poses[..., 7] = -poses[..., 7]   # Negate vx velocity

                    if flip_y:
                        poses[..., 1] = -poses[..., 1]   # Negate y position
                        poses[..., 8] = -poses[..., 8]   # Negate vy velocity

                    result[key] = poses
                else:
                    result[key] = value

            # Pass through other keys unchanged
            else:
                result[key] = value

        return result


class RandomRotate90:
    """
    Randomly rotate occupancy and poses by 90° multiples around Z axis.

    Rotations are sampled from {90°, 180°, 270°} and applied to both
    occupancy grids and pose vectors. This preserves the Manhattan-world
    structure common in urban environments.

    Args:
        p: Probability of applying rotation (default: 0.5)
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def _rotate_position_2d(self, pos: torch.Tensor, k: int) -> torch.Tensor:
        """
        Rotate 2D position (x, y) by k * 90 degrees.

        Args:
            pos: Position tensor [..., 2]
            k: Number of 90° rotations (1, 2, or 3)

        Returns:
            Rotated position
        """
        x, y = pos[..., 0], pos[..., 1]

        if k == 1:  # 90° CCW: (x, y) -> (-y, x)
            return torch.stack([-y, x], dim=-1)
        elif k == 2:  # 180°: (x, y) -> (-x, -y)
            return torch.stack([-x, -y], dim=-1)
        elif k == 3:  # 270° CCW: (x, y) -> (y, -x)
            return torch.stack([y, -x], dim=-1)
        else:
            return pos

    def _get_z_rotation_quaternion(self, k: int) -> np.ndarray:
        """
        Get quaternion for k * 90° rotation around Z axis.

        Args:
            k: Number of 90° rotations (1, 2, or 3)

        Returns:
            Quaternion [w, x, y, z]
        """
        # Rotation around Z axis by angle θ: q = [cos(θ/2), 0, 0, sin(θ/2)]
        angle = k * np.pi / 2  # k * 90° in radians
        w = np.cos(angle / 2)
        z = np.sin(angle / 2)
        return np.array([w, 0.0, 0.0, z], dtype=np.float32)

    def __call__(self, sample: Dict) -> Dict:
        """
        Apply random 90° rotation to sample.

        Args:
            sample: Dictionary with occupancy and pose tensors

        Returns:
            Transformed sample dictionary
        """
        if random.random() >= self.p:
            return sample

        # Sample rotation: k ∈ {1, 2, 3} for {90°, 180°, 270°}
        k = random.randint(1, 3)

        result = {}

        for key, value in sample.items():
            # Handle occupancy tensors
            if key in ('history_occupancy', 'future_occupancy'):
                if isinstance(value, torch.Tensor):
                    occ = value.clone()
                    # Occupancy shape: [T, X, Y, Z]
                    # Rotate in XY plane (dims 1 and 2)
                    occ = torch.rot90(occ, k=k, dims=[1, 2])
                    result[key] = occ
                else:
                    result[key] = value

            # Handle pose tensors
            elif key in ('history_poses', 'future_poses'):
                if isinstance(value, torch.Tensor):
                    poses = value.clone()
                    # Pose format: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]

                    # Rotate position (x, y)
                    pos_2d = poses[..., :2]
                    poses[..., :2] = self._rotate_position_2d(pos_2d, k)

                    # Rotate linear velocity (vx, vy)
                    vel_2d = poses[..., 7:9]
                    poses[..., 7:9] = self._rotate_position_2d(vel_2d, k)

                    # Rotate quaternion
                    q_rot = self._get_z_rotation_quaternion(k)

                    # Apply rotation to each timestep
                    poses_np = poses.numpy()
                    for t in range(poses_np.shape[0]):
                        q_orig = poses_np[t, 3:7]  # [qw, qx, qy, qz]
                        q_new = quaternion_multiply(q_rot, q_orig)
                        # Normalize to prevent numerical drift
                        q_new = q_new / (np.linalg.norm(q_new) + 1e-8)
                        poses_np[t, 3:7] = q_new

                    poses = torch.from_numpy(poses_np).float()
                    result[key] = poses
                else:
                    result[key] = value

            # Pass through other keys unchanged
            else:
                result[key] = value

        return result


class OccupancyDropout:
    """
    Randomly zero out occupied voxels to simulate sensor noise.

    This augmentation simulates various sensor failure modes:
    - LiDAR occlusion
    - Sparse point clouds
    - Measurement dropout

    Only applied to history_occupancy (observations), not future_occupancy (ground truth).

    Args:
        p: Probability of dropping each occupied voxel (default: 0.1)
    """

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        """
        Apply occupancy dropout to sample.

        Args:
            sample: Dictionary with occupancy tensors

        Returns:
            Transformed sample dictionary
        """
        result = {}

        for key, value in sample.items():
            # Only apply to history_occupancy, not future (which is ground truth)
            if key == 'history_occupancy' and isinstance(value, torch.Tensor):
                occ = value.clone()

                # Create dropout mask: keep occupied voxels with probability (1 - p)
                # Only apply to occupied voxels (value > 0.5)
                dropout_mask = torch.rand_like(occ) > self.p
                occupied_mask = occ > 0.5

                # Zero out dropped occupied voxels
                occ = occ * (dropout_mask | ~occupied_mask).float()

                result[key] = occ
            else:
                result[key] = value

        return result


class Compose:
    """
    Compose multiple transforms together.

    Args:
        transforms: List of transform callables
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, sample: Dict) -> Dict:
        """
        Apply all transforms sequentially.

        Args:
            sample: Input sample dictionary

        Returns:
            Transformed sample dictionary
        """
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def get_train_transforms(
    flip_p: float = 0.5,
    rotate_p: float = 0.5,
    dropout_p: float = 0.1,
) -> Compose:
    """
    Get default training augmentation pipeline.

    This pipeline includes:
    1. Random horizontal/vertical flips
    2. Random 90° rotations
    3. Occupancy dropout (sensor noise simulation)

    Args:
        flip_p: Probability of flipping along each axis (default: 0.5)
        rotate_p: Probability of applying 90° rotation (default: 0.5)
        dropout_p: Probability of dropping each occupied voxel (default: 0.1)

    Returns:
        Composed transform pipeline

    Example:
        >>> transforms = get_train_transforms(flip_p=0.5, rotate_p=0.5)
        >>> dataset = MyDataset(data_root, config, transform=transforms)
    """
    return Compose([
        RandomFlip3D(p=flip_p),
        RandomRotate90(p=rotate_p),
        OccupancyDropout(p=dropout_p),
    ])


# Convenience exports
__all__ = [
    'RandomFlip3D',
    'RandomRotate90',
    'OccupancyDropout',
    'Compose',
    'get_train_transforms',
]
