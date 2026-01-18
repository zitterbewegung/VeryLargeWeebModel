#!/usr/bin/env python3
"""
GazeboOccWorldDataset - PyTorch Dataset for OccWorld Training

This module implements a PyTorch Dataset that loads simulation data from
ArduPilot Gazebo recordings for training OccWorld world models.

Data Structure:
    data_root/
    ├── drone_20240101_120000/          # Session directory
    │   ├── images/                      # Camera images
    │   │   ├── 000001_CAM_FRONT.jpg
    │   │   ├── 000001_CAM_FRONT_LEFT.jpg
    │   │   └── ...
    │   ├── lidar/                       # LiDAR point clouds
    │   │   └── 000001_LIDAR.npy
    │   ├── poses/                       # Vehicle poses
    │   │   └── 000001.json
    │   └── occupancy/                   # Ground truth occupancy
    │       └── 000001_occupancy.npz
    └── rover_20240101_130000/
        └── ...

Output Format (compatible with OccWorld training loop):
    {
        'history_images': List[Dict[str, Tensor]],  # [T_h] x {cam_name: [3,H,W]}
        'history_lidar': List[Tensor],              # [T_h] x [N,4]
        'history_poses': Tensor,                    # [T_h, 13]
        'history_occupancy': Tensor,                # [T_h, X, Y, Z]
        'future_occupancy': Tensor,                 # [T_f, X, Y, Z]
        'future_poses': Tensor,                     # [T_f, 13]
        'agent_type': int,                          # 0=ground, 1=aerial
    }
"""

import os
import json
from glob import glob
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

# Thread pool for parallel I/O (shared across dataset instances)
_IO_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _get_io_executor(max_workers: int = None) -> ThreadPoolExecutor:
    """Get or create shared I/O thread pool."""
    global _IO_EXECUTOR
    if _IO_EXECUTOR is None:
        workers = max_workers or min(8, cpu_count())
        _IO_EXECUTOR = ThreadPoolExecutor(max_workers=workers)
    return _IO_EXECUTOR


@dataclass
class DatasetConfig:
    """Configuration for the Gazebo OccWorld dataset."""

    # Temporal configuration
    history_frames: int = 4          # Number of past frames to use
    future_frames: int = 6           # Number of future frames to predict
    frame_skip: int = 1              # Skip frames for temporal diversity

    # Agent filtering
    agent_type: str = 'both'         # 'drone', 'rover', or 'both'

    # Data split
    split: str = 'train'             # 'train', 'val', or 'test'
    val_ratio: float = 0.1           # Validation set ratio
    test_ratio: float = 0.1          # Test set ratio

    # Image configuration
    image_size: Tuple[int, int] = (900, 1600)  # (H, W)
    normalize_images: bool = True

    # Point cloud configuration
    max_points: int = 100000         # Max points per LiDAR scan

    # Occupancy configuration
    point_cloud_range: Tuple[float, ...] = (-40, -40, -2, 40, 40, 150)
    voxel_size: Tuple[float, float, float] = (0.4, 0.4, 1.25)


class GazeboOccWorldDataset(Dataset):
    """
    PyTorch Dataset for OccWorld training using Gazebo simulation data.

    This dataset:
    1. Indexes all valid recording sessions
    2. Builds sequences of (history + future) frames
    3. Loads and preprocesses multi-modal sensor data
    4. Returns tensors compatible with OccWorld's training loop

    Example:
        config = DatasetConfig(history_frames=4, future_frames=6)
        dataset = GazeboOccWorldDataset('/data/occworld_training', config)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        for batch in loader:
            history_occ = batch['history_occupancy']  # [B, T_h, X, Y, Z]
            future_occ = batch['future_occupancy']    # [B, T_f, X, Y, Z]
            # ... training step
    """

    # Camera names matching BEVFusion/nuScenes convention
    CAMERA_NAMES = [
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]

    def __init__(
        self,
        data_root: str,
        config: Optional[DatasetConfig] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_root: Root directory containing session folders
            config: Dataset configuration
            transform: Optional transform to apply to samples
        """
        self.data_root = data_root
        self.config = config or DatasetConfig()
        self.transform = transform

        # Load and validate sessions
        self.sessions = self._discover_sessions()
        print(f"Found {len(self.sessions)} valid sessions")

        # Build frame index
        self.frame_index = self._build_frame_index()
        print(f"Built index with {len(self.frame_index)} valid sequences")

        # Apply data split
        self._apply_split()
        print(f"Split '{self.config.split}': {len(self.frame_index)} sequences")

    def _discover_sessions(self) -> List[str]:
        """
        Discover all valid recording sessions in data_root.

        A valid session must have:
        - images/ directory with camera images
        - lidar/ directory with point clouds
        - poses/ directory with pose JSON files
        - occupancy/ directory with occupancy ground truth

        Returns:
            List of session directory paths
        """
        sessions = []

        for session_dir in sorted(glob(os.path.join(self.data_root, '*'))):
            if not os.path.isdir(session_dir):
                continue

            session_name = os.path.basename(session_dir)

            # Filter by agent type
            if self.config.agent_type == 'drone':
                if not session_name.startswith('drone'):
                    continue
            elif self.config.agent_type == 'rover':
                if not session_name.startswith('rover'):
                    continue
            # 'both' accepts all sessions

            # Verify required directories exist
            required_dirs = ['images', 'lidar', 'poses', 'occupancy']
            if all(os.path.isdir(os.path.join(session_dir, d)) for d in required_dirs):
                sessions.append(session_dir)

        return sessions

    def _build_frame_index(self) -> List[Dict[str, Any]]:
        """
        Build an index of all valid frame sequences.

        Each sequence consists of:
        - history_frames consecutive past frames
        - future_frames consecutive future frames

        Returns:
            List of sequence metadata dicts
        """
        index = []
        total_needed = self.config.history_frames + self.config.future_frames

        for session_dir in self.sessions:
            # Get sorted frame IDs from occupancy files
            occ_files = sorted(glob(os.path.join(session_dir, 'occupancy', '*.npz')))
            frame_ids = [
                os.path.basename(f).replace('_occupancy.npz', '')
                for f in occ_files
            ]

            # Determine agent type from session name
            agent_type = 1 if 'drone' in session_dir else 0  # 1=aerial, 0=ground

            # Build sequences with frame_skip
            step = self.config.frame_skip
            for start_idx in range(0, len(frame_ids) - total_needed * step + 1, step):
                # Get sequence frame IDs
                seq_indices = range(start_idx, start_idx + total_needed * step, step)
                seq_frames = [frame_ids[i] for i in seq_indices]

                # Validate all frames exist
                if self._validate_sequence(session_dir, seq_frames):
                    index.append({
                        'session': session_dir,
                        'frames': seq_frames,
                        'agent_type': agent_type,
                    })

        return index

    def _validate_sequence(self, session_dir: str, frame_ids: List[str]) -> bool:
        """Check if all data exists for a sequence of frames."""
        for fid in frame_ids:
            # Check occupancy
            if not os.path.exists(os.path.join(session_dir, 'occupancy', f'{fid}_occupancy.npz')):
                return False

            # Check pose
            if not os.path.exists(os.path.join(session_dir, 'poses', f'{fid}.json')):
                return False

            # Check at least front camera (others optional)
            if not os.path.exists(os.path.join(session_dir, 'images', f'{fid}_CAM_FRONT.jpg')):
                return False

        return True

    def _apply_split(self):
        """Apply train/val/test split to the frame index."""
        n = len(self.frame_index)
        val_size = int(n * self.config.val_ratio)
        test_size = int(n * self.config.test_ratio)
        train_size = n - val_size - test_size

        # Deterministic split based on index position
        if self.config.split == 'train':
            self.frame_index = self.frame_index[:train_size]
        elif self.config.split == 'val':
            self.frame_index = self.frame_index[train_size:train_size + val_size]
        elif self.config.split == 'test':
            self.frame_index = self.frame_index[train_size + val_size:]

    def __len__(self) -> int:
        return len(self.frame_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and return a training sample.

        Uses parallel I/O to load all frame data concurrently for better performance.

        Args:
            idx: Sample index

        Returns:
            Dict containing history and future data tensors
        """
        item = self.frame_index[idx]
        session_dir = item['session']
        frames = item['frames']
        agent_type = item['agent_type']

        # Split into history and future
        n_history = self.config.history_frames
        history_frames = frames[:n_history]
        future_frames = frames[n_history:]

        executor = _get_io_executor()

        # Submit all I/O operations in parallel
        # History: images, lidar, poses, occupancy
        history_image_futures = [
            executor.submit(self._load_images, session_dir, fid)
            for fid in history_frames
        ]
        history_lidar_futures = [
            executor.submit(self._load_lidar, session_dir, fid)
            for fid in history_frames
        ]
        history_pose_futures = [
            executor.submit(self._load_pose, session_dir, fid)
            for fid in history_frames
        ]
        history_occ_futures = [
            executor.submit(self._load_occupancy, session_dir, fid)
            for fid in history_frames
        ]

        # Future: poses and occupancy only
        future_pose_futures = [
            executor.submit(self._load_pose, session_dir, fid)
            for fid in future_frames
        ]
        future_occ_futures = [
            executor.submit(self._load_occupancy, session_dir, fid)
            for fid in future_frames
        ]

        # Collect results (maintains order)
        history_images = [f.result() for f in history_image_futures]
        history_lidar = [f.result() for f in history_lidar_futures]
        history_poses = [f.result() for f in history_pose_futures]
        history_occupancy = [f.result() for f in history_occ_futures]

        future_poses = [f.result() for f in future_pose_futures]
        future_occupancy = [f.result() for f in future_occ_futures]

        # Build sample dict
        sample = {
            'history_images': history_images,
            'history_lidar': history_lidar,
            'history_poses': torch.stack(history_poses),
            'history_occupancy': torch.stack(history_occupancy),
            'future_occupancy': torch.stack(future_occupancy),
            'future_poses': torch.stack(future_poses),
            'agent_type': agent_type,
            'session': os.path.basename(session_dir),
            'frames': frames,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_single_image(self, img_path: str, cam_name: str) -> Tuple[str, torch.Tensor]:
        """Load a single camera image (for parallel execution)."""
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img.shape[:2] != self.config.image_size:
                img = cv2.resize(img, (self.config.image_size[1], self.config.image_size[0]))

            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

            if self.config.normalize_images:
                img_tensor = img_tensor / 255.0

            return cam_name, img_tensor
        else:
            return cam_name, torch.zeros(
                3, self.config.image_size[0], self.config.image_size[1]
            )

    def _load_images(self, session_dir: str, frame_id: str) -> Dict[str, torch.Tensor]:
        """Load all camera images for a frame (parallel across 6 cameras)."""
        executor = _get_io_executor()

        # Submit all camera loads in parallel
        futures = []
        for cam_name in self.CAMERA_NAMES:
            img_path = os.path.join(session_dir, 'images', f'{frame_id}_{cam_name}.jpg')
            futures.append(executor.submit(self._load_single_image, img_path, cam_name))

        # Collect results
        images = {}
        for future in futures:
            cam_name, img_tensor = future.result()
            images[cam_name] = img_tensor

        return images

    def _load_lidar(self, session_dir: str, frame_id: str) -> torch.Tensor:
        """Load LiDAR point cloud."""
        lidar_path = os.path.join(session_dir, 'lidar', f'{frame_id}_LIDAR.npy')

        if os.path.exists(lidar_path):
            points = np.load(lidar_path)  # [N, 4] - x, y, z, intensity

            # Subsample if too many points
            if len(points) > self.config.max_points:
                indices = np.random.choice(
                    len(points), self.config.max_points, replace=False
                )
                points = points[indices]

            return torch.from_numpy(points).float()
        else:
            return torch.zeros(0, 4)

    def _load_pose(self, session_dir: str, frame_id: str) -> torch.Tensor:
        """
        Load vehicle pose from JSON.

        Returns pose as a 13-element tensor:
        [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        """
        pose_path = os.path.join(session_dir, 'poses', f'{frame_id}.json')

        with open(pose_path, 'r') as f:
            data = json.load(f)

        pos = data['position']
        ori = data['orientation']
        vel = data['velocity']

        return torch.tensor([
            pos['x'], pos['y'], pos['z'],
            ori['x'], ori['y'], ori['z'], ori['w'],
            vel['linear']['x'], vel['linear']['y'], vel['linear']['z'],
            vel['angular']['x'], vel['angular']['y'], vel['angular']['z'],
        ], dtype=torch.float32)

    def _load_occupancy(self, session_dir: str, frame_id: str) -> torch.Tensor:
        """Load occupancy ground truth."""
        occ_path = os.path.join(session_dir, 'occupancy', f'{frame_id}_occupancy.npz')

        data = np.load(occ_path)
        occupancy = data['occupancy']  # [X, Y, Z]

        return torch.from_numpy(occupancy).long()


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for DataLoader.

    Handles variable-size LiDAR point clouds and nested image dicts.
    """
    collated = {}

    # Stack standard tensors
    collated['history_poses'] = torch.stack([s['history_poses'] for s in batch])
    collated['history_occupancy'] = torch.stack([s['history_occupancy'] for s in batch])
    collated['future_occupancy'] = torch.stack([s['future_occupancy'] for s in batch])
    collated['future_poses'] = torch.stack([s['future_poses'] for s in batch])
    collated['agent_type'] = torch.tensor([s['agent_type'] for s in batch])

    # Handle images (list of list of dicts)
    # Output: {cam_name: [B, T, C, H, W]}
    collated['history_images'] = {}
    n_history = len(batch[0]['history_images'])
    cam_names = batch[0]['history_images'][0].keys()

    for cam in cam_names:
        cam_stack = []
        for sample in batch:
            time_stack = torch.stack([sample['history_images'][t][cam] for t in range(n_history)])
            cam_stack.append(time_stack)
        collated['history_images'][cam] = torch.stack(cam_stack)  # [B, T, C, H, W]

    # Handle LiDAR (variable length)
    # Keep as list for now; models should handle padding
    collated['history_lidar'] = [s['history_lidar'] for s in batch]

    return collated


def create_dataloader(
    data_root: str,
    config: Optional[DatasetConfig] = None,
    batch_size: int = 4,
    num_workers: int = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for OccWorld training.

    Args:
        data_root: Path to data directory
        config: Dataset configuration
        batch_size: Batch size
        num_workers: Number of data loading workers (auto-detect if None)
        shuffle: Whether to shuffle data

    Returns:
        DataLoader instance
    """
    dataset = GazeboOccWorldDataset(data_root, config)

    # Auto-detect workers if not specified
    if num_workers is None:
        num_workers = max(4, cpu_count() - 2)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


# Example usage and testing
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test Gazebo OccWorld Dataset')
    parser.add_argument('--data', type=str, required=True, help='Data root directory')
    parser.add_argument('--batch', type=int, default=2, help='Batch size')
    args = parser.parse_args()

    # Create dataset
    config = DatasetConfig(
        history_frames=4,
        future_frames=6,
        agent_type='both',
        split='train',
    )

    dataset = GazeboOccWorldDataset(args.data, config)
    print(f"\nDataset size: {len(dataset)}")

    # Test single sample
    if len(dataset) > 0:
        sample = dataset[0]
        print("\nSample contents:")
        print(f"  history_poses: {sample['history_poses'].shape}")
        print(f"  history_occupancy: {sample['history_occupancy'].shape}")
        print(f"  future_occupancy: {sample['future_occupancy'].shape}")
        print(f"  future_poses: {sample['future_poses'].shape}")
        print(f"  agent_type: {sample['agent_type']}")
        print(f"  session: {sample['session']}")

        if sample['history_images']:
            for cam, img in sample['history_images'][0].items():
                print(f"  {cam}: {img.shape}")

    # Test DataLoader
    loader = create_dataloader(args.data, config, batch_size=args.batch)

    for batch in loader:
        print("\nBatch contents:")
        print(f"  history_poses: {batch['history_poses'].shape}")
        print(f"  history_occupancy: {batch['history_occupancy'].shape}")
        print(f"  future_occupancy: {batch['future_occupancy'].shape}")
        print(f"  agent_type: {batch['agent_type']}")

        for cam, imgs in batch['history_images'].items():
            print(f"  {cam}: {imgs.shape}")

        break  # Just test first batch
