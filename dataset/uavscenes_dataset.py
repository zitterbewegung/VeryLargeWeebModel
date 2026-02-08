#!/usr/bin/env python3
"""
UAVScenes Dataset Loader for 6DoF World Model Training

UAVScenes is an ICCV 2025 benchmark built on MARS-LVIG providing:
- Multi-modal UAV data (LiDAR + Camera)
- 6-DoF poses from aerial platform
- 120k labeled semantic pairs
- 4 diverse scenes: AMtown, AMvalley, HKairport, HKisland

This is REAL aerial 6DoF data, unlike nuScenes (ground vehicles with augmentation).

Dataset structure (actual HuggingFace format):
    data_root/
    ├── interval1_AMtown01/
    │   ├── interval1_CAM/           # Camera images (*.jpg)
    │   ├── interval1_LIDAR/         # LiDAR point clouds (*.txt with x y z)
    │   └── sampleinfos_interpolated.json  # Poses as T4x4 matrices
    ├── interval1_AMvalley01/
    ├── interval1_HKairport01/
    └── interval1_HKisland01/

Download from: https://github.com/sijieaaa/UAVScenes

Usage:
    from dataset.uavscenes_dataset import UAVScenesDataset, UAVScenesConfig

    config = UAVScenesConfig(scenes=['AMtown', 'AMvalley'])
    dataset = UAVScenesDataset('data/uavscenes', config)
"""

import os
import json
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import glob

try:
    from pyquaternion import Quaternion
    HAS_QUATERNION = True
except ImportError:
    HAS_QUATERNION = False
    Quaternion = None


@dataclass
class UAVScenesConfig:
    """Configuration for UAVScenes dataset."""

    # Scene selection
    scenes: List[str] = field(default_factory=lambda: ['AMtown', 'AMvalley', 'HKairport', 'HKisland'])

    # Run selection (None = all runs)
    runs: Optional[List[str]] = None  # e.g., ['run01', 'run02']

    # Interval: 5 = keyframes (available on HuggingFace), 1 = full data (not on HF)
    interval: int = 5

    # Temporal settings
    history_frames: int = 4
    future_frames: int = 6
    frame_skip: int = 1

    # Split (no official splits, we create our own)
    split: str = 'train'
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Occupancy grid settings (aerial-friendly range)
    point_cloud_range: Tuple[float, ...] = (-40.0, -40.0, -10.0, 40.0, 40.0, 50.0)
    voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.5)

    # Transform LiDAR points into ego frame using pose
    ego_frame: bool = True
    fallback_to_lidar_center: bool = True
    min_in_range_ratio: float = 0.01

    # Pose format: 13D = position(3) + quaternion(4) + linear_vel(3) + angular_vel(3)
    pose_dim: int = 13

    # LiDAR settings
    max_points: int = 100000
    use_intensity: bool = True

    # Load semantic labels
    use_semantic_labels: bool = False

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """Calculate grid size from range and voxel size."""
        pc_range = np.array(self.point_cloud_range)
        voxel_sz = np.array(self.voxel_size)
        return tuple(((pc_range[3:] - pc_range[:3]) / voxel_sz).astype(int))


class UAVScenesDataset(Dataset):
    """
    UAVScenes dataset for 6DoF aerial world model training.

    Provides real aerial LiDAR with 6DoF poses from UAV platform.
    """

    def __init__(self, data_root: str, config: UAVScenesConfig):
        self.data_root = Path(data_root)
        self.config = config

        # Validate data exists
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"UAVScenes data not found at: {data_root}\n"
                f"Download from: https://github.com/sijieaaa/UAVScenes"
            )

        # Cache for sampleinfos to avoid repeated file reads
        self._sampleinfos_cache: Dict[str, List[Dict]] = {}

        # Fallback tracking for _align_points
        self._fallback_count = 0
        self._total_align_calls = 0

        # Build sample index
        self.samples = self._build_sample_index()

        # Apply split
        self._apply_split()

        print(f"UAVScenes {config.split}: {len(self.samples)} samples")
        print(f"  Scenes: {config.scenes}")
        print(f"  Grid size: {config.grid_size}")

    def _build_sample_index(self) -> List[Dict]:
        """Build index of all valid temporal windows."""
        samples = []

        for scene in self.config.scenes:
            # Try multiple folder naming patterns
            scene_folders = self._find_scene_folder(scene)

            if not scene_folders:
                print(f"Warning: Scene not found: {scene}")
                continue

            for scene_folder in scene_folders:
                scene_path = self.data_root / scene_folder

                # Load sampleinfos for pose lookup
                sampleinfos_path = scene_path / 'sampleinfos_interpolated.json'
                if sampleinfos_path.exists():
                    self._load_sampleinfos(scene_folder, sampleinfos_path)

                # Find LiDAR files - try different structures
                lidar_files = self._find_lidar_files(scene_path)

                if not lidar_files:
                    print(f"Warning: No LiDAR files in {scene_path}")
                    continue

                print(f"  Found {len(lidar_files)} LiDAR files in {scene_folder}")

                # Get frame info from filenames
                frame_info = []
                for idx, f in enumerate(lidar_files):
                    frame_info.append({
                        'idx': idx,
                        'path': f,
                        'filename': f.stem,
                    })

                # Apply frame skip
                if self.config.frame_skip > 1:
                    frame_info = frame_info[::self.config.frame_skip]

                # Create sliding windows
                total_frames = self.config.history_frames + self.config.future_frames
                for i in range(len(frame_info) - total_frames + 1):
                    window = frame_info[i:i + total_frames]
                    samples.append({
                        'scene': scene,
                        'scene_folder': scene_folder,
                        'run': 'default',  # UAVScenes uses flat structure
                        'frames': window,
                        'history_frames': window[:self.config.history_frames],
                        'future_frames': window[self.config.history_frames:],
                    })

        return samples

    def _find_scene_folder(self, scene: str) -> List[str]:
        """Find actual folder name for a scene (handles different naming conventions).

        Searches both data_root directly and inside the interval{N}_CAM_LIDAR
        subdirectory (the HuggingFace zip extracts with this wrapper folder).
        Returns paths relative to data_root.
        """
        interval = self.config.interval
        folders = []
        seen = set()

        # Directories to search: data_root itself and the CAM_LIDAR subfolder
        search_roots = [self.data_root]
        cam_lidar_dir = self.data_root / f"interval{interval}_CAM_LIDAR"
        if cam_lidar_dir.is_dir():
            search_roots.append(cam_lidar_dir)

        for search_root in search_roots:
            # Relative prefix for paths under search_root
            if search_root == self.data_root:
                prefix = ""
            else:
                prefix = f"{search_root.name}/"

            # Pattern 1: interval{N}_{scene}01 (HuggingFace format)
            p1 = f"interval{interval}_{scene}01"
            if (search_root / p1).exists():
                rel = prefix + p1
                if rel not in seen:
                    folders.append(rel)
                    seen.add(rel)

            # Pattern 2: interval{N}_{scene} (without run number)
            p2 = f"interval{interval}_{scene}"
            if (search_root / p2).exists():
                rel = prefix + p2
                if rel not in seen:
                    folders.append(rel)
                    seen.add(rel)

            # Pattern 3: Just {scene}
            if (search_root / scene).exists():
                rel = prefix + scene
                if rel not in seen:
                    folders.append(rel)
                    seen.add(rel)

            # Pattern 4: Fuzzy match — find all dirs containing scene name
            # Catches variants like HKairport_GNSS01, HKairport_GNSS_Evening
            if search_root.is_dir():
                for d in sorted(search_root.iterdir()):
                    if d.is_dir() and scene.lower() in d.name.lower():
                        rel = prefix + d.name
                        if rel not in seen:
                            folders.append(rel)
                            seen.add(rel)

        return folders

    def _find_lidar_files(self, scene_path: Path) -> List[Path]:
        """Find LiDAR files in a scene folder (handles different structures)."""
        lidar_files = []

        # Pattern 1: interval{N}_LIDAR/*.txt (HuggingFace format)
        lidar_dir = scene_path / f'interval{self.config.interval}_LIDAR'
        if lidar_dir.exists():
            lidar_files = sorted(lidar_dir.glob('*.txt'))
            if lidar_files:
                return lidar_files

        # Pattern 2: interval{N}_CAM_LIDAR/*/lidar/*.pcd (original expected)
        for run_dir in (scene_path / f'interval{self.config.interval}_CAM_LIDAR').glob('*'):
            if run_dir.is_dir():
                lidar_path = run_dir / 'lidar'
                if lidar_path.exists():
                    lidar_files.extend(sorted(lidar_path.glob('*.pcd')))
                    lidar_files.extend(sorted(lidar_path.glob('*.bin')))
                else:
                    lidar_files.extend(sorted(run_dir.glob('*.pcd')))
                    lidar_files.extend(sorted(run_dir.glob('*.bin')))
        if lidar_files:
            return sorted(lidar_files)

        # Pattern 3: lidar/*.pcd or lidar/*.bin
        lidar_dir = scene_path / 'lidar'
        if lidar_dir.exists():
            lidar_files = sorted(lidar_dir.glob('*.pcd')) or sorted(lidar_dir.glob('*.bin'))
            if lidar_files:
                return lidar_files

        # Pattern 4: Direct *.pcd, *.bin, *.txt in scene folder
        lidar_files = sorted(scene_path.glob('*.pcd'))
        if not lidar_files:
            lidar_files = sorted(scene_path.glob('*.bin'))
        if not lidar_files:
            lidar_files = sorted(scene_path.glob('*.txt'))

        return lidar_files

    def _load_sampleinfos(self, scene_folder: str, path: Path):
        """Load and cache sampleinfos_interpolated.json."""
        if scene_folder not in self._sampleinfos_cache:
            try:
                with open(path) as f:
                    self._sampleinfos_cache[scene_folder] = json.load(f)
                print(f"  Loaded {len(self._sampleinfos_cache[scene_folder])} poses from sampleinfos")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load sampleinfos from {path}: {e}")
                self._sampleinfos_cache[scene_folder] = []

    def _apply_split(self):
        """Apply train/val/test split."""
        np.random.seed(42)  # Reproducible split

        n_samples = len(self.samples)
        indices = np.random.permutation(n_samples)

        n_val = int(n_samples * self.config.val_ratio)
        n_test = int(n_samples * self.config.test_ratio)
        n_train = n_samples - n_val - n_test

        if self.config.split == 'train':
            keep_indices = indices[:n_train]
        elif self.config.split == 'val':
            keep_indices = indices[n_train:n_train + n_val]
        else:  # test
            keep_indices = indices[n_train + n_val:]

        self.samples = [self.samples[i] for i in keep_indices]

    def _load_lidar(self, lidar_path: Path) -> np.ndarray:
        """Load LiDAR point cloud."""
        if lidar_path.suffix == '.pcd':
            return self._load_pcd(lidar_path)
        elif lidar_path.suffix == '.bin':
            return self._load_bin(lidar_path)
        elif lidar_path.suffix == '.npy':
            return np.load(lidar_path)
        elif lidar_path.suffix == '.txt':
            return self._load_txt(lidar_path)
        else:
            raise ValueError(f"Unknown LiDAR format: {lidar_path.suffix}")

    def _load_txt(self, path: Path) -> np.ndarray:
        """Load point cloud from space-separated text file (UAVScenes format)."""
        try:
            # UAVScenes format: x y z per line
            points = np.loadtxt(str(path), dtype=np.float32)
            if points.ndim == 1:
                points = points.reshape(1, -1)
            return points[:, :3] if points.shape[1] >= 3 else points
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            return np.zeros((0, 3), dtype=np.float32)

    def _load_pcd(self, path: Path) -> np.ndarray:
        """Load PCD file (simple ASCII/binary PCD parser)."""
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(path))
            points = np.asarray(pcd.points).astype(np.float32)
            return points
        except ImportError:
            # Fallback: simple PCD parser
            with open(path, 'rb') as f:
                header = []
                for _ in range(100):  # PCD headers are typically < 20 lines
                    raw = f.readline()
                    if not raw:  # EOF
                        raise ValueError(f"Invalid PCD file: no DATA line found in {path}")
                    line = raw.decode('utf-8', errors='ignore').strip()
                    header.append(line)
                    if line.startswith('DATA'):
                        break
                else:
                    raise ValueError(f"Invalid PCD file: header too large in {path}")

                # Parse header
                num_points = 0
                data_format = 'ascii'
                num_fields = 4  # Default: XYZI
                for h in header:
                    parts = h.split()
                    if h.startswith('POINTS') and len(parts) >= 2:
                        num_points = int(parts[1])
                    if h.startswith('DATA') and len(parts) >= 2:
                        data_format = parts[1]
                    if h.startswith('FIELDS'):
                        num_fields = len(parts) - 1  # e.g. "FIELDS x y z intensity"

                if data_format == 'binary':
                    # Binary PCD
                    data = np.frombuffer(f.read(), dtype=np.float32)
                    cols = max(num_fields, 3)
                    if len(data) % cols != 0:
                        # Try common column counts
                        for try_cols in [4, 3, 5, 6]:
                            if len(data) % try_cols == 0:
                                cols = try_cols
                                break
                    points = data.reshape(-1, cols)[:, :3]
                else:
                    # ASCII PCD
                    lines = f.read().decode('utf-8', errors='ignore').strip().split('\n')
                    parsed = []
                    for line in lines:
                        if not line.strip():
                            continue
                        try:
                            coords = [float(x) for x in line.split()[:3]]
                            if len(coords) == 3:
                                parsed.append(coords)
                        except (ValueError, IndexError):
                            continue
                    points = np.array(parsed) if parsed else np.zeros((0, 3))

                return points.astype(np.float32)

    def _load_bin(self, path: Path) -> np.ndarray:
        """Load binary point cloud file."""
        points = np.fromfile(str(path), dtype=np.float32)
        # Try common formats
        for cols in [4, 5, 3]:
            if len(points) % cols == 0:
                points = points.reshape(-1, cols)
                break
        return points[:, :3].astype(np.float32)

    def _load_pose(self, scene_folder: str, frame_idx: int, frame_filename: str = None) -> np.ndarray:
        """
        Load 6DoF pose from cached sampleinfos or pose files.

        Args:
            scene_folder: The actual folder name (e.g., 'interval1_AMtown01')
            frame_idx: The frame index in the sequence
            frame_filename: Optional filename for matching (e.g., 'image1658137057.641204937_lidar...')

        Returns 13D pose: position(3) + quaternion(4) + linear_vel(3) + angular_vel(3)
        """
        # Option 1: Use cached sampleinfos (HuggingFace format)
        if scene_folder in self._sampleinfos_cache:
            sampleinfos = self._sampleinfos_cache[scene_folder]

            # Try filename-based matching first (most reliable)
            if frame_filename:
                # Extract image timestamp from LiDAR filename
                # Format: image{ts}_lidar{ts}.txt -> match with OriginalImageName
                if frame_filename.startswith('image'):
                    img_ts = frame_filename.split('_')[0].replace('image', '')
                    for sample in sampleinfos:
                        if img_ts in sample.get('OriginalImageName', ''):
                            if 'T4x4' in sample:
                                return self._parse_t4x4_matrix(sample['T4x4'])

            # Fall back to index-based lookup
            if frame_idx < len(sampleinfos):
                sample = sampleinfos[frame_idx]
                if 'T4x4' in sample:
                    return self._parse_t4x4_matrix(sample['T4x4'])

        # Option 2: Try legacy formats
        scene_path = self.data_root / scene_folder

        # Individual pose JSON files
        pose_file = scene_path / 'poses' / f'{frame_idx:06d}.json'
        if pose_file.exists():
            with open(pose_file) as f:
                pose_data = json.load(f)
                return self._parse_pose_json(pose_data)

        # Aggregated poses.txt file
        poses_file = scene_path / 'poses.txt'
        if poses_file.exists():
            return self._load_pose_from_txt(poses_file, frame_idx)

        # Fallback: return zero pose (warn so users know data is missing)
        import warnings
        warnings.warn(
            f"No pose data found for {scene_folder} frame {frame_idx}. "
            "Using zero pose — this will degrade training quality.",
            stacklevel=2,
        )
        return np.zeros(13, dtype=np.float32)

    def _parse_t4x4_matrix(self, t4x4: List[List[float]]) -> np.ndarray:
        """
        Parse T4x4 transformation matrix to pose vector.

        T4x4 format:
            [[r11, r12, r13, tx],
             [r21, r22, r23, ty],
             [r31, r32, r33, tz],
             [0,   0,   0,   1]]

        Returns 13D pose: position(3) + quaternion(4) + linear_vel(3) + angular_vel(3)
        """
        T = np.array(t4x4, dtype=np.float64)

        # Extract translation
        position = T[:3, 3].astype(np.float32)

        # Extract rotation matrix and convert to quaternion
        R = T[:3, :3]
        quat = self._rotation_matrix_to_quaternion(R)

        # Velocities not available in sampleinfos, will be computed from pose differences
        lin_vel = np.zeros(3, dtype=np.float32)
        ang_vel = np.zeros(3, dtype=np.float32)

        return np.concatenate([position, quat, lin_vel, ang_vel]).astype(np.float32)

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
        if HAS_QUATERNION:
            q = Quaternion(matrix=R)
            return np.array([q.w, q.x, q.y, q.z], dtype=np.float32)

        # Manual conversion (Shepperd's method)
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        # Normalize quaternion (return identity if degenerate)
        quat = np.array([w, x, y, z], dtype=np.float32)
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return quat / norm

    def _parse_pose_json(self, pose_data: Dict) -> np.ndarray:
        """Parse pose from JSON format."""
        # Handle different JSON formats
        if 'position' in pose_data:
            pos = np.array([pose_data['position']['x'],
                          pose_data['position']['y'],
                          pose_data['position']['z']])
        elif 'translation' in pose_data:
            pos = np.array(pose_data['translation'])
        else:
            pos = np.zeros(3)

        if 'orientation' in pose_data:
            quat = np.array([pose_data['orientation']['w'],
                           pose_data['orientation']['x'],
                           pose_data['orientation']['y'],
                           pose_data['orientation']['z']])
        elif 'rotation' in pose_data:
            quat = np.array(pose_data['rotation'])
        else:
            quat = np.array([1, 0, 0, 0])

        # Velocities (if available)
        if 'velocity' in pose_data:
            lin_vel = np.array([pose_data['velocity']['linear']['x'],
                               pose_data['velocity']['linear']['y'],
                               pose_data['velocity']['linear']['z']])
            ang_vel = np.array([pose_data['velocity']['angular']['x'],
                               pose_data['velocity']['angular']['y'],
                               pose_data['velocity']['angular']['z']])
        else:
            lin_vel = np.zeros(3)
            ang_vel = np.zeros(3)

        return np.concatenate([pos, quat, lin_vel, ang_vel]).astype(np.float32)

    def _quaternion_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to rotation matrix."""
        w, x, y, z = quat
        n = w * w + x * x + y * y + z * z
        if n < 1e-8:
            return np.eye(3, dtype=np.float32)
        s = 2.0 / n

        wx = s * w * x
        wy = s * w * y
        wz = s * w * z
        xx = s * x * x
        xy = s * x * y
        xz = s * x * z
        yy = s * y * y
        yz = s * y * z
        zz = s * z * z

        return np.array([
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ], dtype=np.float32)

    def _transform_points_to_ego(self, points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Transform world-frame points into ego frame using pose.

        Convention: pose = [x, y, z, qw, qx, qy, qz, ...] defines the
        ego-to-world transform T4x4 = [R | t; 0 | 1].  To go from world
        to ego we apply the inverse: p_ego = R^T @ (p_world - t).
        R is orthogonal so R^-1 = R^T — no matrix inversion needed.
        """
        if points.shape[0] == 0:
            return points

        position = pose[:3].astype(np.float32)
        quat = pose[3:7].astype(np.float32)
        R = self._quaternion_to_rotation_matrix(quat)

        xyz = points[:, :3].astype(np.float32)
        # Apply world→ego: p_ego = R^T @ (p_world - t)
        xyz_ego = (xyz - position) @ R.T

        if points.shape[1] > 3:
            return np.concatenate([xyz_ego, points[:, 3:]], axis=1)
        return xyz_ego

    def _center_points(self, points: np.ndarray, origin: np.ndarray) -> np.ndarray:
        """Center points by a fixed origin (translation only)."""
        xyz = points[:, :3] - origin
        if points.shape[1] > 3:
            return np.concatenate([xyz, points[:, 3:]], axis=1)
        return xyz

    def _in_range_ratio(self, xyz: np.ndarray) -> float:
        """Compute fraction of points inside the configured point cloud range."""
        if xyz.shape[0] == 0:
            return 0.0
        pc_range = np.array(self.config.point_cloud_range)
        mask = (
            (xyz[:, 0] >= pc_range[0]) & (xyz[:, 0] < pc_range[3]) &
            (xyz[:, 1] >= pc_range[1]) & (xyz[:, 1] < pc_range[4]) &
            (xyz[:, 2] >= pc_range[2]) & (xyz[:, 2] < pc_range[5])
        )
        return float(mask.mean())

    def _align_points(
        self,
        points: np.ndarray,
        pose: np.ndarray,
        sequence_origin: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        """
        Align points to a stable frame.

        Strategy:
        1. Try ego-frame transform if enabled.
        2. If ego works (enough points in range), use it for consistency.
        3. If ego fails AND we already have a sequence_origin, use centering
           (don't mix ego + centering frames in one sequence).
        4. If ego fails AND no origin yet, set origin from LiDAR mean.

        Returns:
            aligned_points, updated_sequence_origin, used_lidar_center
        """
        self._total_align_calls += 1

        if self.config.ego_frame:
            points_pose = self._transform_points_to_ego(points, pose)
            ratio = self._in_range_ratio(points_pose[:, :3])
            if ratio >= self.config.min_in_range_ratio:
                # Ego transform works — use it and don't set sequence_origin
                # so subsequent frames also try ego first
                return points_pose, sequence_origin, False

        # Ego transform failed or not enabled — use centering fallback
        self._fallback_count += 1
        if self._fallback_count <= 5:
            warnings.warn(
                f"UAVScenes _align_points: ego-frame alignment fallback activated "
                f"({self._fallback_count}/{self._total_align_calls} calls). "
                f"LiDAR-center centering used instead.",
                stacklevel=2,
            )
        if self.config.fallback_to_lidar_center:
            if sequence_origin is None and points.shape[0] > 0:
                sequence_origin = points[:, :3].mean(axis=0).astype(np.float32)
            if sequence_origin is not None:
                return self._center_points(points, sequence_origin), sequence_origin, True

        return points, sequence_origin, False

    def _load_pose_from_txt(self, poses_file: Path, frame_id: int) -> np.ndarray:
        """Load pose from TUM-style poses.txt file."""
        # Format: timestamp tx ty tz qx qy qz qw
        with open(poses_file) as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 8:
                ts = int(float(parts[0]))
                if ts == frame_id:
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    return np.array([tx, ty, tz, qw, qx, qy, qz, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        return np.zeros(13, dtype=np.float32)

    def _load_pose_from_sampleinfos(self, sampleinfos_path: Path, run: str, frame_id: int) -> np.ndarray:
        """Load pose from UAVScenes sampleinfos_interpolated.json."""
        with open(sampleinfos_path) as f:
            data = json.load(f)

        # Navigate to correct run and frame
        if run in data:
            frames = data[run]
            frame_key = str(frame_id)
            if frame_key in frames:
                return self._parse_pose_json(frames[frame_key])

        return np.zeros(13, dtype=np.float32)

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

    def _compute_velocity(self, pose_curr: np.ndarray, pose_prev: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """Compute velocities from pose difference."""
        if dt <= 0:
            return pose_curr

        # Linear velocity
        lin_vel = (pose_curr[:3] - pose_prev[:3]) / dt

        # Angular velocity from quaternion difference
        if HAS_QUATERNION:
            q_curr = Quaternion(pose_curr[3], pose_curr[4], pose_curr[5], pose_curr[6])
            q_prev = Quaternion(pose_prev[3], pose_prev[4], pose_prev[5], pose_prev[6])
            q_diff = q_curr * q_prev.inverse
            angle = 2 * np.arccos(np.clip(q_diff.w, -1, 1))
            sin_half = np.sin(angle / 2)
            if sin_half > 1e-6:
                axis = np.array([q_diff.x, q_diff.y, q_diff.z]) / sin_half
                ang_vel = axis * angle / dt
            else:
                ang_vel = np.zeros(3)
        else:
            ang_vel = np.zeros(3)

        # Return a copy with computed velocities to avoid in-place mutation
        result = pose_curr.copy()
        result[7:10] = lin_vel
        result[10:13] = ang_vel

        return result

    @property
    def fallback_stats(self) -> Dict[str, float]:
        """Return statistics about ego-frame alignment fallback usage."""
        total = self._total_align_calls
        fallback = self._fallback_count
        return {
            'fallback_count': fallback,
            'total_align_calls': total,
            'fallback_ratio': fallback / total if total > 0 else 0.0,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]
        scene = sample_info['scene']
        scene_folder = sample_info.get('scene_folder', scene)

        # Load history frames
        history_occ = []
        history_poses = []
        prev_pose = None
        sequence_origin = None

        for frame_info in sample_info['history_frames']:
            # Load pose
            frame_idx = frame_info['idx']
            frame_filename = frame_info.get('filename', '')
            pose = self._load_pose(scene_folder, frame_idx, frame_filename)

            # Load LiDAR
            lidar_path = frame_info['path']
            points = self._load_lidar(lidar_path)
            points, sequence_origin, used_lidar_center = self._align_points(points, pose, sequence_origin)
            if used_lidar_center and sequence_origin is not None:
                pose = pose.copy()
                pose[:3] = pose[:3] - sequence_origin
            occ = self._points_to_occupancy(points)

            # Compute velocities if not provided
            if prev_pose is not None and np.allclose(pose[7:], 0):
                pose = self._compute_velocity(pose, prev_pose)

            history_occ.append(occ)
            history_poses.append(pose)
            prev_pose = pose.copy()

        # Load future frames
        future_occ = []
        future_poses = []

        for frame_info in sample_info['future_frames']:
            frame_idx = frame_info['idx']
            frame_filename = frame_info.get('filename', '')
            pose = self._load_pose(scene_folder, frame_idx, frame_filename)

            lidar_path = frame_info['path']
            points = self._load_lidar(lidar_path)
            points, sequence_origin, used_lidar_center = self._align_points(points, pose, sequence_origin)
            if used_lidar_center and sequence_origin is not None:
                pose = pose.copy()
                pose[:3] = pose[:3] - sequence_origin
            occ = self._points_to_occupancy(points)

            if prev_pose is not None and np.allclose(pose[7:], 0):
                pose = self._compute_velocity(pose, prev_pose)

            future_occ.append(occ)
            future_poses.append(pose)
            prev_pose = pose.copy()

        return {
            'history_occupancy': torch.from_numpy(np.stack(history_occ)).float(),
            'future_occupancy': torch.from_numpy(np.stack(future_occ)).float(),
            'history_poses': torch.from_numpy(np.stack(history_poses)).float(),
            'future_poses': torch.from_numpy(np.stack(future_poses)).float(),
            'agent_type': torch.tensor(1),  # 1 = aerial (real UAV data!)
            'scene': scene,
            'scene_folder': scene_folder,
            'domain_tag': 'real',
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    collated = {}
    for key in ('history_occupancy', 'future_occupancy', 'history_poses', 'future_poses', 'agent_type'):
        tensors = [b[key] for b in batch]
        try:
            collated[key] = torch.stack(tensors)
        except RuntimeError as e:
            shapes = [list(t.shape) for t in tensors]
            raise RuntimeError(
                f"Cannot stack '{key}': shapes {shapes}. "
                f"All samples must have the same shape for '{key}'."
            ) from e
    return collated


def create_dataloader(
    data_root: str,
    config: UAVScenesConfig,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader for UAVScenes dataset."""
    dataset = UAVScenesDataset(data_root, config)
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
# Download Helper
# =============================================================================

def download_uavscenes(output_dir: str, scenes: List[str] = None, interval: int = 5):
    """
    Print download instructions for UAVScenes.

    Full dataset is very large; interval=5 version is ~20% of the size.
    """
    print("=" * 70)
    print("UAVScenes Download Instructions")
    print("=" * 70)
    print()
    print("UAVScenes provides multi-modal UAV data with 6DoF poses.")
    print("Paper: ICCV 2025 | GitHub: https://github.com/sijieaaa/UAVScenes")
    print()
    print("Available scenes: AMtown, AMvalley, HKairport, HKisland")
    print()
    print("Download options (choose one):")
    print("  - OneDrive: See GitHub README")
    print("  - Google Drive: See GitHub README")
    print("  - HuggingFace: huggingface.co/datasets/sijieaaa/UAVScenes")
    print("  - Baidu: See GitHub README")
    print()
    print(f"Recommended for testing: interval={interval} version (~20% size)")
    print()
    print("After download, extract to:")
    print(f"  {output_dir}/")
    print("  ├── interval1_AMtown01/")
    print("  │   ├── interval1_CAM/")
    print("  │   ├── interval1_LIDAR/")
    print("  │   └── sampleinfos_interpolated.json")
    print("  ├── interval1_AMvalley01/")
    print("  ├── interval1_HKairport01/")
    print("  └── interval1_HKisland01/")
    print()


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    import sys

    print("=" * 70)
    print("UAVScenes Dataset - Real Aerial 6DoF Data")
    print("=" * 70)
    print()

    data_root = sys.argv[1] if len(sys.argv) > 1 else 'data/uavscenes'

    if not os.path.exists(data_root):
        print(f"Data not found at: {data_root}")
        download_uavscenes(data_root)
        sys.exit(0)

    print(f"Testing with data from: {data_root}")

    config = UAVScenesConfig(
        scenes=['AMtown'],
        split='train',
    )

    try:
        dataset = UAVScenesDataset(data_root, config)
        print(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"History occupancy: {sample['history_occupancy'].shape}")
            print(f"Future occupancy: {sample['future_occupancy'].shape}")
            print(f"History poses: {sample['history_poses'].shape}")
            print(f"Future poses: {sample['future_poses'].shape}")
            print(f"Agent type: {sample['agent_type']} (1=aerial)")

            # Check pose values
            poses = sample['history_poses'].numpy()
            print(f"Position range: [{poses[:, :3].min():.2f}, {poses[:, :3].max():.2f}]")

    except Exception as e:
        print(f"Error: {e}")
        download_uavscenes(data_root)

    print()
    print("=" * 70)
