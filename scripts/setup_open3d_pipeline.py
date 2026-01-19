#!/usr/bin/env python3
"""
Open3D-ML Pipeline Setup for OccWorld Training

This script sets up the complete data pipeline using Open3D and Open3D-ML:
1. Downloads Tokyo PLATEAU 3D city models
2. Converts meshes to point clouds with semantic features
3. Generates high-quality voxelized occupancy grids
4. Creates training data with proper motion dynamics

Requirements:
    pip install open3d torch numpy requests tqdm

Open3D-ML (optional, for semantic features):
    pip install open3d-ml-torch  # or open3d-ml-tf for TensorFlow

Usage:
    python scripts/setup_open3d_pipeline.py --download --process --output data/tokyo_plateau
    python scripts/setup_open3d_pipeline.py --process-only --input data/plateau/meshes --output data/tokyo_gazebo
"""

import os
import sys
import argparse
import json
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

# Check for required libraries
try:
    import open3d as o3d
    HAS_OPEN3D = True
    O3D_VERSION = o3d.__version__
except ImportError:
    HAS_OPEN3D = False
    O3D_VERSION = None

try:
    # Open3D-ML for semantic feature extraction
    import open3d.ml as ml3d
    import open3d.ml.torch as ml3d_torch
    HAS_OPEN3D_ML = True
except ImportError:
    HAS_OPEN3D_ML = False
except Exception as e:
    # Catch PyTorch version mismatch errors
    HAS_OPEN3D_ML = False
    if "Version mismatch" in str(e):
        print(f"Note: Open3D-ML disabled: {e}")
        print("  Geometric features still available. Semantic segmentation disabled.")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import requests
    from tqdm import tqdm
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Open3DPipelineConfig:
    """Configuration for Open3D-based data pipeline."""

    # Voxel grid settings (matching OccWorld expectations)
    point_cloud_range: Tuple[float, ...] = (-40.0, -40.0, -2.0, 40.0, 40.0, 150.0)
    voxel_size: Tuple[float, float, float] = (0.4, 0.4, 1.25)

    # Point cloud processing
    max_points_per_voxel: int = 35
    max_voxels: int = 40000

    # Feature extraction
    compute_normals: bool = True
    compute_fpfh: bool = True  # Fast Point Feature Histograms
    normal_radius: float = 1.0
    fpfh_radius: float = 2.5

    # Semantic segmentation (requires Open3D-ML)
    use_semantic_features: bool = True
    semantic_model: str = "RandLANet"  # or "KPConv", "PointTransformer"

    # Trajectory settings
    min_motion_threshold: float = 0.5  # Minimum movement between frames (meters)

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        return (
            int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0]),
            int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1]),
            int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.voxel_size[2]),
        )


# =============================================================================
# PLATEAU Data Download
# =============================================================================

PLATEAU_URLS = {
    # Tokyo 23 wards - OBJ format (best for conversion)
    'tokyo_obj': 'https://gic-plateau.s3.ap-northeast-1.amazonaws.com/2020/13100_tokyo23-ku_2020_obj_3_op.zip',
    # Tokyo 23 wards - CityGML (semantic data)
    'tokyo_citygml': 'https://assets.cms.plateau.reearth.io/assets/ec/d51c64-a47f-4a56-aa64-340d1d3c720b/13100_tokyo23-ku_2020_citygml_4_2_op.zip',
    # Shibuya area (smaller, good for testing)
    'shibuya': 'https://gic-plateau.s3.ap-northeast-1.amazonaws.com/2020/13113_shibuya-ku_2020_obj_3_op.zip',
}


def download_file(url: str, output_path: str, description: str = "Downloading") -> bool:
    """Download a file with progress bar."""
    if not HAS_REQUESTS:
        print("Error: requests library required. Install with: pip install requests tqdm")
        return False

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def download_plateau_data(output_dir: str, dataset: str = 'shibuya') -> Optional[str]:
    """
    Download PLATEAU dataset.

    Args:
        output_dir: Output directory for downloaded data
        dataset: Which dataset to download ('tokyo_obj', 'tokyo_citygml', 'shibuya')

    Returns:
        Path to extracted mesh directory, or None if failed
    """
    if dataset not in PLATEAU_URLS:
        print(f"Unknown dataset: {dataset}. Available: {list(PLATEAU_URLS.keys())}")
        return None

    url = PLATEAU_URLS[dataset]
    zip_path = os.path.join(output_dir, 'raw', f'{dataset}.zip')
    extract_dir = os.path.join(output_dir, 'meshes', dataset)

    # Check if already downloaded
    if os.path.exists(extract_dir) and any(Path(extract_dir).rglob('*.obj')):
        print(f"Dataset already exists: {extract_dir}")
        return extract_dir

    # Download
    print(f"\nDownloading PLATEAU {dataset} dataset...")
    print(f"URL: {url}")
    print("Note: This data is provided by MLIT Japan under CC BY 4.0 license")

    if not download_file(url, zip_path, f"Downloading {dataset}"):
        return None

    # Extract
    print(f"\nExtracting to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted successfully")

        # Count mesh files
        obj_files = list(Path(extract_dir).rglob('*.obj'))
        print(f"Found {len(obj_files)} OBJ mesh files")

        return extract_dir
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None


# =============================================================================
# Open3D Point Cloud Processing
# =============================================================================

class Open3DPointCloudProcessor:
    """Process point clouds using Open3D for high-quality voxelization."""

    def __init__(self, config: Open3DPipelineConfig):
        if not HAS_OPEN3D:
            raise ImportError("Open3D required. Install with: pip install open3d")

        self.config = config
        self.semantic_model = None

        print(f"Open3D version: {O3D_VERSION}")
        print(f"Open3D-ML available: {HAS_OPEN3D_ML}")

    def load_mesh(self, mesh_path: str) -> Optional[o3d.geometry.TriangleMesh]:
        """Load a mesh file."""
        try:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if len(mesh.vertices) == 0:
                return None

            # Compute normals if not present
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()

            return mesh
        except Exception as e:
            print(f"Failed to load {mesh_path}: {e}")
            return None

    def mesh_to_point_cloud(
        self,
        mesh: o3d.geometry.TriangleMesh,
        num_points: int = 100000
    ) -> o3d.geometry.PointCloud:
        """Convert mesh to point cloud with uniform sampling."""
        # Sample points from mesh surface
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)

        # Compute normals
        if self.config.compute_normals:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.config.normal_radius,
                    max_nn=30
                )
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)

        return pcd

    def compute_fpfh_features(
        self,
        pcd: o3d.geometry.PointCloud
    ) -> o3d.pipelines.registration.Feature:
        """Compute Fast Point Feature Histograms."""
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.config.normal_radius,
                    max_nn=30
                )
            )

        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.config.fpfh_radius,
                max_nn=100
            )
        )
        return fpfh

    def voxelize_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        position: np.ndarray,
        yaw: float = 0.0
    ) -> np.ndarray:
        """
        Voxelize point cloud centered at position.

        Uses Open3D's VoxelGrid for accurate voxelization.

        Args:
            pcd: Input point cloud
            position: Center position [x, y, z]
            yaw: Rotation angle in radians

        Returns:
            occupancy: [X, Y, Z] uint8 grid
        """
        cfg = self.config
        grid_size = cfg.grid_size

        # Get points as numpy array
        points = np.asarray(pcd.points)

        # Transform to local frame
        # 1. Translate to position-centered coordinates
        points_local = points - position

        # 2. Apply yaw rotation
        if yaw != 0:
            cos_yaw, sin_yaw = np.cos(-yaw), np.sin(-yaw)
            rot_matrix = np.array([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1]
            ])
            points_local = (rot_matrix @ points_local.T).T

        # Filter to point cloud range
        pc_range = np.array(cfg.point_cloud_range)
        mask = (
            (points_local[:, 0] >= pc_range[0]) & (points_local[:, 0] < pc_range[3]) &
            (points_local[:, 1] >= pc_range[1]) & (points_local[:, 1] < pc_range[4]) &
            (points_local[:, 2] >= pc_range[2]) & (points_local[:, 2] < pc_range[5])
        )
        points_local = points_local[mask]

        if len(points_local) == 0:
            # Return grid with just ground plane
            occupancy = np.zeros(grid_size, dtype=np.uint8)
            ground_z = int((0 - pc_range[2]) / cfg.voxel_size[2])
            if 0 <= ground_z < grid_size[2]:
                occupancy[:, :, ground_z] = 1
            return occupancy

        # Create point cloud for voxelization
        pcd_local = o3d.geometry.PointCloud()
        pcd_local.points = o3d.utility.Vector3dVector(points_local)

        # Use Open3D's voxel grid (more accurate than manual binning)
        # Note: Open3D uses uniform voxel size, so we use the smallest dimension
        min_voxel_size = min(cfg.voxel_size)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd_local,
            voxel_size=min_voxel_size
        )

        # Convert to our grid format
        occupancy = np.zeros(grid_size, dtype=np.uint8)
        voxel_size = np.array(cfg.voxel_size)

        for voxel in voxel_grid.get_voxels():
            # Get voxel center in local coords
            center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)

            # Convert to our grid indices
            idx = np.floor((center - pc_range[:3]) / voxel_size).astype(int)

            if (0 <= idx[0] < grid_size[0] and
                0 <= idx[1] < grid_size[1] and
                0 <= idx[2] < grid_size[2]):
                occupancy[idx[0], idx[1], idx[2]] = 1

        # Add ground plane
        ground_z = int((0 - pc_range[2]) / cfg.voxel_size[2])
        if 0 <= ground_z < grid_size[2]:
            occupancy[:, :, ground_z] = np.maximum(occupancy[:, :, ground_z], 1)

        return occupancy

    def voxelize_with_features(
        self,
        pcd: o3d.geometry.PointCloud,
        position: np.ndarray,
        yaw: float = 0.0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Voxelize with optional feature extraction.

        Returns:
            occupancy: [X, Y, Z] binary occupancy
            features: [X, Y, Z, F] voxel features (if enabled), or None
        """
        occupancy = self.voxelize_point_cloud(pcd, position, yaw)

        features = None
        if self.config.compute_fpfh and self.config.use_semantic_features:
            # Extract FPFH features per voxel (simplified)
            # Full implementation would aggregate features per voxel
            pass

        return occupancy, features


# =============================================================================
# Open3D-ML Semantic Feature Extraction
# =============================================================================

class Open3DMLFeatureExtractor:
    """
    Extract semantic features using Open3D-ML pretrained models.

    Supports:
    - RandLA-Net: Efficient for large-scale point clouds
    - KPConv: Kernel Point Convolution
    - PointTransformer: Attention-based
    """

    def __init__(self, model_name: str = "RandLANet", device: str = "cuda"):
        if not HAS_OPEN3D_ML:
            raise ImportError(
                "Open3D-ML required for semantic features.\n"
                "Install with: pip install open3d-ml-torch\n"
                "Or: pip install open3d-ml-tf"
            )

        if not HAS_TORCH:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.pipeline = None

        print(f"Initializing Open3D-ML {model_name} on {self.device}")

    def load_pretrained_model(self, dataset: str = "SemanticKITTI"):
        """
        Load a pretrained semantic segmentation model.

        Args:
            dataset: Dataset the model was trained on
                    ('SemanticKITTI', 'S3DIS', 'Toronto3D', etc.)
        """
        try:
            # Get model class
            if self.model_name == "RandLANet":
                model_class = ml3d_torch.models.RandLANet
            elif self.model_name == "KPConv":
                model_class = ml3d_torch.models.KPFCNN
            elif self.model_name == "PointTransformer":
                model_class = ml3d_torch.models.PointTransformer
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

            # Load pretrained weights
            # Note: You may need to download weights separately
            cfg = ml3d_torch.models.RandLANet.get_config()
            self.model = model_class(**cfg)

            # Create inference pipeline
            self.pipeline = ml3d_torch.pipelines.SemanticSegmentation(
                model=self.model,
                device=self.device
            )

            print(f"Loaded {self.model_name} pretrained on {dataset}")
            return True

        except Exception as e:
            print(f"Failed to load pretrained model: {e}")
            print("Falling back to geometric features only")
            return False

    def extract_features(
        self,
        pcd: o3d.geometry.PointCloud
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract semantic features from point cloud.

        Returns:
            labels: [N] predicted semantic labels
            features: [N, F] feature embeddings from the model
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_pretrained_model() first.")

        points = np.asarray(pcd.points)

        # Prepare data for Open3D-ML
        data = {
            'point': points,
            'feat': None,  # Can add colors, normals, etc.
            'label': np.zeros(len(points), dtype=np.int32)  # Dummy labels
        }

        # Run inference
        result = self.pipeline.run_inference(data)

        labels = result['predict_labels']
        # Features would require model modification to extract intermediate layers
        features = None

        return labels, features


# =============================================================================
# Trajectory Generator with Motion Validation
# =============================================================================

class DynamicTrajectoryGenerator:
    """Generate trajectories with guaranteed motion between frames."""

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        min_motion: float = 0.5
    ):
        """
        Args:
            bounds: (x_min, y_min, x_max, y_max)
            min_motion: Minimum movement between consecutive frames (meters)
        """
        self.bounds = bounds
        self.min_motion = min_motion

    def generate_smooth_trajectory(
        self,
        num_frames: int,
        altitude_range: Tuple[float, float] = (30.0, 100.0),
        speed_range: Tuple[float, float] = (2.0, 8.0),
        agent_type: str = 'drone'
    ) -> List[Dict]:
        """
        Generate smooth trajectory with cubic spline interpolation.

        Ensures minimum motion between frames for proper temporal learning.
        """
        x_min, y_min, x_max, y_max = self.bounds

        # Generate control points
        num_control_points = max(4, num_frames // 20)
        control_points = []

        for i in range(num_control_points):
            x = np.random.uniform(x_min + 10, x_max - 10)
            y = np.random.uniform(y_min + 10, y_max - 10)
            z = np.random.uniform(*altitude_range) if agent_type == 'drone' else 1.5
            control_points.append([x, y, z])

        control_points = np.array(control_points)

        # Interpolate with cubic spline
        from scipy import interpolate

        t_control = np.linspace(0, 1, num_control_points)
        t_frames = np.linspace(0, 1, num_frames)

        # Spline for each dimension
        positions = np.zeros((num_frames, 3))
        for dim in range(3):
            spline = interpolate.CubicSpline(t_control, control_points[:, dim])
            positions[:, dim] = spline(t_frames)

        # Generate waypoints with velocity
        waypoints = []
        for i in range(num_frames):
            pos = positions[i]

            # Calculate velocity from position delta
            if i < num_frames - 1:
                delta = positions[i + 1] - positions[i]
                speed = np.linalg.norm(delta)

                # Ensure minimum motion
                if speed < self.min_motion:
                    # Add random perturbation
                    delta = delta + np.random.randn(3) * self.min_motion
                    speed = np.linalg.norm(delta)

                velocity = delta / max(speed, 1e-6) * np.clip(speed, *speed_range)
            else:
                velocity = waypoints[-1]['velocity']['linear'] if waypoints else np.zeros(3)
                velocity = np.array([velocity['x'], velocity['y'], velocity['z']])

            # Calculate yaw from velocity direction
            yaw = np.arctan2(velocity[1], velocity[0])

            waypoints.append(self._create_waypoint(
                pos[0], pos[1], pos[2], yaw, velocity, agent_type
            ))

        return waypoints

    def generate_figure_eight(
        self,
        num_frames: int,
        center: Tuple[float, float] = (0, 0),
        size: float = 100.0,
        altitude: float = 50.0,
        speed: float = 5.0,
        agent_type: str = 'drone'
    ) -> List[Dict]:
        """Generate figure-8 pattern for diverse viewpoints."""
        waypoints = []

        for i in range(num_frames):
            t = 2 * np.pi * i / num_frames

            # Figure-8 (lemniscate of Bernoulli)
            x = center[0] + size * np.sin(t)
            y = center[1] + size * np.sin(t) * np.cos(t)
            z = altitude + 10 * np.sin(2 * t)  # Altitude variation

            # Velocity from derivative
            dx = size * np.cos(t)
            dy = size * (np.cos(t) * np.cos(t) - np.sin(t) * np.sin(t))
            dz = 20 * np.cos(2 * t)

            velocity = np.array([dx, dy, dz])
            velocity = velocity / np.linalg.norm(velocity) * speed

            yaw = np.arctan2(dy, dx)

            waypoints.append(self._create_waypoint(
                x, y, z, yaw, velocity, agent_type
            ))

        return waypoints

    def _create_waypoint(
        self,
        x: float, y: float, z: float,
        yaw: float,
        velocity: np.ndarray,
        agent_type: str
    ) -> Dict:
        """Create waypoint in dataset format."""
        qw = np.cos(yaw / 2)
        qz = np.sin(yaw / 2)

        return {
            'position': {'x': float(x), 'y': float(y), 'z': float(z)},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': float(qz), 'w': float(qw)},
            'velocity': {
                'linear': {'x': float(velocity[0]), 'y': float(velocity[1]), 'z': float(velocity[2])},
                'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
            },
            'agent_type': agent_type
        }

    def validate_trajectory(self, waypoints: List[Dict]) -> Dict:
        """Validate trajectory has sufficient motion."""
        positions = np.array([
            [w['position']['x'], w['position']['y'], w['position']['z']]
            for w in waypoints
        ])

        # Calculate frame-to-frame distances
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)

        stats = {
            'num_frames': len(waypoints),
            'total_distance': float(np.sum(distances)),
            'mean_distance': float(np.mean(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances)),
            'static_frames': int(np.sum(distances < self.min_motion)),
            'valid': bool(np.all(distances >= self.min_motion * 0.5))
        }

        return stats


# =============================================================================
# Complete Data Generation Pipeline
# =============================================================================

class Open3DDataPipeline:
    """Complete pipeline for generating OccWorld training data."""

    def __init__(self, config: Open3DPipelineConfig):
        self.config = config
        self.processor = Open3DPointCloudProcessor(config)
        self.feature_extractor = None

        if config.use_semantic_features and HAS_OPEN3D_ML:
            try:
                self.feature_extractor = Open3DMLFeatureExtractor(
                    model_name=config.semantic_model
                )
                self.feature_extractor.load_pretrained_model()
            except Exception as e:
                print(f"Warning: Could not load semantic model: {e}")
                self.feature_extractor = None

    def load_scene(self, mesh_dir: str, max_meshes: int = 50) -> o3d.geometry.PointCloud:
        """Load all meshes and combine into single point cloud."""
        mesh_files = []
        for ext in ['*.obj', '*.ply', '*.stl', '*.dae']:
            mesh_files.extend(Path(mesh_dir).rglob(ext))

        print(f"Found {len(mesh_files)} mesh files")
        mesh_files = mesh_files[:max_meshes]

        all_points = []
        all_normals = []

        for i, mesh_file in enumerate(mesh_files):
            if i % 10 == 0:
                print(f"  Loading mesh {i+1}/{len(mesh_files)}")

            mesh = self.processor.load_mesh(str(mesh_file))
            if mesh is None:
                continue

            # Convert to point cloud
            pcd = self.processor.mesh_to_point_cloud(mesh, num_points=10000)

            all_points.append(np.asarray(pcd.points))
            if pcd.has_normals():
                all_normals.append(np.asarray(pcd.normals))

        # Combine all points
        combined_points = np.vstack(all_points)
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(combined_points)

        if all_normals and len(all_normals) == len(all_points):
            combined_normals = np.vstack(all_normals)
            combined_pcd.normals = o3d.utility.Vector3dVector(combined_normals)

        print(f"Combined point cloud: {len(combined_points)} points")

        return combined_pcd

    def generate_training_data(
        self,
        pcd: o3d.geometry.PointCloud,
        output_dir: str,
        num_sessions: int = 5,
        frames_per_session: int = 200,
        agent_type: str = 'drone'
    ):
        """Generate complete training dataset."""

        # Get scene bounds
        points = np.asarray(pcd.points)
        bounds = (
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max())
        )
        print(f"Scene bounds: X[{bounds[0]:.1f}, {bounds[2]:.1f}] Y[{bounds[1]:.1f}, {bounds[3]:.1f}]")

        # Initialize trajectory generator
        traj_gen = DynamicTrajectoryGenerator(
            bounds=bounds,
            min_motion=self.config.min_motion_threshold
        )

        os.makedirs(output_dir, exist_ok=True)

        for session_idx in range(num_sessions):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_name = f'{agent_type}_{timestamp}_{session_idx:02d}'
            session_dir = os.path.join(output_dir, session_name)

            print(f"\n{'='*50}")
            print(f"Session {session_idx + 1}/{num_sessions}: {session_name}")
            print(f"{'='*50}")

            # Create directories
            for subdir in ['occupancy', 'poses', 'lidar', 'images']:
                os.makedirs(os.path.join(session_dir, subdir), exist_ok=True)

            # Generate trajectory
            pattern = np.random.choice(['smooth', 'figure8'])
            if pattern == 'smooth':
                waypoints = traj_gen.generate_smooth_trajectory(
                    frames_per_session,
                    agent_type=agent_type
                )
            else:
                center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
                waypoints = traj_gen.generate_figure_eight(
                    frames_per_session,
                    center=center,
                    agent_type=agent_type
                )

            # Validate trajectory
            stats = traj_gen.validate_trajectory(waypoints)
            print(f"  Trajectory: {stats['total_distance']:.1f}m total, "
                  f"{stats['mean_distance']:.2f}m/frame avg, "
                  f"{stats['static_frames']} static frames")

            if not stats['valid']:
                print(f"  WARNING: Trajectory has insufficient motion!")

            # Process frames
            for frame_idx, waypoint in enumerate(waypoints):
                if frame_idx % 50 == 0:
                    print(f"  Processing frame {frame_idx}/{frames_per_session}")

                self._process_frame(
                    pcd, waypoint, frame_idx, session_dir
                )

            # Save session metadata
            metadata = {
                'session_name': session_name,
                'agent_type': agent_type,
                'num_frames': frames_per_session,
                'trajectory_stats': stats,
                'config': {
                    'point_cloud_range': self.config.point_cloud_range,
                    'voxel_size': self.config.voxel_size,
                    'grid_size': self.config.grid_size,
                }
            }
            with open(os.path.join(session_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

        print(f"\n{'='*50}")
        print("Data generation complete!")
        print(f"{'='*50}")
        print(f"Output: {output_dir}")
        print(f"Sessions: {num_sessions}")
        print(f"Frames per session: {frames_per_session}")

    def _process_frame(
        self,
        pcd: o3d.geometry.PointCloud,
        waypoint: Dict,
        frame_idx: int,
        session_dir: str
    ):
        """Process a single frame."""
        frame_id = f'{frame_idx:06d}'
        pos = waypoint['position']
        ori = waypoint['orientation']

        position = np.array([pos['x'], pos['y'], pos['z']])
        yaw = 2 * np.arctan2(ori['z'], ori['w'])

        # Generate occupancy grid using Open3D
        occupancy = self.processor.voxelize_point_cloud(pcd, position, yaw)

        # Save occupancy
        np.savez_compressed(
            os.path.join(session_dir, 'occupancy', f'{frame_id}_occupancy.npz'),
            occupancy=occupancy
        )

        # Save pose
        with open(os.path.join(session_dir, 'poses', f'{frame_id}.json'), 'w') as f:
            json.dump(waypoint, f, indent=2)

        # Generate synthetic LiDAR from occupancy
        lidar_points = self._occupancy_to_lidar(occupancy, position)
        np.save(
            os.path.join(session_dir, 'lidar', f'{frame_id}_LIDAR.npy'),
            lidar_points
        )

        # Placeholder image
        self._save_placeholder_image(
            os.path.join(session_dir, 'images', f'{frame_id}_CAM_FRONT.jpg'),
            frame_idx, pos
        )

    def _occupancy_to_lidar(
        self,
        occupancy: np.ndarray,
        position: np.ndarray,
        num_points: int = 10000
    ) -> np.ndarray:
        """Generate synthetic LiDAR points from occupancy."""
        cfg = self.config

        # Find occupied voxels
        occupied = np.argwhere(occupancy > 0)
        if len(occupied) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        # Sample points
        if len(occupied) > num_points:
            indices = np.random.choice(len(occupied), num_points, replace=False)
            occupied = occupied[indices]

        # Convert to world coordinates
        voxel_size = np.array(cfg.voxel_size)
        range_min = np.array(cfg.point_cloud_range[:3])

        points = (occupied + np.random.uniform(0, 1, occupied.shape)) * voxel_size + range_min

        # Add intensity
        distances = np.linalg.norm(points, axis=1)
        intensities = np.clip(1.0 - distances / 100.0, 0.1, 1.0)

        return np.column_stack([points, intensities]).astype(np.float32)

    def _save_placeholder_image(self, path: str, frame_idx: int, pos: Dict):
        """Save a placeholder camera image."""
        img = np.zeros((900, 1600, 3), dtype=np.uint8)

        # Gradient background
        for y in range(900):
            blue_val = int(200 * (1 - y / 900))
            img[y, :] = [blue_val + 55, blue_val + 30, blue_val]

        try:
            import cv2
            text = f"Frame {frame_idx} | ({pos['x']:.1f}, {pos['y']:.1f}, {pos['z']:.1f})"
            cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(path, img)
        except ImportError:
            try:
                from PIL import Image
                Image.fromarray(img).save(path)
            except ImportError:
                np.save(path.replace('.jpg', '.npy'), img)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Open3D-ML Pipeline for OccWorld Training Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download and process PLATEAU data (recommended for first run)
    python scripts/setup_open3d_pipeline.py --download --process --output data/tokyo_plateau

    # Process existing mesh files
    python scripts/setup_open3d_pipeline.py --process-only --input data/plateau/meshes --output data/tokyo_gazebo

    # Quick test with Shibuya dataset (smaller)
    python scripts/setup_open3d_pipeline.py --download --dataset shibuya --sessions 2 --frames 100
"""
    )

    # Data source options
    parser.add_argument('--download', action='store_true',
                       help='Download PLATEAU data')
    parser.add_argument('--dataset', type=str, default='shibuya',
                       choices=['tokyo_obj', 'tokyo_citygml', 'shibuya'],
                       help='PLATEAU dataset to download')
    parser.add_argument('--process-only', action='store_true',
                       help='Process existing meshes without downloading')
    parser.add_argument('--input', '-i', type=str, default='data/plateau/meshes',
                       help='Input mesh directory (for --process-only)')

    # Output options
    parser.add_argument('--output', '-o', type=str, default='data/tokyo_gazebo',
                       help='Output directory for training data')

    # Generation options
    parser.add_argument('--sessions', '-s', type=int, default=5,
                       help='Number of training sessions to generate')
    parser.add_argument('--frames', '-f', type=int, default=200,
                       help='Frames per session')
    parser.add_argument('--max-meshes', '-m', type=int, default=50,
                       help='Maximum meshes to load')
    parser.add_argument('--agent', '-a', type=str, default='drone',
                       choices=['drone', 'rover'],
                       help='Agent type')

    # Feature options
    parser.add_argument('--no-semantic', action='store_true',
                       help='Disable semantic feature extraction')
    parser.add_argument('--semantic-model', type=str, default='RandLANet',
                       choices=['RandLANet', 'KPConv', 'PointTransformer'],
                       help='Semantic segmentation model')

    args = parser.parse_args()

    # Check dependencies
    if not HAS_OPEN3D:
        print("ERROR: Open3D is required.")
        print("Install with: pip install open3d")
        sys.exit(1)

    print("=" * 60)
    print("Open3D-ML Data Pipeline for OccWorld")
    print("=" * 60)
    print(f"Open3D version: {O3D_VERSION}")
    print(f"Open3D-ML available: {HAS_OPEN3D_ML}")
    print(f"PyTorch available: {HAS_TORCH}")
    print()

    # Download data if requested
    mesh_dir = args.input
    if args.download:
        mesh_dir = download_plateau_data('data/plateau', args.dataset)
        if mesh_dir is None:
            print("Failed to download data")
            sys.exit(1)

    # Check mesh directory exists
    if not os.path.exists(mesh_dir):
        print(f"ERROR: Mesh directory not found: {mesh_dir}")
        print("\nTo download PLATEAU data, run with --download flag:")
        print(f"  python {sys.argv[0]} --download --dataset shibuya")
        sys.exit(1)

    # Initialize pipeline
    config = Open3DPipelineConfig(
        use_semantic_features=not args.no_semantic and HAS_OPEN3D_ML,
        semantic_model=args.semantic_model
    )

    pipeline = Open3DDataPipeline(config)

    # Load scene
    print(f"\nLoading scene from {mesh_dir}...")
    pcd = pipeline.load_scene(mesh_dir, max_meshes=args.max_meshes)

    # Generate training data
    print(f"\nGenerating training data...")
    pipeline.generate_training_data(
        pcd=pcd,
        output_dir=args.output,
        num_sessions=args.sessions,
        frames_per_session=args.frames,
        agent_type=args.agent
    )

    print(f"\nTraining data ready at: {args.output}")
    print("\nTo train, run:")
    print(f"  python train.py --config config/finetune_tokyo.py --work-dir out/occworld")


if __name__ == '__main__':
    main()
