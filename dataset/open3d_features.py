#!/usr/bin/env python3
"""
Open3D-ML Feature Extraction for OccWorld

This module provides semantic and geometric feature extraction using Open3D-ML.
Features can be used to enhance occupancy grids with semantic information.

Supported models (from Open3D-ML):
- RandLA-Net: Efficient random sampling for large-scale point clouds
- KPConv: Kernel point convolutions for accurate segmentation

Usage:
    from dataset.open3d_features import Open3DFeatureExtractor, FeatureConfig

    # Initialize with geometric features only (no Open3D-ML needed)
    config = FeatureConfig(use_semantic=False)
    extractor = Open3DFeatureExtractor(config)
    features = extractor.extract(points)

    # With semantic features (requires Open3D-ML)
    config = FeatureConfig(use_semantic=True, semantic_model='RandLANet')
    extractor = Open3DFeatureExtractor(config)
    extractor.load_semantic_model()  # Downloads weights if needed
    features = extractor.extract(points)

Requirements:
    pip install open3d

For semantic features (optional):
    pip install open3d-ml-torch

See: https://github.com/isl-org/Open3D-ML
"""

import os
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

# Check for Open3D
try:
    import open3d as o3d
    HAS_OPEN3D = True
    OPEN3D_VERSION = o3d.__version__
except ImportError:
    HAS_OPEN3D = False
    OPEN3D_VERSION = None

# Check for Open3D-ML with PyTorch
HAS_OPEN3D_ML = False
HAS_TORCH = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

if HAS_TORCH:
    try:
        import open3d.ml as _ml3d
        import open3d.ml.torch as ml3d
        HAS_OPEN3D_ML = True
    except ImportError:
        pass


# Pretrained model URLs from Open3D model zoo
PRETRAINED_WEIGHTS = {
    'RandLANet': {
        'SemanticKITTI': 'https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth',
        'S3DIS': 'https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_s3dis_202201071330utc.pth',
    },
    'KPConv': {
        'SemanticKITTI': 'https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_semantickitti_202009090354utc.pth',
        'S3DIS': 'https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_s3dis_202010091238.pth',
    },
}

# SemanticKITTI class labels
SEMANTICKITTI_LABELS = {
    0: 'unlabeled', 1: 'car', 2: 'bicycle', 3: 'motorcycle', 4: 'truck',
    5: 'other-vehicle', 6: 'person', 7: 'bicyclist', 8: 'motorcyclist',
    9: 'road', 10: 'parking', 11: 'sidewalk', 12: 'other-ground',
    13: 'building', 14: 'fence', 15: 'vegetation', 16: 'trunk',
    17: 'terrain', 18: 'pole', 19: 'traffic-sign'
}


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    # Geometric features (Open3D only)
    compute_normals: bool = True
    compute_fpfh: bool = True
    compute_curvature: bool = False  # Slower, disable by default
    normal_radius: float = 1.0
    fpfh_radius: float = 2.5

    # Semantic features (requires Open3D-ML)
    use_semantic: bool = False
    semantic_model: str = 'RandLANet'  # 'RandLANet' or 'KPConv'
    semantic_dataset: str = 'SemanticKITTI'
    cache_dir: str = './pretrained/open3d_ml'

    # Voxel aggregation
    aggregation: str = 'mean'  # 'mean', 'max'

    # Device
    device: str = 'cuda'


class GeometricFeatureExtractor:
    """Extract geometric features using Open3D (no ML dependencies)."""

    def __init__(self, config: FeatureConfig):
        if not HAS_OPEN3D:
            raise ImportError("Open3D required. Install with: pip install open3d")
        self.config = config

    def compute_normals(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """Compute surface normals using PCA."""
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.config.normal_radius,
                    max_nn=30
                )
            )
            # Orient normals consistently
            pcd.orient_normals_consistent_tangent_plane(k=15)
        return np.asarray(pcd.normals)

    def compute_fpfh(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Compute Fast Point Feature Histograms (FPFH).

        FPFH is a 33-dimensional descriptor capturing local geometry.
        Useful for place recognition and registration.

        Returns:
            features: [N, 33] FPFH descriptors
        """
        if not pcd.has_normals():
            self.compute_normals(pcd)

        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.config.fpfh_radius,
                max_nn=100
            )
        )
        # FPFH returns [33, N], transpose to [N, 33]
        return np.asarray(fpfh.data).T

    def compute_curvature(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Estimate surface curvature from covariance analysis.

        Returns:
            curvature: [N] curvature values (0=flat, higher=curved)
        """
        if not pcd.has_normals():
            self.compute_normals(pcd)

        points = np.asarray(pcd.points)
        n_points = len(points)

        # Use covariance eigenvalues as curvature proxy
        curvatures = np.zeros(n_points, dtype=np.float32)

        # Build KD-tree for neighbor search
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        for i in range(n_points):
            [k, idx, _] = pcd_tree.search_radius_vector_3d(
                pcd.points[i], self.config.normal_radius
            )

            if k > 3:
                neighbors = points[idx[1:k], :]  # Exclude query point
                # Covariance of neighbor positions
                centered = neighbors - neighbors.mean(axis=0)
                cov = np.dot(centered.T, centered) / (k - 1)
                eigenvalues = np.linalg.eigvalsh(cov)
                # Curvature = smallest eigenvalue / sum (surface variation)
                curvatures[i] = eigenvalues[0] / (np.sum(eigenvalues) + 1e-8)

        return curvatures

    def extract(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all configured geometric features.

        Args:
            points: [N, 3+] point cloud (xyz required, additional columns ignored)

        Returns:
            Dict with 'points', 'normals', 'fpfh', 'curvature' arrays
        """
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))

        features = {'points': points[:, :3]}

        if self.config.compute_normals:
            features['normals'] = self.compute_normals(pcd)

        if self.config.compute_fpfh:
            features['fpfh'] = self.compute_fpfh(pcd)

        if self.config.compute_curvature:
            features['curvature'] = self.compute_curvature(pcd)

        return features


class SemanticFeatureExtractor:
    """
    Extract semantic features using Open3D-ML pretrained models.

    Requires: pip install open3d-ml-torch
    """

    def __init__(self, config: FeatureConfig):
        if not HAS_OPEN3D_ML:
            raise ImportError(
                "Open3D-ML required for semantic features.\n"
                "Install with: pip install open3d-ml-torch"
            )
        if not HAS_TORCH:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.config = config
        self.device = config.device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.pipeline = None
        self._model_loaded = False

    def _download_weights(self, url: str, save_path: str) -> bool:
        """Download pretrained weights."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            print(f"Using cached weights: {save_path}")
            return True

        print(f"Downloading weights from {url}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, save_path)
            print(f"Saved to {save_path}")
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def load_model(self, weights_path: Optional[str] = None) -> bool:
        """
        Load pretrained semantic segmentation model.

        Args:
            weights_path: Path to weights file (downloads default if None)

        Returns:
            True if model loaded successfully
        """
        model_name = self.config.semantic_model
        dataset_name = self.config.semantic_dataset

        # Get weights URL
        if model_name not in PRETRAINED_WEIGHTS:
            print(f"Unknown model: {model_name}. Available: {list(PRETRAINED_WEIGHTS.keys())}")
            return False

        if dataset_name not in PRETRAINED_WEIGHTS[model_name]:
            print(f"Unknown dataset: {dataset_name}. Available: {list(PRETRAINED_WEIGHTS[model_name].keys())}")
            return False

        # Download weights if needed
        if weights_path is None:
            url = PRETRAINED_WEIGHTS[model_name][dataset_name]
            filename = os.path.basename(url)
            weights_path = os.path.join(self.config.cache_dir, filename)

            if not self._download_weights(url, weights_path):
                return False

        try:
            # Create model based on type
            if model_name == 'RandLANet':
                # RandLANet configuration for SemanticKITTI
                model_cfg = {
                    'name': 'RandLANet',
                    'num_neighbors': 16,
                    'num_layers': 4,
                    'num_points': 45056,
                    'num_classes': 19,
                    'ignored_label_inds': [0],
                    'sub_sampling_ratio': [4, 4, 4, 4],
                    'in_channels': 3,
                    'dim_features': 8,
                    'dim_output': [16, 64, 128, 256],
                    'grid_size': 0.06,
                }
                self.model = ml3d.models.RandLANet(**model_cfg)

            elif model_name == 'KPConv':
                # KPConv configuration
                model_cfg = {
                    'name': 'KPFCNN',
                    'num_classes': 19,
                    'num_layers': 4,
                    'num_points': 65536,
                    'in_channels': 3,
                    'ignored_label_inds': [0],
                }
                self.model = ml3d.models.KPFCNN(**model_cfg)

            else:
                print(f"Model {model_name} not implemented")
                return False

            # Load weights
            self.model.to(self.device)
            checkpoint = torch.load(weights_path, map_location=self.device)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.eval()
            self._model_loaded = True

            print(f"Loaded {model_name} ({dataset_name}) on {self.device}")
            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def segment(self, points: np.ndarray, features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform semantic segmentation on point cloud.

        Args:
            points: [N, 3] xyz coordinates
            features: [N, C] additional features (optional, e.g., RGB)

        Returns:
            labels: [N] predicted semantic class labels (0-18 for SemanticKITTI)
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        with torch.no_grad():
            # Prepare input tensor
            pts = torch.from_numpy(points[:, :3].astype(np.float32)).to(self.device)

            if features is not None:
                feat = torch.from_numpy(features.astype(np.float32)).to(self.device)
            else:
                feat = pts.clone()  # Use xyz as features

            # Add batch dimension
            pts = pts.unsqueeze(0)
            feat = feat.unsqueeze(0)

            # Run inference
            try:
                # Try direct forward pass
                outputs = self.model(pts, feat)
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('semantic_scores'))
                else:
                    logits = outputs

                # Get predictions
                if logits is not None:
                    labels = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
                else:
                    # Fallback: return zeros
                    labels = np.zeros(len(points), dtype=np.int32)

            except Exception as e:
                print(f"Inference failed: {e}")
                # Return zeros as fallback
                labels = np.zeros(len(points), dtype=np.int32)

        return labels

    def get_label_names(self) -> Dict[int, str]:
        """Get mapping from label index to name."""
        return SEMANTICKITTI_LABELS.copy()


class Open3DFeatureExtractor:
    """
    Combined feature extractor using Open3D and Open3D-ML.

    Extracts geometric features (always available) and optionally
    semantic features (requires Open3D-ML).
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

        # Geometric extractor (always available with Open3D)
        if HAS_OPEN3D:
            self.geometric = GeometricFeatureExtractor(self.config)
        else:
            self.geometric = None
            print("Warning: Open3D not available, geometric features disabled")

        # Semantic extractor (requires Open3D-ML)
        self.semantic = None
        if self.config.use_semantic:
            if HAS_OPEN3D_ML:
                try:
                    self.semantic = SemanticFeatureExtractor(self.config)
                except Exception as e:
                    print(f"Warning: Could not initialize semantic extractor: {e}")
            else:
                print("Warning: Open3D-ML not available, semantic features disabled")

    def load_semantic_model(self, weights_path: Optional[str] = None) -> bool:
        """Load semantic segmentation model weights."""
        if self.semantic is None:
            print("Semantic extractor not initialized")
            return False
        return self.semantic.load_model(weights_path)

    def extract(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all available features from point cloud.

        Args:
            points: [N, 3+] point cloud (xyz required)

        Returns:
            Dict with feature arrays:
            - 'points': [N, 3] xyz coordinates
            - 'normals': [N, 3] surface normals (if enabled)
            - 'fpfh': [N, 33] FPFH descriptors (if enabled)
            - 'curvature': [N] curvature values (if enabled)
            - 'semantic_labels': [N] class labels (if semantic enabled)
        """
        features = {}

        # Geometric features
        if self.geometric is not None:
            geo = self.geometric.extract(points)
            features.update(geo)
        else:
            features['points'] = points[:, :3]

        # Semantic features
        if self.semantic is not None and self.semantic._model_loaded:
            try:
                features['semantic_labels'] = self.semantic.segment(points)
            except Exception as e:
                print(f"Warning: Semantic extraction failed: {e}")

        return features

    def aggregate_to_voxels(
        self,
        points: np.ndarray,
        features: np.ndarray,
        grid_size: Tuple[int, int, int],
        point_cloud_range: Tuple[float, ...],
        voxel_size: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Aggregate point features to voxel grid.

        Args:
            points: [N, 3] point coordinates
            features: [N, F] point features to aggregate
            grid_size: (X, Y, Z) voxel grid dimensions
            point_cloud_range: (xmin, ymin, zmin, xmax, ymax, zmax)
            voxel_size: (dx, dy, dz) voxel dimensions

        Returns:
            voxel_features: [X, Y, Z, F] aggregated features
        """
        pc_range = np.array(point_cloud_range)
        voxel_sz = np.array(voxel_size)
        grid_sz = np.array(grid_size)

        # Filter points in range
        mask = (
            (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) &
            (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]) &
            (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
        )
        points = points[mask]
        features = features[mask]

        if len(points) == 0:
            return np.zeros((*grid_size, features.shape[1] if len(features.shape) > 1 else 1), dtype=np.float32)

        # Compute voxel indices
        voxel_indices = np.floor((points - pc_range[:3]) / voxel_sz).astype(np.int32)
        voxel_indices = np.clip(voxel_indices, 0, grid_sz - 1)

        # Aggregate
        feature_dim = features.shape[1] if len(features.shape) > 1 else 1
        if len(features.shape) == 1:
            features = features[:, np.newaxis]

        voxel_features = np.zeros((*grid_size, feature_dim), dtype=np.float32)
        voxel_counts = np.zeros(grid_size, dtype=np.float32)

        for i in range(len(points)):
            idx = tuple(voxel_indices[i])
            if self.config.aggregation == 'mean':
                voxel_features[idx] += features[i]
                voxel_counts[idx] += 1
            elif self.config.aggregation == 'max':
                voxel_features[idx] = np.maximum(voxel_features[idx], features[i])

        # Normalize mean
        if self.config.aggregation == 'mean':
            mask = voxel_counts > 0
            for f in range(feature_dim):
                voxel_features[..., f][mask] /= voxel_counts[mask]

        return voxel_features


def create_occupancy_with_features(
    points: np.ndarray,
    grid_size: Tuple[int, int, int],
    point_cloud_range: Tuple[float, ...],
    voxel_size: Tuple[float, float, float],
    compute_normals: bool = True,
    compute_semantic: bool = False
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Create occupancy grid with optional features.

    Args:
        points: [N, 3+] point cloud
        grid_size: (X, Y, Z) voxel dimensions
        point_cloud_range: (xmin, ymin, zmin, xmax, ymax, zmax)
        voxel_size: (dx, dy, dz)
        compute_normals: Whether to compute surface normals
        compute_semantic: Whether to compute semantic labels (requires Open3D-ML)

    Returns:
        occupancy: [X, Y, Z] binary grid
        features: Dict with additional feature grids
    """
    config = FeatureConfig(
        compute_normals=compute_normals,
        compute_fpfh=False,  # Skip FPFH for speed
        use_semantic=compute_semantic
    )
    extractor = Open3DFeatureExtractor(config)

    if compute_semantic:
        extractor.load_semantic_model()

    # Create binary occupancy
    pc_range = np.array(point_cloud_range)
    voxel_sz = np.array(voxel_size)

    mask = (
        (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) &
        (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]) &
        (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
    )
    valid_points = points[mask]

    occupancy = np.zeros(grid_size, dtype=np.uint8)
    if len(valid_points) > 0:
        voxel_indices = np.floor((valid_points[:, :3] - pc_range[:3]) / voxel_sz).astype(np.int32)
        voxel_indices = np.clip(voxel_indices, 0, np.array(grid_size) - 1)
        occupancy[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1

    # Extract and aggregate features
    features = {}
    point_features = extractor.extract(points)

    if 'normals' in point_features:
        features['normals'] = extractor.aggregate_to_voxels(
            points[:, :3], point_features['normals'],
            grid_size, point_cloud_range, voxel_size
        )

    if 'semantic_labels' in point_features:
        # One-hot encode semantic labels
        num_classes = 19
        semantic_onehot = np.zeros((len(points), num_classes), dtype=np.float32)
        labels = point_features['semantic_labels']
        valid_labels = (labels >= 0) & (labels < num_classes)
        semantic_onehot[valid_labels, labels[valid_labels]] = 1.0

        features['semantic'] = extractor.aggregate_to_voxels(
            points[:, :3], semantic_onehot,
            grid_size, point_cloud_range, voxel_size
        )

    return occupancy, features


# =============================================================================
# Testing and Verification
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Open3D Feature Extraction Module")
    print("=" * 60)
    print(f"Open3D available: {HAS_OPEN3D} (version: {OPEN3D_VERSION})")
    print(f"PyTorch available: {HAS_TORCH}")
    print(f"Open3D-ML available: {HAS_OPEN3D_ML}")
    print()

    if not HAS_OPEN3D:
        print("Install Open3D with: pip install open3d")
        exit(1)

    # Test geometric features
    print("Testing geometric feature extraction...")
    config = FeatureConfig(
        compute_normals=True,
        compute_fpfh=True,
        compute_curvature=False,
        use_semantic=False
    )
    extractor = Open3DFeatureExtractor(config)

    # Create test point cloud (random sphere)
    n_points = 1000
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = 10 + np.random.randn(n_points) * 0.1
    test_points = np.stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi)
    ], axis=1).astype(np.float32)

    features = extractor.extract(test_points)

    print(f"  Points shape: {features['points'].shape}")
    if 'normals' in features:
        print(f"  Normals shape: {features['normals'].shape}")
    if 'fpfh' in features:
        print(f"  FPFH shape: {features['fpfh'].shape}")

    # Test voxel aggregation
    print("\nTesting voxel aggregation...")
    grid_size = (50, 50, 50)
    pc_range = (-20, -20, -20, 20, 20, 20)
    voxel_size = (0.8, 0.8, 0.8)

    if 'normals' in features:
        voxel_normals = extractor.aggregate_to_voxels(
            test_points, features['normals'],
            grid_size, pc_range, voxel_size
        )
        print(f"  Voxel normals shape: {voxel_normals.shape}")
        print(f"  Non-zero voxels: {np.sum(np.any(voxel_normals != 0, axis=-1))}")

    # Test semantic features if available
    if HAS_OPEN3D_ML:
        print("\nTesting semantic feature extraction...")
        config_semantic = FeatureConfig(use_semantic=True, semantic_model='RandLANet')
        sem_extractor = Open3DFeatureExtractor(config_semantic)

        if sem_extractor.load_semantic_model():
            sem_features = sem_extractor.extract(test_points)
            if 'semantic_labels' in sem_features:
                labels = sem_features['semantic_labels']
                print(f"  Semantic labels shape: {labels.shape}")
                print(f"  Unique labels: {np.unique(labels)}")
        else:
            print("  Could not load semantic model (weights download may have failed)")
    else:
        print("\nSemantic features: Skipped (Open3D-ML not installed)")
        print("Install with: pip install open3d-ml-torch")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
