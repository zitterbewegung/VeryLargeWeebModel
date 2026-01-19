#!/usr/bin/env python3
"""
Open3D-ML Feature Extraction for OccWorld

This module provides semantic and geometric feature extraction using Open3D-ML.
Features can be used to enhance occupancy grids with semantic information.

Supported models (from Open3D-ML):
- RandLA-Net: Efficient random sampling for large-scale point clouds
- KPConv: Kernel point convolutions for accurate segmentation
- PointTransformer: Attention-based architecture

Usage:
    from dataset.open3d_features import Open3DFeatureExtractor

    # Initialize extractor
    extractor = Open3DFeatureExtractor(model='RandLANet')

    # Extract features from point cloud
    features = extractor.extract(points)  # [N, F]

    # Get semantic labels
    labels = extractor.segment(points)  # [N]

    # Aggregate features to voxel grid
    voxel_features = extractor.aggregate_to_voxels(points, features, grid_size)

Requirements:
    pip install open3d
    pip install open3d-ml-torch  # or open3d-ml-tf

See: https://github.com/isl-org/Open3D-ML
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Check for Open3D
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

# Check for Open3D-ML
try:
    import open3d.ml as ml3d
    HAS_OPEN3D_ML = True
except ImportError:
    HAS_OPEN3D_ML = False

# Check for PyTorch backend
try:
    import torch
    import open3d.ml.torch as ml3d_torch
    HAS_TORCH_BACKEND = True
except ImportError:
    HAS_TORCH_BACKEND = False


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    # Geometric features
    compute_normals: bool = True
    compute_fpfh: bool = True
    compute_curvature: bool = True
    normal_radius: float = 1.0
    fpfh_radius: float = 2.5

    # Semantic features (requires Open3D-ML)
    use_semantic: bool = True
    semantic_model: str = "RandLANet"  # RandLANet, KPConv, PointTransformer
    semantic_dataset: str = "SemanticKITTI"  # SemanticKITTI, S3DIS, Toronto3D

    # Voxel aggregation
    aggregation: str = "mean"  # mean, max, count


class GeometricFeatureExtractor:
    """Extract geometric features using Open3D."""

    def __init__(self, config: FeatureConfig):
        if not HAS_OPEN3D:
            raise ImportError("Open3D required. Install with: pip install open3d")

        self.config = config

    def extract_normals(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """Compute surface normals."""
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.config.normal_radius,
                    max_nn=30
                )
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)

        return np.asarray(pcd.normals)

    def extract_fpfh(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Compute Fast Point Feature Histograms (FPFH).

        FPFH is a 33-dimensional descriptor that captures local geometric structure.
        Good for place recognition and registration.
        """
        # Ensure normals exist
        if not pcd.has_normals():
            self.extract_normals(pcd)

        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.config.fpfh_radius,
                max_nn=100
            )
        )

        # FPFH returns [33, N], transpose to [N, 33]
        return np.asarray(fpfh.data).T

    def extract_curvature(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Estimate surface curvature from normals.

        Returns principal curvatures approximated from normal variation.
        """
        if not pcd.has_normals():
            self.extract_normals(pcd)

        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        # Build KD-tree
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        curvatures = np.zeros(len(points))

        for i in range(len(points)):
            # Find neighbors
            [k, idx, _] = pcd_tree.search_radius_vector_3d(
                pcd.points[i], self.config.normal_radius
            )

            if k > 3:
                # Compute covariance of neighbor normals
                neighbor_normals = normals[idx[1:], :]
                cov = np.cov(neighbor_normals.T)

                # Smallest eigenvalue indicates curvature
                eigenvalues = np.linalg.eigvalsh(cov)
                curvatures[i] = eigenvalues[0] / (np.sum(eigenvalues) + 1e-8)

        return curvatures

    def extract_all(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all geometric features."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        features = {'points': points[:, :3]}

        if self.config.compute_normals:
            features['normals'] = self.extract_normals(pcd)

        if self.config.compute_fpfh:
            features['fpfh'] = self.extract_fpfh(pcd)

        if self.config.compute_curvature:
            features['curvature'] = self.extract_curvature(pcd)

        return features


class SemanticFeatureExtractor:
    """
    Extract semantic features using Open3D-ML pretrained models.

    Supports:
    - RandLA-Net: Fast random sampling, good for large point clouds
    - KPConv: Accurate kernel point convolutions
    - PointTransformer: Attention-based, best accuracy
    """

    AVAILABLE_MODELS = ['RandLANet', 'KPConv', 'PointTransformer']

    # SemanticKITTI class names (19 classes)
    SEMANTIC_KITTI_CLASSES = [
        'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
        'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
        'sidewalk', 'other-ground', 'building', 'fence', 'vegetation',
        'trunk', 'terrain', 'pole', 'traffic-sign'
    ]

    def __init__(self, config: FeatureConfig, device: str = 'cuda'):
        if not HAS_OPEN3D_ML:
            raise ImportError(
                "Open3D-ML required for semantic features.\n"
                "Install with: pip install open3d-ml-torch"
            )

        if not HAS_TORCH_BACKEND:
            raise ImportError(
                "PyTorch backend required.\n"
                "Install with: pip install torch open3d-ml-torch"
            )

        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.pipeline = None

        print(f"SemanticFeatureExtractor: {config.semantic_model} on {self.device}")

    def load_model(self, weights_path: Optional[str] = None) -> bool:
        """
        Load pretrained semantic segmentation model.

        Args:
            weights_path: Path to pretrained weights (optional, uses default if None)

        Returns:
            True if model loaded successfully
        """
        try:
            model_name = self.config.semantic_model

            if model_name == 'RandLANet':
                model_cfg = ml3d_torch.models.RandLANet.get_config()
                model_class = ml3d_torch.models.RandLANet
            elif model_name == 'KPConv':
                model_cfg = ml3d_torch.models.KPFCNN.get_config()
                model_class = ml3d_torch.models.KPFCNN
            elif model_name == 'PointTransformer':
                model_cfg = ml3d_torch.models.PointTransformer.get_config()
                model_class = ml3d_torch.models.PointTransformer
            else:
                raise ValueError(f"Unknown model: {model_name}")

            self.model = model_class(**model_cfg)

            # Load weights if provided
            if weights_path and torch.cuda.is_available():
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded weights from {weights_path}")

            # Create inference pipeline
            self.pipeline = ml3d_torch.pipelines.SemanticSegmentation(
                model=self.model,
                device=self.device
            )

            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def segment(self, points: np.ndarray) -> np.ndarray:
        """
        Perform semantic segmentation on point cloud.

        Args:
            points: [N, 3+] point cloud (x, y, z, optional features)

        Returns:
            labels: [N] semantic class labels
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Prepare data dict
        data = {
            'point': points[:, :3].astype(np.float32),
            'feat': points[:, 3:].astype(np.float32) if points.shape[1] > 3 else None,
            'label': np.zeros(len(points), dtype=np.int32)
        }

        # Run inference
        result = self.pipeline.run_inference(data)

        return result['predict_labels']

    def extract_embeddings(
        self,
        points: np.ndarray,
        layer: str = 'fc_layer'
    ) -> np.ndarray:
        """
        Extract feature embeddings from intermediate model layer.

        Note: Requires model modification to expose intermediate features.
        This is a placeholder for custom implementation.
        """
        # This would require modifying the model to return intermediate features
        # For now, return one-hot encoding of semantic labels
        labels = self.segment(points)
        num_classes = len(self.SEMANTIC_KITTI_CLASSES)

        embeddings = np.zeros((len(points), num_classes), dtype=np.float32)
        embeddings[np.arange(len(points)), labels] = 1.0

        return embeddings


class Open3DFeatureExtractor:
    """
    Combined feature extractor using Open3D and Open3D-ML.

    Extracts both geometric and semantic features.
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        device: str = 'cuda'
    ):
        self.config = config or FeatureConfig()
        self.device = device

        # Initialize geometric extractor (always available with Open3D)
        self.geometric = GeometricFeatureExtractor(self.config) if HAS_OPEN3D else None

        # Initialize semantic extractor (requires Open3D-ML)
        self.semantic = None
        if self.config.use_semantic and HAS_OPEN3D_ML:
            try:
                self.semantic = SemanticFeatureExtractor(self.config, device)
                self.semantic.load_model()
            except Exception as e:
                print(f"Warning: Could not initialize semantic extractor: {e}")
                self.semantic = None

    def extract(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all available features from point cloud.

        Args:
            points: [N, 3+] point cloud

        Returns:
            Dictionary with feature arrays
        """
        features = {}

        # Geometric features
        if self.geometric is not None:
            geo_features = self.geometric.extract_all(points)
            features.update(geo_features)

        # Semantic features
        if self.semantic is not None:
            try:
                features['semantic_labels'] = self.semantic.segment(points)
                features['semantic_embeddings'] = self.semantic.extract_embeddings(points)
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
            features: [N, F] point features
            grid_size: (X, Y, Z) voxel grid dimensions
            point_cloud_range: (xmin, ymin, zmin, xmax, ymax, zmax)
            voxel_size: (dx, dy, dz) voxel dimensions

        Returns:
            voxel_features: [X, Y, Z, F] voxelized features
        """
        pc_range = np.array(point_cloud_range)
        voxel_sz = np.array(voxel_size)

        # Filter points in range
        mask = (
            (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) &
            (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]) &
            (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
        )
        points = points[mask]
        features = features[mask]

        # Compute voxel indices
        voxel_indices = np.floor((points - pc_range[:3]) / voxel_sz).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, np.array(grid_size) - 1)

        # Initialize output
        feature_dim = features.shape[1]
        voxel_features = np.zeros((*grid_size, feature_dim), dtype=np.float32)
        voxel_counts = np.zeros(grid_size, dtype=np.float32)

        # Aggregate features
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
            voxel_features[mask] /= voxel_counts[mask, np.newaxis]

        return voxel_features


def create_feature_enhanced_occupancy(
    points: np.ndarray,
    grid_size: Tuple[int, int, int],
    point_cloud_range: Tuple[float, ...],
    voxel_size: Tuple[float, float, float],
    use_semantic: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Create occupancy grid with optional semantic features.

    Args:
        points: [N, 3+] point cloud
        grid_size: (X, Y, Z) voxel dimensions
        point_cloud_range: (xmin, ymin, zmin, xmax, ymax, zmax)
        voxel_size: (dx, dy, dz)
        use_semantic: Whether to compute semantic features

    Returns:
        occupancy: [X, Y, Z] binary occupancy grid
        features: [X, Y, Z, F] semantic features (or None)
    """
    config = FeatureConfig(use_semantic=use_semantic)
    extractor = Open3DFeatureExtractor(config)

    # Extract features
    all_features = extractor.extract(points)

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
        voxel_indices = np.floor((valid_points - pc_range[:3]) / voxel_sz).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, np.array(grid_size) - 1)
        occupancy[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1

    # Aggregate features if available
    features = None
    if 'semantic_embeddings' in all_features:
        features = extractor.aggregate_to_voxels(
            points[:, :3],
            all_features['semantic_embeddings'],
            grid_size,
            point_cloud_range,
            voxel_size
        )

    return occupancy, features


# Example usage and testing
if __name__ == '__main__':
    print("Open3D Feature Extraction Module")
    print("=" * 50)
    print(f"Open3D available: {HAS_OPEN3D}")
    print(f"Open3D-ML available: {HAS_OPEN3D_ML}")
    print(f"PyTorch backend available: {HAS_TORCH_BACKEND}")

    if HAS_OPEN3D:
        # Test geometric features
        print("\nTesting geometric feature extraction...")
        config = FeatureConfig(use_semantic=False)
        geo_extractor = GeometricFeatureExtractor(config)

        # Create random test points
        test_points = np.random.randn(1000, 3).astype(np.float32)
        features = geo_extractor.extract_all(test_points)

        print(f"Points: {features['points'].shape}")
        print(f"Normals: {features['normals'].shape}")
        print(f"FPFH: {features['fpfh'].shape}")
        print(f"Curvature: {features['curvature'].shape}")

    if HAS_OPEN3D_ML and HAS_TORCH_BACKEND:
        print("\nTesting semantic feature extraction...")
        config = FeatureConfig(use_semantic=True)

        try:
            sem_extractor = SemanticFeatureExtractor(config)
            if sem_extractor.load_model():
                labels = sem_extractor.segment(test_points)
                print(f"Semantic labels: {labels.shape}")
                print(f"Unique classes: {np.unique(labels)}")
        except Exception as e:
            print(f"Semantic test failed: {e}")
