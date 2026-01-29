#!/usr/bin/env python3
"""
Depth-to-Occupancy Processor for VeryLargeWeebModel Integration

This module converts RGBD depth images from Gazebo simulation into voxelized
occupancy grids compatible with VeryLargeWeebModel's training pipeline.

Depth Occupancy Tiling Algorithm:
1. Unproject depth pixels to 3D points using camera intrinsics
2. Transform points to world frame using camera pose
3. Voxelize points into 3D occupancy grid
4. Apply ray-tracing for free-space carving
5. Generate semantic labels (if available)

Parameters matching VeryLargeWeebModel expectations:
- Voxel size: [0.4, 0.4, 1.25] meters (XYZ) - see utils/voxel_config.py
- Point cloud range: [-40, -40, -2, 40, 40, 150] meters (extended Z for drones)
- Occupancy classes: 0=empty, 1=occupied (binary), or semantic labels

Usage:
    processor = DepthOccupancyProcessor(config)
    occupancy_grid = processor.depth_to_occupancy(depth_image, camera_pose, intrinsics)
"""

import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

# Add scripts directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import HAS_CV2
from utils.voxel_config import DEFAULT_POINT_CLOUD_RANGE, DEFAULT_VOXEL_SIZE

if HAS_CV2:
    import cv2
else:
    cv2 = None


@dataclass
class DepthOccupancyConfig:
    """Configuration for depth-to-occupancy conversion."""

    # Voxel grid parameters (from centralized config)
    voxel_size: Tuple[float, float, float] = DEFAULT_VOXEL_SIZE

    # Point cloud range [xmin, ymin, zmin, xmax, ymax, zmax]
    # Extended Z range for aerial operations
    point_cloud_range: Tuple[float, ...] = DEFAULT_POINT_CLOUD_RANGE

    # Depth camera parameters
    depth_min: float = 0.1    # Minimum valid depth (meters)
    depth_max: float = 10.0   # Maximum valid depth (meters)
    depth_scale: float = 1000.0  # Depth image scale (1000 = millimeters)

    # Camera intrinsics (D435 @ 640x360 defaults)
    fx: float = 343.159
    fy: float = 343.159
    cx: float = 319.5
    cy: float = 179.5
    image_width: int = 640
    image_height: int = 360

    # Ray-tracing for free space
    enable_raytracing: bool = True
    raytracing_step: float = 0.2  # Step size for ray marching (meters)

    # Noise filtering
    depth_noise_threshold: float = 0.05  # Reject noisy depth gradients

    # Tiling parameters for multi-camera fusion
    tile_overlap: float = 0.1  # 10% overlap between tiles

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """Calculate grid dimensions from range and voxel size."""
        return (
            int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0]),
            int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1]),
            int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.voxel_size[2]),
        )


class DepthOccupancyProcessor:
    """
    Converts depth images to 3D occupancy grids for VeryLargeWeebModel training.

    The processor handles:
    - Depth unprojection to 3D points
    - Coordinate frame transformations
    - Voxelization with configurable resolution
    - Free-space carving via ray-tracing
    - Multi-camera tiling and fusion
    """

    def __init__(self, config: Optional[DepthOccupancyConfig] = None):
        """
        Initialize the depth-to-occupancy processor.

        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or DepthOccupancyConfig()
        self._build_projection_matrix()
        self._precompute_pixel_rays()

    def _build_projection_matrix(self):
        """Build the camera intrinsic matrix."""
        self.K = np.array([
            [self.config.fx, 0, self.config.cx],
            [0, self.config.fy, self.config.cy],
            [0, 0, 1]
        ], dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)

    def _precompute_pixel_rays(self):
        """Precompute ray directions for each pixel (optimization)."""
        u = np.arange(self.config.image_width)
        v = np.arange(self.config.image_height)
        u, v = np.meshgrid(u, v)

        # Homogeneous pixel coordinates
        ones = np.ones_like(u)
        pixels_homo = np.stack([u, v, ones], axis=-1).reshape(-1, 3)

        # Unproject to normalized camera rays
        rays = (self.K_inv @ pixels_homo.T).T
        self.pixel_rays = rays.reshape(self.config.image_height, self.config.image_width, 3)

    def depth_to_points(
        self,
        depth_image: np.ndarray,
        transform: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert depth image to 3D point cloud.

        Args:
            depth_image: Depth image [H, W] in meters or scaled units
            transform: Optional 4x4 transformation matrix (camera to world)

        Returns:
            points: [N, 3] array of 3D points in target frame
        """
        # Handle depth scaling
        if depth_image.dtype == np.uint16:
            depth = depth_image.astype(np.float32) / self.config.depth_scale
        else:
            depth = depth_image.astype(np.float32)

        # Create validity mask
        valid_mask = (depth > self.config.depth_min) & (depth < self.config.depth_max)

        # Apply noise filtering (reject large depth discontinuities)
        if self.config.depth_noise_threshold > 0 and HAS_CV2:
            depth_gradient = np.abs(cv2.Sobel(depth, cv2.CV_32F, 1, 1, ksize=3))
            valid_mask &= (depth_gradient < self.config.depth_noise_threshold * depth)

        # Unproject to 3D
        points_camera = self.pixel_rays * depth[:, :, np.newaxis]
        points_camera = points_camera[valid_mask]

        # Transform to world frame if provided
        if transform is not None:
            ones = np.ones((points_camera.shape[0], 1))
            points_homo = np.hstack([points_camera, ones])
            points_world = (transform @ points_homo.T).T[:, :3]
            return points_world

        return points_camera

    def points_to_voxels(self, points: np.ndarray) -> np.ndarray:
        """
        Voxelize point cloud into 3D occupancy grid.

        Args:
            points: [N, 3] array of 3D points

        Returns:
            occupancy: [X, Y, Z] binary occupancy grid
        """
        cfg = self.config
        grid_size = cfg.grid_size

        # Filter points within range
        range_min = np.array(cfg.point_cloud_range[:3])
        range_max = np.array(cfg.point_cloud_range[3:])

        in_range = np.all((points >= range_min) & (points < range_max), axis=1)
        points_filtered = points[in_range]

        if len(points_filtered) == 0:
            return np.zeros(grid_size, dtype=np.uint8)

        # Calculate voxel indices
        voxel_size = np.array(cfg.voxel_size)
        voxel_indices = np.floor((points_filtered - range_min) / voxel_size).astype(np.int32)

        # Clamp to valid range
        voxel_indices = np.clip(
            voxel_indices,
            [0, 0, 0],
            [grid_size[0] - 1, grid_size[1] - 1, grid_size[2] - 1]
        )

        # Create occupancy grid
        occupancy = np.zeros(grid_size, dtype=np.uint8)
        occupancy[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1

        return occupancy

    def raytrace_free_space(
        self,
        occupancy: np.ndarray,
        camera_position: np.ndarray
    ) -> np.ndarray:
        """
        Carve free space using ray-tracing from camera position.

        This marks voxels as 'known empty' if they lie between the camera
        and an occupied voxel along the viewing ray.

        Args:
            occupancy: [X, Y, Z] binary occupancy grid
            camera_position: [3] camera position in world frame

        Returns:
            occupancy_with_free: Grid with free space marked (value=255 for unknown,
                                 0 for free, 1 for occupied)
        """
        if not self.config.enable_raytracing:
            return occupancy

        cfg = self.config
        grid_size = cfg.grid_size
        voxel_size = np.array(cfg.voxel_size)
        range_min = np.array(cfg.point_cloud_range[:3])

        # Initialize output (255 = unknown, 0 = free, 1 = occupied)
        result = np.full(grid_size, 255, dtype=np.uint8)
        result[occupancy == 1] = 1

        # Find occupied voxels
        occupied_indices = np.argwhere(occupancy == 1)

        for idx in occupied_indices:
            # Voxel center in world coordinates
            voxel_center = range_min + (idx + 0.5) * voxel_size

            # Ray from camera to voxel
            ray_dir = voxel_center - camera_position
            ray_length = np.linalg.norm(ray_dir)
            ray_dir = ray_dir / ray_length

            # March along ray
            steps = int(ray_length / cfg.raytracing_step)
            for step in range(1, steps):
                t = step * cfg.raytracing_step
                point = camera_position + t * ray_dir

                # Convert to voxel index
                voxel_idx = np.floor((point - range_min) / voxel_size).astype(np.int32)

                # Check bounds
                if np.any(voxel_idx < 0) or np.any(voxel_idx >= grid_size):
                    continue

                # Mark as free if not already occupied
                if result[voxel_idx[0], voxel_idx[1], voxel_idx[2]] == 255:
                    result[voxel_idx[0], voxel_idx[1], voxel_idx[2]] = 0

        return result

    def depth_to_occupancy(
        self,
        depth_image: np.ndarray,
        camera_pose: np.ndarray,
        intrinsics: Optional[Dict] = None
    ) -> Dict[str, np.ndarray]:
        """
        Complete pipeline: depth image to occupancy grid.

        Args:
            depth_image: [H, W] depth image
            camera_pose: [4, 4] camera-to-world transformation matrix
            intrinsics: Optional dict with fx, fy, cx, cy to override defaults

        Returns:
            dict with keys:
                - 'occupancy': [X, Y, Z] occupancy grid
                - 'points': [N, 3] point cloud
                - 'camera_position': [3] camera position
        """
        # Update intrinsics if provided
        if intrinsics is not None:
            self.config.fx = intrinsics.get('fx', self.config.fx)
            self.config.fy = intrinsics.get('fy', self.config.fy)
            self.config.cx = intrinsics.get('cx', self.config.cx)
            self.config.cy = intrinsics.get('cy', self.config.cy)
            self._build_projection_matrix()
            self._precompute_pixel_rays()

        # Convert depth to points
        points = self.depth_to_points(depth_image, camera_pose)

        # Voxelize
        occupancy = self.points_to_voxels(points)

        # Extract camera position for raytracing
        camera_position = camera_pose[:3, 3]

        # Optional: carve free space
        if self.config.enable_raytracing and len(points) > 0:
            occupancy = self.raytrace_free_space(occupancy, camera_position)

        return {
            'occupancy': occupancy,
            'points': points,
            'camera_position': camera_position,
            'grid_size': self.config.grid_size,
            'voxel_size': self.config.voxel_size,
            'point_cloud_range': self.config.point_cloud_range
        }

    def tile_multi_camera(
        self,
        depth_images: Dict[str, np.ndarray],
        camera_poses: Dict[str, np.ndarray],
        camera_intrinsics: Dict[str, Dict]
    ) -> np.ndarray:
        """
        Fuse multiple depth cameras into a single occupancy grid.

        Args:
            depth_images: Dict mapping camera name to depth image
            camera_poses: Dict mapping camera name to 4x4 pose matrix
            camera_intrinsics: Dict mapping camera name to intrinsics dict

        Returns:
            fused_occupancy: [X, Y, Z] fused occupancy grid
        """
        fused_occupancy = np.zeros(self.config.grid_size, dtype=np.uint8)

        for cam_name in depth_images.keys():
            # Process each camera
            result = self.depth_to_occupancy(
                depth_images[cam_name],
                camera_poses[cam_name],
                camera_intrinsics.get(cam_name)
            )

            # Fuse using max (any camera seeing occupied = occupied)
            fused_occupancy = np.maximum(fused_occupancy, result['occupancy'])

        return fused_occupancy


class VeryLargeWeebModelDepthAdapter:
    """
    Adapter to format depth-derived occupancy for VeryLargeWeebModel training.

    Ensures output format matches VeryLargeWeebModel's expected tensor shapes and
    semantic label conventions.
    """

    # VeryLargeWeebModel semantic class mapping
    SEMANTIC_CLASSES = {
        0: 'empty',
        1: 'barrier',
        2: 'bicycle',
        3: 'bus',
        4: 'car',
        5: 'construction_vehicle',
        6: 'motorcycle',
        7: 'pedestrian',
        8: 'traffic_cone',
        9: 'trailer',
        10: 'truck',
        11: 'driveable_surface',
        12: 'other_flat',
        13: 'sidewalk',
        14: 'terrain',
        15: 'manmade',
        16: 'vegetation',
        17: 'sky',  # aerial extension
    }

    def __init__(self, processor: DepthOccupancyProcessor):
        self.processor = processor

    def to_occworld_format(
        self,
        occupancy: np.ndarray,
        semantic_labels: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Convert occupancy grid to VeryLargeWeebModel training format.

        Args:
            occupancy: [X, Y, Z] binary occupancy grid
            semantic_labels: Optional [X, Y, Z] semantic labels

        Returns:
            dict compatible with VeryLargeWeebModel DataLoader
        """
        cfg = self.processor.config

        # If no semantic labels, use height-based heuristics
        if semantic_labels is None:
            semantic_labels = self._height_based_semantics(occupancy)

        # Convert to VeryLargeWeebModel tensor format [Z, Y, X] (channel-first style)
        occupancy_tensor = np.transpose(occupancy, (2, 1, 0))
        semantic_tensor = np.transpose(semantic_labels, (2, 1, 0))

        return {
            'occ_gt': occupancy_tensor,
            'occ_semantic': semantic_tensor,
            'occ_shape': cfg.grid_size,
            'voxel_size': cfg.voxel_size,
            'pc_range': cfg.point_cloud_range,
        }

    def _height_based_semantics(self, occupancy: np.ndarray) -> np.ndarray:
        """
        Assign semantic labels based on height (simple heuristic).

        For simulation, this provides a baseline. Real deployments should
        use a learned semantic segmentation model.
        """
        cfg = self.processor.config
        semantic = np.zeros_like(occupancy, dtype=np.uint8)

        z_min = cfg.point_cloud_range[2]
        voxel_z = cfg.voxel_size[2]

        for z_idx in range(occupancy.shape[2]):
            height = z_min + (z_idx + 0.5) * voxel_z
            occupied_mask = occupancy[:, :, z_idx] > 0

            if height < 0.3:
                # Ground level
                semantic[:, :, z_idx][occupied_mask] = 11  # driveable_surface
            elif height < 2.0:
                # Low obstacles (could be vehicles, pedestrians)
                semantic[:, :, z_idx][occupied_mask] = 15  # manmade (default)
            elif height < 50:
                # Building level
                semantic[:, :, z_idx][occupied_mask] = 15  # manmade
            else:
                # High altitude structures
                semantic[:, :, z_idx][occupied_mask] = 15  # manmade

        return semantic


# Example usage and testing
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Depth to Occupancy Processor')
    parser.add_argument('--depth', type=str, help='Path to depth image (PNG)')
    parser.add_argument('--output', type=str, default='occupancy.npz', help='Output file')
    args = parser.parse_args()

    # Default configuration for VeryLargeWeebModel
    config = DepthOccupancyConfig(
        voxel_size=(0.4, 0.4, 1.25),  # Coarser Z for extended range
        point_cloud_range=(-40.0, -40.0, -2.0, 40.0, 40.0, 150.0),
        depth_min=0.1,
        depth_max=10.0,
    )

    processor = DepthOccupancyProcessor(config)
    adapter = VeryLargeWeebModelDepthAdapter(processor)

    print(f"Grid size: {config.grid_size}")
    print(f"Voxel size: {config.voxel_size}")
    print(f"Point cloud range: {config.point_cloud_range}")

    if args.depth:
        if not HAS_CV2:
            print("Error: OpenCV (cv2) is required to load depth images.")
            print("Install with: pip install opencv-python")
            sys.exit(1)
        # Load and process depth image
        depth_image = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)

        # Identity pose (camera at origin, looking forward)
        camera_pose = np.eye(4)

        result = processor.depth_to_occupancy(depth_image, camera_pose)
        occworld_data = adapter.to_occworld_format(result['occupancy'])

        np.savez_compressed(args.output, **occworld_data)
        print(f"Saved occupancy to {args.output}")
        print(f"Occupied voxels: {np.sum(result['occupancy'] > 0)}")
