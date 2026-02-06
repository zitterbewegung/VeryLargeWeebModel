"""Centralized voxel configuration for all scripts.

This is the single source of truth for voxel grid parameters.
All scripts should import from here to ensure consistency.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class VoxelConfig:
    """Voxelization configuration matching OccWorld expectations."""

    # Point cloud range [xmin, ymin, zmin, xmax, ymax, zmax]
    # Extended Z range for aerial (drone) operations
    point_cloud_range: Tuple[float, ...] = (-40.0, -40.0, -2.0, 40.0, 40.0, 150.0)

    # Voxel size in meters [x, y, z]
    voxel_size: Tuple[float, float, float] = (0.4, 0.4, 1.25)

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """Calculate grid dimensions from range and voxel size (floor division)."""
        if any(v <= 0 for v in self.voxel_size):
            raise ValueError(f"voxel_size must be positive, got {self.voxel_size}")
        return (
            int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0]),
            int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1]),
            int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.voxel_size[2]),
        )

    @property
    def expected_shape(self) -> Tuple[int, int, int]:
        """Alias for grid_size for compatibility."""
        return self.grid_size


# Default configuration instance
DEFAULT_VOXEL_CONFIG = VoxelConfig()

# Convenience exports
DEFAULT_POINT_CLOUD_RANGE = DEFAULT_VOXEL_CONFIG.point_cloud_range
DEFAULT_VOXEL_SIZE = DEFAULT_VOXEL_CONFIG.voxel_size
DEFAULT_GRID_SIZE = DEFAULT_VOXEL_CONFIG.grid_size  # (200, 200, 121)
