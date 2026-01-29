"""Shared utilities for VeryLargeWeebModel scripts."""

from .logging import Colors, log_info, log_success, log_warn, log_error, log_step
from .s3 import (
    S3_REGION, DEFAULT_BUCKET,
    get_s3_client, s3_file_exists, list_s3_files, ensure_bucket_exists,
    check_aws_credentials, get_transfer_config,
)
from .directories import create_session_dirs
from .voxel_config import (
    VoxelConfig, DEFAULT_VOXEL_CONFIG,
    DEFAULT_POINT_CLOUD_RANGE, DEFAULT_VOXEL_SIZE, DEFAULT_GRID_SIZE,
)

# Re-export dependency flags for convenience
from .dependencies import (
    HAS_BOTO3, HAS_TQDM, HAS_REQUESTS, HAS_OPEN3D, HAS_TRIMESH,
    HAS_CV2, HAS_SCIPY, HAS_NUMBA, HAS_TORCH,
)

# Re-export optional modules (may be None if not installed)
from .dependencies import (
    trimesh, o3d, CubicSpline, jit, prange, tqdm, requests, torch,
)
