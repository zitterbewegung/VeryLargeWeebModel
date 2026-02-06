"""Shared utilities for VeryLargeWeebModel scripts."""

from .logging import Colors, log_info, log_success, log_warn, log_error, log_step
from .s3 import (
    S3_REGION, DEFAULT_BUCKET,
    get_s3_client, s3_file_exists, list_s3_files, ensure_bucket_exists,
    check_aws_credentials, get_transfer_config,
    upload_file, download_file, upload_directory, download_directory,
)
from .trajectory import TrajectoryGenerator, TrajectoryConfig
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

# New utility modules
from .gpu import GPUInfo, detect_gpu_info, gpu_tier, auto_batch_size, select_precision
from .environment import CloudEnvironment, detect_cloud_environment, work_dir_for_provider
from .download import fast_download, verify_download, available_download_tool
from .system_packages import install_system_packages
