#!/usr/bin/env python3
"""
OccWorld Training Script for Tokyo Gazebo Dataset

This script provides a training interface for OccWorld models using
data generated from Gazebo simulation.

It can either:
1. Use an existing OccWorld installation (if available)
2. Run standalone training with our custom dataset

Usage:
    # Fine-tune on Tokyo Gazebo data
    python train.py --config config/finetune_tokyo.py --work-dir out/occworld_tokyo

    # Train from scratch
    python train.py --config config/finetune_tokyo.py --work-dir out/occworld_tokyo --from-scratch

    # Resume training
    python train.py --config config/finetune_tokyo.py --work-dir out/occworld_tokyo --resume

Requirements:
    - PyTorch >= 1.9
    - mmcv-full
    - OccWorld dependencies (or standalone mode)
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
from multiprocessing import cpu_count

# Disable OpenCV threading (must be before cv2 import)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENCV_THREAD_COUNT'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Optional wandb integration
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import our custom datasets
from dataset.gazebo_occworld_dataset import (
    GazeboOccWorldDataset,
    DatasetConfig,
    collate_fn,
    create_dataloader,
)

# Import 6DoF model
from models import (
    OccWorld6DoF,
    OccWorld6DoFConfig,
    OccWorld6DoFLoss,
    count_parameters,
)

# Try to import nuScenes dataset
try:
    from dataset.nuscenes_occworld_dataset import (
        NuScenesOccWorldDataset,
        NuScenesConfig,
        collate_fn as nuscenes_collate_fn,
    )
    HAS_NUSCENES = True
except ImportError:
    HAS_NUSCENES = False

# Try to import nuScenes 6DoF dataset (with geometric augmentation)
try:
    from dataset.nuscenes_6dof_dataset import (
        NuScenes6DoFDataset,
        NuScenes6DoFConfig,
        collate_fn as nuscenes_6dof_collate_fn,
    )
    HAS_NUSCENES_6DOF = True
except ImportError:
    HAS_NUSCENES_6DOF = False

# Try to import UAVScenes dataset (real aerial 6DoF data)
try:
    from dataset.uavscenes_dataset import (
        UAVScenesDataset,
        UAVScenesConfig,
        collate_fn as uavscenes_collate_fn,
    )
    HAS_UAVSCENES = True
except ImportError:
    HAS_UAVSCENES = False


# =============================================================================
# Data Validation
# =============================================================================

S3_BUCKET = "verylargeweebmodel"
S3_REGION = "us-west-2"


def validate_data(data_root: str, dataset_type: str, auto_download: bool = True) -> bool:
    """
    Validate that required training data exists and is complete.

    Args:
        data_root: Path to data directory
        dataset_type: Type of dataset (gazebo, nuscenes, uavscenes, etc.)
        auto_download: If True, attempt to download missing data from S3

    Returns:
        True if data is valid, False otherwise
    """
    data_path = Path(data_root)
    issues = []
    warnings = []

    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)
    print(f"Data root: {data_path.absolute()}")
    print(f"Dataset type: {dataset_type}")

    # Check if data directory exists
    if not data_path.exists():
        issues.append(f"Data directory does not exist: {data_path}")
        if auto_download:
            print(f"\n[AUTO-DOWNLOAD] Creating directory and downloading from S3...")
            return _download_dataset_from_s3(data_root, dataset_type)
        return False

    # Dataset-specific validation
    if dataset_type == 'gazebo':
        valid = _validate_gazebo_data(data_path, issues, warnings)
    elif dataset_type == 'nuscenes':
        valid = _validate_nuscenes_data(data_path, issues, warnings)
    elif dataset_type == 'nuscenes_6dof':
        valid = _validate_nuscenes_data(data_path, issues, warnings)
    elif dataset_type == 'uavscenes':
        valid = _validate_uavscenes_data(data_path, issues, warnings)
    else:
        warnings.append(f"Unknown dataset type: {dataset_type}, skipping validation")
        valid = True

    # Print results
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  [WARN] {w}")

    if issues:
        print("\nIssues found:")
        for i in issues:
            print(f"  [ERROR] {i}")

        if auto_download:
            print(f"\n[AUTO-DOWNLOAD] Attempting to download missing data from S3...")
            if _download_dataset_from_s3(data_root, dataset_type):
                # Re-validate after download
                return validate_data(data_root, dataset_type, auto_download=False)

        print("\nTo fix manually:")
        print(f"  aws s3 sync s3://{S3_BUCKET}/ {data_root}/ --region {S3_REGION}")
        return False

    print("\n[OK] Data validation passed!")
    print("=" * 60 + "\n")
    return True


def _validate_gazebo_data(data_path: Path, issues: list, warnings: list) -> bool:
    """Validate Gazebo/PLATEAU dataset structure."""
    # Expected structure:
    # data_root/
    #   session_name/
    #     occupancy/  (*.npz files)
    #     lidar/      (*.npy files)
    #     poses/      (*.json files)

    sessions = [d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not sessions:
        issues.append("No session directories found")
        return False

    valid_sessions = 0
    for session in sessions:
        has_occupancy = (session / 'occupancy').exists() and any((session / 'occupancy').glob('*.npz'))
        has_lidar = (session / 'lidar').exists() and any((session / 'lidar').glob('*.npy'))

        if has_occupancy or has_lidar:
            valid_sessions += 1
        else:
            warnings.append(f"Session {session.name} missing occupancy/lidar data")

    if valid_sessions == 0:
        issues.append("No valid sessions with occupancy or lidar data")
        return False

    print(f"  Found {valid_sessions}/{len(sessions)} valid sessions")
    return True


def _validate_nuscenes_data(data_path: Path, issues: list, warnings: list) -> bool:
    """Validate nuScenes dataset structure."""
    # Check for required files
    required_files = [
        'nuscenes_infos_train_temporal_v3_scene.pkl',
        'nuscenes_infos_val_temporal_v3_scene.pkl',
    ]

    # Check in data_path and parent
    for req_file in required_files:
        found = False
        for check_path in [data_path / req_file, data_path.parent / req_file]:
            if check_path.exists():
                size_mb = check_path.stat().st_size / 1024 / 1024
                if size_mb < 1:
                    warnings.append(f"{req_file} seems too small ({size_mb:.1f}MB)")
                else:
                    print(f"  Found {req_file} ({size_mb:.1f}MB)")
                found = True
                break
        if not found:
            issues.append(f"Missing: {req_file}")

    # Check for nuScenes data directory
    nuscenes_dirs = ['v1.0-mini', 'v1.0-trainval', 'samples', 'sweeps']
    found_any = False
    for d in nuscenes_dirs:
        if (data_path / d).exists():
            found_any = True
            print(f"  Found nuScenes directory: {d}")

    if not found_any:
        warnings.append("nuScenes raw data directories not found (may not be needed if using pickle files)")

    return len(issues) == 0


def _validate_uavscenes_data(data_path: Path, issues: list, warnings: list) -> bool:
    """Validate UAVScenes dataset structure."""
    # Expected structure:
    # data_root/
    #   interval1_AMtown01/
    #     interval1_LIDAR/  (*.txt files)
    #     interval1_CAM/    (*.jpg files)
    #     sampleinfos_interpolated.json

    scenes = ['AMtown', 'AMvalley', 'HKairport', 'HKisland']
    found_scenes = []

    for scene in scenes:
        # Check multiple naming patterns
        patterns = [
            f"interval1_{scene}01",
            f"interval1_{scene}",
            f"interval5_{scene}01",
            scene,
        ]

        for pattern in patterns:
            scene_path = data_path / pattern
            if scene_path.exists():
                # Check for LiDAR data
                lidar_dirs = [
                    scene_path / 'interval1_LIDAR',
                    scene_path / 'interval5_LIDAR',
                    scene_path / 'lidar',
                ]

                has_lidar = False
                for lidar_dir in lidar_dirs:
                    if lidar_dir.exists():
                        lidar_files = list(lidar_dir.glob('*.txt')) + list(lidar_dir.glob('*.pcd'))
                        if lidar_files:
                            print(f"  Found {scene}: {len(lidar_files)} LiDAR files")
                            found_scenes.append(scene)
                            has_lidar = True
                            break

                if not has_lidar:
                    warnings.append(f"Scene {pattern} exists but no LiDAR data found")
                break

    if not found_scenes:
        issues.append("No UAVScenes data found. Expected interval1_*/interval1_LIDAR/*.txt")
        return False

    print(f"  Found {len(found_scenes)}/4 scenes: {found_scenes}")
    return True


def _download_dataset_from_s3(data_root: str, dataset_type: str) -> bool:
    """Download dataset from S3."""
    import subprocess

    # Map dataset types to S3 prefixes
    s3_prefixes = {
        'gazebo': 'tokyo_gazebo',
        'nuscenes': 'nuscenes',
        'nuscenes_6dof': 'nuscenes',
        'uavscenes': 'uavscenes',
    }

    prefix = s3_prefixes.get(dataset_type, dataset_type)
    s3_uri = f"s3://{S3_BUCKET}/{prefix}/"

    print(f"Downloading from {s3_uri} to {data_root}/")

    try:
        # Check if AWS CLI is available
        result = subprocess.run(
            ["aws", "sts", "get-caller-identity"],
            capture_output=True,
            timeout=10
        )

        if result.returncode != 0:
            print("[ERROR] AWS CLI not configured. Run: aws configure")
            return False

        # Sync from S3
        result = subprocess.run(
            ["aws", "s3", "sync", s3_uri, data_root, "--region", S3_REGION],
            timeout=7200  # 2 hour timeout
        )

        if result.returncode == 0:
            print("[OK] Download complete!")
            return True
        else:
            print("[ERROR] S3 sync failed")
            return False

    except FileNotFoundError:
        print("[ERROR] AWS CLI not installed. Install with: pip install awscli")
        return False
    except subprocess.TimeoutExpired:
        print("[ERROR] Download timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


# =============================================================================
# Pretrained Model Validation
# =============================================================================

# Pretrained models available on S3
PRETRAINED_MODELS = {
    "occworld_checkpoint": {
        "s3_key": "pretrained/occworld/latest.pth",
        "local_path": "pretrained/occworld/latest.pth",
        "min_size": 100_000_000,  # 100MB
        "description": "OccWorld checkpoint (~721MB)",
    },
    "vqvae_checkpoint": {
        "s3_key": "pretrained/vqvae/epoch_125.pth",
        "local_path": "pretrained/vqvae/epoch_125.pth",
        "min_size": 100_000_000,  # 100MB
        "description": "VQVAE checkpoint (~500MB)",
    },
}


def validate_pretrained_models(
    load_from: str = None,
    vqvae_ckpt: str = None,
    auto_download: bool = True
) -> bool:
    """
    Validate that required pretrained models exist.

    Args:
        load_from: Path to main model checkpoint (OccWorld)
        vqvae_ckpt: Path to VQVAE checkpoint
        auto_download: If True, attempt to download missing models from S3

    Returns:
        True if all required models are valid, False otherwise
    """
    import subprocess

    print("\n" + "=" * 60)
    print("PRETRAINED MODEL VALIDATION")
    print("=" * 60)

    models_to_check = []
    if load_from:
        models_to_check.append(("OccWorld checkpoint", load_from, "occworld_checkpoint"))
    if vqvae_ckpt:
        models_to_check.append(("VQVAE checkpoint", vqvae_ckpt, "vqvae_checkpoint"))

    if not models_to_check:
        print("  No pretrained models specified (training from scratch)")
        print("=" * 60 + "\n")
        return True

    all_valid = True
    for name, path, model_key in models_to_check:
        path = Path(path)
        model_info = PRETRAINED_MODELS.get(model_key, {})
        min_size = model_info.get("min_size", 1_000_000)

        if path.exists():
            size = path.stat().st_size
            if size >= min_size:
                print(f"  [OK] {name}: {path} ({size / 1024 / 1024:.1f}MB)")
                continue
            else:
                print(f"  [WARN] {name}: {path} seems incomplete ({size} bytes)")
                # Remove incomplete file
                path.unlink()

        # Model not found or incomplete
        print(f"  [MISSING] {name}: {path}")

        if auto_download and model_key in PRETRAINED_MODELS:
            print(f"  [AUTO-DOWNLOAD] Downloading {name} from S3...")
            s3_key = PRETRAINED_MODELS[model_key]["s3_key"]
            s3_uri = f"s3://{S3_BUCKET}/{s3_key}"

            # Create parent directory
            path.parent.mkdir(parents=True, exist_ok=True)

            try:
                # Check AWS CLI
                result = subprocess.run(
                    ["aws", "sts", "get-caller-identity"],
                    capture_output=True,
                    timeout=10
                )
                if result.returncode != 0:
                    print("    [ERROR] AWS CLI not configured. Run: aws configure")
                    all_valid = False
                    continue

                # Download from S3
                result = subprocess.run(
                    ["aws", "s3", "cp", s3_uri, str(path), "--region", S3_REGION],
                    timeout=3600  # 1 hour timeout
                )

                if result.returncode == 0 and path.exists():
                    size = path.stat().st_size
                    if size >= min_size:
                        print(f"    [OK] Downloaded {name} ({size / 1024 / 1024:.1f}MB)")
                        continue
                    else:
                        print(f"    [ERROR] Downloaded file too small ({size} bytes)")
                        path.unlink()

                print(f"    [ERROR] Failed to download {name}")
                all_valid = False

            except subprocess.TimeoutExpired:
                print(f"    [ERROR] Download timed out for {name}")
                all_valid = False
            except FileNotFoundError:
                print("    [ERROR] AWS CLI not installed. Install with: pip install awscli")
                all_valid = False
            except Exception as e:
                print(f"    [ERROR] Download failed: {e}")
                all_valid = False
        else:
            print(f"  [ERROR] {name} not found and auto-download disabled")
            all_valid = False

    if all_valid:
        print("\n[OK] All pretrained models validated!")
    else:
        print("\n[ERROR] Some pretrained models are missing!")
        print("To download manually:")
        print(f"  python scripts/download_pretrained.py --output .")

    print("=" * 60 + "\n")
    return all_valid


def parse_args():
    parser = argparse.ArgumentParser(description='OccWorld Training for Tokyo Gazebo')

    # Config
    parser.add_argument('--config', '--py-config', type=str,
                        default='config/finetune_tokyo.py',
                        help='Path to config file')
    parser.add_argument('--work-dir', type=str, default='out/occworld_tokyo',
                        help='Directory to save outputs')

    # Training
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from specific checkpoint')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Train from scratch (no pretrained weights)')

    # Hardware
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='GPU IDs to use (comma-separated)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU mode (useful when MPS/CUDA ops are unsupported)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Overrides
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs')

    # Mode
    parser.add_argument('--eval-only', action='store_true',
                        help='Run evaluation only')
    parser.add_argument('--use-occworld', action='store_true',
                        help='Use OccWorld library if installed')

    # Model type
    parser.add_argument('--model-type', type=str, default='simple',
                        choices=['simple', '6dof'],
                        help='Model type: simple (occupancy only) or 6dof (full 6DoF prediction)')
    parser.add_argument('--use-transformer', action='store_true',
                        help='Use Transformer instead of LSTM for temporal modeling (6dof only)')

    # Performance optimizations
    parser.add_argument('--amp', action='store_true',
                        help='Enable automatic mixed precision (FP16/BF16) - much faster on A100')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() for faster training (PyTorch 2.0+)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Data loading workers (default: 1, increase for faster loading)')
    parser.add_argument('--debug-freq', type=int, default=500,
                        help='Debug print frequency (default: 500, higher = faster)')
    parser.add_argument('--save-freq', type=int, default=1,
                        help='Checkpoint save frequency in epochs (default: 1 = every epoch)')

    # Weights & Biases
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='occworld-tokyo',
                        help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='W&B entity (team/username)')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='W&B run name (auto-generated if not provided)')
    parser.add_argument('--wandb-tags', type=str, nargs='+', default=[],
                        help='W&B tags for the run')

    # Data validation
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip data and model validation (use with caution)')
    parser.add_argument('--no-auto-download', action='store_true',
                        help='Disable auto-download from S3 for missing data/models')

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from Python file."""
    import importlib.util

    config_path = Path(config_path).absolute()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    return config


def setup_environment(args):
    """Setup training environment."""
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup GPU/MPS
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    if args.cpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Create work directory
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / 'checkpoints').mkdir(exist_ok=True)
    (work_dir / 'logs').mkdir(exist_ok=True)

    return device, work_dir


def try_import_occworld():
    """Try to import OccWorld library."""
    try:
        # Try importing from installed OccWorld
        sys.path.insert(0, os.path.expanduser('~/OccWorld'))
        from models import TransVQVAE
        from utils import get_logger
        print("Using OccWorld library")
        return True, TransVQVAE
    except ImportError:
        print("OccWorld library not found, using standalone mode")
        return False, None


class SimpleOccupancyModel(nn.Module):
    """
    3DoF Occupancy prediction model for ground vehicles.

    Predicts:
    - Future occupancy grids
    - Future 3DoF poses (x, y, yaw) for ground vehicles

    For full 6DoF (aerial vehicles), use OccWorld6DoF model instead.
    """

    def __init__(self, config):
        super().__init__()

        # Get dimensions from config
        self.history_frames = getattr(config, 'history_frames', 4)
        self.future_frames = getattr(config, 'future_frames', 6)

        grid_size = getattr(config, 'grid_size', [200, 200, 121])
        self.grid_x, self.grid_y, self.grid_z = grid_size

        # 3DoF: x, y, yaw (for ground vehicles)
        # vs 6DoF: x, y, z, qx, qy, qz, qw
        self.pose_dim = 13  # Input pose dim (full format for compatibility)
        self.output_pose_dim = 13  # Output all pose components
        self.latent_dim = 256

        # Simple 3D encoder for occupancy
        self.encoder = nn.Sequential(
            nn.Conv3d(self.history_frames, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        # Adaptive pooling for fixed-size output
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))

        # Pose encoder - processes history poses
        self.pose_encoder = nn.Sequential(
            nn.Linear(self.pose_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim),
        )

        # Fusion layer - combines spatial and pose features
        spatial_features = 256 * 4 * 4 * 4
        self.spatial_proj = nn.Linear(spatial_features, self.latent_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # Temporal modeling (LSTM)
        self.temporal = nn.LSTM(
            input_size=self.latent_dim,
            hidden_size=self.latent_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Decoder for future occupancy
        self.to_spatial = nn.Linear(self.latent_dim, 256 * 4 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, self.future_frames, kernel_size=3, padding=1),
        )

        # Pose decoder - predicts future poses autoregressively
        self.pose_gru = nn.GRUCell(self.pose_dim + self.latent_dim, self.latent_dim)
        self.pose_out = nn.Linear(self.latent_dim, self.output_pose_dim)

    def forward(self, history_occupancy, history_poses=None, future_poses=None):
        """
        Forward pass.

        Args:
            history_occupancy: [B, T_h, X, Y, Z] - Past occupancy grids
            history_poses: [B, T_h, 13] - Past poses
            future_poses: [B, T_f, 13] - Future poses (for teacher forcing, optional)

        Returns:
            Dict with:
                - future_occupancy: [B, T_f, X, Y, Z]
                - future_poses: [B, T_f, 13] (if history_poses provided)
        """
        batch_size = history_occupancy.shape[0]
        device = history_occupancy.device
        target_shape = history_occupancy.shape[2:]  # [X, Y, Z]

        # Encode spatial features from occupancy
        x = history_occupancy.float()
        encoded = self.encoder(x)  # [B, 256, X/4, Y/4, Z/4]
        pooled = self.adaptive_pool(encoded)  # [B, 256, 4, 4, 4]
        spatial_flat = pooled.view(batch_size, -1)  # [B, 256*4*4*4]
        spatial_proj = self.spatial_proj(spatial_flat)  # [B, latent_dim]

        # Encode poses if available
        if history_poses is not None:
            # Encode each pose in history
            B, T, D = history_poses.shape
            poses_flat = history_poses.view(B * T, D)
            pose_features = self.pose_encoder(poses_flat).view(B, T, -1)  # [B, T, latent_dim]
            last_pose_feat = pose_features[:, -1, :]  # [B, latent_dim]

            # Fuse spatial and pose features
            fused = torch.cat([spatial_proj, last_pose_feat], dim=-1)
            fused = self.fusion(fused)  # [B, latent_dim]
        else:
            fused = spatial_proj

        # Temporal modeling
        temporal_out, (h_n, c_n) = self.temporal(fused.unsqueeze(1))
        context = h_n[-1]  # [B, latent_dim]

        # Decode future occupancy
        spatial_for_decode = self.to_spatial(context)  # [B, 256*4*4*4]
        spatial_for_decode = spatial_for_decode.view(batch_size, 256, 4, 4, 4)
        future_occ_logits = self.decoder(spatial_for_decode)  # [B, T_f, X', Y', Z']

        # Resize to match target dimensions
        if future_occ_logits.shape[2:] != target_shape:
            future_occ_logits = F.interpolate(
                future_occ_logits,
                size=target_shape,
                mode='trilinear',
                align_corners=False
            )

        future_occupancy = torch.sigmoid(future_occ_logits)

        # Predict future poses if history poses are available
        predicted_poses = None
        if history_poses is not None:
            predicted_poses = []
            last_pose = history_poses[:, -1, :]  # [B, 13]
            hidden = context  # Use context as initial hidden state

            for t in range(self.future_frames):
                # Combine last pose with context
                inp = torch.cat([last_pose, context], dim=-1)
                hidden = self.pose_gru(inp, hidden)
                pose_delta = self.pose_out(hidden)

                # Residual prediction
                current_pose = last_pose + pose_delta
                predicted_poses.append(current_pose)
                last_pose = current_pose

            predicted_poses = torch.stack(predicted_poses, dim=1)  # [B, T_f, 13]

        # Return dict for consistency with 6DoF model
        if predicted_poses is not None:
            return {
                'future_occupancy': future_occupancy,
                'future_poses': predicted_poses,
            }
        else:
            # For backwards compatibility, return just occupancy
            return future_occupancy


class FocalLoss(nn.Module):
    """
    Focal Loss for extremely imbalanced binary classification.

    Focal loss down-weights easy examples (empty voxels) and focuses on hard examples
    (occupied voxels). Much better than weighted BCE for sparse occupancy grids.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weight for positive class (occupied). Set high for sparse data (e.g., 0.95 for 1% occupancy)
        gamma: Focusing parameter. Higher = more focus on hard examples. Default 2.0 works well.
    """

    def __init__(self, alpha=0.95, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # Clamp predictions to avoid log(0)
        pred = pred.clamp(min=1e-7, max=1 - 1e-7)

        # Binary cross entropy per element
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)

        # Focal weight: (1 - p_t)^gamma
        p_t = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight: alpha for positive, (1-alpha) for negative
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)

        focal_loss = alpha_weight * focal_weight * bce
        return focal_loss.mean()


class OccupancyLoss(nn.Module):
    """
    Combined Focal + Dice + Mean-matching loss for sparse occupancy prediction.

    Uses:
    - Focal Loss: handles extreme class imbalance
    - Dice Loss: optimizes for overlap
    - Mean-matching: prevents collapse to all-zeros by ensuring pred mean ≈ target mean

    Args:
        focal_alpha: Weight for occupied class (0.99 for ~1% occupancy)
        focal_gamma: Focusing parameter (2.0 = standard)
        dice_weight: Weight for Dice loss
        mean_weight: Weight for mean-matching regularization
        smooth: Smoothing factor for Dice loss
    """

    def __init__(self, focal_alpha=0.99, focal_gamma=2.0, dice_weight=1.0, mean_weight=10.0, smooth=1.0):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_weight = dice_weight
        self.mean_weight = mean_weight
        self.smooth = smooth
        self._debug_counter = 0
        # Store last loss components for logging
        self._last_components = {}

    def get_loss_components(self):
        """Return the last computed loss components for logging."""
        return self._last_components

    def forward(self, pred, target):
        # Debug: Print data stats every 100 batches
        self._debug_counter += 1
        if self._debug_counter % 100 == 1:
            print(f"  [LOSS DEBUG] pred shape: {pred.shape}, target shape: {target.shape}")
            print(f"  [LOSS DEBUG] pred dtype: {pred.dtype}, target dtype: {target.dtype}")
            print(f"  [LOSS DEBUG] pred range: [{pred.min().item():.6f}, {pred.max().item():.6f}]")
            print(f"  [LOSS DEBUG] target range: [{target.min().item():.6f}, {target.max().item():.6f}]")
            print(f"  [LOSS DEBUG] target unique values: {torch.unique(target).tolist()[:10]}")

        # Focal loss: handles class imbalance, focuses on hard examples
        focal = self.focal_loss(pred, target)

        # Dice loss: optimizes for overlap, complements focal loss
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        # Mean-matching regularization: prevents collapse to all-zeros
        # Forces model to predict same average occupancy as target
        pred_mean = pred.mean()
        target_mean = target.mean()
        mean_loss = F.mse_loss(pred_mean, target_mean)

        total_loss = focal + self.dice_weight * dice_loss + self.mean_weight * mean_loss

        # Store components for wandb logging
        self._last_components = {
            'focal': focal.item(),
            'dice': dice_loss.item(),
            'mean_match': mean_loss.item(),
            'intersection': intersection.item(),
            'pred_sum': pred_flat.sum().item(),
            'target_sum': target_flat.sum().item(),
        }

        # Debug: Print loss components every 100 batches
        if self._debug_counter % 100 == 1:
            print(f"  [LOSS DEBUG] focal: {focal.item():.6f}, dice: {dice_loss.item():.6f}, mean: {mean_loss.item():.6f}")
            print(f"  [LOSS DEBUG] total: {total_loss.item():.6f}")

        return total_loss


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, writer, use_wandb=False, is_6dof=False, dataset_type='unknown', scaler=None, debug_freq=500):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        history_occ = batch['history_occupancy'].to(device)
        future_occ = batch['future_occupancy'].to(device)
        history_poses = batch['history_poses'].to(device)
        future_poses = batch['future_poses'].to(device)

        # Forward pass with optional mixed precision
        optimizer.zero_grad()

        # Use autocast for mixed precision if scaler is provided
        use_amp = scaler is not None
        amp_context = torch.cuda.amp.autocast() if use_amp else torch.enable_grad()

        with amp_context:
            if is_6dof:
                # 6DoF model returns dict of outputs
                outputs = model(history_occ, history_poses, future_poses)
                pred_occ = outputs['future_occupancy']

                # 6DoF loss expects outputs and targets dicts
                targets = {
                    'future_occupancy': future_occ.float(),
                    'future_poses': future_poses,
                }
                losses = criterion(outputs, targets)
                loss = losses['total']

                # Get debug metrics from loss function
                debug_metrics = criterion.get_debug_metrics()
            else:
                # Simple model - can return dict or tensor depending on pose availability
                output = model(history_occ, history_poses, future_poses)

                if isinstance(output, dict):
                    # Model returned occupancy and poses
                    pred_occ = output['future_occupancy']
                    pred_poses = output.get('future_poses')

                    # Occupancy loss
                    occ_loss = criterion(pred_occ, future_occ.float())

                    # Add pose loss if poses predicted
                    if pred_poses is not None:
                        pose_loss = F.smooth_l1_loss(pred_poses, future_poses)
                        loss = occ_loss + 0.1 * pose_loss
                    else:
                        loss = occ_loss
                else:
                    # Legacy: model returned just occupancy tensor
                    pred_occ = output
                    loss = criterion(pred_occ, future_occ.float())

                debug_metrics = {}

        # Debug: Check occupancy rate and prediction distribution
        # Log at start and every debug_freq batches to track potential collapse
        if batch_idx % debug_freq == 0:
            occ_rate = (future_occ > 0).float().mean().item() * 100
            pred_mean = pred_occ.mean().item()
            pred_max = pred_occ.max().item()
            pred_min = pred_occ.min().item()
            print(f"  DEBUG [{batch_idx}] [{dataset_type}]: Occ: {occ_rate:.2f}%, Pred mean: {pred_mean:.4f}, min: {pred_min:.4f}, max: {pred_max:.4f}")
            # Check if pred and target accidentally share memory
            print(f"  DEBUG [{batch_idx}]: pred_occ.data_ptr={pred_occ.data_ptr()}, future_occ.data_ptr={future_occ.data_ptr()}")
            # Check if predictions are accidentally equal to targets
            target_float = future_occ.float()
            match_rate = (torch.abs(pred_occ - target_float) < 0.01).float().mean().item()
            exact_match = torch.allclose(pred_occ, target_float)
            print(f"  DEBUG [{batch_idx}]: match_rate (<0.01 diff): {match_rate*100:.2f}%, exact_match: {exact_match}")
            # Check if history and future occupancy are identical (static scene issue)
            hist_mean = history_occ.float().mean().item()
            fut_mean = future_occ.float().mean().item()
            hist_fut_match = torch.allclose(history_occ[:, -1].float(), future_occ[:, 0].float())
            print(f"  DEBUG [{batch_idx}]: hist_mean={hist_mean:.4f}, fut_mean={fut_mean:.4f}, last_hist==first_fut: {hist_fut_match}")

            # 6DoF specific debug info
            if is_6dof and debug_metrics:
                print(f"    6DoF: pose_std={debug_metrics.get('pose_pos_std', 0):.4f}, "
                      f"unc_mean={debug_metrics.get('uncertainty_mean', 0):.4f}, "
                      f"emb_std={debug_metrics.get('embedding_std', 0):.4f}")

            # Log prediction stats to wandb
            if use_wandb:
                log_dict = {
                    'debug/pred_mean': pred_mean,
                    'debug/pred_min': pred_min,
                    'debug/pred_max': pred_max,
                    'debug/occupancy_rate': occ_rate,
                    'debug/match_rate': match_rate,
                    'debug/exact_match': 1.0 if exact_match else 0.0,
                    'debug/hist_mean': hist_mean,
                    'debug/fut_mean': fut_mean,
                    'debug/hist_fut_match': 1.0 if hist_fut_match else 0.0,
                }
                # Add loss components from criterion
                if hasattr(criterion, 'get_loss_components'):
                    loss_components = criterion.get_loss_components()
                    for k, v in loss_components.items():
                        log_dict[f'loss_components/{k}'] = v
                # Add 6DoF metrics
                if is_6dof:
                    for k, v in debug_metrics.items():
                        log_dict[f'6dof/{k}'] = v
                    # Log individual loss components
                    for k, v in losses.items():
                        if k != 'total':
                            log_dict[f'loss/{k}'] = v.item()

                wandb.log(log_dict, commit=False)

        # Backward pass with optional mixed precision scaling
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Debug: Check if gradients exist and are non-zero
        if batch_idx % debug_freq == 0:
            total_grad_norm = 0
            num_params_with_grad = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
                    num_params_with_grad += 1
            total_grad_norm = total_grad_norm ** 0.5
            print(f"  DEBUG [{batch_idx}]: grad_norm={total_grad_norm:.6f}, params_with_grad={num_params_with_grad}")
            print(f"  DEBUG [{batch_idx}]: loss.requires_grad={loss.requires_grad}, loss.grad_fn={loss.grad_fn}")

            # Log gradient info to wandb
            if use_wandb:
                wandb.log({
                    'debug/grad_norm': total_grad_norm,
                    'debug/params_with_grad': num_params_with_grad,
                }, commit=False)

        # Gradient clipping and optimizer step with optional scaling
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Log
        if batch_idx % 10 == 0:
            loss_val = loss.item()
            loss_str = f"Loss: {loss_val:.6f}"  # More precision to see small losses
            if is_6dof:
                loss_str += f" (occ={losses['occ'].item():.3f}, pose={losses['pose'].item():.3f})"
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] {loss_str}")

        # TensorBoard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Train/Loss', loss.item(), global_step)
        if is_6dof:
            for k, v in losses.items():
                if k != 'total':
                    writer.add_scalar(f'Train/Loss_{k}', v.item(), global_step)

        # Weights & Biases
        if use_wandb:
            wandb.log({
                'train/loss': loss.item(),
                'train/global_step': global_step,
            })

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device, is_6dof=False):
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            history_occ = batch['history_occupancy'].to(device)
            future_occ = batch['future_occupancy'].to(device)
            history_poses = batch['history_poses'].to(device)
            future_poses = batch['future_poses'].to(device)

            if is_6dof:
                outputs = model(history_occ, history_poses, future_poses)
                targets = {
                    'future_occupancy': future_occ.float(),
                    'future_poses': future_poses,
                }
                losses = criterion(outputs, targets)
                loss = losses['total']
            else:
                output = model(history_occ, history_poses, future_poses)

                if isinstance(output, dict):
                    pred_occ = output['future_occupancy']
                    pred_poses = output.get('future_poses')
                    occ_loss = criterion(pred_occ, future_occ.float())
                    if pred_poses is not None:
                        pose_loss = F.smooth_l1_loss(pred_poses, future_poses)
                        loss = occ_loss + 0.1 * pose_loss
                    else:
                        loss = occ_loss
                else:
                    pred_occ = output
                    loss = criterion(pred_occ, future_occ.float())

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


def main():
    args = parse_args()

    print("=" * 60)
    print("OccWorld Training")
    print("=" * 60)

    # Setup
    device, work_dir = setup_environment(args)
    print(f"Device: {device}")
    print(f"Work directory: {work_dir}")

    # Load config
    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    # Try to use OccWorld library
    use_occworld, OccWorldModel = try_import_occworld()

    if args.use_occworld and not use_occworld:
        print("ERROR: --use-occworld specified but OccWorld library not found")
        print("Install OccWorld: git clone https://github.com/wzzheng/OccWorld.git ~/OccWorld")
        sys.exit(1)

    # Create dataset
    print("Creating dataset...")
    data_root = getattr(config, 'data_root', 'data/tokyo_gazebo')

    # Detect dataset type from config
    dataset_type = getattr(config, 'dataset_type', None)
    if dataset_type is None:
        # Auto-detect from config path or data_root
        if 'uavscenes' in str(args.config).lower():
            dataset_type = 'uavscenes'
        elif 'nuscenes_6dof' in str(args.config).lower():
            dataset_type = 'nuscenes_6dof'
        elif 'nuscenes' in str(args.config).lower() or 'nuscenes' in str(data_root).lower():
            dataset_type = 'nuscenes'
        else:
            dataset_type = 'gazebo'

    print(f"Dataset type: {dataset_type}")

    # Auto-detect 6DoF model for aerial/6DoF datasets
    SIXDOF_DATASETS = ['uavscenes', 'nuscenes_6dof']
    if dataset_type in SIXDOF_DATASETS and args.model_type != '6dof':
        print(f"\n[AUTO-DETECT] Dataset '{dataset_type}' requires 6DoF model.")
        print(f"              Switching from --model-type '{args.model_type}' to '6dof'")
        args.model_type = '6dof'

    # Validate data before creating dataset
    # This will auto-download from S3 if data is missing (unless --no-auto-download)
    if not args.skip_validation:
        auto_download = not args.no_auto_download
        if not validate_data(data_root, dataset_type, auto_download=auto_download):
            print("\n" + "=" * 60)
            print("DATA VALIDATION FAILED")
            print("=" * 60)
            print("Training cannot proceed without valid data.")
            print("\nTo download data manually:")
            print(f"  aws s3 sync s3://{S3_BUCKET}/ . --region {S3_REGION}")
            print("\nOr run the download script:")
            print("  python scripts/download_pretrained.py --all")
            print("\nTo skip validation (use with caution):")
            print("  python train.py --skip-validation ...")
            print("=" * 60)
            sys.exit(1)
    else:
        print("[WARN] Skipping data validation (--skip-validation)")

    if dataset_type == 'uavscenes':
        # UAVScenes - Real aerial 6DoF data from UAV platform
        if not HAS_UAVSCENES:
            raise ImportError(
                "UAVScenes dataset requested but uavscenes_dataset.py not found.\n"
                "Install: pip install pyquaternion open3d"
            )

        ds_cfg = getattr(config, 'dataset_config', {})

        uavscenes_cfg = UAVScenesConfig(
            scenes=ds_cfg.get('scenes', ['AMtown', 'AMvalley', 'HKairport', 'HKisland']),
            interval=ds_cfg.get('interval', 1),
            history_frames=getattr(config, 'history_frames', 4),
            future_frames=getattr(config, 'future_frames', 6),
            frame_skip=ds_cfg.get('frame_skip', 1),
            point_cloud_range=tuple(getattr(config, 'point_cloud_range', (-40, -40, -10, 40, 40, 50))),
            voxel_size=tuple(getattr(config, 'voxel_size', (0.4, 0.4, 0.5))),
            ego_frame=ds_cfg.get('ego_frame', True),
            fallback_to_lidar_center=ds_cfg.get('fallback_to_lidar_center', True),
            min_in_range_ratio=ds_cfg.get('min_in_range_ratio', 0.01),
            val_ratio=ds_cfg.get('val_ratio', 0.1),
            test_ratio=ds_cfg.get('test_ratio', 0.1),
            split='train',
        )
        train_dataset = UAVScenesDataset(data_root, uavscenes_cfg)

        val_uavscenes_cfg = UAVScenesConfig(
            scenes=uavscenes_cfg.scenes,
            interval=uavscenes_cfg.interval,
            history_frames=uavscenes_cfg.history_frames,
            future_frames=uavscenes_cfg.future_frames,
            frame_skip=uavscenes_cfg.frame_skip,
            point_cloud_range=uavscenes_cfg.point_cloud_range,
            voxel_size=uavscenes_cfg.voxel_size,
            ego_frame=uavscenes_cfg.ego_frame,
            fallback_to_lidar_center=uavscenes_cfg.fallback_to_lidar_center,
            min_in_range_ratio=uavscenes_cfg.min_in_range_ratio,
            val_ratio=uavscenes_cfg.val_ratio,
            test_ratio=uavscenes_cfg.test_ratio,
            split='val',
        )
        val_dataset = UAVScenesDataset(data_root, val_uavscenes_cfg)
        dataset_collate_fn = uavscenes_collate_fn

        print(f"UAVScenes scenes: {uavscenes_cfg.scenes}")

    elif dataset_type == 'nuscenes_6dof':
        # nuScenes with 6DoF augmentation (for aerial world model training)
        if not HAS_NUSCENES_6DOF:
            raise ImportError(
                "nuScenes 6DoF dataset requested but nuscenes_6dof_dataset.py not found.\n"
                "Install: pip install nuscenes-devkit pyquaternion"
            )

        # Get config from file or use defaults
        ds_cfg = getattr(config, 'dataset_config', {})

        nuscenes_6dof_cfg = NuScenes6DoFConfig(
            version=ds_cfg.get('version', 'v1.0-mini'),
            history_frames=getattr(config, 'history_frames', 4),
            future_frames=getattr(config, 'future_frames', 6),
            frame_skip=ds_cfg.get('frame_skip', 1),
            point_cloud_range=tuple(getattr(config, 'point_cloud_range', (-40, -40, -1, 40, 40, 5.4))),
            voxel_size=tuple(getattr(config, 'voxel_size', (0.4, 0.4, 0.4))),
            # 6DoF augmentation settings
            augment_6dof=ds_cfg.get('augment_6dof', True),
            max_pitch_deg=ds_cfg.get('max_pitch_deg', 30.0),
            max_roll_deg=ds_cfg.get('max_roll_deg', 45.0),
            max_yaw_deg=ds_cfg.get('max_yaw_deg', 180.0),
            altitude_shift_range=ds_cfg.get('altitude_shift_range', (-2.0, 10.0)),
            consistent_augmentation=ds_cfg.get('consistent_augmentation', True),
            augmentation_prob=ds_cfg.get('augmentation_prob', 0.8),
            split='train',
        )
        train_dataset = NuScenes6DoFDataset(data_root, nuscenes_6dof_cfg)

        # Validation: no augmentation for fair evaluation
        val_6dof_cfg = NuScenes6DoFConfig(
            version=nuscenes_6dof_cfg.version,
            history_frames=nuscenes_6dof_cfg.history_frames,
            future_frames=nuscenes_6dof_cfg.future_frames,
            frame_skip=nuscenes_6dof_cfg.frame_skip,
            point_cloud_range=nuscenes_6dof_cfg.point_cloud_range,
            voxel_size=nuscenes_6dof_cfg.voxel_size,
            augment_6dof=False,  # No augmentation for validation
            split='val',
        )
        val_dataset = NuScenes6DoFDataset(data_root, val_6dof_cfg)
        dataset_collate_fn = nuscenes_6dof_collate_fn

        print(f"6DoF augmentation: pitch=±{nuscenes_6dof_cfg.max_pitch_deg}°, "
              f"roll=±{nuscenes_6dof_cfg.max_roll_deg}°")

    elif dataset_type == 'nuscenes':
        if not HAS_NUSCENES:
            raise ImportError("nuScenes dataset requested but nuscenes_occworld_dataset.py not found. "
                            "Run ./scripts/setup_training_data.sh --nuscenes first.")

        # Create nuScenes dataset
        nuscenes_cfg = NuScenesConfig(
            version=getattr(config, 'dataset_config', {}).get('version', 'v1.0-mini'),
            history_frames=getattr(config, 'history_frames', 4),
            future_frames=getattr(config, 'future_frames', 6),
            point_cloud_range=tuple(getattr(config, 'point_cloud_range', (-40, -40, -1, 40, 40, 5.4))),
            voxel_size=tuple(getattr(config, 'voxel_size', (0.4, 0.4, 0.4))),
            split='train',
        )
        train_dataset = NuScenesOccWorldDataset(data_root, nuscenes_cfg)

        val_nuscenes_cfg = NuScenesConfig(
            version=nuscenes_cfg.version,
            history_frames=nuscenes_cfg.history_frames,
            future_frames=nuscenes_cfg.future_frames,
            point_cloud_range=nuscenes_cfg.point_cloud_range,
            voxel_size=nuscenes_cfg.voxel_size,
            split='val',
        )
        val_dataset = NuScenesOccWorldDataset(data_root, val_nuscenes_cfg)
        dataset_collate_fn = nuscenes_collate_fn
    else:
        # Create Gazebo dataset
        ds_config = getattr(config, 'dataset_config', {})
        dataset_cfg = DatasetConfig(
            history_frames=getattr(config, 'history_frames', 4),
            future_frames=getattr(config, 'future_frames', 6),
            frame_skip=getattr(config, 'frame_skip', 1),
            agent_type=ds_config.get('agent_type', 'both'),
            split='train',
            val_ratio=ds_config.get('val_ratio', 0.1),
            test_ratio=ds_config.get('test_ratio', 0.1),
            exclude_dummy_sessions=ds_config.get('exclude_dummy_sessions', True),
            point_cloud_range=getattr(config, 'point_cloud_range', (-40, -40, -2, 40, 40, 150)),
            voxel_size=getattr(config, 'voxel_size', (0.4, 0.4, 1.25)),
        )

        train_dataset = GazeboOccWorldDataset(data_root, dataset_cfg)

        # Check for empty dataset and provide diagnostic help
        if len(train_dataset) == 0:
            print("\n" + "="*70)
            print("ERROR: No training samples found!")
            print("="*70)
            print(f"\nData root: {data_root}")
            print(f"Absolute path: {os.path.abspath(data_root)}")

            if not os.path.exists(data_root):
                print(f"\n[PROBLEM] Directory does not exist: {data_root}")
                print("\n[FIX] Create the data directory and add training data:")
                print(f"  mkdir -p {data_root}")
            else:
                # Check what's in the directory
                contents = os.listdir(data_root)
                print(f"\nDirectory contents ({len(contents)} items):")
                for item in contents[:10]:
                    item_path = os.path.join(data_root, item)
                    if os.path.isdir(item_path):
                        subdirs = os.listdir(item_path) if os.path.isdir(item_path) else []
                        print(f"  {item}/ -> {subdirs[:5]}{'...' if len(subdirs) > 5 else ''}")
                    else:
                        print(f"  {item}")
                if len(contents) > 10:
                    print(f"  ... and {len(contents) - 10} more")

                print("\n[REQUIRED] Each session directory must have these subdirectories:")
                print("  session_name/")
                print("  ├── images/      (camera images)")
                print("  ├── lidar/       (point cloud .npy files)")
                print("  ├── poses/       (pose .json files)")
                print("  └── occupancy/   (occupancy .npz files)")

                print("\n[POSSIBLE FIXES]")
                print("  1. Check that data was downloaded/generated correctly")
                print("  2. Verify the data_root path in your config file")
                print("  3. If data exists but is being filtered, try adding to config:")
                print("     dataset_config = {'filter_static': False}")
                if dataset_cfg.exclude_dummy_sessions:
                    print("  4. Dummy sessions are excluded - check if all sessions have 'dummy' in name")

            print("\n" + "="*70)
            sys.exit(1)

        val_cfg = DatasetConfig(**vars(dataset_cfg))
        val_cfg.split = 'val'
        val_dataset = GazeboOccWorldDataset(data_root, val_cfg)

        # Warn if validation set is empty (not fatal, but noteworthy)
        if len(val_dataset) == 0:
            print("\nWARNING: Validation dataset is empty. Training will continue without validation.")
            print("This may happen if val_ratio is too small or all data went to training split.\n")

        dataset_collate_fn = collate_fn

    # Create dataloaders
    batch_size = args.batch_size or getattr(config, 'data', {}).get('samples_per_gpu', 1)

    # Data loading workers - use --num-workers to control (default: 4)
    num_workers = args.num_workers
    pin_memory = num_workers > 0 and torch.cuda.is_available()
    print(f"Using {num_workers} data loading workers, pin_memory={pin_memory}")

    # Limit PyTorch threads when using multiprocessing workers
    if num_workers > 0:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
    else:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    # Validate dataset has samples before creating DataLoader
    if len(train_dataset) == 0:
        raise ValueError(
            f"Training dataset has 0 samples!\n"
            f"  Dataset type: {dataset_type}\n"
            f"  Data root: {data_root}\n"
            f"Please verify:\n"
            f"  1. Data exists at the specified path\n"
            f"  2. Data follows the expected directory structure\n"
            f"  3. LiDAR files (*.txt, *.bin, *.pcd) exist in the scene folders"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=dataset_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset_collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Print dataset info banner
    print("\n" + "=" * 60)
    print("DATASET CONFIGURATION")
    print("=" * 60)
    dataset_descriptions = {
        'uavscenes': 'UAVScenes - Real aerial UAV data (LiDAR + Camera, 6DoF poses)',
        'nuscenes': 'nuScenes - Real autonomous driving data',
        'nuscenes_6dof': 'nuScenes 6DoF - Driving data with geometric augmentation',
        'gazebo': 'Tokyo PLATEAU/Gazebo - Synthetic city simulation data',
    }
    print(f"  Dataset:      {dataset_type}")
    print(f"  Description:  {dataset_descriptions.get(dataset_type, 'Custom dataset')}")
    print(f"  Data root:    {data_root}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Grid size:     {getattr(config, 'grid_size', 'default')}")
    if dataset_type == 'uavscenes':
        print(f"  Scenes:       {getattr(config, 'dataset_config', {}).get('scenes', 'all')}")
    print("=" * 60 + "\n")

    # Create model
    print(f"Creating model (type: {args.model_type})...")
    if use_occworld and args.use_occworld:
        # Use OccWorld model
        model_cfg = getattr(config, 'model', {})
        model = OccWorldModel(**model_cfg)
    elif args.model_type == '6dof':
        # Use 6DoF model with full pose prediction
        grid_size = tuple(getattr(config, 'grid_size', [200, 200, 121]))
        model_config = OccWorld6DoFConfig(
            grid_size=grid_size,
            history_frames=getattr(config, 'history_frames', 4),
            future_frames=getattr(config, 'future_frames', 6),
            use_transformer=args.use_transformer,
            pose_dim=13,  # x,y,z + quat(4) + lin_vel(3) + ang_vel(3)
            enable_uncertainty=True,
            enable_relocalization=True,
            enable_place_recognition=True,
        )
        model = OccWorld6DoF(model_config)
        print(f"  6DoF Config: grid={grid_size}, transformer={args.use_transformer}")
    else:
        # Use simple standalone model (occupancy only)
        model = SimpleOccupancyModel(config)

    model = model.to(device)

    # Load pretrained weights or resume checkpoint
    resume_checkpoint = None
    if not args.from_scratch:
        load_from = getattr(config, 'load_from', None)
        vqvae_ckpt = getattr(config, 'vqvae_ckpt', None)

        if args.resume_from:
            load_from = args.resume_from
            resume_checkpoint = load_from
        elif args.resume:
            # Find latest checkpoint
            ckpts = sorted((work_dir / 'checkpoints').glob('epoch_*.pth'))
            if ckpts:
                load_from = str(ckpts[-1])
                resume_checkpoint = load_from

        # Validate pretrained models exist (auto-download from S3 if missing)
        if (load_from or vqvae_ckpt) and not args.skip_validation:
            auto_download = not args.no_auto_download
            if not validate_pretrained_models(load_from, vqvae_ckpt, auto_download=auto_download):
                print("\n" + "=" * 60)
                print("PRETRAINED MODEL VALIDATION FAILED")
                print("=" * 60)
                print("Training cannot proceed without required pretrained models.")
                print("\nOptions:")
                print("  1. Download models: python scripts/download_pretrained.py")
                print("  2. Train from scratch: python train.py --from-scratch ...")
                print("  3. Skip validation: python train.py --skip-validation ...")
                print("=" * 60)
                sys.exit(1)

        if load_from and os.path.exists(load_from):
            print(f"Loading weights from: {load_from}")
            checkpoint = torch.load(load_from, map_location=device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

    # Optional: Compile model for faster training (PyTorch 2.0+)
    if args.compile:
        if hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            model = torch.compile(model)
            print("  Model compiled successfully")
        else:
            print("WARNING: --compile requires PyTorch 2.0+, skipping")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Mixed precision scaler (for --amp)
    scaler = None
    if args.amp:
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision (AMP) enabled - using FP16/BF16 for faster training")
        else:
            print("WARNING: --amp requires CUDA, skipping")

    # Optimizer
    lr = args.lr or getattr(config, 'optimizer', {}).get('lr', 1e-4)
    weight_decay = getattr(config, 'optimizer', {}).get('weight_decay', 0.01)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    max_epochs = args.epochs or getattr(config, 'max_epochs', 50)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    # Loss function - depends on model type
    if args.model_type == '6dof':
        # 6DoF loss with anti-collapse safeguards
        criterion = OccWorld6DoFLoss(
            occ_weight=1.0,
            pose_weight=0.5,
            uncertainty_weight=0.1,
            reloc_weight=0.2,
            place_weight=0.1,
            focal_alpha=0.99,
            focal_gamma=2.0,
            dice_weight=1.0,
            mean_weight=10.0,
            pose_variance_weight=1.0,
            min_pose_std=0.01,
        )
        print("  Using OccWorld6DoFLoss with anti-collapse safeguards")
    else:
        # Simple occupancy loss - Focal + Dice + Mean-matching
        # focal_alpha=0.99 for ~1% occupancy, mean_weight=10 prevents all-zero collapse
        criterion = OccupancyLoss(
            focal_alpha=0.99,
            focal_gamma=2.0,
            dice_weight=1.0,
            mean_weight=10.0,
        )

    # TensorBoard
    writer = SummaryWriter(log_dir=str(work_dir / 'logs'))

    if args.eval_only:
        is_6dof = args.model_type == '6dof'
        val_loss = validate(model, val_loader, criterion, device, is_6dof)
        print(f"Eval-only: Val Loss = {val_loss:.6f}")
        writer.close()
        return

    # Weights & Biases
    use_wandb = args.wandb and HAS_WANDB
    if args.wandb and not HAS_WANDB:
        print("WARNING: --wandb specified but wandb not installed. Run: pip install wandb")

    if use_wandb:
        # Check if wandb is logged in, prompt for API key if not
        if not wandb.api.api_key:
            print("\n" + "=" * 60)
            print("Weights & Biases Login Required")
            print("=" * 60)
            print("Get your API key from: https://wandb.ai/authorize")
            print("")
            api_key = input("Enter your W&B API key (or press Enter to skip W&B): ").strip()
            if api_key:
                wandb.login(key=api_key)
                print("W&B login successful!")
            else:
                print("Skipping W&B logging.")
                use_wandb = False

    if use_wandb:
        wandb_config = {
            # Data
            'data_root': data_root,
            'dataset_type': dataset_type,
            'batch_size': batch_size,
            'history_frames': getattr(config, 'history_frames', 4),
            'future_frames': getattr(config, 'future_frames', 6),
            'grid_size': getattr(config, 'grid_size', [200, 200, 121]),
            'point_cloud_range': getattr(config, 'point_cloud_range', None),
            'voxel_size': getattr(config, 'voxel_size', None),
            # Model
            'model_type': 'OccWorld6DoF' if args.model_type == '6dof' else (
                'TransVQVAE' if (use_occworld and args.use_occworld) else 'SimpleOccupancyModel'
            ),
            'num_params': num_params,
            'use_transformer': args.use_transformer if args.model_type == '6dof' else False,
            # Training
            'lr': lr,
            'weight_decay': weight_decay,
            'max_epochs': max_epochs,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
            # Loss
            'loss_type': 'OccWorld6DoFLoss' if args.model_type == '6dof' else 'Focal+Dice+MeanMatch',
            'focal_alpha': 0.99,
            'focal_gamma': 2.0,
            'dice_weight': 1.0,
            'mean_weight': 10.0,
            # Hardware
            'device': str(device),
            'gpu_ids': args.gpu_ids,
        }

        run_name = args.wandb_run_name or f"occworld_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=wandb_config,
            tags=args.wandb_tags + [dataset_type, 'focal-loss'],
            dir=str(work_dir),
        )
        wandb.watch(model, log='gradients', log_freq=100)
        print(f"W&B run: {wandb.run.url}")

    # Resume optimizer/scheduler state if resuming from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            print(f"Resuming from epoch {start_epoch}")
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("  Restored optimizer state")
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("  Restored scheduler state")
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']

    # Training loop
    print("=" * 60)
    print(f"Starting training on [{dataset_type.upper()}] for {max_epochs} epochs (from epoch {start_epoch})")
    print("=" * 60)

    is_6dof = args.model_type == '6dof'

    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                  device, epoch, writer, use_wandb, is_6dof, dataset_type,
                                  scaler=scaler, debug_freq=args.debug_freq)

        # Validate
        val_loss = validate(model, val_loader, criterion, device, is_6dof)

        # Update scheduler
        scheduler.step()

        # Log
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}/{max_epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"LR: {current_lr:.6f}, Time: {epoch_time:.1f}s")

        writer.add_scalar('Train/EpochLoss', train_loss, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Train/LR', current_lr, epoch)

        # Weights & Biases epoch logging
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'epoch/train_loss': train_loss,
                'epoch/val_loss': val_loss,
                'epoch/lr': current_lr,
                'epoch/time_seconds': epoch_time,
            })

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == max_epochs - 1:
            ckpt_path = work_dir / 'checkpoints' / f'epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

            # Log checkpoint as wandb artifact
            if use_wandb:
                artifact = wandb.Artifact(
                    f'model-checkpoint-epoch{epoch+1}',
                    type='model',
                    metadata={'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss}
                )
                artifact.add_file(str(ckpt_path))
                wandb.log_artifact(artifact)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = work_dir / 'checkpoints' / 'best.pth'
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            print(f"  New best model! Val Loss: {val_loss:.4f}")

            # Log best model as wandb artifact
            if use_wandb:
                best_artifact = wandb.Artifact(
                    'model-best',
                    type='model',
                    metadata={'epoch': epoch + 1, 'val_loss': val_loss}
                )
                best_artifact.add_file(str(best_path))
                wandb.log_artifact(best_artifact)

    writer.close()

    # Finish wandb run
    if use_wandb:
        wandb.finish()

    print("=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {work_dir / 'checkpoints'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
