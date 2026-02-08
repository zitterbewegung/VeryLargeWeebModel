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
import signal
import argparse
import contextlib
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
# Graceful Shutdown Handler
# =============================================================================

class TrainingState:
    """Global training state for graceful shutdown."""
    model = None
    optimizer = None
    scheduler = None
    epoch = 0
    work_dir = None
    best_val_loss = float('inf')
    should_stop = False

    @classmethod
    def save_emergency_checkpoint(cls, signum=None, frame=None):
        """Save checkpoint on SIGTERM/SIGINT for graceful shutdown."""
        sig_name = signal.Signals(signum).name if signum else 'unknown'
        print(f"\n  [SIGNAL] Received {sig_name}, saving emergency checkpoint...")
        if cls.model is not None and cls.work_dir is not None:
            ckpt_path = Path(cls.work_dir) / 'checkpoints' / 'emergency.pth'
            try:
                raw_sd = cls.model.state_dict()
                clean_sd = {}
                for k, v in raw_sd.items():
                    k = k.replace('_orig_mod.', '').replace('module.', '', 1)
                    clean_sd[k] = v
                tmp_path = ckpt_path.with_suffix('.pth.tmp')
                torch.save({
                    'epoch': cls.epoch,
                    'state_dict': clean_sd,
                    'optimizer': cls.optimizer.state_dict() if cls.optimizer else None,
                    'scheduler': cls.scheduler.state_dict() if cls.scheduler else None,
                }, tmp_path)
                tmp_path.replace(ckpt_path)
                print(f"  [SIGNAL] Emergency checkpoint saved: {ckpt_path}")
            except Exception as e:
                print(f"  [SIGNAL] Failed to save emergency checkpoint: {e}")
        cls.should_stop = True


def cleanup_old_checkpoints(ckpt_dir: Path, max_keep: int):
    """Delete old epoch checkpoints, keeping the N most recent + best.pth."""
    if max_keep < 0:
        return  # -1 = keep all
    epoch_ckpts = sorted(ckpt_dir.glob('epoch_*.pth'), key=lambda p: p.stat().st_mtime)
    # Always keep best.pth and emergency.pth
    to_delete = epoch_ckpts[:-max_keep] if len(epoch_ckpts) > max_keep else []
    for ckpt in to_delete:
        ckpt.unlink()
        print(f"  Cleaned up old checkpoint: {ckpt.name}")


# =============================================================================
# Data Validation
# =============================================================================

S3_BUCKET = "verylargeweebmodel"
S3_REGION = "us-west-2"
UAVSCENES_HF_REPO = "sijieaaa/UAVScenes"


def validate_data(data_root: str, dataset_type: str, auto_download: bool = True, interval: int = 1) -> bool:
    """
    Validate that required training data exists and is complete.

    Args:
        data_root: Path to data directory
        dataset_type: Type of dataset (gazebo, nuscenes, uavscenes, etc.)
        auto_download: If True, attempt to download missing data from S3
        interval: UAVScenes interval (1=full, 5=keyframes)

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
            print(f"\n[AUTO-DOWNLOAD] Creating directory and downloading...")
            return _download_dataset_from_s3(data_root, dataset_type, interval=interval)
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
            print(f"\n[AUTO-DOWNLOAD] Attempting to download missing data...")
            if _download_dataset_from_s3(data_root, dataset_type, interval=interval):
                # Re-validate after download
                return validate_data(data_root, dataset_type, auto_download=False, interval=interval)

        print("\nTo fix manually:")
        if dataset_type == 'uavscenes':
            print("  pip install huggingface_hub")
            print(f"  python -c \"from huggingface_hub import snapshot_download; snapshot_download('{UAVSCENES_HF_REPO}', repo_type='dataset', local_dir='{data_root}')\"")
            print("  # Or via S3:")
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
        has_poses = (session / 'poses').exists() and any((session / 'poses').glob('*.json'))

        if has_occupancy and has_poses:
            valid_sessions += 1
        else:
            missing = []
            if not has_occupancy:
                missing.append('occupancy')
            if not has_poses:
                missing.append('poses')
            warnings.append(f"Session {session.name} missing {', '.join(missing)} data")

    if valid_sessions == 0:
        issues.append("No valid sessions with required occupancy + poses data")
        return False

    print(f"  Found {valid_sessions}/{len(sessions)} valid sessions")
    return True


def _validate_nuscenes_data(data_path: Path, issues: list, warnings: list) -> bool:
    """Validate nuScenes dataset structure."""
    # Check for nuScenes raw data directories (used by raw-data loader)
    nuscenes_dirs = ['v1.0-mini', 'v1.0-trainval', 'samples', 'sweeps']
    has_raw_data = False
    for d in nuscenes_dirs:
        if (data_path / d).exists():
            has_raw_data = True
            print(f"  Found nuScenes directory: {d}")

    # Check for OccWorld-style pickle files (optional if raw data exists)
    pkl_files = [
        'nuscenes_infos_train_temporal_v3_scene.pkl',
        'nuscenes_infos_val_temporal_v3_scene.pkl',
    ]
    has_pkls = True
    for req_file in pkl_files:
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
            if has_raw_data:
                # Raw loader doesn't need pkl files — just warn
                warnings.append(f"Missing {req_file} (not needed for raw-data loader)")
            else:
                issues.append(f"Missing: {req_file}")
            has_pkls = False

    if not has_raw_data and not has_pkls:
        issues.append("No nuScenes data found (need raw directories or pickle files)")

    return len(issues) == 0


def _validate_uavscenes_data(data_path: Path, issues: list, warnings: list) -> bool:
    """Validate UAVScenes dataset structure."""
    # Expected structure (after HF zip extraction):
    # data_root/
    #   interval5_CAM_LIDAR/          <- zip wrapper folder
    #     interval5_AMtown01/
    #       interval5_LIDAR/*.txt
    #       interval5_CAM/*.jpg
    #       sampleinfos_interpolated.json
    # OR (flat):
    #   interval5_AMtown01/
    #     interval5_LIDAR/*.txt

    scenes = ['AMtown', 'AMvalley', 'HKairport', 'HKisland']
    found_scenes = []

    # Search both data_root and inside interval{N}_CAM_LIDAR/ wrapper
    search_roots = [data_path]
    for wrapper in data_path.iterdir():
        if wrapper.is_dir() and wrapper.name.startswith('interval') and 'CAM_LIDAR' in wrapper.name:
            search_roots.append(wrapper)

    for scene in scenes:
        scene_found = False
        for search_root in search_roots:
            if scene_found:
                break
            # Check dirs that contain the scene name (handles GNSS variants etc.)
            for d in sorted(search_root.iterdir()) if search_root.is_dir() else []:
                if not d.is_dir() or scene.lower() not in d.name.lower():
                    continue
                # Check for LiDAR data
                for lidar_name in ['interval5_LIDAR', 'interval1_LIDAR', 'lidar']:
                    lidar_dir = d / lidar_name
                    if lidar_dir.exists():
                        lidar_files = list(lidar_dir.glob('*.txt')) + list(lidar_dir.glob('*.pcd'))
                        if lidar_files:
                            print(f"  Found {scene} ({d.name}): {len(lidar_files)} LiDAR files")
                            if scene not in found_scenes:
                                found_scenes.append(scene)
                            scene_found = True
                            break

    if not found_scenes:
        issues.append("No UAVScenes data found. Expected interval5_CAM_LIDAR/interval5_*/interval5_LIDAR/*.txt")
        return False

    print(f"  Found {len(found_scenes)}/4 scenes: {found_scenes}")
    return True


def _download_dataset_from_s3(data_root: str, dataset_type: str, interval: int = 1) -> bool:
    """Download dataset from S3 (or HuggingFace for UAVScenes)."""
    import subprocess

    # For UAVScenes, try HuggingFace first (no credentials needed)
    if dataset_type == 'uavscenes':
        if _download_uavscenes_from_hf(data_root, interval=interval):
            return True
        print("[INFO] Falling back to S3 download...")

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


def _download_uavscenes_from_hf(data_root: str, interval: int = 5) -> bool:
    """Download UAVScenes dataset from HuggingFace Hub and extract zips.

    Args:
        data_root: Path to data directory (files saved to data_root/ directly)
        interval: UAVScenes interval (5=keyframes available on HF, 1=not on HF).

    Returns:
        True if download succeeded, False otherwise.
    """
    import zipfile

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[WARN] huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False

    # HF repo only has interval5_* files
    if interval not in (1, 5):
        print(f"[WARN] Unknown interval {interval}, using 5")
        interval = 5

    try:
        print(f"Downloading UAVScenes from HuggingFace ({UAVSCENES_HF_REPO})...")
        # Download the LiDAR+CAM zip (has both LiDAR point clouds and camera images)
        pattern = f"interval{interval}_CAM_LIDAR.zip"
        print(f"  Downloading: {pattern}")
        snapshot_download(
            repo_id=UAVSCENES_HF_REPO,
            repo_type="dataset",
            local_dir=data_root,
            allow_patterns=[pattern],
        )

        # Extract downloaded zips
        zip_path = Path(data_root) / pattern
        if zip_path.exists():
            print(f"  Extracting {pattern}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(data_root)
            print(f"  Extracted to {data_root}")
        else:
            print(f"[WARN] Expected zip not found: {zip_path}")
            return False

        # Also download the label zip (semantic labels)
        label_pattern = f"interval{interval}_LIDAR_label.zip"
        print(f"  Downloading: {label_pattern}")
        try:
            snapshot_download(
                repo_id=UAVSCENES_HF_REPO,
                repo_type="dataset",
                local_dir=data_root,
                allow_patterns=[label_pattern],
            )
            label_zip = Path(data_root) / label_pattern
            if label_zip.exists():
                print(f"  Extracting {label_pattern}...")
                with zipfile.ZipFile(label_zip, 'r') as zf:
                    zf.extractall(data_root)
        except Exception:
            print(f"  [INFO] Label download optional, continuing without it")

        print("[OK] UAVScenes download complete!")
        return True
    except Exception as e:
        print(f"[WARN] HuggingFace download failed: {e}")
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
    parser.add_argument('--amp-dtype', type=str, default=None, choices=['float16', 'bfloat16'],
                        help='AMP dtype (float16 or bfloat16). Auto-detected if not set.')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() for faster training (PyTorch 2.0+)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Data loading workers (default: 1, increase for faster loading)')
    parser.add_argument('--debug-freq', type=int, default=500,
                        help='Debug print frequency (default: 500, higher = faster)')
    parser.add_argument('--save-freq', type=int, default=1,
                        help='Checkpoint save frequency in epochs (default: 1 = every epoch)')
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='Gradient accumulation steps (default: 1 = no accumulation)')
    parser.add_argument('--max-keep-ckpts', type=int, default=5,
                        help='Max checkpoints to keep (default: 5, -1 = keep all)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='LR warmup epochs (default: 5)')
    parser.add_argument('--strict-load', action='store_true',
                        help='Use strict=True when loading checkpoint weights (fail on any mismatch)')

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
    parser.add_argument('--interval', type=int, default=None,
                        help='UAVScenes interval (1=full, 5=keyframes)')
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
        # Try importing from installed OccWorld (configurable via OCCWORLD_PATH env var)
        occworld_path = os.environ.get('OCCWORLD_PATH', os.path.expanduser('~/OccWorld'))
        # Use importlib to avoid shadowing by our local 'models' package
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "occworld_models", os.path.join(occworld_path, "models", "__init__.py"))
        if spec is None or spec.loader is None:
            raise ImportError("OccWorld models package not found")
        occworld_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(occworld_models)
        TransVQVAE = getattr(occworld_models, 'TransVQVAE')
        print(f"Using OccWorld library from {occworld_path}")
        return True, TransVQVAE
    except (ImportError, FileNotFoundError, AttributeError, OSError):
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
        # GroupNorm: works with batch_size=1 and multi-GPU (no batch statistics)
        self.encoder = nn.Sequential(
            nn.Conv3d(self.history_frames, 64, kernel_size=3, padding=1),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
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
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 64),
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

                # Position: additive residual
                new_pos = last_pose[..., :3] + pose_delta[..., :3]

                # Orientation: quaternion multiplication (proper group operation)
                from models.occworld_6dof import quaternion_multiply
                delta_quat = F.normalize(pose_delta[..., 3:7], p=2, dim=-1)
                current_quat = F.normalize(last_pose[..., 3:7], p=2, dim=-1)
                new_quat = quaternion_multiply(current_quat, delta_quat)
                new_quat = F.normalize(new_quat, p=2, dim=-1)

                # Velocity: additive residual
                new_vel = last_pose[..., 7:] + pose_delta[..., 7:]

                current_pose = torch.cat([new_pos, new_quat, new_vel], dim=-1)
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

    def __init__(self, focal_alpha=0.99, focal_gamma=2.0, dice_weight=1.0, mean_weight=10.0, smooth=1.0, temporal_decay=0.0):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_weight = dice_weight
        self.mean_weight = mean_weight
        self.smooth = smooth
        self.temporal_decay = temporal_decay
        self._debug_counter = 0
        # Store last loss components for logging
        self._last_components = {}

    def get_loss_components(self):
        """Return the last computed loss components for logging."""
        return self._last_components

    def _weighted_focal(self, pred, target):
        """Compute focal loss with optional temporal decay weights.

        When temporal_decay > 0 and input has a time dimension (ndim >= 5),
        earlier timesteps get higher weight via exponential decay:
            w_t = exp(-decay * t), normalized so mean weight = 1.
        """
        if self.temporal_decay > 0 and pred.ndim >= 5:
            # pred shape: [B, T, X, Y, Z] — dim 1 is time
            T = pred.shape[1]
            t_indices = torch.arange(T, device=pred.device, dtype=pred.dtype)
            weights = torch.exp(-self.temporal_decay * t_indices)
            weights = weights / weights.mean()  # normalize so mean = 1
            # Reshape for broadcasting: [1, T, 1, 1, 1]
            weights = weights.view(1, T, *([1] * (pred.ndim - 2)))

            # Compute per-element focal loss and apply weights
            pred_c = pred.clamp(min=1e-7, max=1 - 1e-7)
            bce = -target * torch.log(pred_c) - (1 - target) * torch.log(1 - pred_c)
            p_t = torch.where(target == 1, pred_c, 1 - pred_c)
            focal_weight = (1 - p_t) ** self.focal_loss.gamma
            alpha_weight = torch.where(target == 1, self.focal_loss.alpha, 1 - self.focal_loss.alpha)
            focal_per_elem = alpha_weight * focal_weight * bce
            return (focal_per_elem * weights).mean()
        else:
            return self.focal_loss(pred, target)

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
        focal = self._weighted_focal(pred, target)

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


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, writer, use_wandb=False, is_6dof=False, dataset_type='unknown', scaler=None, debug_freq=500, amp_dtype=None, grad_accum_steps=1):
    """Train for one epoch with optional mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0
    num_batches = 0

    # Occupancy rate tracking
    epoch_pred_occ_sum = 0.0
    epoch_target_occ_sum = 0.0
    epoch_voxel_count = 0

    for batch_idx, batch in enumerate(dataloader):
        # Only zero gradients at accumulation boundaries
        if batch_idx % grad_accum_steps == 0:
            optimizer.zero_grad()

        # Validate batch has required keys
        required_keys = ['history_occupancy', 'future_occupancy', 'history_poses', 'future_poses']
        missing = [k for k in required_keys if k not in batch]
        if missing:
            print(f"  [WARN] Batch {batch_idx} missing keys: {missing}, skipping")
            continue

        # Move to device
        history_occ = batch['history_occupancy'].to(device)
        future_occ = batch['future_occupancy'].to(device)
        history_poses = batch['history_poses'].to(device)
        future_poses = batch['future_poses'].to(device)

        # Validate tensor shapes
        if history_occ.ndim < 3 or future_occ.ndim < 3:
            print(f"  [WARN] Batch {batch_idx} bad occupancy ndim: "
                  f"history={history_occ.ndim}, future={future_occ.ndim}, skipping")
            continue
        if history_poses.ndim < 2 or history_poses.shape[-1] < 7:
            print(f"  [WARN] Batch {batch_idx} bad pose shape: "
                  f"history_poses={list(history_poses.shape)}, need [..., >=7], skipping")
            continue
        if future_poses.ndim < 2 or future_poses.shape[-1] < 7:
            print(f"  [WARN] Batch {batch_idx} bad pose shape: "
                  f"future_poses={list(future_poses.shape)}, need [..., >=7], skipping")
            continue

        # Skip batches with all-zero future occupancy (would train model to predict nothing)
        if future_occ.sum() == 0:
            if batch_idx % debug_freq == 0:
                print(f"  [WARN] Batch {batch_idx} has all-zero future occupancy, skipping")
            continue

        # Use autocast for mixed precision
        use_amp = scaler is not None or amp_dtype is not None
        if use_amp and torch.cuda.is_available():
            amp_context = torch.amp.autocast('cuda', dtype=amp_dtype)
        else:
            amp_context = contextlib.nullcontext()

        with amp_context:
            if is_6dof:
                # 6DoF model returns dict of outputs
                outputs = model(history_occ, history_poses, future_poses)
                pred_occ = outputs['future_occupancy']

                # 6DoF loss expects outputs and targets dicts
                targets = {
                    'future_occupancy': future_occ.float(),
                    'future_poses': future_poses,
                    'global_pose': history_poses[:, -1, :7],
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
            if occ_rate < 0.01:
                print(f"  [WARN] Batch {batch_idx} has {occ_rate:.4f}% occupancy — "
                      f"nearly empty grid may indicate data loading or transform issue")
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

        # Check for NaN/Inf BEFORE backward to prevent corrupting model weights
        if not torch.isfinite(loss):
            print(f"  [WARN] Non-finite loss at batch {batch_idx}: {loss.item()}, skipping")
            optimizer.zero_grad()
            continue

        # Scale loss for gradient accumulation
        scaled_loss = loss / grad_accum_steps

        # Backward pass with optional mixed precision scaling
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

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

        # Only step optimizer at accumulation boundaries
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Track occupancy rates for epoch summary
        with torch.no_grad():
            epoch_pred_occ_sum += (pred_occ > 0.5).float().sum().item()
            epoch_target_occ_sum += (future_occ > 0).float().sum().item()
            epoch_voxel_count += future_occ.numel()

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

    if num_batches == 0:
        print(f"  [ERROR] All batches were skipped! Check data loading.")
        return float('inf')

    # Epoch occupancy rate summary
    if epoch_voxel_count > 0:
        pred_occ_rate = epoch_pred_occ_sum / epoch_voxel_count * 100
        target_occ_rate = epoch_target_occ_sum / epoch_voxel_count * 100
        print(f"  [EPOCH {epoch}] Occupancy rates: pred={pred_occ_rate:.4f}%, target={target_occ_rate:.4f}%")
        writer.add_scalar('Train/PredOccRate', pred_occ_rate, epoch)
        writer.add_scalar('Train/TargetOccRate', target_occ_rate, epoch)
        if use_wandb:
            wandb.log({
                'epoch/pred_occ_rate': pred_occ_rate,
                'epoch/target_occ_rate': target_occ_rate,
            }, commit=False)

    return total_loss / num_batches


def validate(model, dataloader, criterion, device, is_6dof=False):
    """Validate model with task-specific metrics (IoU, precision, recall, pose errors).

    When predictions have a time dimension (ndim >= 5), also computes:
    - Per-timestep IoU (iou_t0, iou_t1, ...)
    - Temporal consistency (cosine similarity of frame-to-frame voxel changes)
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    # Occupancy metrics accumulators
    tp_sum = 0  # true positives
    fp_sum = 0  # false positives
    fn_sum = 0  # false negatives
    tn_sum = 0  # true negatives

    # Per-timestep accumulators (up to 20 timesteps)
    MAX_T = 20
    tp_per_t = [0.0] * MAX_T
    fp_per_t = [0.0] * MAX_T
    fn_per_t = [0.0] * MAX_T
    has_temporal = False

    # Temporal consistency accumulator
    temporal_consistency_sum = 0.0
    temporal_consistency_count = 0

    # Pose metrics accumulators
    pos_errors = []
    quat_errors = []

    with torch.no_grad():
        for batch in dataloader:
            required_keys = ['history_occupancy', 'future_occupancy', 'history_poses', 'future_poses']
            if any(k not in batch for k in required_keys):
                continue

            history_occ = batch['history_occupancy'].to(device)
            future_occ = batch['future_occupancy'].to(device)
            history_poses = batch['history_poses'].to(device)
            future_poses = batch['future_poses'].to(device)

            if is_6dof:
                outputs = model(history_occ, history_poses, future_poses)
                pred_occ = outputs['future_occupancy']
                pred_poses_val = outputs.get('future_poses')
                targets = {
                    'future_occupancy': future_occ.float(),
                    'future_poses': future_poses,
                    'global_pose': history_poses[:, -1, :7],
                }
                losses = criterion(outputs, targets)
                loss = losses['total']
            else:
                output = model(history_occ, history_poses, future_poses)

                if isinstance(output, dict):
                    pred_occ = output['future_occupancy']
                    pred_poses_val = output.get('future_poses')
                    occ_loss = criterion(pred_occ, future_occ.float())
                    if pred_poses_val is not None:
                        pose_loss = F.smooth_l1_loss(pred_poses_val, future_poses)
                        loss = occ_loss + 0.1 * pose_loss
                    else:
                        loss = occ_loss
                else:
                    pred_occ = output
                    pred_poses_val = None
                    loss = criterion(pred_occ, future_occ.float())

            total_loss += loss.item()
            num_batches += 1

            # Compute occupancy metrics (binarize predictions at 0.5 threshold)
            pred_binary = (pred_occ > 0.5).float()
            target_binary = (future_occ > 0).float()
            tp_sum += (pred_binary * target_binary).sum().item()
            fp_sum += (pred_binary * (1 - target_binary)).sum().item()
            fn_sum += ((1 - pred_binary) * target_binary).sum().item()
            tn_sum += ((1 - pred_binary) * (1 - target_binary)).sum().item()

            # Per-timestep metrics when time dimension exists
            if pred_binary.ndim >= 5:
                has_temporal = True
                T = min(pred_binary.shape[1], MAX_T)
                for t in range(T):
                    pred_t = pred_binary[:, t]
                    target_t = target_binary[:, t]
                    tp_per_t[t] += (pred_t * target_t).sum().item()
                    fp_per_t[t] += (pred_t * (1 - target_t)).sum().item()
                    fn_per_t[t] += ((1 - pred_t) * target_t).sum().item()

                # Temporal consistency: cosine similarity of consecutive frame changes
                if T >= 2:
                    for t in range(1, T):
                        diff_pred = pred_binary[:, t].reshape(pred_binary.shape[0], -1) - \
                                    pred_binary[:, t-1].reshape(pred_binary.shape[0], -1)
                        diff_target = target_binary[:, t].reshape(target_binary.shape[0], -1) - \
                                      target_binary[:, t-1].reshape(target_binary.shape[0], -1)
                        # Cosine similarity per sample
                        cos_sim = F.cosine_similarity(diff_pred, diff_target, dim=-1)
                        temporal_consistency_sum += cos_sim.sum().item()
                        temporal_consistency_count += cos_sim.numel()

            # Compute pose metrics
            if pred_poses_val is not None:
                # Position error (Euclidean distance in meters)
                pos_err = (pred_poses_val[..., :3] - future_poses[..., :3]).norm(dim=-1).mean().item()
                pos_errors.append(pos_err)

                # Orientation error (geodesic distance in degrees)
                pred_q = F.normalize(pred_poses_val[..., 3:7], p=2, dim=-1)
                target_q = F.normalize(future_poses[..., 3:7], p=2, dim=-1)
                dot = (pred_q * target_q).sum(dim=-1).abs().clamp(max=1.0)
                angle_rad = 2 * torch.acos(dot)
                quat_errors.append(angle_rad.mean().item() * 180 / 3.14159)

    # Compute final metrics
    metrics = {}
    if num_batches > 0:
        metrics['loss'] = total_loss / num_batches
        eps = 1e-8
        precision = tp_sum / (tp_sum + fp_sum + eps)
        recall = tp_sum / (tp_sum + fn_sum + eps)
        iou = tp_sum / (tp_sum + fp_sum + fn_sum + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        metrics['iou'] = iou
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1

        # Per-timestep IoU
        if has_temporal:
            for t in range(MAX_T):
                if tp_per_t[t] + fp_per_t[t] + fn_per_t[t] > 0:
                    metrics[f'iou_t{t}'] = tp_per_t[t] / (tp_per_t[t] + fp_per_t[t] + fn_per_t[t] + eps)

            # Temporal consistency
            if temporal_consistency_count > 0:
                metrics['temporal_consistency'] = temporal_consistency_sum / temporal_consistency_count

        if pos_errors:
            metrics['pos_error_m'] = sum(pos_errors) / len(pos_errors)
        if quat_errors:
            metrics['orient_error_deg'] = sum(quat_errors) / len(quat_errors)
    else:
        metrics['loss'] = float('inf')

    return metrics


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
        print("Install OccWorld: git clone https://github.com/wzzheng/OccWorld.git <path>")
        print("Then set OCCWORLD_PATH environment variable, or install to ~/OccWorld (default)")
        sys.exit(1)

    if args.use_occworld and args.model_type == '6dof':
        print("ERROR: --use-occworld and --model-type 6dof are incompatible.")
        print("  External OccWorld produces occupancy-only outputs,")
        print("  but 6DoF mode requires pose predictions (future_poses, global_pose).")
        print("  Use --model-type simple with --use-occworld, or remove --use-occworld for 6dof.")
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
        uav_interval = args.interval or 5
        if not validate_data(data_root, dataset_type, auto_download=auto_download, interval=uav_interval):
            print("\n" + "=" * 60)
            print("DATA VALIDATION FAILED")
            print("=" * 60)
            print("Training cannot proceed without valid data.")
            print("\nTo download data manually:")
            if dataset_type == 'uavscenes':
                print("  pip install huggingface_hub")
                print("  python -c \"from huggingface_hub import snapshot_download; "
                      f"snapshot_download('{UAVSCENES_HF_REPO}', repo_type='dataset', local_dir='{data_root}')\"")
                print("  # Or via S3:")
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
            interval=args.interval or ds_cfg.get('interval', 5),
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
            # Training loop only uses occupancy + poses — skip images/lidar I/O
            load_images=ds_config.get('load_images', False),
            load_lidar=ds_config.get('load_lidar', False),
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

    val_has_data = len(val_dataset) > 0
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if val_has_data else 0,
        collate_fn=dataset_collate_fn,
        pin_memory=pin_memory and val_has_data,
        persistent_workers=num_workers > 0 and val_has_data,
        prefetch_factor=2 if num_workers > 0 and val_has_data else None,
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
        print(f"  [MODEL] Using external OccWorld (TransVQVAE) from {os.environ.get('OCCWORLD_PATH', '~/OccWorld')}")
    elif args.model_type == '6dof':
        # Use 6DoF model — read from config if available, else use defaults
        grid_size = tuple(getattr(config, 'grid_size', [200, 200, 121]))
        model_cfg = getattr(config, 'model', {})
        model_config = OccWorld6DoFConfig(
            grid_size=grid_size,
            history_frames=getattr(config, 'history_frames', 4),
            future_frames=getattr(config, 'future_frames', 6),
            use_transformer=args.use_transformer,
            pose_dim=model_cfg.get('pose_dim', 13),
            encoder_channels=model_cfg.get('encoder_channels', (64, 128, 256)),
            num_transformer_layers=model_cfg.get('num_transformer_layers', 4),
            num_heads=model_cfg.get('num_heads', 8),
            transformer_dim=model_cfg.get('transformer_dim', 256),
            dropout=model_cfg.get('dropout', 0.1),
            uncertainty_dim=model_cfg.get('uncertainty_dim', 6),
            place_embedding_dim=model_cfg.get('place_embedding_dim', 256),
            enable_uncertainty=model_cfg.get('enable_uncertainty', True),
            enable_relocalization=model_cfg.get('enable_relocalization', True),
            enable_place_recognition=model_cfg.get('enable_place_recognition', True),
        )
        model = OccWorld6DoF(model_config)
        print(f"  6DoF Config: grid={grid_size}, transformer={args.use_transformer}")
    else:
        # Use simple standalone model (occupancy only)
        model = SimpleOccupancyModel(config)

    model = model.to(device)

    # Wrap in DataParallel if multiple GPUs
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        gpu_ids = [int(g) for g in args.gpu_ids.split(',') if g.strip()]
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
            print(f"  Using DataParallel on GPUs: {gpu_ids}")

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
            else:
                print("\n" + "!" * 60)
                print("WARNING: --resume requested but no checkpoints found in")
                print(f"  {work_dir / 'checkpoints'}")
                print("Training will start from scratch.")
                print("!" * 60 + "\n")

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

        if load_from:
            if not os.path.exists(load_from):
                print("\n" + "!" * 60)
                print(f"WARNING: load_from path does not exist: {load_from}")
                print("Training will proceed WITHOUT pretrained weights.")
                print("!" * 60 + "\n")
            else:
                print(f"Loading weights from: {load_from}")
                checkpoint = torch.load(load_from, map_location=device)
                ckpt_sd = checkpoint.get('state_dict', checkpoint)
                # Strip DataParallel/torch.compile prefixes from checkpoint keys
                cleaned_sd = {}
                for k, v in ckpt_sd.items():
                    clean_k = k.replace('module.', '', 1).replace('_orig_mod.', '')
                    cleaned_sd[clean_k] = v
                ckpt_sd = cleaned_sd
                result = model.load_state_dict(ckpt_sd, strict=args.strict_load)
                # Warn about mismatched keys so partial loads don't go unnoticed
                if result.missing_keys:
                    print(f"  [WARN] Missing keys ({len(result.missing_keys)}): "
                          f"{result.missing_keys[:5]}{'...' if len(result.missing_keys) > 5 else ''}")
                if result.unexpected_keys:
                    print(f"  [WARN] Unexpected keys ({len(result.unexpected_keys)}): "
                          f"{result.unexpected_keys[:5]}{'...' if len(result.unexpected_keys) > 5 else ''}")
                model_key_count = len(dict(model.state_dict()))
                matched = model_key_count - len(result.missing_keys)
                loaded_pct = matched / max(model_key_count, 1)
                if loaded_pct < 0.5:
                    print(f"  [WARN] Only {loaded_pct*100:.0f}% of model weights loaded "
                          f"({matched}/{model_key_count} keys)! "
                          f"Check checkpoint compatibility.")
                else:
                    print(f"  Loaded {loaded_pct*100:.0f}% of model weights "
                          f"({matched}/{model_key_count} keys)")

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
    amp_dtype = None
    if args.amp:
        if torch.cuda.is_available():
            # Determine AMP dtype
            if args.amp_dtype:
                amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
            elif torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16
            # GradScaler is not needed for bfloat16 (it has sufficient dynamic range)
            if amp_dtype == torch.float16:
                scaler = torch.cuda.amp.GradScaler()
            print(f"Mixed precision (AMP) enabled - using {amp_dtype}")
        else:
            print("WARNING: --amp requires CUDA, skipping")

    # Optimizer
    lr = args.lr or getattr(config, 'optimizer', {}).get('lr', 1e-4)
    weight_decay = getattr(config, 'optimizer', {}).get('weight_decay', 0.01)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler with warmup
    max_epochs = args.epochs or getattr(config, 'max_epochs', 50)
    warmup_epochs = min(args.warmup_epochs, max_epochs // 2)  # Don't warmup more than half
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
        print(f"  LR warmup: {warmup_epochs} epochs (0.01x → 1.0x), then cosine decay")
    else:
        scheduler = cosine_scheduler

    # Loss function - depends on model type
    if args.model_type == '6dof':
        # 6DoF loss — read weights from config if available, else use defaults
        loss_cfg = getattr(config, 'loss', {})
        criterion = OccWorld6DoFLoss(
            occ_weight=loss_cfg.get('occ_weight', 1.0),
            pose_weight=loss_cfg.get('pose_weight', 0.5),
            uncertainty_weight=loss_cfg.get('uncertainty_weight', 0.1),
            reloc_weight=loss_cfg.get('reloc_weight', 0.2),
            place_weight=loss_cfg.get('place_weight', 0.1),
            focal_alpha=loss_cfg.get('focal_alpha', 0.99),
            focal_gamma=loss_cfg.get('focal_gamma', 2.0),
            dice_weight=loss_cfg.get('dice_weight', 1.0),
            mean_weight=loss_cfg.get('mean_weight', 10.0),
            pose_variance_weight=loss_cfg.get('pose_variance_weight', 1.0),
            min_pose_std=loss_cfg.get('min_pose_std', 0.01),
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
            temporal_decay=getattr(config, 'temporal_decay', 0.0),
        )

    # TensorBoard
    writer = SummaryWriter(log_dir=str(work_dir / 'logs'))

    if args.eval_only:
        is_6dof = args.model_type == '6dof'
        val_metrics = validate(model, val_loader, criterion, device, is_6dof)
        print(f"Eval-only: Val Loss = {val_metrics['loss']:.6f}")
        print(f"  IoU: {val_metrics.get('iou', 0):.4f}, "
              f"Precision: {val_metrics.get('precision', 0):.4f}, "
              f"Recall: {val_metrics.get('recall', 0):.4f}")
        if 'pos_error_m' in val_metrics:
            print(f"  Position Error: {val_metrics['pos_error_m']:.3f}m, "
                  f"Orientation Error: {val_metrics.get('orient_error_deg', 0):.2f}°")
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

    # Register signal handlers for graceful shutdown
    TrainingState.model = model
    TrainingState.optimizer = optimizer
    TrainingState.scheduler = scheduler
    TrainingState.work_dir = work_dir
    signal.signal(signal.SIGTERM, TrainingState.save_emergency_checkpoint)
    signal.signal(signal.SIGINT, TrainingState.save_emergency_checkpoint)

    # Training loop
    print("=" * 60)
    print(f"Starting training on [{dataset_type.upper()}] for {max_epochs} epochs (from epoch {start_epoch})")
    print("=" * 60)

    is_6dof = args.model_type == '6dof'

    for epoch in range(start_epoch, max_epochs):
        # Check for graceful shutdown
        if TrainingState.should_stop:
            print("  [STOP] Graceful shutdown requested, stopping training.")
            break
        epoch_start = time.time()

        # Update training state for signal handler
        TrainingState.epoch = epoch

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                  device, epoch, writer, use_wandb, is_6dof, dataset_type,
                                  scaler=scaler, debug_freq=args.debug_freq,
                                  amp_dtype=amp_dtype, grad_accum_steps=args.grad_accum)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, is_6dof)
        val_loss = val_metrics['loss']

        # Update scheduler (skip if training failed to avoid incorrect LR decay)
        if train_loss != float('inf') and val_loss != float('inf'):
            scheduler.step()
        else:
            print(f"  [WARN] Skipping scheduler step (training or validation failed this epoch)")

        # Log
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}/{max_epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"LR: {current_lr:.6f}, Time: {epoch_time:.1f}s")
        print(f"  Val Metrics: IoU={val_metrics.get('iou', 0):.4f}, "
              f"Prec={val_metrics.get('precision', 0):.4f}, "
              f"Recall={val_metrics.get('recall', 0):.4f}, "
              f"F1={val_metrics.get('f1', 0):.4f}")
        if 'pos_error_m' in val_metrics:
            print(f"  Pose: pos_err={val_metrics['pos_error_m']:.3f}m, "
                  f"orient_err={val_metrics.get('orient_error_deg', 0):.2f}°")
        # Per-timestep IoU breakdown
        per_t_ious = {k: v for k, v in val_metrics.items() if k.startswith('iou_t')}
        if per_t_ious:
            iou_strs = [f"t{k[5:]}={v:.4f}" for k, v in sorted(per_t_ious.items())]
            print(f"  Per-timestep IoU: {', '.join(iou_strs)}")
        if 'temporal_consistency' in val_metrics:
            print(f"  Temporal consistency: {val_metrics['temporal_consistency']:.4f}")

        writer.add_scalar('Train/EpochLoss', train_loss, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/IoU', val_metrics.get('iou', 0), epoch)
        writer.add_scalar('Val/Precision', val_metrics.get('precision', 0), epoch)
        writer.add_scalar('Val/Recall', val_metrics.get('recall', 0), epoch)
        for k, v in val_metrics.items():
            if k.startswith('iou_t'):
                writer.add_scalar(f'Val/{k}', v, epoch)
        if 'temporal_consistency' in val_metrics:
            writer.add_scalar('Val/TemporalConsistency', val_metrics['temporal_consistency'], epoch)
        writer.add_scalar('Train/LR', current_lr, epoch)

        # Weights & Biases epoch logging
        if use_wandb:
            epoch_log = {
                'epoch': epoch,
                'epoch/train_loss': train_loss,
                'epoch/val_loss': val_loss,
                'epoch/lr': current_lr,
                'epoch/time_seconds': epoch_time,
                'epoch/val_iou': val_metrics.get('iou', 0),
                'epoch/val_precision': val_metrics.get('precision', 0),
                'epoch/val_recall': val_metrics.get('recall', 0),
                'epoch/val_f1': val_metrics.get('f1', 0),
            }
            if 'pos_error_m' in val_metrics:
                epoch_log['epoch/val_pos_error_m'] = val_metrics['pos_error_m']
                epoch_log['epoch/val_orient_error_deg'] = val_metrics.get('orient_error_deg', 0)
            # Per-timestep IoU
            for k, v in val_metrics.items():
                if k.startswith('iou_t'):
                    epoch_log[f'epoch/val_{k}'] = v
            if 'temporal_consistency' in val_metrics:
                epoch_log['epoch/val_temporal_consistency'] = val_metrics['temporal_consistency']
            wandb.log(epoch_log)

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == max_epochs - 1:
            ckpt_path = work_dir / 'checkpoints' / f'epoch_{epoch+1}.pth'
            # Unwrap state_dict keys from torch.compile (_orig_mod.) and DataParallel (module.)
            raw_sd = model.state_dict()
            clean_sd = {}
            for k, v in raw_sd.items():
                k = k.replace('_orig_mod.', '').replace('module.', '', 1)
                clean_sd[k] = v
            # Atomic save: write to tmp then rename to avoid corruption on crash
            tmp_path = ckpt_path.with_suffix('.pth.tmp')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': clean_sd,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, tmp_path)
            tmp_path.replace(ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

            # Clean up old checkpoints
            cleanup_old_checkpoints(work_dir / 'checkpoints', args.max_keep_ckpts)

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
            # Unwrap state_dict keys from torch.compile/DataParallel
            raw_sd = model.state_dict()
            clean_sd = {}
            for k, v in raw_sd.items():
                k = k.replace('_orig_mod.', '').replace('module.', '', 1)
                clean_sd[k] = v
            # Atomic save: write to tmp then rename to avoid corruption on crash
            tmp_path = best_path.with_suffix('.pth.tmp')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': clean_sd,
                'val_loss': val_loss,
            }, tmp_path)
            tmp_path.replace(best_path)
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
