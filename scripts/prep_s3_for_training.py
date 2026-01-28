#!/usr/bin/env python3
"""
Prepare S3 bucket for remote model training.

This script uploads all necessary files to S3 so that training can be
performed on a remote GPU server (Vast.ai, Lambda Cloud, etc.).

What gets uploaded:
  - Pretrained models (OccWorld checkpoints)
  - Training data (Tokyo Gazebo, nuScenes pickles)
  - Config files

Usage:
    # Preview what will be uploaded
    python scripts/prep_s3_for_training.py --dry-run

    # Upload everything
    python scripts/prep_s3_for_training.py

    # Upload to custom bucket
    python scripts/prep_s3_for_training.py --bucket my-bucket

After running, use this on the remote server:
    aws s3 sync s3://verylargeweebmodel/ . --exclude '*.zip'
"""

import os
import sys
import argparse
import subprocess
import hashlib
from pathlib import Path
from typing import Optional

try:
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.exceptions import ClientError, NoCredentialsError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BUCKET = "verylargeweebmodel"
S3_REGION = "us-west-2"

# Files to upload for training
TRAINING_FILES = {
    # Pretrained models (required)
    "pretrained/occworld/latest.pth": {
        "s3_key": "pretrained/occworld/latest.pth",
        "description": "OccWorld checkpoint",
        "required": True,
        "min_size": 100_000_000,  # 100MB
    },
    # nuScenes pickle files
    "data/nuscenes_infos_train_temporal_v3_scene.pkl": {
        "s3_key": "nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl",
        "description": "nuScenes training pickle",
        "required": False,
        "min_size": 10_000_000,  # 10MB
    },
    "data/nuscenes_infos_val_temporal_v3_scene.pkl": {
        "s3_key": "nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl",
        "description": "nuScenes validation pickle",
        "required": False,
        "min_size": 5_000_000,  # 5MB
    },
}

# Directories to upload (recursively)
# Note: nuScenes samples/sweeps are VERY large and should be downloaded
# from the official source on remote servers, not uploaded via this script
TRAINING_DIRS = {
    "data/tokyo_gazebo": {
        "s3_prefix": "tokyo_gazebo",
        "description": "Tokyo Gazebo training data",
        "required": False,
    },
    "data/uavscenes": {
        "s3_prefix": "uavscenes",
        "description": "UAVScenes aerial 6DoF dataset",
        "required": False,
    },
    "config": {
        "s3_prefix": "config",
        "description": "Training configs",
        "required": True,
    },
    "pretrained": {
        "s3_prefix": "pretrained",
        "description": "Pretrained model checkpoints",
        "required": False,
    },
}


# =============================================================================
# Logging
# =============================================================================

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'


def log_info(msg: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}", flush=True)


def log_success(msg: str):
    print(f"{Colors.GREEN}[OK]{Colors.NC} {msg}", flush=True)


def log_warn(msg: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}", flush=True)


def log_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}", flush=True)


def log_step(msg: str):
    print(f"\n{Colors.CYAN}==>{Colors.NC} {Colors.BOLD}{msg}{Colors.NC}", flush=True)


# =============================================================================
# S3 Functions
# =============================================================================

def check_aws_credentials() -> bool:
    """Check if AWS credentials are configured."""
    if not HAS_BOTO3:
        return False
    try:
        import socket
        socket.setdefaulttimeout(10)  # 10 second timeout
        sts = boto3.client('sts')
        sts.get_caller_identity()
        return True
    except Exception as e:
        log_error(f"AWS credential check failed: {e}")
        return False


def get_s3_client():
    """Get S3 client."""
    if not HAS_BOTO3:
        raise ImportError("boto3 required: pip install boto3")
    return boto3.client('s3', region_name=S3_REGION)


def ensure_bucket_exists(s3_client, bucket: str, dry_run: bool = False) -> bool:
    """Create bucket if it doesn't exist."""
    try:
        s3_client.head_bucket(Bucket=bucket)
        log_info(f"Bucket exists: s3://{bucket}")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            log_info(f"Creating bucket: {bucket}")
            if dry_run:
                log_info(f"[DRY RUN] Would create bucket: {bucket}")
                return True
            try:
                if S3_REGION != 'us-east-1':
                    s3_client.create_bucket(
                        Bucket=bucket,
                        CreateBucketConfiguration={'LocationConstraint': S3_REGION}
                    )
                else:
                    s3_client.create_bucket(Bucket=bucket)
                log_success(f"Created bucket: {bucket}")
                return True
            except Exception as create_err:
                log_error(f"Failed to create bucket: {create_err}")
                return False
        elif error_code == '403':
            log_error(f"Access denied to bucket: {bucket}")
            return False
        else:
            log_error(f"Error checking bucket: {e}")
            return False


def s3_object_exists(s3_client, bucket: str, key: str) -> tuple[bool, int]:
    """Check if S3 object exists and return its size."""
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return True, response['ContentLength']
    except ClientError:
        return False, 0


def upload_file(
    s3_client,
    local_path: Path,
    bucket: str,
    s3_key: str,
    dry_run: bool = False,
    force: bool = False
) -> bool:
    """Upload a single file to S3."""
    if not local_path.exists():
        return False

    local_size = local_path.stat().st_size
    size_mb = local_size / 1024 / 1024

    # Check if already exists with same size
    if not force:
        exists, s3_size = s3_object_exists(s3_client, bucket, s3_key)
        if exists and s3_size == local_size:
            log_info(f"Already uploaded (same size): {s3_key}")
            return True

    if dry_run:
        log_info(f"[DRY RUN] Would upload: {local_path.name} ({size_mb:.1f}MB) -> s3://{bucket}/{s3_key}")
        return True

    log_info(f"Uploading: {local_path.name} ({size_mb:.1f}MB)")

    try:
        config = TransferConfig(
            multipart_threshold=50 * 1024 * 1024,
            max_concurrency=10,
            multipart_chunksize=50 * 1024 * 1024,
        )

        if HAS_TQDM:
            with tqdm(total=local_size, unit='B', unit_scale=True, desc=local_path.name[:30]) as pbar:
                s3_client.upload_file(
                    str(local_path),
                    bucket,
                    s3_key,
                    Config=config,
                    Callback=lambda b: pbar.update(b)
                )
        else:
            s3_client.upload_file(str(local_path), bucket, s3_key, Config=config)

        log_success(f"Uploaded: s3://{bucket}/{s3_key}")
        return True

    except Exception as e:
        log_error(f"Upload failed: {e}")
        return False


def upload_directory(
    s3_client,
    local_dir: Path,
    bucket: str,
    s3_prefix: str,
    dry_run: bool = False,
    force: bool = False,
    max_files: int = 1000,
    exclude_patterns: list = None
) -> dict:
    """Upload a directory recursively to S3."""
    results = {"uploaded": 0, "skipped": 0, "failed": 0, "total_size": 0}

    if exclude_patterns is None:
        exclude_patterns = [".zip", ".tar.gz", ".tar", ".pyc", "__pycache__"]

    if not local_dir.exists():
        log_warn(f"Directory not found: {local_dir}")
        return results

    # Collect files efficiently (with limit for dry-run)
    files = []
    for f in local_dir.rglob("*"):
        if f.is_file():
            # Skip excluded patterns
            skip = False
            for pattern in exclude_patterns:
                if pattern in str(f):
                    skip = True
                    break
            if skip:
                continue
            files.append(f)
            if dry_run and len(files) >= max_files:
                break

    if not files:
        log_warn(f"No files in: {local_dir}")
        return results

    total_note = f" (showing first {max_files})" if dry_run and len(files) >= max_files else ""
    log_info(f"Found {len(files)} files in {local_dir}{total_note}")

    for filepath in files:
        rel_path = filepath.relative_to(local_dir)
        s3_key = f"{s3_prefix}/{rel_path}".replace("\\", "/")

        local_size = filepath.stat().st_size

        # Skip check
        if not force:
            exists, s3_size = s3_object_exists(s3_client, bucket, s3_key)
            if exists and s3_size == local_size:
                results["skipped"] += 1
                results["total_size"] += local_size
                continue

        if dry_run:
            results["uploaded"] += 1
            results["total_size"] += local_size
            continue

        if upload_file(s3_client, filepath, bucket, s3_key, dry_run=False, force=True):
            results["uploaded"] += 1
            results["total_size"] += local_size
        else:
            results["failed"] += 1

    return results


# =============================================================================
# Main Prep Function
# =============================================================================

def prep_s3_bucket(
    project_root: Path,
    bucket: str,
    dry_run: bool = False,
    force: bool = False,
    include_optional: bool = True
) -> dict:
    """Prepare S3 bucket with all training files."""

    results = {
        "files": {"uploaded": 0, "skipped": 0, "failed": 0, "missing": 0},
        "dirs": {"uploaded": 0, "skipped": 0, "failed": 0},
    }

    # Check credentials
    log_step("Checking AWS credentials")
    if not check_aws_credentials():
        log_error("AWS credentials not configured!")
        log_info("Run: aws configure")
        log_info("Or set: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return results

    log_success("AWS credentials OK")

    # Get S3 client
    s3_client = get_s3_client()

    # Ensure bucket exists
    log_step("Checking S3 bucket")
    if not ensure_bucket_exists(s3_client, bucket, dry_run):
        return results

    # Upload individual files
    log_step("Uploading training files")
    for local_rel_path, config in TRAINING_FILES.items():
        local_path = project_root / local_rel_path

        if not include_optional and not config.get("required", False):
            continue

        if not local_path.exists():
            if config.get("required", False):
                log_error(f"Required file missing: {local_rel_path}")
                results["files"]["missing"] += 1
            else:
                log_warn(f"Optional file missing: {local_rel_path}")
            continue

        # Check minimum size
        min_size = config.get("min_size", 0)
        if local_path.stat().st_size < min_size:
            log_warn(f"File too small (possibly corrupt): {local_rel_path}")
            continue

        success = upload_file(
            s3_client,
            local_path,
            bucket,
            config["s3_key"],
            dry_run,
            force
        )

        if success:
            results["files"]["uploaded"] += 1
        else:
            results["files"]["failed"] += 1

    # Upload directories
    log_step("Uploading training directories")
    for local_rel_dir, config in TRAINING_DIRS.items():
        local_dir = project_root / local_rel_dir

        if not include_optional and not config.get("required", False):
            continue

        if not local_dir.exists():
            if config.get("required", False):
                log_warn(f"Required directory missing: {local_rel_dir}")
            continue

        log_info(f"Processing: {config['description']}")

        dir_results = upload_directory(
            s3_client,
            local_dir,
            bucket,
            config["s3_prefix"],
            dry_run,
            force
        )

        results["dirs"]["uploaded"] += dir_results["uploaded"]
        results["dirs"]["skipped"] += dir_results["skipped"]
        results["dirs"]["failed"] += dir_results["failed"]

        size_mb = dir_results["total_size"] / 1024 / 1024
        log_info(f"  {dir_results['uploaded']} uploaded, {dir_results['skipped']} skipped ({size_mb:.1f}MB total)")

    return results


def print_remote_instructions(bucket: str):
    """Print instructions for the remote server."""
    print()
    print("=" * 70)
    print(f"  {Colors.GREEN}S3 Bucket Ready for Remote Training{Colors.NC}")
    print("=" * 70)
    print()
    print(f"  Bucket: {Colors.CYAN}s3://{bucket}/{Colors.NC}")
    print()
    print(f"  {Colors.BOLD}On your remote GPU server, run:{Colors.NC}")
    print()
    print(f"    # Clone the repo")
    print(f"    git clone https://github.com/yourusername/VeryLargeWeebModel.git")
    print(f"    cd VeryLargeWeebModel")
    print()
    print(f"    # Download data from S3")
    print(f"    pip install boto3")
    print(f"    python scripts/download_pretrained.py --all")
    print()
    print(f"    # Or use AWS CLI directly")
    print(f"    aws s3 sync s3://{bucket}/ . --exclude '*.zip'")
    print()
    print(f"    # Start training")
    print(f"    python train.py --config config/finetune_tokyo.py")
    print()
    print("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare S3 bucket for remote model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what will be uploaded
  python scripts/prep_s3_for_training.py --dry-run

  # Upload everything
  python scripts/prep_s3_for_training.py

  # Force re-upload all files
  python scripts/prep_s3_for_training.py --force

  # Upload only required files
  python scripts/prep_s3_for_training.py --required-only
        """
    )

    parser.add_argument(
        "--bucket", "-b",
        default=DEFAULT_BUCKET,
        help=f"S3 bucket name (default: {DEFAULT_BUCKET})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-upload even if files exist"
    )
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="Only upload required files (skip optional data)"
    )

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent

    print(flush=True)
    print("=" * 70, flush=True)
    print(f"  {Colors.BOLD}VeryLargeWeebModel - S3 Training Prep{Colors.NC}", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    print(f"  Project:  {project_root}", flush=True)
    print(f"  Bucket:   s3://{args.bucket}", flush=True)
    print(f"  Dry run:  {args.dry_run}", flush=True)
    print(f"  Force:    {args.force}", flush=True)
    print(flush=True)

    if not HAS_BOTO3:
        log_error("boto3 is required. Install with: pip install boto3")
        sys.exit(1)

    # Run prep
    results = prep_s3_bucket(
        project_root,
        args.bucket,
        dry_run=args.dry_run,
        force=args.force,
        include_optional=not args.required_only
    )

    # Summary
    log_step("Summary")
    print(f"  Files:  {results['files']['uploaded']} uploaded, {results['files']['skipped']} skipped, {results['files']['failed']} failed, {results['files']['missing']} missing")
    print(f"  Dirs:   {results['dirs']['uploaded']} uploaded, {results['dirs']['skipped']} skipped, {results['dirs']['failed']} failed")

    if not args.dry_run and results['files']['failed'] == 0 and results['files']['missing'] == 0:
        print_remote_instructions(args.bucket)
    elif args.dry_run:
        print()
        log_info("This was a dry run. Run without --dry-run to actually upload.")


if __name__ == "__main__":
    main()
