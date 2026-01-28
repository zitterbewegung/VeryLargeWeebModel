#!/usr/bin/env python3
"""
Download VeryLargeWeebModel pretrained models and data from S3.

Primary source: s3://verylargeweebmodel/
Fallback: Tsinghua Cloud (Seafile)

Usage:
    python scripts/download_pretrained.py
    python scripts/download_pretrained.py --output pretrained/
    python scripts/download_pretrained.py --all  # Download everything (models + data)
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

S3_BUCKET = "verylargeweebmodel"
S3_REGION = "us-west-2"

# Files available on S3
S3_FILES = {
    # Pretrained models
    "occworld_checkpoint": {
        "s3_key": "pretrained/occworld/latest.pth",
        "local_path": "pretrained/occworld/latest.pth",
        "description": "OccWorld checkpoint (~721MB)",
        "required": True,
    },
    "vqvae_checkpoint": {
        "s3_key": "pretrained/vqvae/epoch_125.pth",
        "local_path": "pretrained/vqvae/epoch_125.pth",
        "description": "VQVAE checkpoint (~500MB)",
        "required": False,
    },
    # PLATEAU data
    "plateau_obj": {
        "s3_key": "plateau/tokyo23ku_obj.zip",
        "local_path": "data/plateau/raw/tokyo23ku_obj.zip",
        "description": "Tokyo PLATEAU OBJ models (~2.1GB)",
        "required": False,
    },
    "plateau_3dtiles": {
        "s3_key": "plateau/tokyo23ku_3dtiles.zip",
        "local_path": "data/plateau/raw/tokyo23ku_3dtiles.zip",
        "description": "Tokyo PLATEAU 3D Tiles (~4GB)",
        "required": False,
    },
    "plateau_citygml": {
        "s3_key": "plateau/tokyo23ku_citygml.zip",
        "local_path": "data/plateau/raw/tokyo23ku_citygml.zip",
        "description": "Tokyo PLATEAU CityGML (~5GB)",
        "required": False,
    },
    # nuScenes pickle files
    "nuscenes_train_pkl": {
        "s3_key": "nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl",
        "local_path": "data/nuscenes_infos_train_temporal_v3_scene.pkl",
        "description": "nuScenes training pickle (~100MB)",
        "required": False,
    },
    "nuscenes_val_pkl": {
        "s3_key": "nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl",
        "local_path": "data/nuscenes_infos_val_temporal_v3_scene.pkl",
        "description": "nuScenes validation pickle (~30MB)",
        "required": False,
    },
}

# Directories to sync from S3 (use aws s3 sync instead of single file download)
S3_DIRS = {
    "uavscenes": {
        "s3_prefix": "uavscenes",
        "local_path": "data/uavscenes",
        "description": "UAVScenes aerial 6DoF dataset (~20GB)",
        "required": False,
    },
    "tokyo_gazebo": {
        "s3_prefix": "tokyo_gazebo",
        "local_path": "data/tokyo_gazebo",
        "description": "Tokyo Gazebo training data (~15MB)",
        "required": False,
    },
}

# Minimum file sizes (bytes) to consider a file valid
MIN_FILE_SIZES = {
    "occworld_checkpoint": 100_000_000,  # 100MB
    "vqvae_checkpoint": 100_000_000,     # 100MB
    "plateau_obj": 1_000_000_000,        # 1GB
    "plateau_3dtiles": 1_000_000_000,    # 1GB
    "plateau_citygml": 1_000_000_000,    # 1GB
    "nuscenes_train_pkl": 10_000_000,    # 10MB
    "nuscenes_val_pkl": 5_000_000,       # 5MB
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
    NC = '\033[0m'


def log_info(msg: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")


def log_success(msg: str):
    print(f"{Colors.GREEN}[OK]{Colors.NC} {msg}")


def log_warn(msg: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")


def log_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")


def log_step(msg: str):
    print(f"\n{Colors.CYAN}==>{Colors.NC} {msg}")


# =============================================================================
# S3 Download Functions
# =============================================================================

def check_aws_cli() -> bool:
    """Check if AWS CLI is installed and configured."""
    try:
        result = subprocess.run(
            ["aws", "sts", "get-caller-identity"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def download_from_s3(s3_key: str, local_path: str, description: str = "") -> bool:
    """Download a file from S3 using AWS CLI."""
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"

    log_info(f"Downloading {description or s3_key}...")
    log_info(f"  Source: {s3_uri}")
    log_info(f"  Destination: {local_path}")

    # Create parent directory
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["aws", "s3", "cp", s3_uri, local_path, "--region", S3_REGION],
            capture_output=False,
            timeout=3600  # 1 hour timeout for large files
        )

        if result.returncode == 0 and Path(local_path).exists():
            size_mb = Path(local_path).stat().st_size / 1024 / 1024
            log_success(f"Downloaded {description} ({size_mb:.1f}MB)")
            return True
        else:
            log_error(f"Failed to download {description}")
            return False

    except subprocess.TimeoutExpired:
        log_error(f"Download timed out for {description}")
        return False
    except Exception as e:
        log_error(f"Download error: {e}")
        return False


def sync_from_s3(s3_prefix: str, local_dir: str, description: str = "") -> bool:
    """Sync a directory from S3 using AWS CLI."""
    s3_uri = f"s3://{S3_BUCKET}/{s3_prefix}"

    log_info(f"Syncing {description or s3_prefix}...")
    log_info(f"  Source: {s3_uri}")
    log_info(f"  Destination: {local_dir}")

    Path(local_dir).mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["aws", "s3", "sync", s3_uri, local_dir, "--region", S3_REGION],
            capture_output=False,
            timeout=7200  # 2 hour timeout
        )

        if result.returncode == 0:
            log_success(f"Synced {description}")
            return True
        else:
            log_error(f"Failed to sync {description}")
            return False

    except subprocess.TimeoutExpired:
        log_error(f"Sync timed out for {description}")
        return False
    except Exception as e:
        log_error(f"Sync error: {e}")
        return False


# =============================================================================
# Validation Functions
# =============================================================================

def validate_file(name: str, local_path: str) -> bool:
    """Check if a downloaded file is valid."""
    path = Path(local_path)

    if not path.exists():
        return False

    size = path.stat().st_size
    min_size = MIN_FILE_SIZES.get(name, 1000)

    if size < min_size:
        log_warn(f"{name} seems incomplete ({size} bytes, expected >= {min_size})")
        return False

    # Check if it's an HTML error page
    if size < 10_000_000:  # Only check smaller files
        try:
            with open(path, 'rb') as f:
                header = f.read(100)
                if b'<!DOCTYPE' in header or b'<html' in header:
                    log_warn(f"{name} appears to be an HTML error page")
                    return False
        except Exception:
            pass

    return True


def check_existing_files(output_dir: str, files: list) -> dict:
    """Check which files already exist and are valid."""
    status = {}

    for name in files:
        if name not in S3_FILES:
            continue

        info = S3_FILES[name]
        local_path = Path(output_dir) / info["local_path"]

        if local_path.exists() and validate_file(name, str(local_path)):
            size_mb = local_path.stat().st_size / 1024 / 1024
            status[name] = {"exists": True, "valid": True, "size_mb": size_mb}
        elif local_path.exists():
            status[name] = {"exists": True, "valid": False, "size_mb": 0}
        else:
            status[name] = {"exists": False, "valid": False, "size_mb": 0}

    return status


# =============================================================================
# Main Download Function
# =============================================================================

def download_files(
    output_dir: str,
    files: list = None,
    force: bool = False,
    dry_run: bool = False,
    include_dirs: bool = False
) -> dict:
    """
    Download specified files from S3.

    Args:
        output_dir: Base directory for downloads
        files: List of file names to download (None = required files only)
        force: Re-download even if file exists
        dry_run: Show what would be downloaded without downloading
        include_dirs: Also sync directories from S3_DIRS

    Returns:
        Dict with download status for each file
    """
    if files is None:
        files = [name for name, info in S3_FILES.items() if info.get("required", False)]

    # Check AWS CLI
    if not dry_run and not check_aws_cli():
        log_error("AWS CLI not configured. Please run: aws configure")
        log_info("Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        return {}

    # Check existing files
    existing = check_existing_files(output_dir, files)

    results = {}

    # Download individual files
    for name in files:
        if name not in S3_FILES:
            # Check if it's a directory
            if name in S3_DIRS:
                continue  # Handle below
            log_warn(f"Unknown file: {name}")
            continue

        info = S3_FILES[name]
        local_path = Path(output_dir) / info["local_path"]

        # Skip if already valid
        if not force and existing.get(name, {}).get("valid", False):
            log_info(f"{info['description']} already exists ({existing[name]['size_mb']:.1f}MB)")
            results[name] = True
            continue

        if dry_run:
            log_info(f"[DRY RUN] Would download: {info['description']}")
            log_info(f"  s3://{S3_BUCKET}/{info['s3_key']} -> {local_path}")
            results[name] = True
            continue

        # Download from S3
        success = download_from_s3(
            info["s3_key"],
            str(local_path),
            info["description"]
        )

        results[name] = success

    # Sync directories if requested
    if include_dirs:
        for name in files:
            if name in S3_DIRS:
                info = S3_DIRS[name]
                local_path = Path(output_dir) / info["local_path"]

                # Check if directory exists and has content
                if not force and local_path.exists() and any(local_path.iterdir()):
                    log_info(f"{info['description']} already exists at {local_path}")
                    results[name] = True
                    continue

                if dry_run:
                    log_info(f"[DRY RUN] Would sync: {info['description']}")
                    log_info(f"  s3://{S3_BUCKET}/{info['s3_prefix']}/ -> {local_path}/")
                    results[name] = True
                    continue

                # Sync from S3
                success = sync_from_s3(
                    info["s3_prefix"],
                    str(local_path),
                    info["description"]
                )
                results[name] = success

        # Also sync any dirs specified with --all
        if files == list(S3_FILES.keys()):
            for name, info in S3_DIRS.items():
                if name in results:
                    continue
                local_path = Path(output_dir) / info["local_path"]

                if not force and local_path.exists() and any(local_path.iterdir()):
                    log_info(f"{info['description']} already exists at {local_path}")
                    results[name] = True
                    continue

                if dry_run:
                    log_info(f"[DRY RUN] Would sync: {info['description']}")
                    results[name] = True
                    continue

                success = sync_from_s3(
                    info["s3_prefix"],
                    str(local_path),
                    info["description"]
                )
                results[name] = success

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Download VeryLargeWeebModel pretrained models from S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download required pretrained models
  python scripts/download_pretrained.py

  # Download everything (models + data)
  python scripts/download_pretrained.py --all

  # Download specific files
  python scripts/download_pretrained.py --files occworld_checkpoint vqvae_checkpoint

  # Force re-download
  python scripts/download_pretrained.py --force

  # Preview what would be downloaded
  python scripts/download_pretrained.py --dry-run --all

Available files:
""" + "\n".join([f"  {name}: {info['description']}" for name, info in S3_FILES.items()])
    + "\n\nAvailable directories:\n"
    + "\n".join([f"  {name}: {info['description']}" for name, info in S3_DIRS.items()])
    )

    all_names = list(S3_FILES.keys()) + list(S3_DIRS.keys())
    parser.add_argument('--output', '-o', default='.', help='Output directory (default: current)')
    parser.add_argument('--all', '-a', action='store_true', help='Download all files and directories')
    parser.add_argument('--files', '-f', nargs='+', choices=all_names,
                       help='Specific files/directories to download')
    parser.add_argument('--force', action='store_true', help='Force re-download even if files exist')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be downloaded')
    parser.add_argument('--list', '-l', action='store_true', help='List available files and exit')

    args = parser.parse_args()

    # List mode
    if args.list:
        print("\nAvailable files on S3:\n")
        for name, info in S3_FILES.items():
            required = " (required)" if info.get("required") else ""
            print(f"  {name}{required}")
            print(f"    {info['description']}")
            print(f"    s3://{S3_BUCKET}/{info['s3_key']}")
            print()
        print("\nAvailable directories on S3:\n")
        for name, info in S3_DIRS.items():
            print(f"  {name}")
            print(f"    {info['description']}")
            print(f"    s3://{S3_BUCKET}/{info['s3_prefix']}/")
            print()
        return

    print("=" * 60)
    print("VeryLargeWeebModel - Download from S3")
    print("=" * 60)
    print(f"S3 Bucket: s3://{S3_BUCKET}")
    print(f"Output: {os.path.abspath(args.output)}")
    print()

    # Determine which files to download
    include_dirs = False
    if args.files:
        files = args.files
        # Check if any directories are requested
        include_dirs = any(f in S3_DIRS for f in files)
    elif args.all:
        files = list(S3_FILES.keys()) + list(S3_DIRS.keys())
        include_dirs = True
    else:
        files = None  # Will default to required files

    # Download
    results = download_files(
        args.output,
        files=files,
        force=args.force,
        dry_run=args.dry_run,
        include_dirs=include_dirs
    )

    # Summary
    log_step("Download Summary")
    success_count = sum(1 for v in results.values() if v)
    fail_count = len(results) - success_count

    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")

    print()
    if fail_count == 0:
        log_success(f"All {success_count} files downloaded successfully!")
    else:
        log_warn(f"{success_count} succeeded, {fail_count} failed")

    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
