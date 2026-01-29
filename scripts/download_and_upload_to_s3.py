#!/usr/bin/env python3
"""
Download data locally and upload to S3 bucket.

Downloads:
  - Tokyo PLATEAU 3D city models
  - Pretrained models (OccWorld/VQVAE)
  - nuScenes pickle files
  - VisDrone dataset (optional)

Then uploads everything to s3://verylargeweebmodel/

Usage:
    python scripts/download_and_upload_to_s3.py
    python scripts/download_and_upload_to_s3.py --check-only     # Check which files exist locally
    python scripts/download_and_upload_to_s3.py --skip-download  # Upload existing files only
    python scripts/download_and_upload_to_s3.py --dry-run        # Show what would be uploaded
    python scripts/download_and_upload_to_s3.py --parallel-upload # Upload each file as it downloads
    python scripts/download_and_upload_to_s3.py --bucket my-bucket --prefix data/

Requirements:
    pip install boto3 requests tqdm
"""

import os
import sys
import argparse
import subprocess
import hashlib
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Tuple

try:
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.exceptions import ClientError, NoCredentialsError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    TransferConfig = None

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BUCKET = "verylargeweebmodel"
DEFAULT_LOCAL_DIR = "data"
DEFAULT_PRETRAINED_DIR = "pretrained"
S3_REGION = "us-west-2"  # Must match prep_s3_for_training.py

# Download URLs
DOWNLOAD_SOURCES = {
    "plateau_obj": {
        "url": "https://gic-plateau.s3.ap-northeast-1.amazonaws.com/2020/13100_tokyo23-ku_2020_obj_3_op.zip",
        "local_path": "data/plateau/raw/tokyo23ku_obj.zip",
        "s3_key": "plateau/tokyo23ku_obj.zip",
        "description": "Tokyo PLATEAU OBJ models (~2.1GB)",
    },
    "plateau_3dtiles": {
        "url": "https://gic-plateau.s3.ap-northeast-1.amazonaws.com/2020/13100_tokyo23ku_2020_3Dtiles_etc_1_op.zip",
        "local_path": "data/plateau/raw/tokyo23ku_3dtiles.zip",
        "s3_key": "plateau/tokyo23ku_3dtiles.zip",
        "description": "Tokyo PLATEAU 3D Tiles",
    },
    "plateau_citygml": {
        "url": "https://assets.cms.plateau.reearth.io/assets/ec/d51c64-a47f-4a56-aa64-340d1d3c720b/13100_tokyo23-ku_2020_citygml_4_2_op.zip",
        "local_path": "data/plateau/raw/tokyo23ku_citygml.zip",
        "s3_key": "plateau/tokyo23ku_citygml.zip",
        "description": "Tokyo PLATEAU CityGML",
    },
    "occworld_checkpoint": {
        "url": "https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/files/?p=/latest.pth&dl=1",
        "local_path": "pretrained/occworld/latest.pth",
        "s3_key": "pretrained/occworld/latest.pth",
        "description": "OccWorld checkpoint (~721MB)",
    },
    "nuscenes_train_pkl": {
        "url": "https://cloud.tsinghua.edu.cn/f/a05c25067a864e0eb7d0/?dl=1",
        "local_path": "data/nuscenes_infos_train_temporal_v3_scene.pkl",
        "s3_key": "nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl",
        "description": "nuScenes training pickle (~100MB)",
    },
    "nuscenes_val_pkl": {
        "url": "https://cloud.tsinghua.edu.cn/f/8c8f1e9b5f4a47a3b7c2/?dl=1",
        "local_path": "data/nuscenes_infos_val_temporal_v3_scene.pkl",
        "s3_key": "nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl",
        "description": "nuScenes validation pickle (~30MB)",
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
# File Checking Functions
# =============================================================================

def check_local_files(project_root: Path, sources: Optional[List[str]] = None) -> dict:
    """Check which files already exist locally."""

    if sources is None:
        sources = list(DOWNLOAD_SOURCES.keys())

    status = {}
    for name in sources:
        if name not in DOWNLOAD_SOURCES:
            continue

        source = DOWNLOAD_SOURCES[name]
        local_path = project_root / source["local_path"]

        exists = local_path.exists()
        size = local_path.stat().st_size if exists else 0
        is_complete = exists and size > 1_000_000  # > 1MB considered complete

        status[name] = {
            "exists": exists,
            "path": str(local_path),
            "size": size,
            "size_mb": size / 1024 / 1024 if size > 0 else 0,
            "is_complete": is_complete,
            "s3_key": source["s3_key"],
            "description": source["description"],
        }

    return status


def print_file_status(status: dict):
    """Print status of local files."""
    log_step("Local File Status")
    for name, info in status.items():
        if info["is_complete"]:
            log_success(f"{name}: {info['size_mb']:.1f}MB - {info['path']}")
        elif info["exists"]:
            log_warn(f"{name}: {info['size_mb']:.1f}MB (incomplete) - {info['path']}")
        else:
            log_info(f"{name}: NOT FOUND - {info['path']}")


# =============================================================================
# Download Functions
# =============================================================================

def download_file(url: str, local_path: str, description: str = "") -> bool:
    """Download a file from URL to local path."""

    local_path = Path(local_path)

    # Skip if already exists and has reasonable size
    if local_path.exists():
        size = local_path.stat().st_size
        if size > 1_000_000:  # > 1MB
            log_info(f"{description} already exists ({size / 1024 / 1024:.1f}MB), skipping download")
            return True
        else:
            log_warn(f"{description} exists but seems incomplete, re-downloading...")
            local_path.unlink()

    local_path.parent.mkdir(parents=True, exist_ok=True)

    log_info(f"Downloading {description}...")
    log_info(f"  URL: {url}")
    log_info(f"  Output: {local_path}")

    # Try aria2c first (fastest)
    if subprocess.run(["which", "aria2c"], capture_output=True).returncode == 0:
        log_info("  Using aria2c (16 connections)...")
        result = subprocess.run(
            ["aria2c", "-x", "16", "-s", "16", "--file-allocation=none",
             "-d", str(local_path.parent), "-o", local_path.name, url],
            capture_output=True
        )
        if result.returncode == 0 and local_path.exists():
            log_success(f"Downloaded {description}")
            return True

    # Try curl
    if subprocess.run(["which", "curl"], capture_output=True).returncode == 0:
        log_info("  Using curl...")
        result = subprocess.run(
            ["curl", "-L", "-C", "-", "--progress-bar", "-o", str(local_path), url]
        )
        if result.returncode == 0 and local_path.exists():
            log_success(f"Downloaded {description}")
            return True

    # Try wget
    if subprocess.run(["which", "wget"], capture_output=True).returncode == 0:
        log_info("  Using wget...")
        result = subprocess.run(
            ["wget", "-c", "-O", str(local_path), url]
        )
        if result.returncode == 0 and local_path.exists():
            log_success(f"Downloaded {description}")
            return True

    # Try requests library
    if HAS_REQUESTS:
        log_info("  Using requests library...")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(local_path, 'wb') as f:
                if HAS_TQDM and total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        print(f"\r  Downloaded: {downloaded / 1024 / 1024:.1f} MB", end='')
                    print()

            log_success(f"Downloaded {description}")
            return True

        except Exception as e:
            log_error(f"Download failed: {e}")
            if local_path.exists():
                local_path.unlink()

    log_error(f"Failed to download {description}")
    return False


def download_all(project_root: Path, sources: Optional[list] = None) -> dict:
    """Download all data files."""

    log_step("Downloading data files...")

    results = {}

    if sources is None:
        sources = list(DOWNLOAD_SOURCES.keys())

    for name in sources:
        if name not in DOWNLOAD_SOURCES:
            log_warn(f"Unknown source: {name}")
            continue

        source = DOWNLOAD_SOURCES[name]
        local_path = project_root / source["local_path"]

        success = download_file(
            url=source["url"],
            local_path=str(local_path),
            description=source["description"]
        )

        results[name] = {
            "success": success,
            "local_path": str(local_path),
            "s3_key": source["s3_key"],
        }

    return results


# =============================================================================
# S3 Upload Functions
# =============================================================================

def get_s3_client():
    """Get S3 client with proper credentials."""
    if not HAS_BOTO3:
        raise ImportError("boto3 is required. Install with: pip install boto3")

    return boto3.client('s3', region_name=S3_REGION)


def file_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Calculate MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def s3_object_exists(s3_client, bucket: str, key: str, local_path: Path) -> bool:
    """Check if S3 object exists and matches local file."""
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        s3_size = response['ContentLength']
        local_size = local_path.stat().st_size

        # Check if sizes match
        if s3_size == local_size:
            return True

        return False
    except ClientError:
        return False


def upload_file_to_s3(
    s3_client,
    local_path: Path,
    bucket: str,
    s3_key: str,
    dry_run: bool = False,
    skip_existing: bool = True
) -> bool:
    """Upload a single file to S3."""

    if not local_path.exists():
        log_warn(f"Local file not found: {local_path}")
        return False

    file_size = local_path.stat().st_size
    size_mb = file_size / 1024 / 1024

    # Check if already exists
    if skip_existing and s3_object_exists(s3_client, bucket, s3_key, local_path):
        log_info(f"Already in S3 (same size): s3://{bucket}/{s3_key}")
        return True

    if dry_run:
        log_info(f"[DRY RUN] Would upload: {local_path} ({size_mb:.1f}MB) -> s3://{bucket}/{s3_key}")
        return True

    log_info(f"Uploading: {local_path.name} ({size_mb:.1f}MB) -> s3://{bucket}/{s3_key}")

    try:
        # Use multipart upload for large files
        config = TransferConfig(
            multipart_threshold=100 * 1024 * 1024,  # 100MB
            max_concurrency=10,
            multipart_chunksize=100 * 1024 * 1024,  # 100MB chunks
        )

        # Progress callback
        if HAS_TQDM:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=local_path.name) as pbar:
                s3_client.upload_file(
                    str(local_path),
                    bucket,
                    s3_key,
                    Config=config,
                    Callback=lambda b: pbar.update(b)
                )
        else:
            uploaded = [0]
            def progress_callback(bytes_transferred):
                uploaded[0] += bytes_transferred
                print(f"\r  Uploaded: {uploaded[0] / 1024 / 1024:.1f} MB / {size_mb:.1f} MB", end='')

            s3_client.upload_file(
                str(local_path),
                bucket,
                s3_key,
                Config=config,
                Callback=progress_callback
            )
            print()

        log_success(f"Uploaded: s3://{bucket}/{s3_key}")
        return True

    except NoCredentialsError:
        log_error("AWS credentials not found. Configure with:")
        log_error("  aws configure")
        log_error("  or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return False
    except ClientError as e:
        log_error(f"S3 upload failed: {e}")
        return False


def upload_directory_to_s3(
    s3_client,
    local_dir: Path,
    bucket: str,
    s3_prefix: str,
    dry_run: bool = False,
    skip_existing: bool = True,
    max_workers: int = 4
) -> dict:
    """Upload entire directory to S3."""

    if not local_dir.exists():
        log_warn(f"Directory not found: {local_dir}")
        return {"success": 0, "failed": 0, "skipped": 0}

    # Collect all files
    files = []
    for filepath in local_dir.rglob("*"):
        if filepath.is_file():
            relative_path = filepath.relative_to(local_dir)
            s3_key = f"{s3_prefix}/{relative_path}".replace("\\", "/")
            files.append((filepath, s3_key))

    log_info(f"Found {len(files)} files to upload from {local_dir}")

    results = {"success": 0, "failed": 0, "skipped": 0}

    # Upload files
    for filepath, s3_key in files:
        if skip_existing and s3_object_exists(s3_client, bucket, s3_key, filepath):
            results["skipped"] += 1
            continue

        if upload_file_to_s3(s3_client, filepath, bucket, s3_key, dry_run, skip_existing):
            results["success"] += 1
        else:
            results["failed"] += 1

    return results


def upload_file_subprocess(
    local_path: str,
    bucket: str,
    s3_key: str,
    dry_run: bool = False
) -> bool:
    """Upload a file to S3 in a subprocess (for parallel execution)."""

    local_path = Path(local_path)
    if not local_path.exists():
        log_warn(f"Local file not found for upload: {local_path}")
        return False

    if dry_run:
        log_info(f"[DRY RUN] Would upload: {local_path} -> s3://{bucket}/{s3_key}")
        return True

    try:
        s3_client = get_s3_client()
        return upload_file_to_s3(s3_client, local_path, bucket, s3_key, dry_run)
    except Exception as e:
        log_error(f"Upload subprocess failed for {local_path}: {e}")
        return False


def download_and_upload_single(
    name: str,
    source: dict,
    project_root: Path,
    bucket: str,
    s3_prefix: str,
    dry_run: bool = False,
    skip_upload: bool = False
) -> Tuple[str, dict]:
    """Download a single file and immediately upload it to S3.

    Returns tuple of (name, result_dict).
    """
    local_path = project_root / source["local_path"]
    s3_key = source["s3_key"]
    if s3_prefix:
        s3_key = f"{s3_prefix}/{s3_key}"

    result = {
        "download_success": False,
        "upload_success": False,
        "local_path": str(local_path),
        "s3_key": s3_key,
    }

    # Download
    download_success = download_file(
        url=source["url"],
        local_path=str(local_path),
        description=source["description"]
    )
    result["download_success"] = download_success

    # Upload immediately after download if successful
    if download_success and not skip_upload:
        log_info(f"Starting upload for {name} in subprocess...")

        # Use multiprocessing to upload in parallel
        upload_process = multiprocessing.Process(
            target=upload_file_subprocess,
            args=(str(local_path), bucket, s3_key, dry_run)
        )
        upload_process.start()
        upload_process.join()  # Wait for upload to complete

        result["upload_success"] = upload_process.exitcode == 0

        if result["upload_success"]:
            log_success(f"Uploaded {name} to s3://{bucket}/{s3_key}")
        else:
            log_warn(f"Upload may have had issues for {name}")

    return name, result


def upload_all_to_s3(
    project_root: Path,
    bucket: str,
    s3_prefix: str = "",
    download_results: Optional[dict] = None,
    dry_run: bool = False
) -> dict:
    """Upload all downloaded data to S3."""

    log_step(f"Uploading to S3 bucket: {bucket}")

    if not HAS_BOTO3:
        log_error("boto3 is required. Install with: pip install boto3")
        return {}

    try:
        s3_client = get_s3_client()

        # Verify bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket)
            log_info(f"Bucket exists: s3://{bucket}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                log_info(f"Creating bucket: {bucket}")
                if not dry_run:
                    # Use consistent region (S3_REGION)
                    # us-east-1 doesn't allow LocationConstraint parameter
                    if S3_REGION and S3_REGION != 'us-east-1':
                        s3_client.create_bucket(
                            Bucket=bucket,
                            CreateBucketConfiguration={'LocationConstraint': S3_REGION}
                        )
                    else:
                        s3_client.create_bucket(Bucket=bucket)
            else:
                raise
    except NoCredentialsError:
        log_error("AWS credentials not found. Configure with: aws configure")
        return {}

    results = {}

    # Upload downloaded files
    if download_results:
        for name, info in download_results.items():
            if info.get("success") and Path(info["local_path"]).exists():
                s3_key = info["s3_key"]
                if s3_prefix:
                    s3_key = f"{s3_prefix}/{s3_key}"

                success = upload_file_to_s3(
                    s3_client,
                    Path(info["local_path"]),
                    bucket,
                    s3_key,
                    dry_run
                )
                results[name] = success

    # Also upload any extracted data directories
    data_dirs = [
        ("data/plateau/meshes", "plateau/meshes"),
        ("data/tokyo_gazebo", "tokyo_gazebo"),
        ("data/nuscenes", "nuscenes"),
        ("pretrained", "pretrained"),
    ]

    for local_subdir, s3_subdir in data_dirs:
        local_path = project_root / local_subdir
        if local_path.exists() and any(local_path.iterdir()):
            s3_key_prefix = s3_subdir
            if s3_prefix:
                s3_key_prefix = f"{s3_prefix}/{s3_subdir}"

            log_info(f"Uploading directory: {local_subdir} -> s3://{bucket}/{s3_key_prefix}")

            dir_results = upload_directory_to_s3(
                s3_client,
                local_path,
                bucket,
                s3_key_prefix,
                dry_run
            )

            results[local_subdir] = dir_results

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download data locally and upload to S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check which files already exist locally
  python scripts/download_and_upload_to_s3.py --check-only

  # Download everything and upload to default bucket
  python scripts/download_and_upload_to_s3.py

  # Upload existing files only (skip download)
  python scripts/download_and_upload_to_s3.py --skip-download

  # Preview what would be uploaded
  python scripts/download_and_upload_to_s3.py --dry-run

  # Download and upload in parallel (upload starts as each download completes)
  python scripts/download_and_upload_to_s3.py --parallel-upload

  # Use different bucket
  python scripts/download_and_upload_to_s3.py --bucket my-bucket

  # Download specific sources only
  python scripts/download_and_upload_to_s3.py --sources plateau_obj occworld_checkpoint
        """
    )

    parser.add_argument(
        "--bucket", "-b",
        default=DEFAULT_BUCKET,
        help=f"S3 bucket name (default: {DEFAULT_BUCKET})"
    )
    parser.add_argument(
        "--prefix", "-p",
        default="",
        help="S3 key prefix for all uploads"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step, only upload existing files"
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip upload step, only download files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(DOWNLOAD_SOURCES.keys()),
        help="Specific sources to download (default: all)"
    )
    parser.add_argument(
        "--output", "-o",
        default=".",
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check which files exist locally, don't download or upload"
    )
    parser.add_argument(
        "--parallel-upload",
        action="store_true",
        help="Upload files in parallel as each download completes"
    )

    args = parser.parse_args()

    project_root = Path(args.output).resolve()

    print("=" * 60)
    print("  VeryLargeWeebModel Data Download & S3 Upload")
    print("=" * 60)
    print()
    print(f"Project root:  {project_root}")
    print(f"S3 bucket:     s3://{args.bucket}")
    if args.prefix:
        print(f"S3 prefix:     {args.prefix}")
    print(f"Dry run:       {args.dry_run}")
    print(f"Parallel upload: {args.parallel_upload}")
    print()

    # Get list of sources to process
    sources_to_process = args.sources if args.sources else list(DOWNLOAD_SOURCES.keys())

    # First, check which files already exist locally
    file_status = check_local_files(project_root, sources_to_process)
    print_file_status(file_status)

    # Check-only mode - just show status and exit
    if args.check_only:
        complete_count = sum(1 for s in file_status.values() if s["is_complete"])
        total_count = len(file_status)
        print()
        print(f"Files complete: {complete_count}/{total_count}")
        print()
        return

    # Determine what needs downloading
    files_to_download = [
        name for name, info in file_status.items()
        if not info["is_complete"]
    ]
    files_already_complete = [
        name for name, info in file_status.items()
        if info["is_complete"]
    ]

    if files_already_complete:
        log_info(f"{len(files_already_complete)} files already complete locally")
    if files_to_download:
        log_info(f"{len(files_to_download)} files need to be downloaded")

    # Process files with parallel upload mode
    download_results = {}

    if args.parallel_upload and not args.skip_download and not args.skip_upload:
        # Combined download + upload mode: upload each file as soon as it's downloaded
        log_step("Download & Upload (parallel upload mode)")

        if not HAS_BOTO3:
            log_error("boto3 is required for S3 upload. Install with: pip install boto3")
            sys.exit(1)

        # Process files that need downloading
        for name in files_to_download:
            source = DOWNLOAD_SOURCES[name]
            _, result = download_and_upload_single(
                name, source, project_root,
                args.bucket, args.prefix,
                args.dry_run, args.skip_upload
            )
            download_results[name] = {
                "success": result["download_success"],
                "local_path": result["local_path"],
                "s3_key": result["s3_key"],
                "upload_success": result["upload_success"],
            }

        # Also upload files that were already complete locally
        if files_already_complete and not args.skip_upload:
            log_step("Uploading pre-existing complete files")
            try:
                s3_client = get_s3_client()
                for name in files_already_complete:
                    info = file_status[name]
                    s3_key = info["s3_key"]
                    if args.prefix:
                        s3_key = f"{args.prefix}/{s3_key}"

                    success = upload_file_to_s3(
                        s3_client,
                        Path(info["path"]),
                        args.bucket,
                        s3_key,
                        args.dry_run
                    )
                    download_results[name] = {
                        "success": True,
                        "local_path": info["path"],
                        "s3_key": s3_key,
                        "upload_success": success,
                    }
            except Exception as e:
                log_error(f"Failed to upload pre-existing files: {e}")

        # Print combined summary
        log_step("Download & Upload Summary")
        for name, info in download_results.items():
            dl_status = "OK" if info.get("success") else "FAILED"
            ul_status = "OK" if info.get("upload_success") else "FAILED"
            print(f"  {name}: download={dl_status}, upload={ul_status}")

    else:
        # Original sequential mode
        # Download
        if not args.skip_download:
            download_results = download_all(project_root, sources_to_process)

            # Print download summary
            log_step("Download Summary")
            for name, info in download_results.items():
                status = "OK" if info["success"] else "FAILED"
                print(f"  {name}: {status}")
        else:
            # Build results from existing files
            for name, source in DOWNLOAD_SOURCES.items():
                if args.sources and name not in args.sources:
                    continue
                local_path = project_root / source["local_path"]
                download_results[name] = {
                    "success": local_path.exists(),
                    "local_path": str(local_path),
                    "s3_key": source["s3_key"],
                }

        # Upload to S3
        if not args.skip_upload:
            if not HAS_BOTO3:
                log_error("boto3 is required for S3 upload. Install with: pip install boto3")
                sys.exit(1)

            upload_results = upload_all_to_s3(
                project_root,
                args.bucket,
                args.prefix,
                download_results,
                args.dry_run
            )

            # Print upload summary
            log_step("Upload Summary")
            for name, result in upload_results.items():
                if isinstance(result, dict):
                    print(f"  {name}: {result.get('success', 0)} uploaded, {result.get('skipped', 0)} skipped, {result.get('failed', 0)} failed")
                else:
                    status = "OK" if result else "FAILED"
                    print(f"  {name}: {status}")

    print()
    print("=" * 60)
    print("  Complete!")
    print("=" * 60)
    print()
    print(f"Data available at: s3://{args.bucket}/")
    print()
    print("To sync from S3 on remote server:")
    print(f"  aws s3 sync s3://{args.bucket}/ ./data/ --exclude '*.zip'")
    print()


if __name__ == "__main__":
    main()
