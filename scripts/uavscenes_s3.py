#!/usr/bin/env python3
"""
UAVScenes S3 Upload/Download Script

Downloads UAVScenes from Google Drive, uploads to S3, and syncs to remote servers.
Handles Google Drive quota limits by caching to S3 for reliable distribution.

Usage:
    # Download from GDrive and upload to S3 (run locally once)
    python scripts/uavscenes_s3.py --upload

    # Download from S3 (run on remote GPU servers)
    python scripts/uavscenes_s3.py --download

    # Check what's available
    python scripts/uavscenes_s3.py --status

    # Custom bucket
    python scripts/uavscenes_s3.py --download --bucket my-bucket

Requirements:
    pip install boto3 gdown tqdm
"""

import os
import sys
import argparse
import subprocess
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict

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
S3_PREFIX = "uavscenes"
S3_REGION = "us-west-2"

GDRIVE_FOLDER_ID = "1HSJWc5qmIKLdpaS8w8pqrWch4F9MHIeN"
GDRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"

# Expected scenes in the dataset
EXPECTED_SCENES = ["AMtown", "AMvalley", "HKairport", "HKisland"]


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
# Google Drive Download
# =============================================================================

def download_from_gdrive(output_dir: Path, use_rclone: bool = True) -> bool:
    """Download UAVScenes from Google Drive using gdown or rclone."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Try rclone first (best for large files, no quota limits)
    if use_rclone and subprocess.run(["which", "rclone"], capture_output=True).returncode == 0:
        # Check if gdrive remote exists
        result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
        if "gdrive:" in result.stdout:
            log_info("Using rclone (4 parallel transfers, no quota limits)...")

            cmd = [
                "rclone", "copy",
                "--progress",
                "--transfers=4",
                "--drive-acknowledge-abuse",
                f"gdrive:/{GDRIVE_FOLDER_ID}",
                str(output_dir)
            ]

            result = subprocess.run(cmd)
            if result.returncode == 0:
                log_success("Downloaded from Google Drive via rclone")
                return True
            else:
                log_warn("rclone download failed, trying gdown...")
        else:
            log_warn("rclone 'gdrive' remote not configured, trying gdown...")

    # Try gdown (simpler but may hit quota limits)
    try:
        import gdown
        log_info("Using gdown (may hit quota limits on large files)...")

        gdown.download_folder(
            url=GDRIVE_FOLDER_URL,
            output=str(output_dir),
            quiet=False,
            remaining_ok=True  # Continue even if some files fail
        )

        log_success("Downloaded from Google Drive via gdown")
        return True

    except ImportError:
        log_error("gdown not installed. Install with: pip install gdown")
    except Exception as e:
        log_warn(f"gdown download failed: {e}")

    # Try gdown CLI as fallback
    if subprocess.run(["which", "gdown"], capture_output=True).returncode == 0:
        log_info("Trying gdown CLI...")

        cmd = [
            "gdown", "--folder", "--remaining-ok",
            GDRIVE_FOLDER_URL,
            "-O", str(output_dir)
        ]

        result = subprocess.run(cmd)
        if result.returncode == 0:
            log_success("Downloaded from Google Drive via gdown CLI")
            return True

    log_error("Failed to download from Google Drive")
    log_info("Install gdown: pip install gdown")
    log_info("Or configure rclone: rclone config (create 'gdrive' remote)")
    return False


# =============================================================================
# S3 Functions
# =============================================================================

def get_s3_client():
    """Get S3 client."""
    if not HAS_BOTO3:
        raise ImportError("boto3 required. Install with: pip install boto3")
    return boto3.client('s3', region_name=S3_REGION)


def s3_file_exists(s3_client, bucket: str, key: str, local_size: Optional[int] = None) -> bool:
    """Check if file exists in S3 (optionally check size matches)."""
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        if local_size is not None:
            return response['ContentLength'] == local_size
        return True
    except ClientError:
        return False


def list_s3_files(s3_client, bucket: str, prefix: str) -> List[Dict]:
    """List all files in S3 under prefix."""
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'modified': obj['LastModified']
                })
    except ClientError as e:
        log_error(f"Failed to list S3 bucket: {e}")

    return files


def upload_to_s3(
    local_dir: Path,
    bucket: str,
    prefix: str,
    dry_run: bool = False,
    max_workers: int = 4
) -> Dict:
    """Upload directory to S3 with parallel uploads."""

    if not HAS_BOTO3:
        log_error("boto3 required. Install with: pip install boto3")
        return {"success": 0, "failed": 0, "skipped": 0}

    s3_client = get_s3_client()

    # Ensure bucket exists
    try:
        s3_client.head_bucket(Bucket=bucket)
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            log_info(f"Creating bucket: {bucket}")
            if not dry_run:
                if S3_REGION != 'us-east-1':
                    s3_client.create_bucket(
                        Bucket=bucket,
                        CreateBucketConfiguration={'LocationConstraint': S3_REGION}
                    )
                else:
                    s3_client.create_bucket(Bucket=bucket)
        else:
            raise

    # Collect files to upload
    files_to_upload = []
    for filepath in local_dir.rglob("*"):
        if filepath.is_file():
            relative = filepath.relative_to(local_dir)
            s3_key = f"{prefix}/{relative}".replace("\\", "/")
            files_to_upload.append((filepath, s3_key))

    log_info(f"Found {len(files_to_upload)} files to upload")

    results = {"success": 0, "failed": 0, "skipped": 0}

    # Transfer config for large files
    transfer_config = TransferConfig(
        multipart_threshold=100 * 1024 * 1024,  # 100MB
        max_concurrency=10,
        multipart_chunksize=100 * 1024 * 1024,
    )

    def upload_file(args):
        filepath, s3_key = args
        file_size = filepath.stat().st_size

        # Skip if already exists with same size
        if s3_file_exists(s3_client, bucket, s3_key, file_size):
            return ("skipped", filepath, s3_key)

        if dry_run:
            return ("success", filepath, s3_key)

        try:
            s3_client.upload_file(
                str(filepath),
                bucket,
                s3_key,
                Config=transfer_config
            )
            return ("success", filepath, s3_key)
        except Exception as e:
            return ("failed", filepath, s3_key, str(e))

    # Upload with progress
    if HAS_TQDM:
        with tqdm(total=len(files_to_upload), desc="Uploading") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(upload_file, f): f for f in files_to_upload}
                for future in as_completed(futures):
                    result = future.result()
                    status = result[0]
                    results[status] += 1
                    pbar.update(1)

                    if status == "failed":
                        log_warn(f"Failed: {result[1].name} - {result[3]}")
    else:
        for i, f in enumerate(files_to_upload):
            result = upload_file(f)
            status = result[0]
            results[status] += 1
            print(f"\r  Uploading: {i+1}/{len(files_to_upload)}", end="")
        print()

    return results


def download_from_s3(
    bucket: str,
    prefix: str,
    local_dir: Path,
    dry_run: bool = False,
    max_workers: int = 4
) -> Dict:
    """Download from S3 to local directory with parallel downloads."""

    if not HAS_BOTO3:
        log_error("boto3 required. Install with: pip install boto3")
        return {"success": 0, "failed": 0, "skipped": 0}

    s3_client = get_s3_client()
    local_dir.mkdir(parents=True, exist_ok=True)

    # List files in S3
    s3_files = list_s3_files(s3_client, bucket, prefix)

    if not s3_files:
        log_warn(f"No files found in s3://{bucket}/{prefix}")
        return {"success": 0, "failed": 0, "skipped": 0}

    log_info(f"Found {len(s3_files)} files in S3")

    # Calculate total size
    total_size = sum(f['size'] for f in s3_files)
    log_info(f"Total size: {total_size / 1024 / 1024 / 1024:.2f} GB")

    results = {"success": 0, "failed": 0, "skipped": 0}

    # Transfer config
    transfer_config = TransferConfig(
        max_concurrency=10,
        multipart_chunksize=100 * 1024 * 1024,
    )

    def download_file(s3_file):
        s3_key = s3_file['key']
        s3_size = s3_file['size']

        # Calculate local path
        relative_key = s3_key[len(prefix):].lstrip('/')
        local_path = local_dir / relative_key

        # Skip if already exists with same size
        if local_path.exists() and local_path.stat().st_size == s3_size:
            return ("skipped", local_path)

        if dry_run:
            return ("success", local_path)

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(
                bucket,
                s3_key,
                str(local_path),
                Config=transfer_config
            )
            return ("success", local_path)
        except Exception as e:
            return ("failed", local_path, str(e))

    # Download with progress
    if HAS_TQDM:
        with tqdm(total=len(s3_files), desc="Downloading") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(download_file, f): f for f in s3_files}
                for future in as_completed(futures):
                    result = future.result()
                    status = result[0]
                    results[status] += 1
                    pbar.update(1)

                    if status == "failed":
                        log_warn(f"Failed: {result[1].name} - {result[2]}")
    else:
        for i, f in enumerate(s3_files):
            result = download_file(f)
            status = result[0]
            results[status] += 1
            print(f"\r  Downloading: {i+1}/{len(s3_files)}", end="")
        print()

    return results


def show_status(bucket: str, prefix: str, local_dir: Path):
    """Show status of S3 and local files."""

    log_step("Checking status...")

    # Check local files
    print(f"\n{Colors.CYAN}Local ({local_dir}):{Colors.NC}")
    if local_dir.exists():
        local_files = list(local_dir.rglob("*"))
        local_files = [f for f in local_files if f.is_file()]
        local_size = sum(f.stat().st_size for f in local_files)
        print(f"  Files: {len(local_files)}")
        print(f"  Size:  {local_size / 1024 / 1024 / 1024:.2f} GB")

        # Check for expected scenes
        for scene in EXPECTED_SCENES:
            scene_dir = local_dir / f"interval1_{scene}01"
            if scene_dir.exists():
                lidar_dir = scene_dir / "interval1_LIDAR"
                if lidar_dir.exists():
                    frames = len(list(lidar_dir.glob("*")))
                    print(f"  {Colors.GREEN}[OK]{Colors.NC} {scene}: {frames} LiDAR frames")
                else:
                    print(f"  {Colors.YELLOW}[!]{Colors.NC} {scene}: no LiDAR data")
            else:
                print(f"  {Colors.RED}[X]{Colors.NC} {scene}: not found")
    else:
        print(f"  {Colors.RED}Directory does not exist{Colors.NC}")

    # Check S3
    print(f"\n{Colors.CYAN}S3 (s3://{bucket}/{prefix}):{Colors.NC}")
    if HAS_BOTO3:
        try:
            s3_client = get_s3_client()
            s3_files = list_s3_files(s3_client, bucket, prefix)

            if s3_files:
                s3_size = sum(f['size'] for f in s3_files)
                print(f"  Files: {len(s3_files)}")
                print(f"  Size:  {s3_size / 1024 / 1024 / 1024:.2f} GB")

                # Check for scenes
                for scene in EXPECTED_SCENES:
                    scene_files = [f for f in s3_files if f"interval1_{scene}01" in f['key']]
                    if scene_files:
                        print(f"  {Colors.GREEN}[OK]{Colors.NC} {scene}: {len(scene_files)} files")
                    else:
                        print(f"  {Colors.RED}[X]{Colors.NC} {scene}: not found")
            else:
                print(f"  {Colors.YELLOW}No files found{Colors.NC}")

        except NoCredentialsError:
            print(f"  {Colors.RED}AWS credentials not configured{Colors.NC}")
        except ClientError as e:
            print(f"  {Colors.RED}Error: {e}{Colors.NC}")
    else:
        print(f"  {Colors.YELLOW}boto3 not installed{Colors.NC}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="UAVScenes S3 Upload/Download",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from GDrive and upload to S3 (run once locally)
  python scripts/uavscenes_s3.py --upload

  # Download from S3 to remote server
  python scripts/uavscenes_s3.py --download

  # Check status
  python scripts/uavscenes_s3.py --status

  # Use custom bucket
  python scripts/uavscenes_s3.py --download --bucket my-training-data
        """
    )

    # Actions
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--upload", action="store_true",
                       help="Download from GDrive and upload to S3")
    action.add_argument("--download", action="store_true",
                       help="Download from S3 to local")
    action.add_argument("--status", action="store_true",
                       help="Show status of local and S3 files")

    # Options
    parser.add_argument("--bucket", "-b", default=DEFAULT_BUCKET,
                       help=f"S3 bucket (default: {DEFAULT_BUCKET})")
    parser.add_argument("--prefix", "-p", default=S3_PREFIX,
                       help=f"S3 prefix (default: {S3_PREFIX})")
    parser.add_argument("--output", "-o", default="data/uavscenes",
                       help="Local output directory (default: data/uavscenes)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without doing it")
    parser.add_argument("--workers", "-w", type=int, default=4,
                       help="Parallel upload/download workers (default: 4)")
    parser.add_argument("--skip-gdrive", action="store_true",
                       help="Skip GDrive download (upload existing files only)")
    parser.add_argument("--no-rclone", action="store_true",
                       help="Don't use rclone even if available")

    args = parser.parse_args()

    # Determine project root and output dir
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / args.output

    print("=" * 60)
    print("  UAVScenes S3 Manager")
    print("=" * 60)
    print()
    print(f"Local dir:  {output_dir}")
    print(f"S3 bucket:  s3://{args.bucket}/{args.prefix}")
    print(f"Dry run:    {args.dry_run}")
    print()

    if args.status:
        show_status(args.bucket, args.prefix, output_dir)
        return

    if args.upload:
        # Step 1: Download from Google Drive
        if not args.skip_gdrive:
            log_step("Step 1: Download from Google Drive")
            if not download_from_gdrive(output_dir, use_rclone=not args.no_rclone):
                log_error("Google Drive download failed")
                log_info("You can retry with existing files: --skip-gdrive")
                if not output_dir.exists() or not any(output_dir.iterdir()):
                    sys.exit(1)
        else:
            log_info("Skipping Google Drive download")

        # Step 2: Upload to S3
        log_step("Step 2: Upload to S3")
        if not output_dir.exists():
            log_error(f"Output directory not found: {output_dir}")
            sys.exit(1)

        results = upload_to_s3(
            output_dir,
            args.bucket,
            args.prefix,
            dry_run=args.dry_run,
            max_workers=args.workers
        )

        print()
        log_step("Upload Summary")
        print(f"  Uploaded: {results['success']}")
        print(f"  Skipped:  {results['skipped']}")
        print(f"  Failed:   {results['failed']}")

        print()
        print("=" * 60)
        print("  Upload Complete!")
        print("=" * 60)
        print()
        print("To download on remote servers:")
        print(f"  python scripts/uavscenes_s3.py --download --bucket {args.bucket}")
        print()
        print("Or using AWS CLI:")
        print(f"  aws s3 sync s3://{args.bucket}/{args.prefix}/ {args.output}/")
        print()

    elif args.download:
        log_step("Downloading from S3")

        results = download_from_s3(
            args.bucket,
            args.prefix,
            output_dir,
            dry_run=args.dry_run,
            max_workers=args.workers
        )

        print()
        log_step("Download Summary")
        print(f"  Downloaded: {results['success']}")
        print(f"  Skipped:    {results['skipped']}")
        print(f"  Failed:     {results['failed']}")

        # Verify
        print()
        log_step("Verification")
        for scene in EXPECTED_SCENES:
            lidar_dir = output_dir / f"interval1_{scene}01" / "interval1_LIDAR"
            if lidar_dir.exists():
                frames = len(list(lidar_dir.glob("*")))
                print(f"  {Colors.GREEN}[OK]{Colors.NC} {scene}: {frames} LiDAR frames")
            else:
                print(f"  {Colors.RED}[X]{Colors.NC} {scene}: not found")

        print()
        print("=" * 60)
        print("  Download Complete!")
        print("=" * 60)
        print()
        print("To start training:")
        print("  python train.py --py-config config/finetune_uavscenes.py")
        print()


if __name__ == "__main__":
    main()
