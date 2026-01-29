"""S3 utilities for uploading and downloading data."""

from typing import Any, Optional, List, Dict

from .logging import log_info, log_success, log_error
from .dependencies import HAS_BOTO3

if HAS_BOTO3:
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.exceptions import ClientError, NoCredentialsError
else:
    boto3 = None
    TransferConfig = None
    ClientError = Exception
    NoCredentialsError = Exception

# Default configuration
S3_REGION: str = "us-west-2"
DEFAULT_BUCKET: str = "verylargeweebmodel"


def get_s3_client(region: Optional[str] = None) -> Any:
    """Get an S3 client.

    Args:
        region: AWS region name (defaults to S3_REGION)

    Returns:
        boto3 S3 client

    Raises:
        ImportError: If boto3 is not installed
    """
    if not HAS_BOTO3:
        raise ImportError("boto3 is required. Install with: pip install boto3")
    return boto3.client('s3', region_name=region or S3_REGION)


def check_aws_credentials() -> bool:
    """Check if AWS credentials are configured.

    Returns:
        True if credentials are valid, False otherwise
    """
    if not HAS_BOTO3:
        return False
    try:
        import socket
        socket.setdefaulttimeout(10)
        sts = boto3.client('sts')
        sts.get_caller_identity()
        return True
    except Exception as e:
        log_error(f"AWS credential check failed: {e}")
        return False


def s3_file_exists(
    s3_client: Any,
    bucket: str,
    key: str,
    local_size: Optional[int] = None
) -> bool:
    """Check if a file exists in S3, optionally checking size matches.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key
        local_size: If provided, also verify file size matches

    Returns:
        True if file exists (and size matches if specified)
    """
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        if local_size is not None:
            return response['ContentLength'] == local_size
        return True
    except ClientError:
        return False


def list_s3_files(s3_client: Any, bucket: str, prefix: str) -> List[Dict[str, Any]]:
    """List all files in S3 under a prefix.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 key prefix to list

    Returns:
        List of dicts with 'key', 'size', and 'modified' for each file
    """
    files: List[Dict[str, Any]] = []
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


def ensure_bucket_exists(
    s3_client: Any,
    bucket: str,
    region: Optional[str] = None,
    dry_run: bool = False
) -> bool:
    """Create S3 bucket if it doesn't exist.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        region: AWS region for bucket creation
        dry_run: If True, don't actually create

    Returns:
        True if bucket exists or was created successfully
    """
    region = region or S3_REGION

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
                if region != 'us-east-1':
                    s3_client.create_bucket(
                        Bucket=bucket,
                        CreateBucketConfiguration={'LocationConstraint': region}
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


def get_transfer_config(
    multipart_threshold_mb: int = 100,
    max_concurrency: int = 10,
    multipart_chunksize_mb: int = 100
) -> Any:
    """Get a TransferConfig for large file uploads/downloads.

    Args:
        multipart_threshold_mb: Size threshold for multipart uploads (MB)
        max_concurrency: Maximum concurrent transfers
        multipart_chunksize_mb: Size of each multipart chunk (MB)

    Returns:
        boto3 TransferConfig object

    Raises:
        ImportError: If boto3 is not installed
    """
    if not HAS_BOTO3:
        raise ImportError("boto3 is required. Install with: pip install boto3")

    return TransferConfig(
        multipart_threshold=multipart_threshold_mb * 1024 * 1024,
        max_concurrency=max_concurrency,
        multipart_chunksize=multipart_chunksize_mb * 1024 * 1024,
    )


def upload_file(
    s3_client: Any,
    local_path: str,
    bucket: str,
    s3_key: str,
    transfer_config: Optional[Any] = None,
    dry_run: bool = False
) -> bool:
    """Upload a single file to S3.

    Args:
        s3_client: boto3 S3 client
        local_path: Local file path
        bucket: S3 bucket name
        s3_key: S3 object key
        transfer_config: Optional TransferConfig for large files
        dry_run: If True, don't actually upload

    Returns:
        True if successful, False otherwise
    """
    if dry_run:
        log_info(f"[DRY RUN] Would upload: {local_path} -> s3://{bucket}/{s3_key}")
        return True

    try:
        if transfer_config:
            s3_client.upload_file(local_path, bucket, s3_key, Config=transfer_config)
        else:
            s3_client.upload_file(local_path, bucket, s3_key)
        return True
    except Exception as e:
        log_error(f"Failed to upload {local_path}: {e}")
        return False


def download_file(
    s3_client: Any,
    bucket: str,
    s3_key: str,
    local_path: str,
    transfer_config: Optional[Any] = None,
    dry_run: bool = False
) -> bool:
    """Download a single file from S3.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        s3_key: S3 object key
        local_path: Local file path to save to
        transfer_config: Optional TransferConfig for large files
        dry_run: If True, don't actually download

    Returns:
        True if successful, False otherwise
    """
    import os
    if dry_run:
        log_info(f"[DRY RUN] Would download: s3://{bucket}/{s3_key} -> {local_path}")
        return True

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if transfer_config:
            s3_client.download_file(bucket, s3_key, local_path, Config=transfer_config)
        else:
            s3_client.download_file(bucket, s3_key, local_path)
        return True
    except Exception as e:
        log_error(f"Failed to download {s3_key}: {e}")
        return False


def upload_directory(
    s3_client: Any,
    local_dir: str,
    bucket: str,
    prefix: str,
    transfer_config: Optional[Any] = None,
    dry_run: bool = False,
    skip_existing: bool = True
) -> Dict[str, int]:
    """Upload a directory to S3.

    Args:
        s3_client: boto3 S3 client
        local_dir: Local directory path
        bucket: S3 bucket name
        prefix: S3 key prefix
        transfer_config: Optional TransferConfig for large files
        dry_run: If True, don't actually upload
        skip_existing: Skip files that already exist with same size

    Returns:
        Dict with counts: {"success": N, "failed": N, "skipped": N}
    """
    from pathlib import Path

    local_path = Path(local_dir)
    results: Dict[str, int] = {"success": 0, "failed": 0, "skipped": 0}

    for filepath in local_path.rglob("*"):
        if not filepath.is_file():
            continue

        relative = filepath.relative_to(local_path)
        s3_key = f"{prefix}/{relative}".replace("\\", "/")

        # Check if already exists
        if skip_existing:
            file_size = filepath.stat().st_size
            if s3_file_exists(s3_client, bucket, s3_key, file_size):
                results["skipped"] += 1
                continue

        if upload_file(s3_client, str(filepath), bucket, s3_key, transfer_config, dry_run):
            results["success"] += 1
        else:
            results["failed"] += 1

    return results


def download_directory(
    s3_client: Any,
    bucket: str,
    prefix: str,
    local_dir: str,
    transfer_config: Optional[Any] = None,
    dry_run: bool = False,
    skip_existing: bool = True
) -> Dict[str, int]:
    """Download all files under an S3 prefix to a local directory.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 key prefix
        local_dir: Local directory path
        transfer_config: Optional TransferConfig for large files
        dry_run: If True, don't actually download
        skip_existing: Skip files that already exist with same size

    Returns:
        Dict with counts: {"success": N, "failed": N, "skipped": N}
    """
    import os

    results: Dict[str, int] = {"success": 0, "failed": 0, "skipped": 0}

    # List all files under prefix
    files = list_s3_files(s3_client, bucket, prefix)

    for file_info in files:
        s3_key = file_info['key']
        relative_path = s3_key[len(prefix):].lstrip('/')
        local_path = os.path.join(local_dir, relative_path)

        # Check if already exists
        if skip_existing and os.path.exists(local_path):
            if os.path.getsize(local_path) == file_info['size']:
                results["skipped"] += 1
                continue

        if download_file(s3_client, bucket, s3_key, local_path, transfer_config, dry_run):
            results["success"] += 1
        else:
            results["failed"] += 1

    return results
