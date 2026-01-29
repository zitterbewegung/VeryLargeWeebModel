"""S3 utilities for uploading and downloading data."""

from typing import Optional, List, Dict

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
S3_REGION = "us-west-2"
DEFAULT_BUCKET = "verylargeweebmodel"


def get_s3_client(region: str = None):
    """Get an S3 client."""
    if not HAS_BOTO3:
        raise ImportError("boto3 is required. Install with: pip install boto3")
    return boto3.client('s3', region_name=region or S3_REGION)


def check_aws_credentials() -> bool:
    """Check if AWS credentials are configured."""
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
    s3_client,
    bucket: str,
    key: str,
    local_size: Optional[int] = None
) -> bool:
    """Check if a file exists in S3, optionally checking size matches."""
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        if local_size is not None:
            return response['ContentLength'] == local_size
        return True
    except ClientError:
        return False


def list_s3_files(s3_client, bucket: str, prefix: str) -> List[Dict]:
    """List all files in S3 under a prefix."""
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


def ensure_bucket_exists(
    s3_client,
    bucket: str,
    region: str = None,
    dry_run: bool = False
) -> bool:
    """Create S3 bucket if it doesn't exist."""
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
):
    """Get a TransferConfig for large file uploads/downloads."""
    if not HAS_BOTO3:
        raise ImportError("boto3 is required. Install with: pip install boto3")

    return TransferConfig(
        multipart_threshold=multipart_threshold_mb * 1024 * 1024,
        max_concurrency=max_concurrency,
        multipart_chunksize=multipart_chunksize_mb * 1024 * 1024,
    )
