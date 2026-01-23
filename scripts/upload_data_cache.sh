#!/bin/bash
# =============================================================================
# Upload PLATEAU data to S3/Cloud Storage for faster access on Vast.ai
# =============================================================================
#
# This script uploads the PLATEAU data to your own S3 bucket or cloud storage
# so future Vast.ai instances can download from a faster/closer location.
#
# Usage:
#   ./scripts/upload_data_cache.sh s3://your-bucket/plateau-cache
#   ./scripts/upload_data_cache.sh gs://your-bucket/plateau-cache  # GCS
#
# Then on Vast.ai:
#   export DATA_MIRROR="https://your-bucket.s3.amazonaws.com/plateau-cache"
#   ./scripts/setup_and_train.sh
#
# =============================================================================

set -e

DEST="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"

if [ -z "$DEST" ]; then
    echo "Usage: $0 <destination>"
    echo ""
    echo "Examples:"
    echo "  $0 s3://my-bucket/plateau-cache"
    echo "  $0 gs://my-bucket/plateau-cache"
    echo ""
    echo "Prerequisites:"
    echo "  - AWS CLI configured (aws configure) for S3"
    echo "  - gsutil configured for GCS"
    echo ""
    echo "Files to upload:"
    echo "  data/plateau/raw/tokyo23ku_obj.zip (~2.1GB)"
    echo ""
    exit 1
fi

# Check if source file exists
PLATEAU_ZIP="${DATA_DIR}/plateau/raw/tokyo23ku_obj.zip"

if [ ! -f "$PLATEAU_ZIP" ]; then
    echo "Error: PLATEAU data not found at $PLATEAU_ZIP"
    echo ""
    echo "Download it first:"
    echo "  wget -O $PLATEAU_ZIP \\"
    echo "    'https://gic-plateau.s3.ap-northeast-1.amazonaws.com/2020/13100_tokyo23-ku_2020_obj_3_op.zip'"
    exit 1
fi

FILE_SIZE=$(stat -c%s "$PLATEAU_ZIP" 2>/dev/null || stat -f%z "$PLATEAU_ZIP" 2>/dev/null)
echo "Uploading PLATEAU data ($(numfmt --to=iec $FILE_SIZE 2>/dev/null || echo "$FILE_SIZE bytes"))..."

# Detect storage type and upload
if [[ "$DEST" == s3://* ]]; then
    echo "Uploading to AWS S3..."
    aws s3 cp "$PLATEAU_ZIP" "${DEST}/tokyo23ku_obj.zip" --progress

    # Make public readable (optional - remove if you want private)
    echo ""
    echo "To make the file publicly accessible:"
    echo "  aws s3api put-object-acl --bucket YOUR_BUCKET --key plateau-cache/tokyo23ku_obj.zip --acl public-read"
    echo ""
    echo "Your mirror URL:"
    BUCKET=$(echo "$DEST" | sed 's|s3://||' | cut -d'/' -f1)
    PATH_PREFIX=$(echo "$DEST" | sed 's|s3://[^/]*/||')
    echo "  https://${BUCKET}.s3.amazonaws.com/${PATH_PREFIX}"

elif [[ "$DEST" == gs://* ]]; then
    echo "Uploading to Google Cloud Storage..."
    gsutil cp "$PLATEAU_ZIP" "${DEST}/tokyo23ku_obj.zip"

    echo ""
    echo "To make the file publicly accessible:"
    echo "  gsutil acl ch -u AllUsers:R ${DEST}/tokyo23ku_obj.zip"
    echo ""
    echo "Your mirror URL:"
    BUCKET=$(echo "$DEST" | sed 's|gs://||' | cut -d'/' -f1)
    PATH_PREFIX=$(echo "$DEST" | sed 's|gs://[^/]*/||')
    echo "  https://storage.googleapis.com/${BUCKET}/${PATH_PREFIX}"

elif [[ "$DEST" == /* ]] || [[ "$DEST" == ./* ]]; then
    echo "Copying to local path..."
    mkdir -p "$DEST"
    cp "$PLATEAU_ZIP" "${DEST}/tokyo23ku_obj.zip"
    echo "Copied to ${DEST}/tokyo23ku_obj.zip"

else
    echo "Unknown destination type: $DEST"
    echo "Supported: s3://, gs://, or local path"
    exit 1
fi

echo ""
echo "=============================================="
echo "Upload complete!"
echo "=============================================="
echo ""
echo "On your Vast.ai instance, use:"
echo ""
echo "  export DATA_MIRROR=\"YOUR_MIRROR_URL\""
echo "  ./scripts/setup_and_train.sh"
echo ""
echo "Or:"
echo ""
echo "  ./scripts/setup_and_train.sh --mirror YOUR_MIRROR_URL"
echo ""
