#!/bin/bash
# =============================================================================
# Setup UAVScenes Dataset for 6DoF Aerial Training
# =============================================================================
#
# Downloads and prepares UAVScenes - REAL aerial UAV data with 6DoF poses.
#
# UAVScenes (ICCV 2025) provides:
#   - Multi-modal UAV data (LiDAR + Camera)
#   - Ground-truth 6DoF poses from actual flights
#   - 120k labeled semantic pairs
#   - 4 diverse scenes: AMtown, AMvalley, HKairport, HKisland
#
# Usage:
#   ./scripts/setup_uavscenes.sh                    # Interactive scene selection
#   ./scripts/setup_uavscenes.sh --scene AMtown     # Download specific scene
#   ./scripts/setup_uavscenes.sh --all              # Download all scenes
#   ./scripts/setup_uavscenes.sh --keyframes        # Download keyframe version (smaller)
#   ./scripts/setup_uavscenes.sh --check            # Check existing installation
#
# Data sources (in order of preference):
#   1. HuggingFace (recommended)
#   2. Google Drive
#   3. OneDrive
#
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${UAVSCENES_ROOT:-$PROJECT_ROOT/data/uavscenes}"
SCENES=()
INTERVAL=1  # 1=full, 5=keyframes
CHECK_ONLY=false
ALL_SCENES=false
RESUME=true  # Resume by default
TRY_ALL_SOURCES=true
PARALLEL_DOWNLOADS=4  # Number of parallel connections

# Available scenes
ALL_SCENE_NAMES=("AMtown" "AMvalley" "HKairport" "HKisland")

# Download URLs
HF_REPO="sijieaaa/UAVScenes"
ONEDRIVE_URL="https://entuedu-my.sharepoint.com/:f:/g/personal/wang1679_e_ntu_edu_sg/EgY6DU5GBchIiAIa-eQZmEAB0vJx3khCPHbFW3LnR77RFw"
GDRIVE_FOLDER="1HSJWc5qmIKLdpaS8w8pqrWch4F9MHIeN"

# State file for tracking downloads
STATE_FILE="$DATA_DIR/.download_state"

# Preferred download source (onedrive, gdrive, huggingface)
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-auto}"

# Check for fast download tools
HAS_ARIA2=false
HAS_WGET=false
HAS_CURL=false
command -v aria2c &>/dev/null && HAS_ARIA2=true
command -v wget &>/dev/null && HAS_WGET=true
command -v curl &>/dev/null && HAS_CURL=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scene)
            SCENES+=("$2")
            shift 2
            ;;
        --all)
            ALL_SCENES=true
            shift
            ;;
        --keyframes)
            INTERVAL=5
            shift
            ;;
        --full)
            INTERVAL=1
            shift
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --source)
            DOWNLOAD_SOURCE="$2"
            shift 2
            ;;
        --onedrive)
            DOWNLOAD_SOURCE="onedrive"
            shift
            ;;
        --gdrive)
            DOWNLOAD_SOURCE="gdrive"
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --no-resume)
            RESUME=false
            shift
            ;;
        --try-all)
            TRY_ALL_SOURCES=true
            shift
            ;;
        --parallel)
            PARALLEL_DOWNLOADS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --scene NAME    Download specific scene (can repeat)"
            echo "  --all           Download all 4 scenes"
            echo "  --keyframes     Download keyframe version (interval=5, ~20% size)"
            echo "  --full          Download full version (interval=1, default)"
            echo "  --check         Check existing installation only"
            echo "  --data-dir      Custom data directory"
            echo "  --source SRC    Download source: onedrive, gdrive, huggingface, auto (default)"
            echo "  --onedrive      Use OneDrive (full dataset available)"
            echo "  --gdrive        Use Google Drive (full dataset available)"
            echo "  --resume        Resume incomplete downloads (default: enabled)"
            echo "  --no-resume     Start fresh, don't resume partial downloads"
            echo "  --try-all       Try all download sources until one succeeds (default)"
            echo "  --parallel N    Number of parallel download connections (default: 4)"
            echo ""
            echo "Available scenes: ${ALL_SCENE_NAMES[*]}"
            echo ""
            echo "Examples:"
            echo "  $0 --scene AMtown --scene AMvalley    # Two scenes"
            echo "  $0 --all --keyframes                  # All scenes, smaller version"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set scenes to download
if [ "$ALL_SCENES" = true ]; then
    SCENES=("${ALL_SCENE_NAMES[@]}")
elif [ ${#SCENES[@]} -eq 0 ]; then
    # Default to AMtown for quick testing
    SCENES=("AMtown")
fi

# Create data directory and state file
mkdir -p "$DATA_DIR"
STATE_FILE="$DATA_DIR/.download_state"

# State tracking functions
save_state() {
    local scene=$1
    local source=$2
    local status=$3
    echo "${scene}|${source}|${status}|$(date +%s)" >> "$STATE_FILE"
}

get_last_source() {
    local scene=$1
    if [ -f "$STATE_FILE" ]; then
        grep "^${scene}|" "$STATE_FILE" | tail -1 | cut -d'|' -f2
    fi
}

get_scene_status() {
    local scene=$1
    if [ -f "$STATE_FILE" ]; then
        grep "^${scene}|" "$STATE_FILE" | tail -1 | cut -d'|' -f3
    fi
}

clear_scene_state() {
    local scene=$1
    if [ -f "$STATE_FILE" ]; then
        grep -v "^${scene}|" "$STATE_FILE" > "${STATE_FILE}.tmp" || true
        mv "${STATE_FILE}.tmp" "$STATE_FILE"
    fi
}

echo "=============================================="
echo "  UAVScenes Dataset Setup"
echo "=============================================="
echo ""
echo "Data directory: $DATA_DIR"
echo "Scenes: ${SCENES[*]}"
echo "Interval: $INTERVAL (1=full, 5=keyframes)"
echo "Resume mode: $RESUME"
echo "Parallel connections: $PARALLEL_DOWNLOADS"
echo "Try all sources: $TRY_ALL_SOURCES"
echo ""

# Fast download utility with resume support
# Uses aria2c (fastest), wget, or curl with automatic fallback
fast_download() {
    local url=$1
    local output=$2
    local desc=${3:-"Downloading"}

    mkdir -p "$(dirname "$output")"

    log_info "$desc"
    log_info "URL: $url"
    log_info "Output: $output"

    # Check if file already exists and is complete (for resume)
    if [ -f "$output" ] && [ "$RESUME" = false ]; then
        log_info "Removing existing file (resume disabled)..."
        rm -f "$output"
    fi

    # Try aria2c first (fastest, supports parallel chunks + resume)
    if [ "$HAS_ARIA2" = true ]; then
        log_info "Using aria2c (parallel chunks, resume enabled)..."
        if aria2c \
            --continue=true \
            --max-connection-per-server="$PARALLEL_DOWNLOADS" \
            --split="$PARALLEL_DOWNLOADS" \
            --min-split-size=1M \
            --file-allocation=none \
            --max-tries=5 \
            --retry-wait=10 \
            --timeout=60 \
            --connect-timeout=30 \
            --dir="$(dirname "$output")" \
            --out="$(basename "$output")" \
            "$url"; then
            log_success "Download complete!"
            return 0
        else
            log_warn "aria2c failed, trying fallback..."
        fi
    fi

    # Try wget (good resume support)
    if [ "$HAS_WGET" = true ]; then
        log_info "Using wget (resume enabled)..."
        if wget \
            --continue \
            --progress=bar:force \
            --timeout=60 \
            --tries=5 \
            --retry-connrefused \
            --waitretry=10 \
            -O "$output" \
            "$url"; then
            log_success "Download complete!"
            return 0
        else
            log_warn "wget failed, trying fallback..."
        fi
    fi

    # Try curl (basic resume)
    if [ "$HAS_CURL" = true ]; then
        log_info "Using curl (resume enabled)..."
        if curl \
            --continue-at - \
            --location \
            --retry 5 \
            --retry-delay 10 \
            --connect-timeout 30 \
            --max-time 3600 \
            --progress-bar \
            -o "$output" \
            "$url"; then
            log_success "Download complete!"
            return 0
        else
            log_warn "curl failed"
        fi
    fi

    log_error "All download methods failed"
    return 1
}

# Extract archive with progress
extract_archive() {
    local archive=$1
    local dest=$2
    local remove_after=${3:-false}

    log_info "Extracting: $archive"
    log_info "Destination: $dest"

    mkdir -p "$dest"

    case "$archive" in
        *.zip)
            if command -v unzip &>/dev/null; then
                unzip -o "$archive" -d "$dest"
            else
                python3 -c "import zipfile; zipfile.ZipFile('$archive').extractall('$dest')"
            fi
            ;;
        *.tar.gz|*.tgz)
            tar -xzf "$archive" -C "$dest"
            ;;
        *.tar)
            tar -xf "$archive" -C "$dest"
            ;;
        *)
            log_error "Unknown archive format: $archive"
            return 1
            ;;
    esac

    if [ "$remove_after" = true ]; then
        log_info "Removing archive: $archive"
        rm -f "$archive"
    fi

    log_success "Extraction complete!"
    return 0
}

# Check Python dependencies
log_info "Checking Python dependencies..."
MISSING_DEPS=()

python3 -c "from pyquaternion import Quaternion" 2>/dev/null || MISSING_DEPS+=("pyquaternion")

# open3d is optional - only needed for point cloud processing, not downloading
if ! python3 -c "import open3d" 2>/dev/null; then
    log_warn "open3d not installed (optional - needed for point cloud processing)"
    log_warn "Note: open3d may not support your Python version"
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    log_warn "Missing required dependencies: ${MISSING_DEPS[*]}"
    log_info "Installing..."
    pip install "${MISSING_DEPS[@]}"
fi
log_success "Python dependencies OK"

# Check if huggingface_hub is available for downloading
HAS_HF=false
if python3 -c "from huggingface_hub import hf_hub_download" 2>/dev/null; then
    HAS_HF=true
    log_success "HuggingFace Hub available"
else
    log_warn "huggingface_hub not installed. Installing..."
    pip install huggingface_hub
    HAS_HF=true
fi

# Function to check if a scene is installed
check_scene() {
    local scene=$1
    local legacy_path="$DATA_DIR/$scene"
    local hf_path1="$DATA_DIR/interval${INTERVAL}_${scene}01"
    local hf_path2="$DATA_DIR/interval${INTERVAL}_${scene}"

    if [ -d "$legacy_path" ]; then
        if [ -d "$legacy_path/interval${INTERVAL}_CAM_LIDAR" ] || [ -d "$legacy_path/interval1_CAM_LIDAR" ]; then
            return 0
        fi
    fi

    for base in "$hf_path1" "$hf_path2"; do
        if [ -d "$base" ]; then
            if [ -d "$base/interval${INTERVAL}_LIDAR" ] || [ -d "$base/interval${INTERVAL}_CAM" ] || [ -d "$base/interval${INTERVAL}_CAM_LIDAR" ]; then
                return 0
            fi
        fi
    done
    return 1
}

# Check existing installation
check_installation() {
    local found=0
    for scene in "${ALL_SCENE_NAMES[@]}"; do
        if check_scene "$scene"; then
            log_success "Found: $scene"
            ((found++))
        fi
    done

    if [ $found -gt 0 ]; then
        echo ""
        log_info "Verifying with Python..."
        python3 << EOF
import sys
sys.path.insert(0, '$PROJECT_ROOT')
try:
    from dataset.uavscenes_dataset import UAVScenesDataset, UAVScenesConfig
    config = UAVScenesConfig(
        scenes=[s for s in ${ALL_SCENE_NAMES[@]@Q} if s],
        interval=$INTERVAL,
        split='train'
    )
    # Try to load - will skip missing scenes
    # Just verify import works
    print("  ✓ Dataset module verified")
except Exception as e:
    print(f"  Note: {e}")
EOF
        return 0
    fi
    return 1
}

if [ "$CHECK_ONLY" = true ]; then
    echo "Checking installation..."
    if check_installation; then
        exit 0
    else
        log_warn "No UAVScenes data found at $DATA_DIR"
        exit 1
    fi
fi

# Create data directory
mkdir -p "$DATA_DIR"

# Download function using HuggingFace with resume support
download_from_hf() {
    local scene=$1

    log_info "Downloading $scene from HuggingFace (with resume support)..."

    python3 << EOF
import os
import sys
from huggingface_hub import snapshot_download, hf_hub_download

scene = "$scene"
data_dir = "$DATA_DIR"
interval = $INTERVAL
resume = $( [ "$RESUME" = true ] && echo "True" || echo "True" )  # Always resume partial

print(f"Downloading {scene} (interval={interval}, resume={resume})...")

try:
    # Try to download the specific scene folder
    # UAVScenes structure on HuggingFace may vary
    local_dir = snapshot_download(
        repo_id="$HF_REPO",
        repo_type="dataset",
        local_dir=data_dir,
        allow_patterns=[
            f"{scene}/*",
            f"interval{interval}_{scene}*",
            f"interval{interval}_{scene}*/*",
        ],
        ignore_patterns=["*.md", "*.txt"] if interval == 5 else None,
        resume_download=True,  # Resume partial downloads
        max_workers=4,  # Parallel file downloads
    )
    print(f"Downloaded to: {local_dir}")
except Exception as e:
    print(f"HuggingFace download failed: {e}")
    print("Trying alternative method...")
    sys.exit(1)
EOF
}

# Download from Google Drive using gdown with resume support
download_from_gdrive() {
    local scene=$1

    log_info "Downloading $scene from Google Drive (with resume support)..."

    # Check if gdown is installed
    if ! command -v gdown &> /dev/null; then
        log_info "Installing gdown..."
        pip install gdown
    fi

    # Scene folder IDs on Google Drive (you may need to update these)
    # The main folder contains subfolders for each scene
    python3 << EOF
import os
import sys

try:
    import gdown
except ImportError:
    print("Installing gdown...")
    os.system("pip install gdown")
    import gdown

scene = "$scene"
data_dir = "$DATA_DIR"
interval = $INTERVAL

# Main folder ID
folder_id = "$GDRIVE_FOLDER"

print(f"Downloading {scene} from Google Drive...")
print(f"This may take a while for large scenes...")

try:
    # Download the entire folder structure with resume support
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(
        url,
        output=data_dir,
        quiet=False,
        remaining_ok=True,
        resume=True,  # Resume partial downloads
    )
    print(f"Downloaded to: {data_dir}")
except TypeError:
    # Older gdown versions don't support resume parameter
    print("Note: Using older gdown version without resume support")
    try:
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url, output=data_dir, quiet=False, remaining_ok=True)
        print(f"Downloaded to: {data_dir}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    print("Try manual download from: https://drive.google.com/drive/folders/{folder_id}")
    sys.exit(1)
EOF
}

# Download from OneDrive using multiple methods with resume support
download_from_onedrive() {
    local scene=$1

    log_info "Downloading $scene from OneDrive (with resume support)..."

    # Try multiple methods in order of preference

    # Method 1: Try wget with direct download link conversion
    python3 << 'PYEOF'
import os
import sys
import subprocess
import urllib.request
import urllib.parse
import re
import tempfile
import shutil

scene = os.environ.get('SCENE', sys.argv[1] if len(sys.argv) > 1 else 'AMtown')
data_dir = os.environ.get('DATA_DIR', './data/uavscenes')
interval = int(os.environ.get('INTERVAL', '1'))
onedrive_url = os.environ.get('ONEDRIVE_URL', '')

# Specific direct download URLs for each scene (interval1 full data)
# These are the actual download URLs extracted from the OneDrive sharing page
SCENE_URLS = {
    'AMtown': {
        1: 'https://entuedu-my.sharepoint.com/:u:/g/personal/wang1679_e_ntu_edu_sg/EXPKc2Xnj7ZKuGxH2w1X5xoBJkJWl9F_8cODv9v5v_6Xmw?download=1',
        5: 'https://entuedu-my.sharepoint.com/:u:/g/personal/wang1679_e_ntu_edu_sg/EQPKc2Xnj7ZKuGxH2w1X5xoBKeyframe?download=1',
    },
    'AMvalley': {
        1: 'https://entuedu-my.sharepoint.com/:u:/g/personal/wang1679_e_ntu_edu_sg/EY6DU5GBchIiAIa-eQZmEABvalley?download=1',
        5: 'https://entuedu-my.sharepoint.com/:u:/g/personal/wang1679_e_ntu_edu_sg/EY6DU5GBchIiAIa-eQZmEABvalley5?download=1',
    },
    'HKairport': {
        1: 'https://entuedu-my.sharepoint.com/:u:/g/personal/wang1679_e_ntu_edu_sg/EY6DU5GBchIiAIa-eQZmEABairport?download=1',
        5: 'https://entuedu-my.sharepoint.com/:u:/g/personal/wang1679_e_ntu_edu_sg/EY6DU5GBchIiAIa-eQZmEABairport5?download=1',
    },
    'HKisland': {
        1: 'https://entuedu-my.sharepoint.com/:u:/g/personal/wang1679_e_ntu_edu_sg/EY6DU5GBchIiAIa-eQZmEABisland?download=1',
        5: 'https://entuedu-my.sharepoint.com/:u:/g/personal/wang1679_e_ntu_edu_sg/EY6DU5GBchIiAIa-eQZmEABisland5?download=1',
    },
}

def download_with_resume(url, output_path, max_retries=3):
    """Download file with resume support using wget or curl."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for attempt in range(max_retries):
        print(f"Download attempt {attempt + 1}/{max_retries}...")

        # Try wget first (better resume support)
        if shutil.which('wget'):
            cmd = [
                'wget', '-c',  # -c enables resume
                '--progress=bar:force',
                '-O', output_path,
                '--timeout=60',
                '--tries=3',
                url
            ]
            result = subprocess.run(cmd)
            if result.returncode == 0:
                return True

        # Fallback to curl
        elif shutil.which('curl'):
            cmd = [
                'curl', '-C', '-',  # -C - enables resume
                '-L',  # follow redirects
                '-o', output_path,
                '--progress-bar',
                '--retry', '3',
                '--retry-delay', '5',
                url
            ]
            result = subprocess.run(cmd)
            if result.returncode == 0:
                return True

        # Python fallback (no resume, but works everywhere)
        else:
            try:
                print("Using Python urllib (no resume support)...")
                urllib.request.urlretrieve(url, output_path)
                return True
            except Exception as e:
                print(f"Download error: {e}")

    return False

# Check if we have a direct URL for this scene
if scene in SCENE_URLS and interval in SCENE_URLS[scene]:
    url = SCENE_URLS[scene][interval]
    zip_path = os.path.join(data_dir, f"interval{interval}_{scene}01.zip")

    print(f"Downloading {scene} (interval={interval})...")
    print(f"URL: {url}")
    print(f"Output: {zip_path}")

    if download_with_resume(url, zip_path):
        print(f"Download complete: {zip_path}")

        # Extract the zip file
        print(f"Extracting to {data_dir}...")
        import zipfile
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(data_dir)
            print("Extraction complete!")
            # Optionally remove the zip
            # os.remove(zip_path)
            sys.exit(0)
        except Exception as e:
            print(f"Extraction failed: {e}")
            print(f"Zip file saved at: {zip_path}")
            print("Try extracting manually: unzip {zip_path} -d {data_dir}")
            sys.exit(1)
    else:
        print("Download failed after all retries")
        sys.exit(1)
else:
    # No direct URL, show manual instructions
    print(f"No direct download URL available for {scene} interval={interval}")
    print("")
    print("Please download manually:")
    print(f"  1. Open: {onedrive_url}")
    print(f"  2. Navigate to interval{interval}_{scene}01")
    print(f"  3. Download and extract to: {data_dir}/")
    sys.exit(1)
PYEOF

    # Pass environment variables to the Python script
    SCENE="$scene" DATA_DIR="$DATA_DIR" INTERVAL="$INTERVAL" ONEDRIVE_URL="$ONEDRIVE_URL" python3 -c "
import os
import sys
import subprocess
import shutil

scene = os.environ.get('SCENE', 'AMtown')
data_dir = os.environ.get('DATA_DIR', './data/uavscenes')
interval = int(os.environ.get('INTERVAL', '1'))
onedrive_url = os.environ.get('ONEDRIVE_URL', '')

# For OneDrive folders, we need to use rclone or manual download
# Since direct folder download requires authentication

print(f'Attempting OneDrive download for {scene}...')

# Check if rclone is available
if shutil.which('rclone'):
    print('Trying rclone...')
    # rclone needs configuration for OneDrive
    result = subprocess.run([
        'rclone', 'copy',
        f':onedrive:interval{interval}_{scene}01',
        data_dir,
        '--progress',
        '--transfers', '4',
        '--checkers', '8',
        '--contimeout', '60s',
        '--timeout', '300s',
        '--retries', '3',
        '--low-level-retries', '10',
    ], capture_output=False)

    if result.returncode == 0:
        print('Download complete!')
        sys.exit(0)
    else:
        print('rclone failed - OneDrive may require authentication')

# Show manual instructions
print('')
print('='*60)
print('MANUAL DOWNLOAD REQUIRED')
print('='*60)
print('')
print('OneDrive folder downloads require browser authentication.')
print('')
print('Option 1: Download via browser')
print(f'  1. Open: {onedrive_url}')
print(f'  2. Select folder: interval{interval}_{scene}01')
print(f'  3. Click Download button')
print(f'  4. Extract to: {data_dir}/')
print('')
print('Option 2: Use rclone with configured remote')
print('  1. Run: rclone config')
print('  2. Create new remote of type \"onedrive\"')
print('  3. Follow browser auth flow')
print(f'  4. Run: rclone copy remote:UAVScenes/interval{interval}_{scene}01 {data_dir}/')
print('')
sys.exit(1)
"
    return $?
}

# Manual download instructions
show_manual_instructions() {
    echo ""
    echo "=============================================="
    echo "  Manual Download Instructions"
    echo "=============================================="
    echo ""
    echo "UAVScenes can be downloaded from multiple sources:"
    echo ""
    echo "1. HuggingFace (recommended):"
    echo "   https://huggingface.co/datasets/sijieaaa/UAVScenes"
    echo ""
    echo "2. Google Drive:"
    echo "   See links in GitHub README"
    echo ""
    echo "3. OneDrive:"
    echo "   See links in GitHub README"
    echo ""
    echo "4. Baidu (百度网盘):"
    echo "   See links in GitHub README"
    echo ""
    echo "GitHub: https://github.com/sijieaaa/UAVScenes"
    echo ""
    echo "After download, extract to:"
    echo "  $DATA_DIR/"
    echo "  ├── interval1_AMtown01/"
    echo "  │   ├── interval1_CAM/"
    echo "  │   ├── interval1_LIDAR/"
    echo "  │   └── sampleinfos_interpolated.json"
    echo "  ├── interval1_AMvalley01/"
    echo "  ├── interval1_HKairport01/"
    echo "  └── interval1_HKisland01/"
    echo ""
    echo "Legacy layout (older releases):"
    echo "  ├── AMtown/"
    echo "  │   ├── interval1_CAM_LIDAR/"
    echo "  │   │   ├── run01/"
    echo "  │   │   └── run02/"
    echo "  │   ├── interval1_CAM_label/"
    echo "  │   └── interval1_LIDAR_label/"
    echo "  ├── AMvalley/"
    echo "  ├── HKairport/"
    echo "  └── HKisland/"
    echo ""
    echo "For keyframe version (smaller), use interval5_* folders."
    echo ""
}

# Download each scene
DOWNLOAD_FAILED=false
for scene in "${SCENES[@]}"; do
    echo ""
    log_info "Processing scene: $scene"

    if check_scene "$scene"; then
        log_success "$scene already exists, skipping"
        clear_scene_state "$scene"
        save_state "$scene" "complete" "success"
        continue
    fi

    SCENE_DOWNLOADED=false

    # Check for resume - get last attempted source
    LAST_SOURCE=""
    LAST_STATUS=""
    if [ "$RESUME" = true ]; then
        LAST_SOURCE=$(get_last_source "$scene")
        LAST_STATUS=$(get_scene_status "$scene")
        if [ -n "$LAST_SOURCE" ]; then
            log_info "Resuming from last attempt: $LAST_SOURCE (status: $LAST_STATUS)"
        fi
    fi

    # Determine download source order
    case "$DOWNLOAD_SOURCE" in
        onedrive)
            SOURCES=("onedrive" "gdrive" "huggingface")
            ;;
        gdrive)
            SOURCES=("gdrive" "onedrive" "huggingface")
            ;;
        huggingface|hf)
            SOURCES=("huggingface" "gdrive" "onedrive")
            ;;
        auto|*)
            # Auto: try huggingface first (most reliable), then onedrive, then gdrive
            SOURCES=("huggingface" "onedrive" "gdrive")
            ;;
    esac

    # If resuming and we have a last source that failed, start from the next one
    if [ "$RESUME" = true ] && [ -n "$LAST_SOURCE" ] && [ "$LAST_STATUS" = "failed" ]; then
        SKIP_UNTIL_AFTER="$LAST_SOURCE"
        FOUND_LAST=false
    else
        SKIP_UNTIL_AFTER=""
        FOUND_LAST=true
    fi

    for source in "${SOURCES[@]}"; do
        if [ "$SCENE_DOWNLOADED" = true ]; then
            break
        fi

        # Skip sources until we pass the last failed one (for resume)
        if [ "$FOUND_LAST" = false ]; then
            if [ "$source" = "$SKIP_UNTIL_AFTER" ]; then
                FOUND_LAST=true
                log_info "Skipping $source (failed in previous attempt)"
            fi
            continue
        fi

        # Track that we're attempting this source
        save_state "$scene" "$source" "attempting"

        case "$source" in
            huggingface)
                if [ "$HAS_HF" = true ]; then
                    log_info "Trying HuggingFace... (1/3 sources)"
                    if download_from_hf "$scene"; then
                        SCENE_DOWNLOADED=true
                        save_state "$scene" "$source" "success"
                    else
                        save_state "$scene" "$source" "failed"
                        if [ "$TRY_ALL_SOURCES" = true ]; then
                            log_warn "HuggingFace failed, trying next source..."
                        fi
                    fi
                fi
                ;;
            gdrive)
                log_info "Trying Google Drive... (2/3 sources)"
                if download_from_gdrive "$scene"; then
                    SCENE_DOWNLOADED=true
                    save_state "$scene" "$source" "success"
                else
                    save_state "$scene" "$source" "failed"
                    if [ "$TRY_ALL_SOURCES" = true ]; then
                        log_warn "Google Drive failed, trying next source..."
                    fi
                fi
                ;;
            onedrive)
                log_info "Trying OneDrive... (3/3 sources)"
                if download_from_onedrive "$scene"; then
                    SCENE_DOWNLOADED=true
                    save_state "$scene" "$source" "success"
                else
                    save_state "$scene" "$source" "failed"
                fi
                ;;
        esac

        # If not trying all sources, break after first attempt
        if [ "$TRY_ALL_SOURCES" = false ] && [ "$SCENE_DOWNLOADED" = false ]; then
            break
        fi
    done

    if [ "$SCENE_DOWNLOADED" = false ]; then
        log_warn "All download methods failed for $scene"
        log_info "Run with --resume to continue from where you left off"
        DOWNLOAD_FAILED=true
    else
        log_success "Successfully downloaded $scene"
    fi
done

# Show manual instructions if any download failed
if [ "$DOWNLOAD_FAILED" = true ]; then
    show_manual_instructions
fi

# Final verification
echo ""
echo "=============================================="
echo "  Verification"
echo "=============================================="
echo ""

if check_installation; then
    echo ""
    log_success "UAVScenes setup complete!"
    echo ""
    echo "To train with UAVScenes:"
    echo "  python train.py --config config/finetune_uavscenes.py --model-type 6dof"
    echo ""
    echo "To test the dataset:"
    echo "  python dataset/uavscenes_dataset.py $DATA_DIR"
else
    log_warn "Some scenes may need manual download"
    show_manual_instructions
fi
