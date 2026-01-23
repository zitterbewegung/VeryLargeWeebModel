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

# Available scenes
ALL_SCENE_NAMES=("AMtown" "AMvalley" "HKairport" "HKisland")

# Download URLs
HF_REPO="sijieaaa/UAVScenes"
ONEDRIVE_URL="https://entuedu-my.sharepoint.com/:f:/g/personal/wang1679_e_ntu_edu_sg/EgY6DU5GBchIiAIa-eQZmEAB0vJx3khCPHbFW3LnR77RFw"
GDRIVE_FOLDER="1HSJWc5qmIKLdpaS8w8pqrWch4F9MHIeN"

# Preferred download source (onedrive, gdrive, huggingface)
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-auto}"

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

echo "=============================================="
echo "  UAVScenes Dataset Setup"
echo "=============================================="
echo ""
echo "Data directory: $DATA_DIR"
echo "Scenes: ${SCENES[*]}"
echo "Interval: $INTERVAL (1=full, 5=keyframes)"
echo ""

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

# Download function using HuggingFace
download_from_hf() {
    local scene=$1

    log_info "Downloading $scene from HuggingFace..."

    python3 << EOF
import os
import sys
from huggingface_hub import snapshot_download, hf_hub_download

scene = "$scene"
data_dir = "$DATA_DIR"
interval = $INTERVAL

print(f"Downloading {scene} (interval={interval})...")

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
    )
    print(f"Downloaded to: {local_dir}")
except Exception as e:
    print(f"HuggingFace download failed: {e}")
    print("Trying alternative method...")
    sys.exit(1)
EOF
}

# Download from Google Drive using gdown
download_from_gdrive() {
    local scene=$1

    log_info "Downloading $scene from Google Drive..."

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
    # Download the entire folder structure
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url, output=data_dir, quiet=False, remaining_ok=True)
    print(f"Downloaded to: {data_dir}")
except Exception as e:
    print(f"Error: {e}")
    print("Try manual download from: https://drive.google.com/drive/folders/{folder_id}")
    sys.exit(1)
EOF
}

# Download from OneDrive using rclone or direct download
download_from_onedrive() {
    local scene=$1

    log_info "Downloading $scene from OneDrive..."

    # Install rclone if not available
    if ! command -v rclone &> /dev/null; then
        log_info "Installing rclone..."
        curl -s https://rclone.org/install.sh | sudo bash || {
            log_warn "Failed to install rclone automatically"
            # Try alternative install
            if command -v apt-get &> /dev/null; then
                sudo apt-get update && sudo apt-get install -y rclone
            fi
        }
    fi

    # Check if rclone is now available
    if command -v rclone &> /dev/null; then
        log_info "Using rclone for OneDrive download..."

        # Create a temporary rclone config for the public OneDrive link
        # This uses the :onedrive: backend with the shared link
        mkdir -p "$DATA_DIR"

        # OneDrive shared folder URL
        ONEDRIVE_SHARE="$ONEDRIVE_URL"

        # Try to download using rclone with the link
        # Format: rclone copy ":onedrive,shared_url=URL:" dest
        log_info "Attempting download from OneDrive shared link..."

        # Method 1: Try rclone with onedrive backend and shared URL
        if rclone copy ":onedrive:interval${INTERVAL}_${scene}01" "$DATA_DIR/" \
            --onedrive-link-type="view" \
            --onedrive-link-scope="anonymous" \
            --progress 2>/dev/null; then
            log_success "Downloaded $scene from OneDrive"
            return 0
        fi

        # Method 2: Try using rclone http backend with OneDrive direct URL
        log_info "Trying alternative download method..."

        # Convert OneDrive share URL to downloadable format
        python3 << EOF
import os
import sys
import subprocess
import urllib.request
import urllib.parse

scene = "$scene"
data_dir = "$DATA_DIR"
interval = $INTERVAL
onedrive_url = "$ONEDRIVE_URL"

# Convert OneDrive sharing URL to direct download URL
# Format: https://...sharepoint.com/:f:/g/personal/USER/HASH?e=XXX
# To: https://...sharepoint.com/personal/USER/_layouts/15/download.aspx?share=HASH

print(f"Converting OneDrive URL for direct download...")

# Extract components from the sharing URL
import re

# Try to construct a direct download URL
try:
    # Parse the URL
    # Example: https://entuedu-my.sharepoint.com/:f:/g/personal/wang1679_e_ntu_edu_sg/EgY6DU5GBchIiAIa-eQZmEAB0vJx3khCPHbFW3LnR77RFw

    match = re.match(r'https://([^/]+)/:f:/g/personal/([^/]+)/([^?]+)', onedrive_url)
    if match:
        host = match.group(1)
        user = match.group(2)
        share_id = match.group(3)

        # Construct download URL for folder
        # Note: This may prompt for browser download for folders
        download_url = f"https://{host}/personal/{user}/_layouts/15/download.aspx?share={share_id}"

        print(f"Download URL: {download_url}")
        print("")
        print("OneDrive folder downloads require browser interaction.")
        print("")
        print("Please download manually:")
        print(f"  1. Open in browser: {onedrive_url}")
        print(f"  2. Select folder: interval{interval}_{scene}01")
        print(f"  3. Click 'Download' button")
        print(f"  4. Extract to: {data_dir}/")
        print("")
        print("Or use wget with cookies from browser session.")
        sys.exit(1)
    else:
        print(f"Could not parse OneDrive URL: {onedrive_url}")
        sys.exit(1)

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF
        return 1
    else
        log_warn "rclone not available"
        log_info "Install with: curl https://rclone.org/install.sh | sudo bash"
        return 1
    fi
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
        continue
    fi

    SCENE_DOWNLOADED=false

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

    for source in "${SOURCES[@]}"; do
        if [ "$SCENE_DOWNLOADED" = true ]; then
            break
        fi

        case "$source" in
            huggingface)
                if [ "$HAS_HF" = true ]; then
                    log_info "Trying HuggingFace..."
                    if download_from_hf "$scene"; then
                        SCENE_DOWNLOADED=true
                    fi
                fi
                ;;
            gdrive)
                log_info "Trying Google Drive..."
                if download_from_gdrive "$scene"; then
                    SCENE_DOWNLOADED=true
                fi
                ;;
            onedrive)
                log_info "Trying OneDrive..."
                if download_from_onedrive "$scene"; then
                    SCENE_DOWNLOADED=true
                fi
                ;;
        esac
    done

    if [ "$SCENE_DOWNLOADED" = false ]; then
        log_warn "All download methods failed for $scene"
        DOWNLOAD_FAILED=true
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
