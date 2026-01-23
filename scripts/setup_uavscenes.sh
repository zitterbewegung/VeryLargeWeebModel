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

# HuggingFace dataset URL
HF_REPO="sijieaaa/UAVScenes"

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

    # Try HuggingFace download
    if [ "$HAS_HF" = true ]; then
        download_from_hf "$scene" || {
            log_warn "Automatic download failed for $scene"
            DOWNLOAD_FAILED=true
        }
    else
        log_warn "HuggingFace not available"
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
