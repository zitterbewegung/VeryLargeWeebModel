#!/bin/bash
# =============================================================================
# Setup nuScenes Dataset for 6DoF Training
# =============================================================================
#
# Downloads and prepares nuScenes data for OccWorld 6DoF training.
#
# Usage:
#   ./scripts/setup_nuscenes.sh              # Download mini version (recommended for testing)
#   ./scripts/setup_nuscenes.sh --full       # Download full trainval (requires account)
#   ./scripts/setup_nuscenes.sh --check      # Check existing installation
#
# Requirements:
#   - pip install nuscenes-devkit pyquaternion
#   - ~4GB disk space for mini, ~300GB for full
#
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${NUSCENES_ROOT:-$PROJECT_ROOT/data/nuscenes}"
VERSION="mini"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            VERSION="trainval"
            shift
            ;;
        --mini)
            VERSION="mini"
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
            echo "Usage: $0 [--mini|--full] [--check] [--data-dir PATH]"
            echo ""
            echo "Options:"
            echo "  --mini      Download mini version (~4GB, for testing)"
            echo "  --full      Download full trainval (~300GB, requires account)"
            echo "  --check     Check existing installation only"
            echo "  --data-dir  Custom data directory (default: data/nuscenes)"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "  nuScenes Dataset Setup"
echo "=============================================="
echo ""
echo "Version: v1.0-$VERSION"
echo "Data directory: $DATA_DIR"
echo ""

# Check Python dependencies
log_info "Checking Python dependencies..."
python3 -c "import nuscenes" 2>/dev/null || {
    log_warn "nuscenes-devkit not installed"
    log_info "Installing: pip install nuscenes-devkit pyquaternion"
    pip install nuscenes-devkit pyquaternion
}
log_success "Python dependencies OK"

# Check if data already exists
check_installation() {
    if [ -d "$DATA_DIR/v1.0-$VERSION" ] || [ -d "$DATA_DIR/samples" ]; then
        log_success "nuScenes data found at $DATA_DIR"

        # Verify with Python
        python3 << EOF
import sys
sys.path.insert(0, '$PROJECT_ROOT')
try:
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-$VERSION', dataroot='$DATA_DIR', verbose=False)
    print(f"  Scenes: {len(nusc.scene)}")
    print(f"  Samples: {len(nusc.sample)}")
    print(f"  ✓ Dataset verified")
except Exception as e:
    print(f"  ✗ Verification failed: {e}")
    sys.exit(1)
EOF
        return 0
    fi
    return 1
}

if [ "$CHECK_ONLY" = true ]; then
    if check_installation; then
        exit 0
    else
        log_error "nuScenes not found at $DATA_DIR"
        exit 1
    fi
fi

# Check if already installed
if check_installation; then
    log_info "nuScenes already installed. Use --check to verify."
    exit 0
fi

# Create data directory
mkdir -p "$DATA_DIR"

# Download based on version
if [ "$VERSION" = "mini" ]; then
    log_info "Downloading nuScenes mini..."

    # Mini dataset URLs (public, no account needed)
    MINI_URL="https://www.nuscenes.org/data/v1.0-mini.tgz"

    # Check if wget or curl available
    if command -v wget &> /dev/null; then
        DOWNLOAD_CMD="wget -O"
    elif command -v curl &> /dev/null; then
        DOWNLOAD_CMD="curl -L -o"
    else
        log_error "Neither wget nor curl found. Please install one."
        exit 1
    fi

    # Download mini metadata
    log_info "Downloading mini metadata..."
    $DOWNLOAD_CMD "$DATA_DIR/v1.0-mini.tgz" "$MINI_URL" || {
        log_error "Download failed. The mini dataset may require manual download."
        echo ""
        echo "Manual download instructions:"
        echo "1. Go to: https://www.nuscenes.org/nuscenes#download"
        echo "2. Download 'Mini' split (Metadata + All sensor data)"
        echo "3. Extract to: $DATA_DIR/"
        echo ""
        exit 1
    }

    # Extract
    log_info "Extracting..."
    cd "$DATA_DIR"
    tar -xzf v1.0-mini.tgz
    rm -f v1.0-mini.tgz

    log_success "nuScenes mini downloaded and extracted"

else
    # Full version requires account
    log_warn "Full nuScenes requires an account at nuscenes.org"
    echo ""
    echo "Manual download instructions:"
    echo "1. Create account at: https://www.nuscenes.org/sign-up"
    echo "2. Go to: https://www.nuscenes.org/nuscenes#download"
    echo "3. Download 'Full dataset (v1.0)' - Trainval split"
    echo "   - Metadata (~1GB)"
    echo "   - Sensor blobs (multiple files, ~300GB total)"
    echo "4. Extract all to: $DATA_DIR/"
    echo ""
    echo "Expected structure:"
    echo "  $DATA_DIR/"
    echo "  ├── v1.0-trainval/"
    echo "  │   ├── attribute.json"
    echo "  │   ├── sample.json"
    echo "  │   └── ..."
    echo "  ├── samples/"
    echo "  │   ├── CAM_FRONT/"
    echo "  │   ├── LIDAR_TOP/"
    echo "  │   └── ..."
    echo "  └── sweeps/"
    echo ""
    exit 0
fi

# Verify installation
echo ""
log_info "Verifying installation..."
if check_installation; then
    log_success "Setup complete!"
else
    log_error "Verification failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "To train with nuScenes 6DoF:"
echo "  python train.py --config config/finetune_nuscenes_6dof.py --model-type 6dof"
echo ""
echo "To test the dataset:"
echo "  python dataset/nuscenes_6dof_dataset.py $DATA_DIR"
echo ""
