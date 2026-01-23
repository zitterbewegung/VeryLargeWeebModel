#!/bin/bash
#
# VeryLargeWeebModel - One-Command Setup and Training
#
# Downloads data, converts to training format, and starts training.
#
# Usage:
#   ./scripts/setup_and_train.sh              # Full setup with real PLATEAU data
#   ./scripts/setup_and_train.sh --test       # Quick test with synthetic data
#   ./scripts/setup_and_train.sh --6dof       # Train with 6DoF enhanced model
#   ./scripts/setup_and_train.sh --nuscenes   # Use nuScenes mini dataset
#   ./scripts/setup_and_train.sh --skip-train # Setup data only, don't train
#
# Options:
#   --test          Use synthetic buildings for quick pipeline testing (~5 min)
#   --6dof          Use 6DoF enhanced model (pose uncertainty, relocalization)
#   --nuscenes      Download and use nuScenes mini dataset instead
#   --uavscenes     Download UAVScenes aerial drone dataset (real 6DoF data)
#   --plateau       Download real Tokyo PLATEAU 3D city data (default)
#   --all           Download all available datasets
#   --skip-train    Only setup data, don't start training
#   --sessions N    Number of training sessions to generate (default: 50)
#   --frames N      Frames per session (default: 300)
#   --epochs N      Training epochs (default: 50)
#   --batch-size N  Batch size (default: auto-detect)
#   --work-dir DIR  Output directory (default: /workspace/checkpoints or ./out)
#   --mirror URL    Custom data mirror URL (faster than official source)
#
# Caching for Vast.ai:
#   1. Upload data to your S3: ./scripts/upload_data_cache.sh s3://your-bucket/plateau
#   2. On Vast.ai: ./scripts/setup_and_train.sh --mirror https://your-bucket.s3.amazonaws.com/plateau
#
#   Or set environment variable:
#     export DATA_MIRROR="https://your-bucket.s3.amazonaws.com/plateau"
#
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()    { echo -e "\n${CYAN}==>${NC} ${GREEN}$1${NC}"; }

# Defaults
MODE="plateau"
SKIP_TRAIN=false
USE_6DOF=false
SESSIONS=50
FRAMES=300
EPOCHS=50
BATCH_SIZE=""
WORK_DIR=""
SYNTHETIC_BUILDINGS=200

# Custom data mirror (set via environment or --mirror flag)
# Examples:
#   export DATA_MIRROR="https://your-bucket.s3.amazonaws.com/plateau"
#   export DATA_MIRROR="https://huggingface.co/datasets/your-name/plateau/resolve/main"
DATA_MIRROR="${DATA_MIRROR:-}"

# Detect environment
if [ -d "/workspace" ]; then
    # Cloud GPU instance (Vast.ai, Lambda, etc.)
    DEFAULT_WORK_DIR="/workspace/checkpoints"
    PROJECT_DIR="/workspace/VeryLargeWeebModel"
else
    # Local machine
    DEFAULT_WORK_DIR="./out"
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

WORK_DIR="${WORK_DIR:-$DEFAULT_WORK_DIR}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            MODE="test"
            SESSIONS=10
            FRAMES=100
            EPOCHS=5
            shift
            ;;
        --6dof)
            USE_6DOF=true
            shift
            ;;
        --nuscenes)
            MODE="nuscenes"
            shift
            ;;
        --uavscenes)
            MODE="uavscenes"
            USE_6DOF=true  # UAVScenes works best with 6DoF model
            shift
            ;;
        --plateau)
            MODE="plateau"
            shift
            ;;
        --all)
            MODE="all"
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --sessions)
            SESSIONS="$2"
            shift 2
            ;;
        --frames)
            FRAMES="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --mirror)
            DATA_MIRROR="$2"
            shift 2
            ;;
        --help|-h)
            head -25 "$0" | tail -22
            exit 0
            ;;
        *)
            log_warn "Unknown option: $1"
            shift
            ;;
    esac
done

# Header
echo ""
echo "============================================================"
echo "   VeryLargeWeebModel - Automated Setup & Training"
echo "============================================================"
echo ""
echo "  Mode:        $MODE"
echo "  6DoF Model:  $USE_6DOF"
echo "  Sessions:    $SESSIONS"
echo "  Frames:      $FRAMES"
echo "  Epochs:      $EPOCHS"
echo "  Work Dir:    $WORK_DIR"
echo "  Skip Train:  $SKIP_TRAIN"
echo ""

cd "$PROJECT_DIR"

# =============================================================================
# Step 1: Install Dependencies
# =============================================================================
log_step "Step 1/4: Installing dependencies..."

# Check for pip
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    log_error "pip not found. Please install Python first."
    exit 1
fi

PIP_CMD="pip"
command -v pip3 &> /dev/null && PIP_CMD="pip3"

# Install required packages
$PIP_CMD install -q trimesh numpy opencv-python torch torchvision tqdm scipy 2>/dev/null || true
log_success "Dependencies installed"

# =============================================================================
# Step 2: Download/Generate Data
# =============================================================================
log_step "Step 2/4: Setting up training data..."

DATA_DIR="$PROJECT_DIR/data"
mkdir -p "$DATA_DIR"

case $MODE in
    test)
        log_info "Generating SYNTHETIC test data (for pipeline testing only)"
        log_warn "This data will cause overfitting - use only for testing!"

        mkdir -p "$DATA_DIR/plateau"

        python scripts/plateau_to_occworld.py \
            --input "$DATA_DIR/plateau" \
            --output "$DATA_DIR/tokyo_gazebo" \
            --sessions "$SESSIONS" \
            --frames "$FRAMES" \
            --allow-synthetic \
            --synthetic-buildings "$SYNTHETIC_BUILDINGS"

        log_success "Synthetic data generated: $SESSIONS sessions x $FRAMES frames"
        CONFIG="config/finetune_tokyo.py"
        ;;

    plateau)
        log_info "Downloading Tokyo PLATEAU 3D city data (~2.1GB)..."

        PLATEAU_DIR="$DATA_DIR/plateau"
        PLATEAU_RAW="$PLATEAU_DIR/raw"
        PLATEAU_MESHES="$PLATEAU_DIR/meshes/obj"

        mkdir -p "$PLATEAU_RAW" "$PLATEAU_MESHES"

        PLATEAU_ARCHIVE="$PLATEAU_RAW/tokyo23ku_obj.zip"

        # Use custom mirror if set, otherwise use official PLATEAU source
        if [ -n "$DATA_MIRROR" ]; then
            PLATEAU_URL="${DATA_MIRROR}/tokyo23ku_obj.zip"
            log_info "Using custom mirror: $DATA_MIRROR"
        else
            PLATEAU_URL="https://gic-plateau.s3.ap-northeast-1.amazonaws.com/2020/13100_tokyo23-ku_2020_obj_3_op.zip"
        fi

        # Check if already downloaded
        MESH_COUNT=$(find "$PLATEAU_MESHES" -name "*.obj" 2>/dev/null | wc -l)

        if [ "$MESH_COUNT" -gt 10 ]; then
            log_info "PLATEAU meshes already exist: $MESH_COUNT files"
        else
            # Expected size ~2.1GB (2100000000 bytes minimum)
            MIN_SIZE=2000000000

            # Check if file exists and is large enough
            if [ -f "$PLATEAU_ARCHIVE" ]; then
                FILE_SIZE=$(stat -c%s "$PLATEAU_ARCHIVE" 2>/dev/null || stat -f%z "$PLATEAU_ARCHIVE" 2>/dev/null || echo 0)
                if [ "$FILE_SIZE" -lt "$MIN_SIZE" ]; then
                    log_warn "Existing archive is incomplete (${FILE_SIZE} bytes, expected ~2.1GB)"
                    log_info "Removing corrupted file and re-downloading..."
                    rm -f "$PLATEAU_ARCHIVE"
                fi
            fi

            # Download if file doesn't exist
            if [ ! -f "$PLATEAU_ARCHIVE" ]; then
                log_info "Downloading from Project PLATEAU (MLIT Japan)..."
                log_info "This is a large file (~2.1GB), please wait..."

                # Try wget with resume support first
                if command -v wget &> /dev/null; then
                    wget -c --show-progress -O "$PLATEAU_ARCHIVE" "$PLATEAU_URL" || {
                        log_warn "wget failed, trying curl..."
                        rm -f "$PLATEAU_ARCHIVE"
                        curl -L -C - --progress-bar -o "$PLATEAU_ARCHIVE" "$PLATEAU_URL"
                    }
                else
                    curl -L -C - --progress-bar -o "$PLATEAU_ARCHIVE" "$PLATEAU_URL"
                fi

                # Verify download completed
                if [ -f "$PLATEAU_ARCHIVE" ]; then
                    FILE_SIZE=$(stat -c%s "$PLATEAU_ARCHIVE" 2>/dev/null || stat -f%z "$PLATEAU_ARCHIVE" 2>/dev/null || echo 0)
                    if [ "$FILE_SIZE" -lt "$MIN_SIZE" ]; then
                        log_error "Download incomplete (${FILE_SIZE} bytes, expected ~2.1GB)"
                        log_info "Try downloading manually:"
                        log_info "  wget -c -O $PLATEAU_ARCHIVE $PLATEAU_URL"
                        exit 1
                    fi
                    log_success "Download complete ($(numfmt --to=iec $FILE_SIZE 2>/dev/null || echo "$FILE_SIZE bytes"))"
                else
                    log_error "Download failed. Try manually:"
                    log_info "  wget -c -O $PLATEAU_ARCHIVE $PLATEAU_URL"
                    exit 1
                fi
            fi

            # Verify it's a valid zip before extracting
            if ! unzip -t "$PLATEAU_ARCHIVE" > /dev/null 2>&1; then
                log_error "Archive is corrupted. Removing and please re-run script."
                rm -f "$PLATEAU_ARCHIVE"
                exit 1
            fi

            log_info "Extracting meshes..."
            unzip -q -o "$PLATEAU_ARCHIVE" -d "$PLATEAU_MESHES/" 2>/dev/null || \
            unzip -o "$PLATEAU_ARCHIVE" -d "$PLATEAU_MESHES/"

            MESH_COUNT=$(find "$PLATEAU_MESHES" -name "*.obj" 2>/dev/null | wc -l)
            log_success "Extracted $MESH_COUNT OBJ mesh files"
        fi

        log_info "Converting PLATEAU meshes to training format..."
        python scripts/plateau_to_occworld.py \
            --input "$PLATEAU_MESHES" \
            --output "$DATA_DIR/tokyo_gazebo" \
            --sessions "$SESSIONS" \
            --frames "$FRAMES" \
            --pattern random

        log_success "Training data generated: $SESSIONS sessions x $FRAMES frames"
        CONFIG="config/finetune_tokyo.py"
        ;;

    nuscenes)
        log_info "Setting up nuScenes mini dataset..."

        NUSCENES_DIR="$DATA_DIR/nuscenes"
        mkdir -p "$NUSCENES_DIR"

        # Check if already exists
        if [ -d "$NUSCENES_DIR/v1.0-mini" ]; then
            log_info "nuScenes mini already exists"
        else
            log_info "Downloading nuScenes mini (~4GB)..."
            log_warn "Note: Full nuScenes requires registration at nuscenes.org"

            cd "$NUSCENES_DIR"

            # Try to download mini dataset
            MINI_URL="https://www.nuscenes.org/data/v1.0-mini.tgz"
            if wget -q --spider "$MINI_URL" 2>/dev/null; then
                wget -q --show-progress -O v1.0-mini.tgz "$MINI_URL"
                tar -xzf v1.0-mini.tgz
                rm v1.0-mini.tgz
            else
                log_error "Cannot download nuScenes automatically."
                log_info "Please register at https://www.nuscenes.org/ and download manually."
                log_info "Then extract to: $NUSCENES_DIR/v1.0-mini/"
                exit 1
            fi

            cd "$PROJECT_DIR"
        fi

        # Install nuscenes-devkit
        $PIP_CMD install -q nuscenes-devkit

        log_success "nuScenes mini ready"
        CONFIG="config/finetune_nuscenes.py"
        ;;

    uavscenes)
        log_info "Setting up UAVScenes aerial drone dataset..."
        log_info "This is REAL aerial 6DoF data from UAV flights"

        # Install gdown for Google Drive downloads
        $PIP_CMD install -q gdown pyquaternion

        # Run UAVScenes setup script
        UAVSCENES_SCENE="${UAVSCENES_SCENE:-AMtown}"

        if [ -n "$DATA_MIRROR" ]; then
            log_info "Using custom mirror for UAVScenes: $DATA_MIRROR"
            # If custom mirror is set, download from there
            UAVSCENES_DIR="$DATA_DIR/uavscenes"
            mkdir -p "$UAVSCENES_DIR"
            wget -c -O "$UAVSCENES_DIR/uavscenes.zip" "$DATA_MIRROR/uavscenes.zip" || \
            curl -L -C - -o "$UAVSCENES_DIR/uavscenes.zip" "$DATA_MIRROR/uavscenes.zip"
            unzip -q -o "$UAVSCENES_DIR/uavscenes.zip" -d "$UAVSCENES_DIR/"
        else
            # Use the setup script with Google Drive (most reliable)
            ./scripts/setup_uavscenes.sh --gdrive --scene "$UAVSCENES_SCENE" --keyframes
        fi

        log_success "UAVScenes ready"
        CONFIG="config/finetune_uavscenes.py"
        ;;

    all)
        log_info "Downloading ALL available datasets..."

        # Run research data download script
        ./scripts/download_research_data.sh --all --output "$DATA_DIR"

        # Convert PLATEAU
        PLATEAU_MESHES="$DATA_DIR/plateau/meshes/obj"
        if [ -d "$PLATEAU_MESHES" ]; then
            log_info "Converting PLATEAU meshes..."
            python scripts/plateau_to_occworld.py \
                --input "$PLATEAU_MESHES" \
                --output "$DATA_DIR/tokyo_gazebo" \
                --sessions "$SESSIONS" \
                --frames "$FRAMES"
        fi

        log_success "All datasets ready"
        CONFIG="config/finetune_tokyo.py"
        ;;
esac

# =============================================================================
# Step 3: Verify Data
# =============================================================================
log_step "Step 3/4: Verifying training data..."

case "$MODE" in
    nuscenes)
        DATA_CHECK_DIR="$DATA_DIR/nuscenes"
        ;;
    uavscenes)
        DATA_CHECK_DIR="$DATA_DIR/uavscenes"
        ;;
    *)
        DATA_CHECK_DIR="$DATA_DIR/tokyo_gazebo"
        ;;
esac

# Verify data exists
case "$MODE" in
    nuscenes)
        if [ -d "$DATA_CHECK_DIR/v1.0-mini" ]; then
            log_success "nuScenes data directory verified"
        else
            log_error "nuScenes data not found"
            exit 1
        fi
        ;;
    uavscenes)
        # Check for UAVScenes directory structure
        SCENE_COUNT=$(find "$DATA_CHECK_DIR" -maxdepth 1 -type d -name "interval*" 2>/dev/null | wc -l)
        if [ "$SCENE_COUNT" -gt 0 ]; then
            log_success "Found $SCENE_COUNT UAVScenes scene(s)"
        else
            log_warn "UAVScenes data may still be downloading or not found"
            log_info "Check: $DATA_CHECK_DIR"
        fi
        ;;
    *)
        # Count sessions for Gazebo/PLATEAU data
        SESSION_COUNT=$(find "$DATA_CHECK_DIR" -maxdepth 1 -type d -name "*_*" 2>/dev/null | wc -l)
        if [ "$SESSION_COUNT" -eq 0 ]; then
            log_error "No training sessions found in $DATA_CHECK_DIR"
            log_info "Data generation may have failed. Check errors above."
            exit 1
        fi
        FRAME_COUNT=$(find "$DATA_CHECK_DIR" -name "*_occupancy.npz" 2>/dev/null | wc -l)
        log_success "Found $SESSION_COUNT sessions with $FRAME_COUNT total frames"
        ;;
esac

# =============================================================================
# Step 4: Start Training
# =============================================================================
if [ "$SKIP_TRAIN" = true ]; then
    log_step "Step 4/4: Skipping training (--skip-train)"
    echo ""
    echo "============================================================"
    echo "   Data Setup Complete!"
    echo "============================================================"
    echo ""
    if [ "$USE_6DOF" = true ]; then
        echo "To start 6DoF training manually:"
        echo "  python train_6dof.py --config config/finetune_6dof.py --work-dir $WORK_DIR"
    else
        echo "To start training manually:"
        echo "  python train.py --config $CONFIG --work-dir $WORK_DIR"
    fi
    echo ""
    exit 0
fi

log_step "Step 4/4: Starting training..."

mkdir -p "$WORK_DIR"

# Build training command
if [ "$USE_6DOF" = true ]; then
    log_info "Using 6DoF enhanced model with:"
    log_info "  - Pose uncertainty estimation"
    log_info "  - Relocalization head"
    log_info "  - Place recognition embeddings"

    # Use dataset-specific 6DoF config
    if [ "$MODE" = "uavscenes" ]; then
        SIXDOF_CONFIG="config/finetune_uavscenes.py"
    else
        SIXDOF_CONFIG="config/finetune_6dof.py"
    fi
    TRAIN_CMD="python train.py --config $SIXDOF_CONFIG --work-dir $WORK_DIR --epochs $EPOCHS --model-type 6dof"
else
    TRAIN_CMD="python train.py --config $CONFIG --work-dir $WORK_DIR --epochs $EPOCHS"
fi

if [ -n "$BATCH_SIZE" ]; then
    TRAIN_CMD="$TRAIN_CMD --batch-size $BATCH_SIZE"
fi

echo ""
echo "============================================================"
echo "   Training Configuration"
echo "============================================================"
echo ""
echo "  Config:     $CONFIG"
echo "  Work Dir:   $WORK_DIR"
echo "  Epochs:     $EPOCHS"
echo "  Command:    $TRAIN_CMD"
echo ""
echo "============================================================"
echo ""

# Run training
$TRAIN_CMD

log_success "Training complete!"
echo ""
echo "Checkpoints saved to: $WORK_DIR"
echo "Best model: $WORK_DIR/checkpoints/best.pth"
