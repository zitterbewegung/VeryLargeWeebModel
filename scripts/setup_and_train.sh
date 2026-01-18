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
#   --plateau       Download real Tokyo PLATEAU 3D city data (default)
#   --all           Download all available datasets
#   --skip-train    Only setup data, don't start training
#   --sessions N    Number of training sessions to generate (default: 50)
#   --frames N      Frames per session (default: 300)
#   --epochs N      Training epochs (default: 50)
#   --batch-size N  Batch size (default: auto-detect)
#   --work-dir DIR  Output directory (default: /workspace/checkpoints or ./out)
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
        PLATEAU_URL="https://gic-plateau.s3.ap-northeast-1.amazonaws.com/2020/13100_tokyo23-ku_2020_obj_3_op.zip"

        # Check if already downloaded
        MESH_COUNT=$(find "$PLATEAU_MESHES" -name "*.obj" 2>/dev/null | wc -l)

        if [ "$MESH_COUNT" -gt 10 ]; then
            log_info "PLATEAU meshes already exist: $MESH_COUNT files"
        else
            if [ ! -f "$PLATEAU_ARCHIVE" ]; then
                log_info "Downloading from Project PLATEAU (MLIT Japan)..."
                wget -q --show-progress -O "$PLATEAU_ARCHIVE" "$PLATEAU_URL" || \
                curl -L --progress-bar -o "$PLATEAU_ARCHIVE" "$PLATEAU_URL"
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

if [ "$MODE" = "nuscenes" ]; then
    DATA_CHECK_DIR="$DATA_DIR/nuscenes"
else
    DATA_CHECK_DIR="$DATA_DIR/tokyo_gazebo"
fi

# Count sessions
SESSION_COUNT=$(find "$DATA_CHECK_DIR" -maxdepth 1 -type d -name "*_*" 2>/dev/null | wc -l)

if [ "$SESSION_COUNT" -eq 0 ] && [ "$MODE" != "nuscenes" ]; then
    log_error "No training sessions found in $DATA_CHECK_DIR"
    log_info "Data generation may have failed. Check errors above."
    exit 1
fi

if [ "$MODE" != "nuscenes" ]; then
    # Count frames
    FRAME_COUNT=$(find "$DATA_CHECK_DIR" -name "*_occupancy.npz" 2>/dev/null | wc -l)
    log_success "Found $SESSION_COUNT sessions with $FRAME_COUNT total frames"
else
    log_success "nuScenes data directory verified"
fi

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
    TRAIN_CMD="python train_6dof.py --config config/finetune_6dof.py --work-dir $WORK_DIR --epochs $EPOCHS"
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
