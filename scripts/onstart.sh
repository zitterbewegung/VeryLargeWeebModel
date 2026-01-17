#!/bin/bash
# =============================================================================
# OccWorld Auto-Setup and Training Script
# =============================================================================
# One-shot script for cloud GPU instances (Vast.ai, Lambda, RunPod, etc.)
# Automatically sets up environment, downloads data, and starts training.
#
# Usage:
#   # As onstart script (runs everything automatically):
#   curl -sSL https://raw.githubusercontent.com/YOUR_USERNAME/VeryLargeWeebModel/main/scripts/onstart.sh | bash
#
#   # Or clone first and run:
#   ./scripts/onstart.sh [OPTIONS]
#
# Options:
#   --no-train          Setup only, don't start training
#   --resume            Resume from existing checkpoint
#   --epochs N          Set number of epochs (default: 50)
#   --batch-size N      Override batch size
#   --clean-data        Remove existing training data before setup
#   --clean-checkpoints Remove existing checkpoints before setup
#   --clean-all         Remove both data and checkpoints
#   --skip-nuscenes     Skip nuScenes mini download
#   --skip-plateau      Skip PLATEAU data download
#   --skip-real-data    Skip all real data downloads (use dummy data)
#   --help              Show this help
#
# Environment Variables:
#   GITHUB_REPO     Repository to clone (default: zitterbewegung/VeryLargeWeebModel)
#   WORK_DIR        Working directory (auto-detected)
#   CHECKPOINT_DIR  Checkpoint directory (default: $WORK_DIR/checkpoints)
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
GITHUB_REPO="${GITHUB_REPO:-zitterbewegung/VeryLargeWeebModel}"
GITHUB_URL="https://github.com/${GITHUB_REPO}.git"
SCRIPT_START_TIME=$(date +%s)

# Show all commands if VERBOSE is set
if [ "${VERBOSE:-0}" = "1" ]; then
    set -x
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Logging
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# Step tracking with timing
STEP_NUM=0
STEP_START=0
log_step() {
    # Print elapsed time for previous step
    if [ "$STEP_START" -gt 0 ]; then
        local elapsed=$(($(date +%s) - STEP_START))
        echo -e "${GREEN}    ✓ Completed in ${elapsed}s${NC}"
    fi

    STEP_NUM=$((STEP_NUM + 1))
    STEP_START=$(date +%s)
    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${BOLD}  Step $STEP_NUM: $1${NC}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Default options
START_TRAINING=true
RESUME_TRAINING=false
EPOCHS=50
BATCH_SIZE=""
CLEAN_DATA=false
CLEAN_CHECKPOINTS=false
DOWNLOAD_NUSCENES=true
DOWNLOAD_PLATEAU=true
SKIP_REAL_DATA=false

# =============================================================================
# Banner
# =============================================================================
echo ""
echo -e "${CYAN}${BOLD}"
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                       ║"
echo "║     ██████╗  ██████╗ ██████╗██╗    ██╗ ██████╗ ██████╗ ██╗     ██████╗║"
echo "║    ██╔═══██╗██╔════╝██╔════╝██║    ██║██╔═══██╗██╔══██╗██║     ██╔══██╗"
echo "║    ██║   ██║██║     ██║     ██║ █╗ ██║██║   ██║██████╔╝██║     ██║  ██║"
echo "║    ██║   ██║██║     ██║     ██║███╗██║██║   ██║██╔══██╗██║     ██║  ██║"
echo "║    ╚██████╔╝╚██████╗╚██████╗╚███╔███╔╝╚██████╔╝██║  ██║███████╗██████╔╝"
echo "║     ╚═════╝  ╚═════╝ ╚═════╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═════╝ "
echo "║                                                                       ║"
echo "║              Auto-Setup & Training Script for Cloud GPUs              ║"
echo "║                                                                       ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo -e "${BLUE}Started at:${NC} $(date)"
echo -e "${BLUE}Repository:${NC} $GITHUB_REPO"
echo ""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-train)         START_TRAINING=false; shift ;;
        --resume)           RESUME_TRAINING=true; shift ;;
        --epochs)           EPOCHS="$2"; shift 2 ;;
        --batch-size)       BATCH_SIZE="$2"; shift 2 ;;
        --clean-data)       CLEAN_DATA=true; shift ;;
        --clean-checkpoints) CLEAN_CHECKPOINTS=true; shift ;;
        --clean-all)        CLEAN_DATA=true; CLEAN_CHECKPOINTS=true; shift ;;
        --skip-nuscenes)    DOWNLOAD_NUSCENES=false; shift ;;
        --skip-plateau)     DOWNLOAD_PLATEAU=false; shift ;;
        --skip-real-data)   SKIP_REAL_DATA=true; DOWNLOAD_NUSCENES=false; DOWNLOAD_PLATEAU=false; shift ;;
        --help|-h)          head -40 "$0" | tail -35; exit 0 ;;
        *)                  log_warn "Unknown option: $1"; shift ;;
    esac
done

# =============================================================================
# Environment Detection
# =============================================================================
log_step "Detecting environment..."

# Detect cloud provider
if [ -d "/workspace" ]; then
    CLOUD_ENV="vastai"
    WORK_DIR="/workspace"
    log_info "Detected: Vast.ai"
elif [ -f "/etc/lambda-stack-version" ]; then
    CLOUD_ENV="lambda"
    WORK_DIR="/home/ubuntu"
    log_info "Detected: Lambda Cloud ($(cat /etc/lambda-stack-version))"
elif [ -d "/root/workspace" ]; then
    CLOUD_ENV="runpod"
    WORK_DIR="/root/workspace"
    log_info "Detected: RunPod"
else
    CLOUD_ENV="generic"
    WORK_DIR="${HOME}"
    log_info "Detected: Generic Linux"
fi

PROJECT_DIR="${WORK_DIR}/VeryLargeWeebModel"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORK_DIR}/checkpoints}"
DATA_DIR="${PROJECT_DIR}/data"
PRETRAINED_DIR="${PROJECT_DIR}/pretrained"

log_info "Working directory: $WORK_DIR"
log_info "Project directory: $PROJECT_DIR"
log_info "Checkpoint directory: $CHECKPOINT_DIR"

# =============================================================================
# Clean Data/Checkpoints (if requested)
# =============================================================================
if [ "$CLEAN_DATA" = true ] || [ "$CLEAN_CHECKPOINTS" = true ]; then
    log_step "Cleaning up..."

    if [ "$CLEAN_DATA" = true ]; then
        if [ -d "${PROJECT_DIR}/data/tokyo_gazebo" ]; then
            log_warn "Removing training data: ${PROJECT_DIR}/data/tokyo_gazebo"
            rm -rf "${PROJECT_DIR}/data/tokyo_gazebo"
            log_success "Training data removed"
        else
            log_info "No training data to remove"
        fi
    fi

    if [ "$CLEAN_CHECKPOINTS" = true ]; then
        if [ -d "$CHECKPOINT_DIR" ]; then
            log_warn "Removing checkpoints: $CHECKPOINT_DIR"
            rm -rf "$CHECKPOINT_DIR"
            log_success "Checkpoints removed"
        else
            log_info "No checkpoints to remove"
        fi
    fi

    echo ""
    log_success "Cleanup complete"
fi

# =============================================================================
# GPU Check
# =============================================================================
log_step "Checking GPU..."

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "1")
    log_success "GPU: $GPU_NAME ($GPU_MEM) x $GPU_COUNT"

    # Recommend batch size based on GPU memory
    MEM_GB=$(echo "$GPU_MEM" | grep -oE '[0-9]+' | head -1)
    if [ -n "$MEM_GB" ] && [ -z "$BATCH_SIZE" ]; then
        if [ "$MEM_GB" -ge 80 ]; then
            BATCH_SIZE=4
        elif [ "$MEM_GB" -ge 40 ]; then
            BATCH_SIZE=2
        else
            BATCH_SIZE=1
        fi
        log_info "Auto-selected batch size: $BATCH_SIZE (based on ${MEM_GB}GB VRAM)"
    fi
else
    log_error "No GPU detected! Training will be very slow."
    GPU_NAME="CPU"
fi

# =============================================================================
# System Packages
# =============================================================================
log_step "Installing system packages..."

# Detect package manager
if command -v apt-get &> /dev/null; then
    PKG_MGR="apt"
    log_info "Updating package lists..."
    sudo apt-get update 2>/dev/null || apt-get update 2>/dev/null || true
    log_info "Installing: git wget curl unzip screen tmux htop..."
    sudo apt-get install -y git wget curl unzip screen tmux htop 2>/dev/null || \
        apt-get install -y git wget curl unzip screen tmux htop 2>/dev/null || true
elif command -v yum &> /dev/null; then
    PKG_MGR="yum"
    log_info "Installing: git wget curl unzip screen tmux htop..."
    sudo yum install -y git wget curl unzip screen tmux htop 2>/dev/null || true
fi

log_success "System packages ready"

# =============================================================================
# Clone Repository
# =============================================================================
log_step "Setting up project repository..."

if [ -d "$PROJECT_DIR/.git" ]; then
    log_info "Repository already exists at: $PROJECT_DIR"
    cd "$PROJECT_DIR"
    log_info "Pulling latest changes..."
    git fetch --all
    git status
    git pull --ff-only || log_warn "Could not pull latest (may have local changes)"
else
    log_info "Cloning repository from: $GITHUB_URL"
    log_info "Destination: $PROJECT_DIR"
    git clone --progress "$GITHUB_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi
log_info "Current commit:"
git log --oneline -1

log_success "Repository ready: $PROJECT_DIR"

# =============================================================================
# Python Environment
# =============================================================================
log_step "Setting up Python environment..."

# Check for conda
CONDA_FOUND=false
log_info "Searching for conda installation..."
for conda_path in ~/miniconda3 /opt/miniconda3 ~/anaconda3 /opt/conda; do
    log_info "  Checking: $conda_path"
    if [ -f "${conda_path}/etc/profile.d/conda.sh" ]; then
        source "${conda_path}/etc/profile.d/conda.sh"
        CONDA_FOUND=true
        log_success "Found conda at: $conda_path"
        break
    fi
done

if [ "$CONDA_FOUND" = true ]; then
    # Create/activate conda environment
    log_info "Checking for existing 'occworld' environment..."
    conda env list
    echo ""

    if conda env list | grep -q "^occworld "; then
        log_info "Activating existing conda environment 'occworld'..."
        conda activate occworld
        log_success "Activated conda environment"
    else
        log_info "Creating new conda environment 'occworld' with Python 3.10..."
        conda create -n occworld python=3.10 -y
        log_info "Activating conda environment..."
        conda activate occworld
        log_success "Created and activated conda environment"
    fi
else
    log_warn "Conda not found, using system Python"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
log_success "Python: $PYTHON_VERSION"

# =============================================================================
# PyTorch Installation
# =============================================================================
log_step "Installing PyTorch..."

# Check if PyTorch already installed with CUDA
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
    log_success "PyTorch $TORCH_VERSION with CUDA $CUDA_VERSION already installed"
else
    log_info "Installing PyTorch with CUDA support..."

    # Detect CUDA version
    if command -v nvcc &> /dev/null; then
        NVCC_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        log_info "NVCC version: $NVCC_VERSION"
    fi

    # Install PyTorch (use pip for broadest compatibility)
    log_info "Upgrading pip..."
    pip install --upgrade pip

    log_info "Installing PyTorch (trying CUDA 12.1 first)..."
    if pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121; then
        log_success "PyTorch installed with CUDA 12.1"
    else
        log_warn "CUDA 12.1 failed, trying CUDA 11.8..."
        if pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118; then
            log_success "PyTorch installed with CUDA 11.8"
        else
            log_warn "CUDA versions failed, installing default PyTorch..."
            pip install torch torchvision
        fi
    fi

    # Verify
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        log_success "PyTorch $TORCH_VERSION installed with CUDA"
    else
        log_warn "PyTorch installed but CUDA not available"
    fi
fi

# =============================================================================
# Project Dependencies
# =============================================================================
log_step "Installing project dependencies..."

cd "$PROJECT_DIR"

# Install from requirements if exists
if [ -f "requirements.txt" ]; then
    log_info "Installing from requirements.txt..."
    pip install -r requirements.txt
else
    log_info "Installing core dependencies..."

    DEPS=(tqdm scipy opencv-python pillow tensorboard einops timm open3d trimesh pyvista)
    TOTAL=${#DEPS[@]}
    CURRENT=0

    for dep in "${DEPS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo -e "${BLUE}[$CURRENT/$TOTAL]${NC} Installing $dep..."
        pip install "$dep" || log_warn "Failed to install $dep"
    done
fi

log_success "Dependencies installed"

# =============================================================================
# Download Pretrained Models
# =============================================================================
log_step "Downloading pretrained models..."

mkdir -p "${PRETRAINED_DIR}/occworld"

OCCWORLD_MODEL="${PRETRAINED_DIR}/occworld/latest.pth"
OCCWORLD_URL="https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/files/?p=/latest.pth&dl=1"

if [ -f "$OCCWORLD_MODEL" ]; then
    MODEL_SIZE=$(stat -f%z "$OCCWORLD_MODEL" 2>/dev/null || stat -c%s "$OCCWORLD_MODEL" 2>/dev/null || echo 0)
    if [ "$MODEL_SIZE" -gt 700000000 ]; then
        log_success "OccWorld checkpoint already downloaded ($(numfmt --to=iec $MODEL_SIZE 2>/dev/null || echo "${MODEL_SIZE} bytes"))"
    else
        log_warn "Existing checkpoint seems incomplete, re-downloading..."
        rm -f "$OCCWORLD_MODEL"
    fi
fi

if [ ! -f "$OCCWORLD_MODEL" ]; then
    log_info "Downloading OccWorld checkpoint (~721MB)..."
    log_info "URL: $OCCWORLD_URL"
    log_info "Destination: $OCCWORLD_MODEL"
    echo ""

    # Use curl with full progress display
    if curl -L -o "$OCCWORLD_MODEL" "$OCCWORLD_URL"; then
        echo ""
        MODEL_SIZE=$(stat -f%z "$OCCWORLD_MODEL" 2>/dev/null || stat -c%s "$OCCWORLD_MODEL" 2>/dev/null || echo 0)
        MODEL_SIZE_MB=$((MODEL_SIZE / 1024 / 1024))
        if [ "$MODEL_SIZE" -gt 700000000 ]; then
            log_success "OccWorld checkpoint downloaded! (${MODEL_SIZE_MB}MB)"
        else
            log_error "Download seems incomplete (${MODEL_SIZE_MB}MB, expected ~721MB)"
            log_info "Try manual download: curl -L -o ${OCCWORLD_MODEL} '${OCCWORLD_URL}'"
        fi
    else
        echo ""
        log_error "Download failed. Training will start from scratch."
    fi
fi

# =============================================================================
# Download nuScenes Mini Dataset (Real Driving Data)
# =============================================================================
if [ "$DOWNLOAD_NUSCENES" = true ] && [ "$SKIP_REAL_DATA" = false ]; then
    log_step "Downloading nuScenes Mini dataset (~4GB)..."

    NUSCENES_DIR="${DATA_DIR}/nuscenes"
    NUSCENES_ARCHIVE="${DATA_DIR}/v1.0-mini.tgz"

    mkdir -p "$NUSCENES_DIR"

    # Check if already downloaded
    if [ -d "${NUSCENES_DIR}/v1.0-mini" ] && [ -d "${NUSCENES_DIR}/samples" ]; then
        log_success "nuScenes Mini already downloaded"
        ls -la "$NUSCENES_DIR"
    else
        log_info "Downloading nuScenes Mini from nuscenes.org..."
        log_info "This is real autonomous driving data (~4GB)"
        log_warn "Note: nuScenes requires registration. If download fails, visit https://www.nuscenes.org/"
        echo ""

        # nuScenes direct download link (may require auth)
        NUSCENES_URL="https://www.nuscenes.org/data/v1.0-mini.tgz"

        if [ ! -f "$NUSCENES_ARCHIVE" ]; then
            log_info "Downloading: $NUSCENES_URL"
            if curl -L -o "$NUSCENES_ARCHIVE" "$NUSCENES_URL" 2>&1; then
                ARCHIVE_SIZE=$(stat -f%z "$NUSCENES_ARCHIVE" 2>/dev/null || stat -c%s "$NUSCENES_ARCHIVE" 2>/dev/null || echo 0)
                ARCHIVE_SIZE_MB=$((ARCHIVE_SIZE / 1024 / 1024))
                log_info "Downloaded: ${ARCHIVE_SIZE_MB}MB"
            else
                log_warn "Direct download failed. You may need to register at nuscenes.org"
                log_info "Manual download: https://www.nuscenes.org/nuscenes#download"
                log_info "Place v1.0-mini.tgz in: ${DATA_DIR}/"
            fi
        fi

        # Extract if archive exists and is valid size
        if [ -f "$NUSCENES_ARCHIVE" ]; then
            ARCHIVE_SIZE=$(stat -f%z "$NUSCENES_ARCHIVE" 2>/dev/null || stat -c%s "$NUSCENES_ARCHIVE" 2>/dev/null || echo 0)
            if [ "$ARCHIVE_SIZE" -gt 100000000 ]; then  # > 100MB
                log_info "Extracting nuScenes Mini..."
                cd "$NUSCENES_DIR"
                tar -xzf "$NUSCENES_ARCHIVE" --strip-components=0 || tar -xzf "$NUSCENES_ARCHIVE"
                cd "$PROJECT_DIR"
                log_success "nuScenes Mini extracted"

                # Verify extraction
                if [ -d "${NUSCENES_DIR}/samples" ] || [ -d "${NUSCENES_DIR}/v1.0-mini/samples" ]; then
                    log_success "nuScenes Mini ready!"
                    # Move contents up if nested
                    if [ -d "${NUSCENES_DIR}/v1.0-mini/samples" ] && [ ! -d "${NUSCENES_DIR}/samples" ]; then
                        mv ${NUSCENES_DIR}/v1.0-mini/* ${NUSCENES_DIR}/ 2>/dev/null || true
                    fi
                else
                    log_warn "Extraction may have failed - check ${NUSCENES_DIR}"
                fi
            else
                log_warn "Archive too small (${ARCHIVE_SIZE} bytes), download may have failed"
            fi
        fi
    fi

    # Install nuscenes-devkit if needed
    if ! python3 -c "from nuscenes.nuscenes import NuScenes" 2>/dev/null; then
        log_info "Installing nuscenes-devkit..."
        pip install nuscenes-devkit pyquaternion
    fi
else
    log_info "Skipping nuScenes download (--skip-nuscenes or --skip-real-data)"
fi

# =============================================================================
# Download PLATEAU Tokyo 3D City Data
# =============================================================================
if [ "$DOWNLOAD_PLATEAU" = true ] && [ "$SKIP_REAL_DATA" = false ]; then
    log_step "Downloading Tokyo PLATEAU 3D city data..."

    PLATEAU_DIR="${DATA_DIR}/plateau"
    PLATEAU_RAW="${PLATEAU_DIR}/raw"
    PLATEAU_MESHES="${PLATEAU_DIR}/meshes"

    mkdir -p "$PLATEAU_RAW"
    mkdir -p "$PLATEAU_MESHES"

    # PLATEAU OBJ download URL
    PLATEAU_OBJ_URL="https://gic-plateau.s3.ap-northeast-1.amazonaws.com/2020/13100_tokyo23-ku_2020_obj_3_op.zip"
    PLATEAU_OBJ_ARCHIVE="${PLATEAU_RAW}/tokyo23ku_obj.zip"

    if [ -d "${PLATEAU_MESHES}/obj" ] && [ "$(ls -A ${PLATEAU_MESHES}/obj 2>/dev/null)" ]; then
        log_success "PLATEAU meshes already extracted"
        MESH_COUNT=$(find "${PLATEAU_MESHES}/obj" -name "*.obj" 2>/dev/null | wc -l)
        log_info "Found $MESH_COUNT OBJ files"
    else
        # Download PLATEAU OBJ files (~2.1GB)
        if [ ! -f "$PLATEAU_OBJ_ARCHIVE" ]; then
            log_info "Downloading Tokyo PLATEAU OBJ models (~2.1GB)..."
            log_info "Source: Project PLATEAU (MLIT Japan) - CC BY 4.0"
            log_info "URL: $PLATEAU_OBJ_URL"
            echo ""

            if curl -L -o "$PLATEAU_OBJ_ARCHIVE" "$PLATEAU_OBJ_URL"; then
                ARCHIVE_SIZE=$(stat -f%z "$PLATEAU_OBJ_ARCHIVE" 2>/dev/null || stat -c%s "$PLATEAU_OBJ_ARCHIVE" 2>/dev/null || echo 0)
                ARCHIVE_SIZE_MB=$((ARCHIVE_SIZE / 1024 / 1024))
                log_success "Downloaded PLATEAU OBJ: ${ARCHIVE_SIZE_MB}MB"
            else
                log_error "PLATEAU download failed"
            fi
        else
            log_info "PLATEAU archive already exists"
        fi

        # Extract
        if [ -f "$PLATEAU_OBJ_ARCHIVE" ]; then
            ARCHIVE_SIZE=$(stat -f%z "$PLATEAU_OBJ_ARCHIVE" 2>/dev/null || stat -c%s "$PLATEAU_OBJ_ARCHIVE" 2>/dev/null || echo 0)
            if [ "$ARCHIVE_SIZE" -gt 100000000 ]; then  # > 100MB
                log_info "Extracting PLATEAU meshes..."
                mkdir -p "${PLATEAU_MESHES}/obj"
                unzip -q -o "$PLATEAU_OBJ_ARCHIVE" -d "${PLATEAU_MESHES}/obj/" || {
                    log_warn "Unzip had issues, trying alternate method..."
                    cd "${PLATEAU_MESHES}/obj" && unzip -o "$PLATEAU_OBJ_ARCHIVE"
                    cd "$PROJECT_DIR"
                }
                MESH_COUNT=$(find "${PLATEAU_MESHES}/obj" -name "*.obj" 2>/dev/null | wc -l)
                log_success "Extracted $MESH_COUNT OBJ mesh files"
            fi
        fi
    fi

    # Convert PLATEAU to OccWorld training format
    if [ -f "${PROJECT_DIR}/scripts/plateau_to_occworld.py" ]; then
        PLATEAU_TRAINING_DIR="${DATA_DIR}/tokyo_gazebo"

        # Check if already converted
        EXISTING_PLATEAU_SESSIONS=$(ls -d ${PLATEAU_TRAINING_DIR}/drone_* ${PLATEAU_TRAINING_DIR}/rover_* 2>/dev/null | wc -l || echo 0)

        if [ "$EXISTING_PLATEAU_SESSIONS" -gt 3 ]; then
            log_success "PLATEAU training data already generated: $EXISTING_PLATEAU_SESSIONS sessions"
        else
            log_info "Converting PLATEAU meshes to OccWorld training format..."
            log_info "This generates occupancy grids and synthetic trajectories"
            echo ""

            # Install trimesh if needed
            if ! python3 -c "import trimesh" 2>/dev/null; then
                log_info "Installing trimesh for mesh processing..."
                pip install trimesh
            fi

            python3 "${PROJECT_DIR}/scripts/plateau_to_occworld.py" \
                --input "${PLATEAU_MESHES}/obj" \
                --output "$PLATEAU_TRAINING_DIR" \
                --frames 500 \
                --sessions 10 \
                --pattern survey \
                --max-meshes 30 || {
                log_warn "PLATEAU conversion had issues, trying with fewer meshes..."
                python3 "${PROJECT_DIR}/scripts/plateau_to_occworld.py" \
                    --input "${PLATEAU_MESHES}/obj" \
                    --output "$PLATEAU_TRAINING_DIR" \
                    --frames 200 \
                    --sessions 5 \
                    --max-meshes 10 || log_warn "PLATEAU conversion failed"
            }

            GENERATED_SESSIONS=$(ls -d ${PLATEAU_TRAINING_DIR}/drone_* ${PLATEAU_TRAINING_DIR}/rover_* 2>/dev/null | wc -l || echo 0)
            log_success "Generated $GENERATED_SESSIONS training sessions from PLATEAU data"
        fi
    else
        log_warn "plateau_to_occworld.py not found, skipping conversion"
    fi
else
    log_info "Skipping PLATEAU download (--skip-plateau or --skip-real-data)"
fi

# =============================================================================
# Setup Training Data (Fallback to Dummy if No Real Data)
# =============================================================================
log_step "Verifying training data..."

log_info "Creating data directory: ${DATA_DIR}/tokyo_gazebo"
mkdir -p "${DATA_DIR}/tokyo_gazebo"

# Check if data already exists
log_info "Checking for existing training data..."
EXISTING_SESSIONS=$(ls -d ${DATA_DIR}/tokyo_gazebo/drone_* ${DATA_DIR}/tokyo_gazebo/rover_* 2>/dev/null | wc -l || echo 0)

if [ "$EXISTING_SESSIONS" -gt 0 ]; then
    log_success "Training data exists: $EXISTING_SESSIONS sessions"
    log_info "Existing sessions:"
    ls -la ${DATA_DIR}/tokyo_gazebo/ | head -20
else
    log_info "No existing training data found. Generating..."

    # Try to run the data generation script
    if [ -f "${PROJECT_DIR}/scripts/download_and_prepare_data.sh" ]; then
        log_info "Running download_and_prepare_data.sh..."
        chmod +x "${PROJECT_DIR}/scripts/download_and_prepare_data.sh"
        "${PROJECT_DIR}/scripts/download_and_prepare_data.sh" --models --skip-plateau --generate-data || {
            log_warn "Data generation had issues, creating minimal dataset..."
        }
    else
        log_warn "download_and_prepare_data.sh not found"
    fi

    # Fallback: create dummy data for testing
    if [ ! -d "${DATA_DIR}/tokyo_gazebo/drone_session_001" ]; then
        log_info "Creating minimal test dataset..."
        python3 -c "
import os
import numpy as np
import json

data_dir = '${DATA_DIR}/tokyo_gazebo'
for session_type in ['drone', 'rover']:
    session_dir = os.path.join(data_dir, f'{session_type}_session_001')
    for subdir in ['images', 'lidar', 'poses', 'occupancy']:
        os.makedirs(os.path.join(session_dir, subdir), exist_ok=True)

    # Create 20 frames
    for i in range(1, 21):
        fid = f'{i:06d}'

        # Dummy occupancy
        occ = np.random.rand(200, 200, 16) > 0.95  # Sparse occupancy
        np.savez_compressed(os.path.join(session_dir, 'occupancy', f'{fid}_occupancy.npz'), occupancy=occ)

        # Dummy pose
        pose = {
            'position': [float(i), 0.0, 10.0],
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'timestamp': float(i) * 0.1,
            'linear_velocity': [1.0, 0.0, 0.0],
            'angular_velocity': [0.0, 0.0, 0.0]
        }
        with open(os.path.join(session_dir, 'poses', f'{fid}.json'), 'w') as f:
            json.dump(pose, f)

        # Dummy lidar
        points = np.random.rand(1000, 4).astype(np.float32)
        np.save(os.path.join(session_dir, 'lidar', f'{fid}_LIDAR.npy'), points)

print('Created minimal test dataset')
" 2>/dev/null || log_warn "Could not create test dataset"
    fi

    log_success "Training data ready"
fi

# =============================================================================
# Create Directories
# =============================================================================
log_step "Creating checkpoint directories..."

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "${CHECKPOINT_DIR}/logs"

log_success "Directories ready"

# =============================================================================
# Shell Configuration
# =============================================================================
log_step "Configuring shell..."

SHELL_CONFIG=""
case "$CLOUD_ENV" in
    vastai)  SHELL_CONFIG="$HOME/.bashrc" ;;
    lambda)  SHELL_CONFIG="$HOME/.bashrc" ;;
    runpod)  SHELL_CONFIG="$HOME/.bashrc" ;;
    *)       SHELL_CONFIG="$HOME/.bashrc" ;;
esac

if [ -n "$SHELL_CONFIG" ] && ! grep -q "# OccWorld Training Setup" "$SHELL_CONFIG" 2>/dev/null; then
    cat >> "$SHELL_CONFIG" << EOF

# OccWorld Training Setup
# -----------------------
export PROJECT_DIR="$PROJECT_DIR"
export CHECKPOINT_DIR="$CHECKPOINT_DIR"

# Aliases
alias train='cd \$PROJECT_DIR && python train.py --config config/finetune_tokyo.py --work-dir \$CHECKPOINT_DIR'
alias gpu='watch -n 1 nvidia-smi'
alias tb='tensorboard --logdir \$CHECKPOINT_DIR --port 6006 --bind_all'
alias logs='tail -f \$CHECKPOINT_DIR/*.log 2>/dev/null || echo "No logs found"'
alias attach='screen -r training'

# Conda activation (if available)
[ -f ~/miniconda3/etc/profile.d/conda.sh ] && source ~/miniconda3/etc/profile.d/conda.sh && conda activate occworld 2>/dev/null
EOF
    log_success "Shell configuration added"
fi

# =============================================================================
# Verification
# =============================================================================
log_step "Verifying installation..."

echo ""
echo "=============================================="
echo "         Installation Summary                "
echo "=============================================="
echo ""
echo "  Cloud:        $CLOUD_ENV"
echo "  GPU:          $GPU_NAME"
echo "  Python:       $PYTHON_VERSION"
echo "  PyTorch:      $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "  CUDA:         $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
echo ""
echo "  Project:      $PROJECT_DIR"
echo "  Checkpoints:  $CHECKPOINT_DIR"
echo "  Data:         $DATA_DIR"
echo ""

# Check pretrained model
if [ -f "$OCCWORLD_MODEL" ]; then
    echo -e "  ${GREEN}✓${NC} Pretrained model: OK"
else
    echo -e "  ${RED}✗${NC} Pretrained model: MISSING"
fi

# Check training data
DATA_SESSIONS=$(ls -d ${DATA_DIR}/tokyo_gazebo/*_session_* ${DATA_DIR}/tokyo_gazebo/drone_* ${DATA_DIR}/tokyo_gazebo/rover_* 2>/dev/null | wc -l || echo 0)
if [ "$DATA_SESSIONS" -gt 0 ]; then
    echo -e "  ${GREEN}✓${NC} Training data: $DATA_SESSIONS sessions"
else
    echo -e "  ${YELLOW}!${NC} Training data: minimal/test only"
fi

echo ""
echo "=============================================="

# =============================================================================
# Start Training
# =============================================================================
if [ "$START_TRAINING" = true ]; then
    log_step "Starting training..."

    cd "$PROJECT_DIR"

    # Build training command
    TRAIN_CMD="python train.py --config config/finetune_tokyo.py --work-dir $CHECKPOINT_DIR"

    if [ "$RESUME_TRAINING" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --resume"
    fi

    if [ -n "$BATCH_SIZE" ]; then
        TRAIN_CMD="$TRAIN_CMD --batch-size $BATCH_SIZE"
    fi

    if [ -n "$EPOCHS" ]; then
        TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
    fi

    log_info "Training command: $TRAIN_CMD"

    # Start in screen session
    screen -dmS training bash -c "
        cd $PROJECT_DIR
        source ~/.bashrc 2>/dev/null || true
        echo 'Starting training at \$(date)...'
        echo 'Command: $TRAIN_CMD'
        echo ''
        $TRAIN_CMD 2>&1 | tee ${CHECKPOINT_DIR}/training_\$(date +%Y%m%d_%H%M%S).log
        echo ''
        echo 'Training completed at \$(date)'
        echo 'Press Enter to exit...'
        read
    "

    sleep 2

    if screen -list | grep -q "training"; then
        log_success "Training started in screen session 'training'"
        echo ""
        echo "=============================================="
        echo "              Training Started!              "
        echo "=============================================="
        echo ""
        echo "  Monitor training:"
        echo "    screen -r training     # Attach to training session"
        echo "    Ctrl+A, D              # Detach from session"
        echo ""
        echo "  Monitor GPU:"
        echo "    nvidia-smi -l 1"
        echo "    nvtop"
        echo ""
        echo "  TensorBoard:"
        echo "    tensorboard --logdir $CHECKPOINT_DIR --port 6006 --bind_all"
        echo "    Then open http://<IP>:6006"
        echo ""
        echo "  Checkpoints saved to: $CHECKPOINT_DIR"
        echo ""
        echo "=============================================="
    else
        log_error "Failed to start training screen session"
        log_info "Try running manually:"
        echo "  cd $PROJECT_DIR"
        echo "  $TRAIN_CMD"
    fi
else
    echo ""
    echo "=============================================="
    echo "            Setup Complete!                  "
    echo "=============================================="
    echo ""
    echo "  Start training manually:"
    echo "    cd $PROJECT_DIR"
    echo "    python train.py --config config/finetune_tokyo.py \\"
    echo "        --work-dir $CHECKPOINT_DIR"
    echo ""
    echo "  Or use screen for persistent session:"
    echo "    screen -S training"
    echo "    # Then run training command"
    echo "    # Detach: Ctrl+A, D"
    echo ""
    echo "=============================================="
fi

# =============================================================================
# Timing
# =============================================================================
SCRIPT_END_TIME=$(date +%s)
ELAPSED=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
log_info "Setup completed in ${ELAPSED} seconds"
