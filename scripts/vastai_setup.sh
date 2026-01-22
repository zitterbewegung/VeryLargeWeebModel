#!/bin/bash
# =============================================================================
# Vast.ai Setup Script for VeryLargeWeebModel Training
# =============================================================================
# Comprehensive setup script optimized for Vast.ai GPU instances.
#
# Features:
#   - Auto-detects Vast.ai environment and template type
#   - Configures persistent storage for checkpoints
#   - Installs optimized dependencies with parallel downloads
#   - Sets up TensorBoard with port forwarding instructions
#   - GPU-specific optimizations (A100, H100, RTX 4090, etc.)
#   - Disk space verification and cleanup
#
# Usage:
#   # Run after SSH into Vast.ai instance:
#   ./scripts/vastai_setup.sh [OPTIONS]
#
#   # Or with curl (remote setup):
#   curl -sSL https://raw.githubusercontent.com/zitterbewegung/VeryLargeWeebModel/main/scripts/vastai_setup.sh | bash
#
# Options:
#   --full              Full setup including data download (default)
#   --minimal           Minimal setup (dependencies only)
#   --deps-only         Install dependencies only (no data)
#   --skip-data         Skip data download
#   --skip-models       Skip pretrained model download
#   --tensorboard       Start TensorBoard after setup
#   --jupyter           Setup Jupyter notebook server
#   --clean             Clean previous installation
#   --help              Show this help
#
# Prerequisites:
#   - Vast.ai instance (recommended: PyTorch template)
#   - GPU with 24GB+ VRAM (A100/H100/RTX 4090 recommended)
#   - 50GB+ disk space for full setup
#
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_VERSION="2.0.0"
GITHUB_REPO="${GITHUB_REPO:-zitterbewegung/VeryLargeWeebModel}"
GITHUB_URL="https://github.com/${GITHUB_REPO}.git"
MIN_DISK_GB=30
MIN_GPU_MEM_MB=16000

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Logging functions
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()    {
    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Default options
SETUP_MODE="full"
SKIP_DATA=false
SKIP_MODELS=false
START_TENSORBOARD=false
SETUP_JUPYTER=false
CLEAN_INSTALL=false

# =============================================================================
# Banner
# =============================================================================
show_banner() {
    echo ""
    echo -e "${MAGENTA}${BOLD}"
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                            ║"
    echo "║   ██╗   ██╗██╗    ██╗    ██╗███╗   ███╗  ╔═╗ VAST.AI                       ║"
    echo "║   ██║   ██║██║    ██║    ██║████╗ ████║  ╚═╗ SETUP                         ║"
    echo "║   ██║   ██║██║    ██║ █╗ ██║██╔████╔██║  ╔═╝ v${SCRIPT_VERSION}                          ║"
    echo "║   ╚██╗ ██╔╝██║    ██║███╗██║██║╚██╔╝██║  ║                                 ║"
    echo "║    ╚████╔╝ ███████╗╚███╔███╔╝██║ ╚═╝ ██║  ╚═══════════════════════════════╝║"
    echo "║     ╚═══╝  ╚══════╝ ╚══╝╚══╝ ╚═╝     ╚═╝                                   ║"
    echo "║                                                                            ║"
    echo "║          VeryLargeWeebModel Training Environment for Vast.ai              ║"
    echo "║                                                                            ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# =============================================================================
# Parse Arguments
# =============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)         SETUP_MODE="full"; shift ;;
            --minimal)      SETUP_MODE="minimal"; shift ;;
            --deps-only)    SETUP_MODE="deps"; shift ;;
            --skip-data)    SKIP_DATA=true; shift ;;
            --skip-models)  SKIP_MODELS=true; shift ;;
            --tensorboard)  START_TENSORBOARD=true; shift ;;
            --jupyter)      SETUP_JUPYTER=true; shift ;;
            --clean)        CLEAN_INSTALL=true; shift ;;
            --help|-h)      head -45 "$0" | tail -40; exit 0 ;;
            *)              log_warn "Unknown option: $1"; shift ;;
        esac
    done
}

# =============================================================================
# Environment Detection
# =============================================================================
detect_environment() {
    log_step "Detecting Vast.ai Environment"

    # Detect Vast.ai by environment variables and paths
    VASTAI_DETECTED=false

    # Check for common Vast.ai indicators
    if [ -d "/workspace" ]; then
        VASTAI_DETECTED=true
        WORK_DIR="/workspace"
        log_success "Vast.ai detected (found /workspace)"
    elif [ -n "$VAST_CONTAINERLABEL" ] || [ -n "$VAST_TCP_PORT_22" ]; then
        VASTAI_DETECTED=true
        WORK_DIR="${HOME}"
        log_success "Vast.ai detected (environment variables)"
    else
        WORK_DIR="${HOME}"
        log_warn "Vast.ai not detected, using home directory"
    fi

    # Detect template type
    TEMPLATE_TYPE="unknown"
    if python3 -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
            TEMPLATE_TYPE="pytorch-cuda"
            log_success "Template: PyTorch ${TORCH_VERSION} with CUDA ${CUDA_VERSION}"
        else
            TEMPLATE_TYPE="pytorch-cpu"
            log_warn "Template: PyTorch ${TORCH_VERSION} (CPU only)"
        fi
    elif [ -f "/etc/nvidia/cuda-version" ]; then
        TEMPLATE_TYPE="cuda-only"
        log_info "Template: CUDA base image"
    else
        log_warn "Template: Unknown (no PyTorch/CUDA detected)"
    fi

    # Set directories
    PROJECT_DIR="${WORK_DIR}/VeryLargeWeebModel"
    CHECKPOINT_DIR="${WORK_DIR}/checkpoints"
    DATA_DIR="${PROJECT_DIR}/data"
    PRETRAINED_DIR="${PROJECT_DIR}/pretrained"

    log_info "Working directory: $WORK_DIR"
    log_info "Project directory: $PROJECT_DIR"
    log_info "Checkpoint directory: $CHECKPOINT_DIR"
}

# =============================================================================
# System Checks
# =============================================================================
check_system() {
    log_step "System Requirements Check"

    # Check disk space
    log_info "Checking disk space..."
    DISK_FREE_GB=$(df -BG "$WORK_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$DISK_FREE_GB" -lt "$MIN_DISK_GB" ]; then
        log_error "Insufficient disk space: ${DISK_FREE_GB}GB free (need ${MIN_DISK_GB}GB)"
        log_info "Consider:"
        log_info "  - Using a larger instance"
        log_info "  - Running with --minimal flag"
        log_info "  - Cleaning up: rm -rf /workspace/cache/*"
        exit 1
    fi
    log_success "Disk space: ${DISK_FREE_GB}GB free"

    # Check GPU
    log_info "Checking GPU..."
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 || echo "1")
        GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")

        log_success "GPU: $GPU_NAME (${GPU_MEM}MB VRAM) x $GPU_COUNT"
        log_info "Driver: $GPU_DRIVER"

        # GPU memory warning
        if [ "$GPU_MEM" -lt "$MIN_GPU_MEM_MB" ]; then
            log_warn "GPU memory below recommended ${MIN_GPU_MEM_MB}MB"
            log_info "Training may be slow or fail with OOM errors"
        fi

        # Auto-detect optimal settings based on GPU
        if [ "$GPU_MEM" -ge 70000 ]; then
            RECOMMENDED_BATCH_SIZE=12
            GPU_TIER="high-end"
        elif [ "$GPU_MEM" -ge 40000 ]; then
            RECOMMENDED_BATCH_SIZE=6
            GPU_TIER="high"
        elif [ "$GPU_MEM" -ge 24000 ]; then
            RECOMMENDED_BATCH_SIZE=4
            GPU_TIER="mid-high"
        elif [ "$GPU_MEM" -ge 16000 ]; then
            RECOMMENDED_BATCH_SIZE=2
            GPU_TIER="mid"
        else
            RECOMMENDED_BATCH_SIZE=1
            GPU_TIER="low"
        fi
        log_info "GPU tier: $GPU_TIER (recommended batch size: $RECOMMENDED_BATCH_SIZE)"

        # Detect precision support
        if echo "$GPU_NAME" | grep -qiE "A100|H100|H200|A6000|RTX 40|RTX 30"; then
            RECOMMENDED_PRECISION="bf16"
            log_info "Precision: BF16 supported (native)"
        else
            RECOMMENDED_PRECISION="fp16"
            log_info "Precision: FP16 recommended"
        fi
    else
        log_error "nvidia-smi not found - no GPU available!"
        GPU_NAME="None"
        GPU_MEM=0
        RECOMMENDED_BATCH_SIZE=1
        RECOMMENDED_PRECISION="fp32"
    fi

    # Check internet connectivity
    log_info "Checking internet connectivity..."
    if curl -s --connect-timeout 5 https://github.com > /dev/null 2>&1; then
        log_success "Internet: Connected"
    else
        log_warn "Internet connectivity issues detected"
    fi
}

# =============================================================================
# Install System Packages
# =============================================================================
install_system_packages() {
    log_step "Installing System Packages"

    # Detect package manager
    if command -v apt-get &> /dev/null; then
        log_info "Updating package lists..."
        sudo apt-get update -qq 2>/dev/null || apt-get update -qq 2>/dev/null || true

        PACKAGES="git wget curl unzip screen tmux htop aria2 axel rsync jq"
        log_info "Installing: $PACKAGES"
        sudo apt-get install -y -qq $PACKAGES 2>/dev/null || \
            apt-get install -y -qq $PACKAGES 2>/dev/null || true
    elif command -v yum &> /dev/null; then
        log_info "Installing packages via yum..."
        sudo yum install -y git wget curl unzip screen tmux htop aria2 rsync jq 2>/dev/null || true
    fi

    log_success "System packages installed"
}

# =============================================================================
# Fast Download Function
# =============================================================================
fast_download() {
    local url="$1"
    local output="$2"
    local description="${3:-file}"
    local max_retries=3

    log_info "Downloading $description..."

    mkdir -p "$(dirname "$output")"

    for attempt in $(seq 1 $max_retries); do
        # Try aria2c first (fastest - 16 parallel connections)
        if command -v aria2c &> /dev/null; then
            log_info "  Using aria2c (16 connections) - attempt $attempt/$max_retries"
            if aria2c -x 16 -s 16 --file-allocation=none \
                      --max-tries=3 --retry-wait=2 \
                      -d "$(dirname "$output")" -o "$(basename "$output")" "$url" 2>/dev/null; then
                log_success "Downloaded $description"
                return 0
            fi
        fi

        # Try axel (also fast)
        if command -v axel &> /dev/null; then
            log_info "  Using axel - attempt $attempt/$max_retries"
            if axel -n 16 -o "$output" "$url" 2>/dev/null; then
                log_success "Downloaded $description"
                return 0
            fi
        fi

        # Fallback to curl with resume support
        log_info "  Using curl - attempt $attempt/$max_retries"
        if curl -L -C - --retry 3 --retry-delay 2 -o "$output" "$url" 2>/dev/null; then
            log_success "Downloaded $description"
            return 0
        fi

        log_warn "Attempt $attempt failed, retrying..."
        rm -f "$output"
        sleep $((attempt * 2))
    done

    log_error "Failed to download $description after $max_retries attempts"
    return 1
}

# =============================================================================
# Setup Python Environment
# =============================================================================
setup_python() {
    log_step "Setting Up Python Environment"

    # Check existing Python/PyTorch
    PYTHON_CMD="python3"
    if ! command -v python3 &> /dev/null; then
        PYTHON_CMD="python"
    fi

    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    log_info "Python: $PYTHON_VERSION"

    # Check/install PyTorch
    if ! $PYTHON_CMD -c "import torch" 2>/dev/null; then
        log_info "Installing PyTorch..."

        # Detect best CUDA version
        if command -v nvcc &> /dev/null; then
            NVCC_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' | cut -d. -f1,2)
            log_info "Detected CUDA: $NVCC_VERSION"
        fi

        # Install PyTorch with appropriate CUDA
        pip install --upgrade pip
        if pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null; then
            log_success "PyTorch installed with CUDA 12.1"
        elif pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null; then
            log_success "PyTorch installed with CUDA 11.8"
        else
            pip install torch torchvision
            log_warn "PyTorch installed (default version)"
        fi
    fi

    # Verify PyTorch CUDA
    if $PYTHON_CMD -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)")
        CUDA_VERSION=$($PYTHON_CMD -c "import torch; print(torch.version.cuda)")
        log_success "PyTorch $TORCH_VERSION with CUDA $CUDA_VERSION"
    else
        log_warn "PyTorch installed but CUDA not available"
    fi
}

# =============================================================================
# Install Project Dependencies
# =============================================================================
install_dependencies() {
    log_step "Installing Project Dependencies"

    cd "$PROJECT_DIR" 2>/dev/null || cd "$WORK_DIR"

    # Core ML dependencies
    CORE_DEPS=(
        "tqdm"
        "scipy"
        "opencv-python"
        "pillow"
        "tensorboard"
        "einops"
        "timm"
        "open3d"
        "trimesh"
        "pyvista"
        "wandb"
    )

    # Mesh processing dependencies
    MESH_DEPS=(
        "fast_simplification"
        "numpy-stl"
        "pycollada"
    )

    # Optional optimizations
    OPT_DEPS=(
        "flash-attn"
        "xformers"
    )

    log_info "Installing core dependencies..."
    pip install --quiet "${CORE_DEPS[@]}" || {
        log_warn "Some core dependencies failed, installing individually..."
        for dep in "${CORE_DEPS[@]}"; do
            pip install --quiet "$dep" 2>/dev/null || log_warn "Failed: $dep"
        done
    }

    log_info "Installing mesh processing dependencies..."
    pip install --quiet "${MESH_DEPS[@]}" 2>/dev/null || true

    # Install requirements.txt if exists
    if [ -f "$PROJECT_DIR/requirements.txt" ]; then
        log_info "Installing from requirements.txt..."
        pip install --quiet -r "$PROJECT_DIR/requirements.txt" 2>/dev/null || true
    fi

    # Try optional optimizations (may fail on some GPUs)
    if [ "$GPU_TIER" != "low" ]; then
        log_info "Installing GPU optimizations (optional)..."
        pip install --quiet flash-attn --no-build-isolation 2>/dev/null && \
            log_success "Flash Attention 2 installed" || \
            log_info "Flash Attention not available (requires CUDA 11.6+)"

        pip install --quiet xformers 2>/dev/null && \
            log_success "xFormers installed" || \
            log_info "xFormers not available"
    fi

    log_success "Dependencies installed"
}

# =============================================================================
# Clone/Update Repository
# =============================================================================
setup_repository() {
    log_step "Setting Up Repository"

    if [ -d "$PROJECT_DIR/.git" ]; then
        log_info "Repository exists, updating..."
        cd "$PROJECT_DIR"
        git fetch --all 2>/dev/null || true
        git status
        git pull --ff-only 2>/dev/null || log_warn "Could not pull (may have local changes)"
    else
        log_info "Cloning repository..."
        git clone --progress "$GITHUB_URL" "$PROJECT_DIR"
        cd "$PROJECT_DIR"
    fi

    log_info "Current commit: $(git log --oneline -1)"
    log_success "Repository ready: $PROJECT_DIR"
}

# =============================================================================
# Download Pretrained Models
# =============================================================================
download_models() {
    if [ "$SKIP_MODELS" = true ]; then
        log_info "Skipping model download (--skip-models)"
        return 0
    fi

    log_step "Downloading Pretrained Models"

    mkdir -p "${PRETRAINED_DIR}/occworld"

    local MODEL_FILE="${PRETRAINED_DIR}/occworld/latest.pth"
    local MODEL_URL="https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/files/?p=/latest.pth&dl=1"

    # Check if already downloaded
    if [ -f "$MODEL_FILE" ]; then
        local size=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE" 2>/dev/null || echo 0)
        if [ "$size" -gt 700000000 ]; then
            local size_mb=$((size / 1024 / 1024))
            log_success "Pretrained model already downloaded (${size_mb}MB)"
            return 0
        else
            log_warn "Existing model seems incomplete, re-downloading..."
            rm -f "$MODEL_FILE"
        fi
    fi

    # Download
    fast_download "$MODEL_URL" "$MODEL_FILE" "OccWorld pretrained model (~721MB)"

    # Verify
    if [ -f "$MODEL_FILE" ]; then
        local size=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE" 2>/dev/null || echo 0)
        if [ "$size" -gt 700000000 ]; then
            log_success "Pretrained model ready!"
        else
            log_error "Download incomplete. Manual download:"
            log_info "  curl -L -o ${MODEL_FILE} '${MODEL_URL}'"
        fi
    fi
}

# =============================================================================
# Setup Training Data
# =============================================================================
setup_data() {
    if [ "$SKIP_DATA" = true ] || [ "$SETUP_MODE" = "minimal" ] || [ "$SETUP_MODE" = "deps" ]; then
        log_info "Skipping data setup"
        return 0
    fi

    log_step "Setting Up Training Data"

    mkdir -p "${DATA_DIR}/tokyo_gazebo"

    # Check for existing data
    local existing_sessions=$(ls -d ${DATA_DIR}/tokyo_gazebo/drone_* ${DATA_DIR}/tokyo_gazebo/rover_* 2>/dev/null | wc -l || echo 0)

    if [ "$existing_sessions" -gt 10 ]; then
        log_success "Training data exists: $existing_sessions sessions"
        return 0
    fi

    # Run data preparation script if available
    if [ -f "${PROJECT_DIR}/scripts/download_and_prepare_data.sh" ]; then
        log_info "Running data preparation script..."
        chmod +x "${PROJECT_DIR}/scripts/download_and_prepare_data.sh"
        "${PROJECT_DIR}/scripts/download_and_prepare_data.sh" --models --generate-data || {
            log_warn "Data preparation had issues"
        }
    else
        log_warn "Data preparation script not found"
        log_info "Generate data manually after setup"
    fi
}

# =============================================================================
# Create Directories
# =============================================================================
setup_directories() {
    log_step "Creating Directories"

    mkdir -p "$CHECKPOINT_DIR"
    mkdir -p "${CHECKPOINT_DIR}/logs"
    mkdir -p "${DATA_DIR}"
    mkdir -p "${PRETRAINED_DIR}"

    # Symlink checkpoints to persistent storage if available
    if [ -d "/workspace" ] && [ "$WORK_DIR" = "/workspace" ]; then
        # Vast.ai persistent storage setup
        if [ ! -L "${PROJECT_DIR}/checkpoints" ]; then
            ln -sf "$CHECKPOINT_DIR" "${PROJECT_DIR}/checkpoints" 2>/dev/null || true
        fi
    fi

    log_success "Directories ready"
}

# =============================================================================
# Configure Shell
# =============================================================================
configure_shell() {
    log_step "Configuring Shell Environment"

    local SHELL_RC="$HOME/.bashrc"

    # Check if already configured
    if grep -q "# VeryLargeWeebModel Vast.ai Setup" "$SHELL_RC" 2>/dev/null; then
        log_info "Shell already configured"
        return 0
    fi

    cat >> "$SHELL_RC" << EOF

# VeryLargeWeebModel Vast.ai Setup
# ================================
export PROJECT_DIR="$PROJECT_DIR"
export CHECKPOINT_DIR="$CHECKPOINT_DIR"
export DATA_DIR="$DATA_DIR"

# Quick aliases
alias train='cd \$PROJECT_DIR && python train.py --config config/finetune_tokyo.py --work-dir \$CHECKPOINT_DIR'
alias train-optimized='\$PROJECT_DIR/scripts/train_optimized.sh'
alias gpu='watch -n 1 nvidia-smi'
alias gpumem='nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1'
alias tb='tensorboard --logdir \$CHECKPOINT_DIR --port 6006 --bind_all'
alias logs='tail -f \$CHECKPOINT_DIR/*.log 2>/dev/null || echo "No logs found"'
alias attach='screen -r training 2>/dev/null || tmux attach -t training 2>/dev/null'
alias status='nvidia-smi && echo "" && df -h /workspace && echo "" && ls -la \$CHECKPOINT_DIR/*.pth 2>/dev/null | tail -5'

# Project navigation
alias vlwm='cd \$PROJECT_DIR'
alias data='cd \$DATA_DIR'
alias ckpt='cd \$CHECKPOINT_DIR'

# Auto-activate if in project
cd_vlwm() {
    cd "\$PROJECT_DIR" && echo "VeryLargeWeebModel project directory"
}
EOF

    log_success "Shell aliases configured"
    log_info "Reload with: source ~/.bashrc"
}

# =============================================================================
# Setup TensorBoard
# =============================================================================
setup_tensorboard() {
    if [ "$START_TENSORBOARD" != true ]; then
        return 0
    fi

    log_step "Starting TensorBoard"

    # Kill existing TensorBoard
    pkill -f tensorboard 2>/dev/null || true

    # Start TensorBoard in background
    nohup tensorboard --logdir "$CHECKPOINT_DIR" --port 6006 --bind_all > /tmp/tensorboard.log 2>&1 &

    sleep 2

    if pgrep -f tensorboard > /dev/null; then
        log_success "TensorBoard started on port 6006"
        echo ""
        echo "Access TensorBoard:"
        echo "  1. In Vast.ai console, add port 6006 to 'Direct Port HTTP'"
        echo "  2. Or use SSH tunnel: ssh -L 6006:localhost:6006 root@<instance-ip>"
        echo "  3. Then open: http://localhost:6006"
    else
        log_warn "TensorBoard failed to start"
        cat /tmp/tensorboard.log 2>/dev/null || true
    fi
}

# =============================================================================
# Setup Jupyter
# =============================================================================
setup_jupyter() {
    if [ "$SETUP_JUPYTER" != true ]; then
        return 0
    fi

    log_step "Setting Up Jupyter"

    pip install --quiet jupyterlab notebook 2>/dev/null

    # Generate config
    jupyter notebook --generate-config -y 2>/dev/null || true

    log_success "Jupyter installed"
    echo ""
    echo "Start Jupyter with:"
    echo "  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
    echo ""
    echo "Access via Vast.ai port forwarding or SSH tunnel"
}

# =============================================================================
# Print Summary
# =============================================================================
print_summary() {
    echo ""
    echo -e "${GREEN}${BOLD}"
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                         SETUP COMPLETE!                                    ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo "  Environment:"
    echo "    GPU:           $GPU_NAME (${GPU_MEM}MB)"
    echo "    GPU Tier:      $GPU_TIER"
    echo "    Python:        $PYTHON_VERSION"
    echo "    PyTorch:       ${TORCH_VERSION:-N/A}"
    echo "    CUDA:          ${CUDA_VERSION:-N/A}"
    echo ""
    echo "  Directories:"
    echo "    Project:       $PROJECT_DIR"
    echo "    Checkpoints:   $CHECKPOINT_DIR"
    echo "    Data:          $DATA_DIR"
    echo ""
    echo "  Recommended Settings:"
    echo "    Batch Size:    $RECOMMENDED_BATCH_SIZE"
    echo "    Precision:     $RECOMMENDED_PRECISION"
    echo ""

    # Check pretrained model
    if [ -f "${PRETRAINED_DIR}/occworld/latest.pth" ]; then
        echo -e "  ${GREEN}✓${NC} Pretrained model: Ready"
    else
        echo -e "  ${RED}✗${NC} Pretrained model: Missing"
    fi

    # Check training data
    local data_sessions=$(ls -d ${DATA_DIR}/tokyo_gazebo/drone_* ${DATA_DIR}/tokyo_gazebo/rover_* 2>/dev/null | wc -l || echo 0)
    if [ "$data_sessions" -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} Training data: $data_sessions sessions"
    else
        echo -e "  ${YELLOW}!${NC} Training data: Not generated"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Quick Start:"
    echo ""
    echo "    1. Start training in screen (recommended):"
    echo "       screen -S training"
    echo "       cd $PROJECT_DIR"
    echo "       python train.py --config config/finetune_tokyo.py \\"
    echo "           --work-dir $CHECKPOINT_DIR"
    echo ""
    echo "    2. Or use optimized training script:"
    echo "       ./scripts/train_optimized.sh --batch-size $RECOMMENDED_BATCH_SIZE"
    echo ""
    echo "    3. Detach from screen: Ctrl+A, then D"
    echo "       Reattach: screen -r training"
    echo ""
    echo "    4. Monitor:"
    echo "       nvidia-smi -l 1          # GPU usage"
    echo "       tb                        # TensorBoard (alias)"
    echo "       logs                      # Training logs (alias)"
    echo ""
    echo "  Port Forwarding (for TensorBoard/Jupyter):"
    echo "    - Add port 6006 in Vast.ai console under 'Direct Port HTTP'"
    echo "    - Or SSH tunnel: ssh -L 6006:localhost:6006 root@<ip> -p <port>"
    echo ""
    echo "  IMPORTANT: Download checkpoints before destroying instance!"
    echo "    scp -r root@<ip>:$CHECKPOINT_DIR ./local_checkpoints/"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# =============================================================================
# Clean Installation
# =============================================================================
clean_install() {
    if [ "$CLEAN_INSTALL" != true ]; then
        return 0
    fi

    log_step "Cleaning Previous Installation"

    log_warn "This will remove:"
    log_warn "  - $PROJECT_DIR"
    log_warn "  - Python packages"
    echo ""
    read -p "Continue? [y/N] " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$PROJECT_DIR"
        log_success "Cleaned previous installation"
    else
        log_info "Clean cancelled"
    fi
}

# =============================================================================
# Main
# =============================================================================
main() {
    show_banner
    parse_args "$@"

    echo ""
    log_info "Setup mode: $SETUP_MODE"
    log_info "Repository: $GITHUB_REPO"
    echo ""

    # Clean if requested
    clean_install

    # Core setup steps
    detect_environment
    check_system
    install_system_packages
    setup_python

    # Repository setup
    setup_repository

    # Dependencies (always)
    install_dependencies

    # Create directories
    setup_directories

    # Mode-specific setup
    case $SETUP_MODE in
        full)
            download_models
            setup_data
            ;;
        minimal)
            download_models
            ;;
        deps)
            log_info "Dependencies-only mode, skipping data/models"
            ;;
    esac

    # Configure shell
    configure_shell

    # Optional services
    setup_tensorboard
    setup_jupyter

    # Summary
    print_summary
}

# Run main
main "$@"
