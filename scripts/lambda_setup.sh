#!/bin/bash
# =============================================================================
# Lambda Cloud Setup Script for VeryLargeWeebModel Training
# =============================================================================
# This script sets up a Lambda Cloud GPU instance for VeryLargeWeebModel training.
#
# Usage:
#   ./scripts/lambda_setup.sh [OPTIONS]
#
# Options:
#   --full          Full setup including data download
#   --minimal       Minimal setup (dependencies only)
#   --skip-pytorch  Skip PyTorch installation (use Lambda's pre-installed)
#   --help          Show this help message
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()    { echo -e "${CYAN}[STEP]${NC} $1"; }

# Default options
FULL_SETUP=false
MINIMAL_SETUP=false
SKIP_PYTORCH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)         FULL_SETUP=true; shift ;;
        --minimal)      MINIMAL_SETUP=true; shift ;;
        --skip-pytorch) SKIP_PYTORCH=true; shift ;;
        --help|-h)      head -20 "$0" | tail -15; exit 0 ;;
        *)              log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# Detect Environment
# =============================================================================
echo ""
echo "=============================================="
echo "    VeryLargeWeebModel Lambda Cloud Setup Script       "
echo "=============================================="
echo ""

log_step "Detecting environment..."

# Check if Lambda instance
if [ -f /etc/lambda-stack-version ]; then
    LAMBDA_VERSION=$(cat /etc/lambda-stack-version)
    log_success "Lambda Stack detected: $LAMBDA_VERSION"
else
    log_warn "Not a Lambda instance - some features may not work"
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    log_success "GPU detected: $GPU_NAME ($GPU_MEM)"
else
    log_error "No GPU detected!"
    exit 1
fi

# Check CUDA
if [ -n "$CUDA_HOME" ]; then
    log_success "CUDA_HOME: $CUDA_HOME"
else
    log_warn "CUDA_HOME not set"
fi

# =============================================================================
# System Packages
# =============================================================================
log_step "Installing system packages..."

sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    htop \
    screen \
    nvtop \
    tree \
    rsync

log_success "System packages installed"

# =============================================================================
# Conda Environment
# =============================================================================
log_step "Setting up conda environment..."

# Find conda
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    log_warn "Conda not found, installing miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p ~/miniconda3
    source ~/miniconda3/etc/profile.d/conda.sh
fi

# Create environment
if conda env list | grep -q "^vlwm "; then
    log_warn "Conda environment 'vlwm' already exists"
    conda activate vlwm
else
    log_info "Creating conda environment 'vlwm'..."
    conda create -n vlwm python=3.8 -y
    conda activate vlwm
fi

log_success "Conda environment ready"

# =============================================================================
# PyTorch Installation
# =============================================================================
if [ "$SKIP_PYTORCH" = false ]; then
    log_step "Installing PyTorch..."

    # Check CUDA version
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    log_info "CUDA version: $CUDA_VERSION"

    # Install appropriate PyTorch
    if [[ "$CUDA_VERSION" == 12.* ]]; then
        pip install -q torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == 11.8* ]]; then
        pip install -q torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
    else
        pip install -q torch torchvision
    fi

    # Verify PyTorch CUDA
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"
    log_success "PyTorch installed with CUDA support"
else
    log_warn "Skipping PyTorch installation"
fi

# =============================================================================
# MMDetection3D Stack
# =============================================================================
log_step "Installing MMDetection3D stack..."

# Detect PyTorch and CUDA versions for mmcv
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION_SHORT=$(python -c "import torch; print(torch.version.cuda.replace('.', '')[:3])")

log_info "Installing mmcv for PyTorch $TORCH_VERSION, CUDA $CUDA_VERSION_SHORT"

# Install mmcv (this can take a while)
pip install -q mmcv-full -f "https://download.openmmlab.com/mmcv/dist/cu${CUDA_VERSION_SHORT}/torch${TORCH_VERSION%.*}.0/index.html" || {
    log_warn "mmcv-full install failed, trying mmcv..."
    pip install -q mmcv
}

# Install mmdet and mmsegmentation
pip install -q mmdet mmsegmentation

log_success "MMDetection3D stack installed"

# =============================================================================
# VeryLargeWeebModel Dependencies
# =============================================================================
log_step "Installing VeryLargeWeebModel dependencies..."

pip install -q \
    nuscenes-devkit \
    torchpack \
    tqdm \
    open3d \
    scipy \
    opencv-python \
    pillow \
    trimesh \
    tensorboard \
    einops \
    timm \
    yapf==0.40.1

log_success "VeryLargeWeebModel dependencies installed"

# =============================================================================
# Mesh Processing Dependencies
# =============================================================================
log_step "Installing mesh processing dependencies..."

pip install -q \
    pyvista \
    fast_simplification \
    numpy-stl \
    pycollada

log_success "Mesh processing dependencies installed"

# =============================================================================
# Clone VeryLargeWeebModel Repository
# =============================================================================
log_step "Setting up VeryLargeWeebModel repository..."

if [ ! -d ~/VeryLargeWeebModel ]; then
    log_info "Cloning VeryLargeWeebModel..."
    git clone https://github.com/wzzheng/VeryLargeWeebModel.git ~/VeryLargeWeebModel
    cd ~/VeryLargeWeebModel
    pip install -q -e .
    cd -
else
    log_warn "VeryLargeWeebModel already exists at ~/VeryLargeWeebModel"
fi

log_success "VeryLargeWeebModel repository ready"

# =============================================================================
# Setup Persistent Storage
# =============================================================================
log_step "Setting up storage..."

# Create directories
mkdir -p ~/checkpoints
mkdir -p ~/data

# Check for Lambda persistent storage
if [ -d /home/ubuntu/persistent ]; then
    log_info "Persistent storage detected at /home/ubuntu/persistent"
    mkdir -p /home/ubuntu/persistent/checkpoints
    mkdir -p /home/ubuntu/persistent/data

    # Create symlinks
    if [ ! -L ~/checkpoints ] && [ -d ~/checkpoints ]; then
        rm -rf ~/checkpoints
        ln -s /home/ubuntu/persistent/checkpoints ~/checkpoints
    fi

    log_success "Persistent storage configured"
else
    log_warn "No persistent storage found - data will be lost on termination!"
    log_warn "Attach persistent storage in Lambda console for data preservation"
fi

# =============================================================================
# Project Setup
# =============================================================================
log_step "Setting up project..."

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Create required directories
mkdir -p pretrained/vqvae
mkdir -p pretrained/vlwm
mkdir -p pretrained/bevfusion
mkdir -p data/tokyo_gazebo
mkdir -p config

log_success "Project directories ready"

# =============================================================================
# Shell Configuration
# =============================================================================
log_step "Configuring shell..."

# Add aliases and environment to bashrc
if ! grep -q "# VeryLargeWeebModel Lambda Setup" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# VeryLargeWeebModel Lambda Setup
# ----------------------

# Conda activation
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null

# Aliases
alias oc='conda activate vlwm'
alias gpu='watch -n 1 nvidia-smi'
alias gpumon='nvtop'
alias tb='tensorboard --logdir ~/checkpoints --port 6006 --bind_all'
alias train='cd ~/VeryLargeWeebModel && conda activate vlwm'
alias logs='tail -f ~/checkpoints/*/training.log 2>/dev/null || echo "No training logs found"'

# Quick functions
start_training() {
    screen -dmS training bash -c "conda activate vlwm && cd ~/VeryLargeWeebModel && python train.py --config config/finetune_tokyo.py --work-dir ~/checkpoints/vlwm_tokyo"
    echo "Training started in screen session 'training'"
    echo "Attach with: screen -r training"
}

backup_checkpoints() {
    local dest="${1:-/tmp/checkpoint_backup}"
    mkdir -p "$dest"
    cp -r ~/checkpoints/* "$dest/"
    echo "Checkpoints backed up to $dest"
}

# Auto-activate vlwm environment
conda activate vlwm 2>/dev/null
EOF

    log_success "Shell configuration added"
else
    log_warn "Shell configuration already exists"
fi

# =============================================================================
# Download Data (Full Setup Only)
# =============================================================================
if [ "$FULL_SETUP" = true ]; then
    log_step "Downloading data (full setup)..."

    if [ -f "$PROJECT_DIR/scripts/download_and_prepare_data.sh" ]; then
        chmod +x "$PROJECT_DIR/scripts/download_and_prepare_data.sh"
        "$PROJECT_DIR/scripts/download_and_prepare_data.sh" --models --skip-plateau
    else
        log_warn "Download script not found"
    fi
fi

# =============================================================================
# Verification
# =============================================================================
log_step "Verifying installation..."

echo ""
echo "=============================================="
echo "         Installation Verification           "
echo "=============================================="

# Check GPU
echo -n "GPU: "
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# Check Python
echo -n "Python: "
python --version 2>&1 | cut -d' ' -f2

# Check PyTorch
echo -n "PyTorch: "
python -c "import torch; print(f'{torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>/dev/null || echo "Not installed"

# Check mmcv
echo -n "mmcv: "
python -c "import mmcv; print(mmcv.__version__)" 2>/dev/null || echo "Not installed"

# Check VeryLargeWeebModel
echo -n "VeryLargeWeebModel: "
if [ -d ~/VeryLargeWeebModel ]; then echo "Installed"; else echo "Not found"; fi

# Check storage
echo -n "Persistent Storage: "
if [ -d /home/ubuntu/persistent ]; then echo "Available"; else echo "Not available"; fi

echo "=============================================="

# =============================================================================
# Summary
# =============================================================================
echo ""
log_success "Lambda Cloud setup complete!"
echo ""
echo "=============================================="
echo "                 Next Steps                  "
echo "=============================================="
echo ""
echo "1. Reload shell configuration:"
echo "   source ~/.bashrc"
echo ""
echo "2. Download pretrained models (manual - Tsinghua Cloud):"
echo "   Visit: https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/"
echo "   Download to: ~/VeryLargeWeebModel/pretrained/"
echo ""
echo "3. Start training:"
echo "   screen -S training"
echo "   conda activate vlwm"
echo "   cd ~/VeryLargeWeebModel"
echo "   python train.py --config config/finetune_tokyo.py \\"
echo "       --work-dir ~/checkpoints/vlwm_tokyo"
echo "   # Detach: Ctrl+A, then D"
echo "   # Reattach: screen -r training"
echo ""
echo "4. Monitor training:"
echo "   - GPU: nvtop"
echo "   - TensorBoard: tb (then open http://<IP>:6006)"
echo "   - Logs: logs"
echo ""
echo "5. IMPORTANT - Don't forget to:"
echo "   - Use screen to prevent training loss on disconnect"
echo "   - Download checkpoints before terminating instance"
echo "   - Terminate instance when done (billing is per-minute!)"
echo ""
echo "=============================================="
