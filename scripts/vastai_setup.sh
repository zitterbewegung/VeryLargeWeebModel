#!/bin/bash
# =============================================================================
# Vast.ai Setup Script for OccWorld Training
# =============================================================================
# Run this after SSHing into your Vast.ai instance.
#
# Usage:
#   ./scripts/vastai_setup.sh
#
# Prerequisites:
#   - Vast.ai instance with PyTorch template
#   - A100 or similar GPU with 40GB+ VRAM
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }

echo ""
echo "=============================================="
echo "    OccWorld Vast.ai Setup                   "
echo "=============================================="
echo ""

# Check if we're on Vast.ai (usually has /workspace)
if [ -d "/workspace" ]; then
    WORK_DIR="/workspace"
    log_info "Vast.ai detected, using /workspace"
else
    WORK_DIR="$HOME"
    log_warn "Not on Vast.ai, using $HOME"
fi

# Check GPU
log_info "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    log_success "GPU: $GPU_NAME ($GPU_MEM)"
else
    log_warn "nvidia-smi not found"
fi

# Install system packages
log_info "Installing system packages..."
apt-get update -qq
apt-get install -y -qq screen tmux htop git wget unzip > /dev/null 2>&1
log_success "System packages installed"

# Check PyTorch (should be pre-installed on Vast.ai)
log_info "Checking PyTorch..."
if python -c "import torch; print(torch.__version__)" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_AVAIL=$(python -c "import torch; print(torch.cuda.is_available())")
    log_success "PyTorch $TORCH_VERSION (CUDA: $CUDA_AVAIL)"
else
    log_warn "PyTorch not found, installing..."
    pip install torch torchvision --quiet
fi

# Install ML dependencies
log_info "Installing ML dependencies..."
pip install --quiet \
    tqdm \
    scipy \
    opencv-python \
    pillow \
    tensorboard \
    open3d \
    trimesh \
    einops \
    timm

log_success "ML dependencies installed"

# Install mesh processing dependencies (for PLATEAU data conversion)
log_info "Installing mesh processing dependencies..."
pip install --quiet \
    pyvista \
    fast_simplification \
    numpy-stl \
    pycollada

log_success "Mesh processing dependencies installed"

# Setup project directory
PROJECT_DIR="${WORK_DIR}/VeryLargeWeebModel"
if [ ! -d "$PROJECT_DIR" ]; then
    log_warn "Project not found at $PROJECT_DIR"
    log_info "Make sure you cloned the repo first:"
    echo "  git clone https://github.com/YOUR_USERNAME/VeryLargeWeebModel.git"
fi

# Create checkpoint directory
mkdir -p "${WORK_DIR}/checkpoints"
log_success "Checkpoint directory: ${WORK_DIR}/checkpoints"

# Create convenience aliases
cat >> ~/.bashrc << 'EOF'

# OccWorld Vast.ai aliases
alias gpu='watch -n 1 nvidia-smi'
alias tb='tensorboard --logdir /workspace/checkpoints --port 6006 --bind_all'
alias train='cd /workspace/VeryLargeWeebModel && python train.py --config config/finetune_tokyo.py --work-dir /workspace/checkpoints'
alias logs='tail -f /workspace/checkpoints/*/training.log 2>/dev/null || tail -f /workspace/checkpoints/*.log 2>/dev/null || echo "No logs found"'
EOF

# Verify setup
echo ""
echo "=============================================="
echo "         Setup Complete!                     "
echo "=============================================="
echo ""
echo "GPU: $GPU_NAME"
echo "PyTorch: $TORCH_VERSION"
echo "Project: $PROJECT_DIR"
echo "Checkpoints: ${WORK_DIR}/checkpoints"
echo ""
echo "Next steps:"
echo ""
echo "  1. Start screen (keeps training alive if disconnected):"
echo "     screen -S training"
echo ""
echo "  2. Start training:"
echo "     cd ${PROJECT_DIR}"
echo "     python train.py \\"
echo "         --config config/finetune_tokyo.py \\"
echo "         --work-dir ${WORK_DIR}/checkpoints"
echo ""
echo "  3. Detach from screen: Ctrl+A, then D"
echo "     Reattach later: screen -r training"
echo ""
echo "  4. Monitor GPU: nvidia-smi -l 1"
echo ""
echo "  5. IMPORTANT: Download checkpoints before destroying instance!"
echo ""
echo "=============================================="
