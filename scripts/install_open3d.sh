#!/bin/bash
# =============================================================================
# Install Open3D-ML Pipeline Dependencies
# =============================================================================
#
# Usage:
#   ./scripts/install_open3d.sh          # Install with CUDA 11.8
#   ./scripts/install_open3d.sh cpu      # Install CPU-only
#   ./scripts/install_open3d.sh cu121    # Install with CUDA 12.1
#
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Detect CUDA version or use argument
CUDA_VERSION="${1:-cu118}"

echo "=============================================="
echo "  Open3D-ML Pipeline Installation"
echo "=============================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
log_info "Python version: $PYTHON_VERSION"

# Install PyTorch first (with appropriate CUDA version)
log_info "Installing PyTorch with $CUDA_VERSION..."

if [ "$CUDA_VERSION" = "cpu" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
elif [ "$CUDA_VERSION" = "cu118" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
elif [ "$CUDA_VERSION" = "cu121" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [ "$CUDA_VERSION" = "cu124" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
    log_warn "Unknown CUDA version: $CUDA_VERSION, trying default..."
    pip install torch torchvision
fi

log_success "PyTorch installed"

# Verify PyTorch CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Install Open3D
log_info "Installing Open3D..."
pip install open3d>=0.18.0
log_success "Open3D installed"

# Install other dependencies
log_info "Installing other dependencies..."
pip install numpy'>=1.24.0,<2.0.0' scipy>=1.10.0 requests>=2.28.0 tqdm>=4.65.0
pip install trimesh>=4.0.0 numba>=0.57.0
pip install opencv-python>=4.8.0 pillow>=9.0.0
pip install wandb>=0.15.0 tensorboard>=2.12.0
pip install pyyaml>=6.0 rich>=13.0.0 pyquaternion>=0.9.9
log_success "Dependencies installed"

# Try to install Open3D-ML (optional)
log_info "Attempting to install Open3D-ML (optional)..."
if pip install open3d-ml-torch 2>/dev/null; then
    log_success "Open3D-ML installed (semantic features available)"
else
    log_warn "Open3D-ML installation failed (semantic features disabled)"
    log_warn "This is optional - geometric features still work"
    log_warn "To install manually: pip install open3d-ml-torch"
fi

echo ""
echo "=============================================="
echo "  Installation Complete!"
echo "=============================================="
echo ""

# Verify installation
log_info "Verifying installation..."
python3 << 'EOF'
import sys

def check(name, import_name=None):
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {name}: {version}")
        return True
    except ImportError:
        print(f"  ✗ {name}: NOT INSTALLED")
        return False

print("\nInstalled packages:")
check("PyTorch", "torch")
check("Open3D", "open3d")
check("NumPy", "numpy")
check("SciPy", "scipy")
check("Trimesh", "trimesh")
check("Numba", "numba")
check("OpenCV", "cv2")
check("W&B", "wandb")

# Check Open3D-ML
try:
    import open3d.ml
    print("  ✓ Open3D-ML: available")
except ImportError:
    print("  ○ Open3D-ML: not installed (optional)")

# Check CUDA
import torch
if torch.cuda.is_available():
    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA: {torch.version.cuda}")
else:
    print("\n  GPU: Not available (CPU mode)")

print("\nReady to run:")
print("  python scripts/setup_open3d_pipeline.py --download --dataset shibuya")
EOF

echo ""
