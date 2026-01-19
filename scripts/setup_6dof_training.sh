#!/bin/bash
# =============================================================================
# Setup 6DoF World Model Training Environment
# =============================================================================
#
# One-command setup for training 6DoF world models with multiple datasets.
#
# Usage:
#   ./scripts/setup_6dof_training.sh                    # Full setup guide
#   ./scripts/setup_6dof_training.sh --nuscenes         # Setup nuScenes only
#   ./scripts/setup_6dof_training.sh --uavscenes        # Setup UAVScenes only
#   ./scripts/setup_6dof_training.sh --all              # Setup all datasets
#   ./scripts/setup_6dof_training.sh --deps-only        # Install dependencies only
#
# Datasets supported:
#   - nuScenes + 6DoF augmentation (ground vehicles with aerial simulation)
#   - UAVScenes (real aerial UAV data)
#   - PLATEAU/Gazebo (synthetic Tokyo city data)
#
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() { echo -e "\n${BOLD}${CYAN}=== $1 ===${NC}\n"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
SETUP_NUSCENES=false
SETUP_UAVSCENES=false
SETUP_PLATEAU=false
DEPS_ONLY=false
SHOW_GUIDE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --nuscenes)
            SETUP_NUSCENES=true
            SHOW_GUIDE=false
            shift
            ;;
        --uavscenes)
            SETUP_UAVSCENES=true
            SHOW_GUIDE=false
            shift
            ;;
        --plateau)
            SETUP_PLATEAU=true
            SHOW_GUIDE=false
            shift
            ;;
        --all)
            SETUP_NUSCENES=true
            SETUP_UAVSCENES=true
            SETUP_PLATEAU=true
            SHOW_GUIDE=false
            shift
            ;;
        --deps-only)
            DEPS_ONLY=true
            SHOW_GUIDE=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --nuscenes    Setup nuScenes dataset"
            echo "  --uavscenes   Setup UAVScenes dataset"
            echo "  --plateau     Setup PLATEAU/Gazebo data"
            echo "  --all         Setup all datasets"
            echo "  --deps-only   Install Python dependencies only"
            echo "  -h, --help    Show this help"
            echo ""
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           6DoF World Model Training Setup                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
# Install Dependencies
# =============================================================================

log_header "Installing Python Dependencies"

log_info "Installing core dependencies..."
pip install -q torch torchvision --upgrade 2>/dev/null || true

log_info "Installing dataset dependencies..."
pip install -q \
    nuscenes-devkit \
    pyquaternion \
    open3d \
    huggingface_hub \
    scipy \
    numpy \
    tqdm \
    wandb \
    2>/dev/null || {
    log_warn "Some packages may need manual installation"
}

log_success "Dependencies installed"

if [ "$DEPS_ONLY" = true ]; then
    echo ""
    log_success "Dependencies installed. Run without --deps-only to setup datasets."
    exit 0
fi

# =============================================================================
# Dataset Setup
# =============================================================================

if [ "$SETUP_NUSCENES" = true ]; then
    log_header "Setting up nuScenes"
    bash "$SCRIPT_DIR/setup_nuscenes.sh" --mini || log_warn "nuScenes setup needs attention"
fi

if [ "$SETUP_UAVSCENES" = true ]; then
    log_header "Setting up UAVScenes"
    bash "$SCRIPT_DIR/setup_uavscenes.sh" --scene AMtown || log_warn "UAVScenes setup needs attention"
fi

if [ "$SETUP_PLATEAU" = true ]; then
    log_header "Setting up PLATEAU"
    if [ -f "$SCRIPT_DIR/setup_open3d_pipeline.py" ]; then
        python3 "$SCRIPT_DIR/setup_open3d_pipeline.py" --download --dataset shibuya --output "$PROJECT_ROOT/data/plateau" || {
            log_warn "PLATEAU setup needs attention"
        }
    else
        log_warn "PLATEAU setup script not found"
    fi
fi

# =============================================================================
# Show Guide
# =============================================================================

if [ "$SHOW_GUIDE" = true ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║           6DoF Training Strategy Guide                           ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "┌─────────────────────────────────────────────────────────────────┐"
    echo "│  RECOMMENDED TRAINING PIPELINE FOR RESEARCH PAPER               │"
    echo "└─────────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  Step 1: Pre-train on nuScenes + 6DoF Augmentation"
    echo "  ─────────────────────────────────────────────────"
    echo "  • Large dataset, well-validated ground truth"
    echo "  • Pitch/roll augmentation simulates aerial viewpoints"
    echo "  • Establishes strong occupancy prediction baseline"
    echo ""
    echo "  Command:"
    echo "    python train.py --config config/finetune_nuscenes_6dof.py \\"
    echo "                    --model-type 6dof --wandb"
    echo ""
    echo "  Step 2: Fine-tune on UAVScenes (Real Aerial)"
    echo "  ─────────────────────────────────────────────"
    echo "  • Domain adaptation to real UAV sensor data"
    echo "  • True 6DoF motion from aerial platform"
    echo "  • Validates cross-domain transfer"
    echo ""
    echo "  Command:"
    echo "    python train.py --config config/finetune_uavscenes.py \\"
    echo "                    --model-type 6dof --resume-from out/nuscenes_6dof/best.pth"
    echo ""
    echo "  Step 3: Evaluate on UAVScenes Test Set"
    echo "  ─────────────────────────────────────────────"
    echo "  • Report aerial world model metrics"
    echo "  • Compare with/without augmentation (ablation)"
    echo ""
    echo "┌─────────────────────────────────────────────────────────────────┐"
    echo "│  DATASET COMPARISON                                             │"
    echo "└─────────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  ┌──────────────┬────────────────────┬────────────────────┐"
    echo "  │ Dataset      │ Platform           │ 6DoF Source        │"
    echo "  ├──────────────┼────────────────────┼────────────────────┤"
    echo "  │ nuScenes     │ Ground vehicle     │ Augmented          │"
    echo "  │ UAVScenes    │ Real UAV           │ Ground truth       │"
    echo "  │ PLATEAU      │ Synthetic          │ Generated          │"
    echo "  └──────────────┴────────────────────┴────────────────────┘"
    echo ""
    echo "┌─────────────────────────────────────────────────────────────────┐"
    echo "│  QUICK START COMMANDS                                           │"
    echo "└─────────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  # Setup nuScenes (mini for testing)"
    echo "  ./scripts/setup_nuscenes.sh --mini"
    echo ""
    echo "  # Setup UAVScenes (AMtown scene)"
    echo "  ./scripts/setup_uavscenes.sh --scene AMtown"
    echo ""
    echo "  # Install all dependencies"
    echo "  ./scripts/setup_6dof_training.sh --deps-only"
    echo ""
    echo "  # Setup everything"
    echo "  ./scripts/setup_6dof_training.sh --all"
    echo ""
fi

# =============================================================================
# Verify Installation
# =============================================================================

log_header "Verification"

echo "Checking dataset availability..."
echo ""

python3 << 'EOF'
import sys
sys.path.insert(0, '.')

datasets = []

# Check nuScenes
try:
    from dataset.nuscenes_6dof_dataset import NuScenes6DoFDataset
    datasets.append(("nuScenes 6DoF", "✓", "config/finetune_nuscenes_6dof.py"))
except ImportError as e:
    datasets.append(("nuScenes 6DoF", "✗", str(e)[:40]))

# Check UAVScenes
try:
    from dataset.uavscenes_dataset import UAVScenesDataset
    datasets.append(("UAVScenes", "✓", "config/finetune_uavscenes.py"))
except ImportError as e:
    datasets.append(("UAVScenes", "✗", str(e)[:40]))

# Check Gazebo
try:
    from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset
    datasets.append(("Gazebo/PLATEAU", "✓", "config/finetune_6dof.py"))
except ImportError as e:
    datasets.append(("Gazebo/PLATEAU", "✗", str(e)[:40]))

# Print table
print("  Dataset Loaders:")
for name, status, config in datasets:
    print(f"    {status} {name:<20} → {config}")

print()
EOF

log_success "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Download a dataset using the setup scripts"
echo "  2. Run training with the appropriate config"
echo "  3. Monitor with wandb (--wandb flag)"
echo ""
