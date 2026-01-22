#!/bin/bash
# =============================================================================
# Optimized Training Script for VeryLargeWeebModel
# =============================================================================
# GPU auto-detection, multi-GPU support, and cloud-aware defaults.
#
# Usage:
#   ./scripts/train_optimized.sh [options]
#
# Options:
#   --config FILE       Config file (default: config/finetune_tokyo.py)
#   --work-dir DIR      Output directory (auto-detected: /workspace/checkpoints)
#   --epochs N          Number of epochs (default: 50)
#   --batch-size N      Override batch size (auto-detected based on GPU)
#   --gpus N            Number of GPUs to use (default: all available)
#   --resume PATH       Resume from specific checkpoint
#   --auto-resume       Auto-resume from latest checkpoint in work-dir
#   --install-deps      Install optimized dependencies (flash-attn, xformers)
#   --dry-run           Show config without training
#   --help              Show this help
#
# Examples:
#   # Basic training (auto-detects everything)
#   ./scripts/train_optimized.sh
#
#   # Training with specific batch size
#   ./scripts/train_optimized.sh --batch-size 4
#
#   # Resume from latest checkpoint
#   ./scripts/train_optimized.sh --auto-resume
#
#   # Install optimizations and train
#   ./scripts/train_optimized.sh --install-deps --epochs 100
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
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# Auto-detect work directory based on environment
if [ -d "/workspace" ]; then
    DEFAULT_WORK_DIR="/workspace/checkpoints"
elif [ -d "/home/ubuntu" ]; then
    DEFAULT_WORK_DIR="/home/ubuntu/checkpoints"
else
    DEFAULT_WORK_DIR="${HOME}/checkpoints"
fi

# Defaults
CONFIG="config/finetune_tokyo.py"
WORK_DIR="${DEFAULT_WORK_DIR}"
EPOCHS=""
BATCH_SIZE=""
NUM_GPUS=""
RESUME=""
INSTALL_DEPS=false
DRY_RUN=false
AUTO_RESUME=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)       CONFIG="$2"; shift 2 ;;
        --work-dir)     WORK_DIR="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        --gpus)         NUM_GPUS="$2"; shift 2 ;;
        --resume)       RESUME="$2"; shift 2 ;;
        --auto-resume)  AUTO_RESUME=true; shift ;;
        --install-deps) INSTALL_DEPS=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --help|-h)      head -20 "$0" | tail -15; exit 0 ;;
        *)              log_warn "Unknown option: $1"; shift ;;
    esac
done

# Auto-resume: find latest checkpoint
if [ "$AUTO_RESUME" = true ] && [ -z "$RESUME" ]; then
    LATEST_CKPT=$(ls -t ${WORK_DIR}/*.pth 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        RESUME="$LATEST_CKPT"
        log_info "Auto-resume from: $RESUME"
    fi
fi

echo "=============================================="
echo "  AerialWorld Optimized Training"
echo "=============================================="
echo ""

# =============================================================================
# GPU Detection
# =============================================================================
detect_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Is CUDA installed?"
        exit 1
    fi

    # Get GPU info
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

    log_info "Detected GPU(s):"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read line; do
        echo "  $line"
    done
    echo ""

    # Determine optimal batch size based on GPU memory
    if [ -z "$BATCH_SIZE" ]; then
        if [ "$GPU_MEMORY" -ge 70000 ]; then
            # A100-80GB, H100-80GB
            BATCH_SIZE=12
            GPU_TIER="high"
        elif [ "$GPU_MEMORY" -ge 35000 ]; then
            # A100-40GB, A6000
            BATCH_SIZE=6
            GPU_TIER="mid-high"
        elif [ "$GPU_MEMORY" -ge 20000 ]; then
            # RTX 3090, RTX 4090, A5000
            BATCH_SIZE=4
            GPU_TIER="mid"
        elif [ "$GPU_MEMORY" -ge 10000 ]; then
            # RTX 3080, RTX 4080
            BATCH_SIZE=2
            GPU_TIER="low"
        else
            BATCH_SIZE=1
            GPU_TIER="minimal"
        fi
        log_info "Auto-detected batch size: $BATCH_SIZE (${GPU_MEMORY}MB VRAM)"
    fi

    # Determine precision based on GPU architecture
    if echo "$GPU_NAME" | grep -qiE "A100|H100|H200|A6000|RTX 40|RTX 30"; then
        PRECISION="bf16"
        log_info "Using BF16 precision (native support)"
    else
        PRECISION="fp16"
        log_info "Using FP16 precision"
    fi

    # Set number of GPUs
    if [ -z "$NUM_GPUS" ]; then
        NUM_GPUS=$GPU_COUNT
    fi
    log_info "Using $NUM_GPUS GPU(s)"
}

# =============================================================================
# Install Optimized Dependencies
# =============================================================================
install_optimized_deps() {
    log_info "Installing optimized dependencies..."

    # Flash Attention 2 (huge speedup for transformers)
    if ! python3 -c "import flash_attn" 2>/dev/null; then
        log_info "Installing Flash Attention 2..."
        pip install flash-attn --no-build-isolation 2>/dev/null || {
            log_warn "Flash Attention install failed (requires CUDA 11.6+)"
            log_info "Training will work but slower without Flash Attention"
        }
    else
        log_success "Flash Attention already installed"
    fi

    # xFormers (alternative efficient attention)
    if ! python3 -c "import xformers" 2>/dev/null; then
        log_info "Installing xFormers..."
        pip install xformers 2>/dev/null || log_warn "xFormers install failed"
    else
        log_success "xFormers already installed"
    fi

    # Apex for fused optimizers
    if ! python3 -c "import apex" 2>/dev/null; then
        log_info "Installing NVIDIA Apex..."
        pip install nvidia-apex 2>/dev/null || log_warn "Apex install failed (optional)"
    fi

    # DeepSpeed for multi-GPU efficiency
    if ! python3 -c "import deepspeed" 2>/dev/null; then
        log_info "Installing DeepSpeed..."
        pip install deepspeed 2>/dev/null || log_warn "DeepSpeed install failed (optional)"
    fi

    log_success "Optimized dependencies installed"
}

# =============================================================================
# Generate Optimized Config
# =============================================================================
generate_runtime_config() {
    RUNTIME_CONFIG="/tmp/runtime_config.py"

    cat > "$RUNTIME_CONFIG" << EOF
# Auto-generated runtime config for GPU: $GPU_NAME
# Generated by train_optimized.sh

# Import base config
_base_ = ['../$CONFIG']

# Override with optimized settings
data = dict(
    samples_per_gpu=$BATCH_SIZE,
    workers_per_gpu=$BATCH_SIZE,
)

# Precision settings
EOF

    if [ "$PRECISION" = "bf16" ]; then
        cat >> "$RUNTIME_CONFIG" << EOF
bf16 = dict(enabled=True)
fp16 = dict(enabled=False)
EOF
    else
        cat >> "$RUNTIME_CONFIG" << EOF
bf16 = dict(enabled=False)
fp16 = dict(loss_scale='dynamic', enabled=True)
EOF
    fi

    # Add epochs override if specified
    if [ -n "$EPOCHS" ]; then
        cat >> "$RUNTIME_CONFIG" << EOF

max_epochs = $EPOCHS
runner = dict(type='EpochBasedRunner', max_epochs=$EPOCHS)
EOF
    fi

    echo ""
    log_info "Runtime config generated:"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Precision: $PRECISION"
    echo "  GPUs: $NUM_GPUS"
    [ -n "$EPOCHS" ] && echo "  Epochs: $EPOCHS"
    echo ""
}

# =============================================================================
# Training Time Estimation
# =============================================================================
estimate_training_time() {
    # Rough estimates based on benchmarks
    # Base: ~26 min/epoch with BS=1 on single GPU

    local base_time_per_epoch=26  # minutes
    local speedup_from_batch=$((BATCH_SIZE > 1 ? BATCH_SIZE / 2 : 1))
    local speedup_from_gpus=$NUM_GPUS
    local total_speedup=$((speedup_from_batch * speedup_from_gpus))

    local effective_epochs=${EPOCHS:-50}
    local time_per_epoch=$((base_time_per_epoch / total_speedup))
    local total_minutes=$((time_per_epoch * effective_epochs))
    local total_hours=$((total_minutes / 60))
    local remaining_minutes=$((total_minutes % 60))

    echo "=============================================="
    echo "  Estimated Training Time"
    echo "=============================================="
    echo ""
    echo "  Epochs: $effective_epochs"
    echo "  Time per epoch: ~${time_per_epoch} min"
    echo "  Total: ~${total_hours}h ${remaining_minutes}m"
    echo ""
    echo "  Speedup breakdown:"
    echo "    - Batch size ($BATCH_SIZE): ~${speedup_from_batch}x"
    echo "    - Multi-GPU ($NUM_GPUS): ~${speedup_from_gpus}x"
    echo "    - Total speedup: ~${total_speedup}x"
    echo ""
}

# =============================================================================
# Run Training
# =============================================================================
run_training() {
    mkdir -p "$WORK_DIR"

    local TRAIN_CMD=""

    if [ "$NUM_GPUS" -gt 1 ]; then
        # Multi-GPU training with torchrun
        TRAIN_CMD="torchrun --nproc_per_node=$NUM_GPUS train.py"
    else
        # Single GPU training
        TRAIN_CMD="python train.py"
    fi

    # Build full command
    TRAIN_CMD="$TRAIN_CMD --py-config $CONFIG --work-dir $WORK_DIR"

    # Add resume if specified
    if [ -n "$RESUME" ]; then
        TRAIN_CMD="$TRAIN_CMD --resume $RESUME"
    fi

    echo "=============================================="
    echo "  Training Command"
    echo "=============================================="
    echo ""
    echo "  $TRAIN_CMD"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        log_warn "Dry run - not starting training"
        exit 0
    fi

    # Set optimal CUDA settings
    export CUDA_LAUNCH_BLOCKING=0
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

    # Enable TF32 for A100/H100
    export NVIDIA_TF32_OVERRIDE=1

    log_info "Starting training..."
    echo ""

    eval $TRAIN_CMD
}

# =============================================================================
# Main
# =============================================================================

# Detect GPU and set optimal parameters
detect_gpu

# Install dependencies if requested
if [ "$INSTALL_DEPS" = true ]; then
    install_optimized_deps
fi

# Generate runtime config
generate_runtime_config

# Estimate training time
estimate_training_time

# Run training
run_training
