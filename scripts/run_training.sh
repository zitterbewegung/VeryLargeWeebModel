#!/bin/bash
# =============================================================================
# OccWorld Training Runner
# =============================================================================
# Automates the full training workflow:
#   - Downloads pretrained models (if needed)
#   - Generates training data (if needed)
#   - Runs training with monitoring
#   - Handles resume on crash
#   - Runs evaluation after training
#
# Usage:
#   ./scripts/run_training.sh                    # Full training
#   ./scripts/run_training.sh --quick            # Quick test (2 epochs)
#   ./scripts/run_training.sh --resume           # Resume interrupted training
#   ./scripts/run_training.sh --eval             # Evaluation only
#   ./scripts/run_training.sh --tensorboard      # Start tensorboard
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
CONFIG="config/finetune_tokyo.py"
WORK_DIR="${PROJECT_ROOT}/checkpoints"
EPOCHS=50
BATCH_SIZE=""
LEARNING_RATE=""
QUICK_MODE=false
RESUME=false
EVAL_ONLY=false
TENSORBOARD_ONLY=false
GENERATE_DATA=false
DATA_FRAMES=200
DATA_SESSIONS=5

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)       CONFIG="$2"; shift 2 ;;
        --work-dir)     WORK_DIR="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        --lr)           LEARNING_RATE="$2"; shift 2 ;;
        --quick)        QUICK_MODE=true; EPOCHS=2; shift ;;
        --resume)       RESUME=true; shift ;;
        --eval)         EVAL_ONLY=true; shift ;;
        --tensorboard)  TENSORBOARD_ONLY=true; shift ;;
        --generate-data) GENERATE_DATA=true; shift ;;
        --data-frames)  DATA_FRAMES="$2"; shift 2 ;;
        --data-sessions) DATA_SESSIONS="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE       Config file (default: config/finetune_tokyo.py)"
            echo "  --work-dir DIR      Output directory (default: ./checkpoints)"
            echo "  --epochs N          Number of epochs (default: 50)"
            echo "  --batch-size N      Override batch size"
            echo "  --lr RATE           Override learning rate"
            echo "  --quick             Quick test mode (2 epochs)"
            echo "  --resume            Resume from last checkpoint"
            echo "  --eval              Evaluation only"
            echo "  --tensorboard       Start tensorboard only"
            echo "  --generate-data     Generate training data before training"
            echo "  --data-frames N     Frames per session (default: 200)"
            echo "  --data-sessions N   Number of sessions (default: 5)"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Functions
# =============================================================================

check_gpu() {
    log_info "Checking GPU..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
        GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        log_info "Free GPU memory: ${GPU_MEM}MB"

        if [ "$GPU_MEM" -lt 8000 ]; then
            log_warn "Low GPU memory. Consider reducing batch size."
        fi
    else
        log_warn "nvidia-smi not found. Running on CPU?"
    fi
}

check_pretrained() {
    log_info "Checking pretrained models..."

    local CHECKPOINT="${PROJECT_ROOT}/pretrained/occworld/latest.pth"

    if [ -f "$CHECKPOINT" ]; then
        local size=$(stat -f%z "$CHECKPOINT" 2>/dev/null || stat -c%s "$CHECKPOINT" 2>/dev/null || echo 0)
        local size_mb=$((size / 1024 / 1024))

        if [ "$size" -gt 100000000 ]; then
            log_success "OccWorld checkpoint found (${size_mb}MB)"
            return 0
        fi
    fi

    log_warn "Pretrained model not found. Downloading..."
    mkdir -p "${PROJECT_ROOT}/pretrained/occworld"

    curl -L --progress-bar -o "$CHECKPOINT" \
        "https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/files/?p=/latest.pth&dl=1"

    if [ -f "$CHECKPOINT" ]; then
        log_success "Downloaded pretrained model"
    else
        log_error "Failed to download. Training will start from scratch."
    fi
}

check_data() {
    log_info "Checking training data..."

    local DATA_DIR="${PROJECT_ROOT}/data/tokyo_gazebo"
    local num_sessions=$(ls -d ${DATA_DIR}/drone_* ${DATA_DIR}/rover_* 2>/dev/null | wc -l | tr -d ' ')

    if [ "$num_sessions" -gt 0 ]; then
        log_success "Found $num_sessions training sessions"

        # Count frames
        local total_frames=0
        for session in ${DATA_DIR}/drone_* ${DATA_DIR}/rover_*; do
            if [ -d "$session/occupancy" ]; then
                local frames=$(ls "$session/occupancy" 2>/dev/null | wc -l)
                total_frames=$((total_frames + frames))
            fi
        done
        log_info "Total frames: $total_frames"
        return 0
    else
        log_warn "No training data found"
        return 1
    fi
}

generate_data() {
    log_info "Generating training data..."

    local CONVERTER="${SCRIPT_DIR}/plateau_to_occworld.py"
    local DATA_DIR="${PROJECT_ROOT}/data/tokyo_gazebo"
    local MESH_DIR="${PROJECT_ROOT}/data/plateau/meshes/obj"

    # Try PLATEAU converter first, fall back to dummy data
    if [ -f "$CONVERTER" ]; then
        python3 "$CONVERTER" \
            --input "$MESH_DIR" \
            --output "$DATA_DIR" \
            --frames "$DATA_FRAMES" \
            --sessions "$DATA_SESSIONS" \
            --pattern survey
    else
        log_warn "PLATEAU converter not found, using dummy data"
        python3 "${SCRIPT_DIR}/create_dummy_data.py" \
            --output "$DATA_DIR" \
            --frames "$DATA_FRAMES" \
            --sessions "$DATA_SESSIONS"
    fi

    log_success "Data generation complete"
}

start_tensorboard() {
    log_info "Starting TensorBoard..."

    local TB_PORT=6007
    local TB_DIR="$WORK_DIR"

    # Kill existing tensorboard
    pkill -f "tensorboard.*${TB_DIR}" 2>/dev/null || true

    # Start tensorboard in background
    tensorboard --logdir "$TB_DIR" --port $TB_PORT --bind_all &
    TB_PID=$!

    sleep 2

    if kill -0 $TB_PID 2>/dev/null; then
        log_success "TensorBoard running at http://localhost:${TB_PORT}"
        echo $TB_PID > "${WORK_DIR}/.tensorboard.pid"
    else
        log_warn "TensorBoard failed to start"
    fi
}

run_training() {
    log_info "Starting training..."

    mkdir -p "$WORK_DIR"

    # Build command
    local CMD="python3 ${PROJECT_ROOT}/train.py"
    CMD="$CMD --py-config ${PROJECT_ROOT}/${CONFIG}"
    CMD="$CMD --work-dir $WORK_DIR"
    CMD="$CMD --epochs $EPOCHS"

    if [ -n "$BATCH_SIZE" ]; then
        CMD="$CMD --batch-size $BATCH_SIZE"
    fi

    if [ -n "$LEARNING_RATE" ]; then
        CMD="$CMD --lr $LEARNING_RATE"
    fi

    if [ "$RESUME" = true ]; then
        CMD="$CMD --resume"
    fi

    echo ""
    log_info "Command: $CMD"
    echo ""

    # Run with logging
    local LOG_FILE="${WORK_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

    $CMD 2>&1 | tee "$LOG_FILE"

    local EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        log_success "Training completed!"
    else
        log_error "Training failed with exit code $EXIT_CODE"
        log_info "Log file: $LOG_FILE"
        log_info "To resume: $0 --resume --work-dir $WORK_DIR"
        return $EXIT_CODE
    fi
}

run_evaluation() {
    log_info "Running evaluation..."

    python3 "${PROJECT_ROOT}/train.py" \
        --py-config "${PROJECT_ROOT}/${CONFIG}" \
        --work-dir "$WORK_DIR" \
        --eval-only

    log_success "Evaluation complete"
}

print_summary() {
    echo ""
    echo "=============================================================================="
    echo "                         Training Summary                                     "
    echo "=============================================================================="
    echo ""
    echo "Work directory: $WORK_DIR"
    echo ""

    # List checkpoints
    if [ -d "$WORK_DIR" ]; then
        echo "Checkpoints:"
        ls -lh "$WORK_DIR"/*.pth 2>/dev/null | while read line; do
            echo "  $line"
        done
        echo ""
    fi

    # Show best metrics if available
    if [ -f "${WORK_DIR}/best_metrics.json" ]; then
        echo "Best metrics:"
        cat "${WORK_DIR}/best_metrics.json"
        echo ""
    fi

    echo "=============================================================================="
    echo ""
    echo "Next steps:"
    echo "  - View tensorboard: tensorboard --logdir $WORK_DIR"
    echo "  - Run evaluation:   $0 --eval --work-dir $WORK_DIR"
    echo "  - Copy model:       scp $WORK_DIR/latest.pth local:trained_models/"
    echo ""
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo ""
    echo "=============================================================================="
    echo "                    OccWorld Training Runner                                  "
    echo "=============================================================================="
    echo ""

    cd "$PROJECT_ROOT"

    # Tensorboard only mode
    if [ "$TENSORBOARD_ONLY" = true ]; then
        start_tensorboard
        log_info "Press Ctrl+C to stop"
        wait
        exit 0
    fi

    # Eval only mode
    if [ "$EVAL_ONLY" = true ]; then
        run_evaluation
        exit 0
    fi

    # Check environment
    check_gpu

    # Check/download pretrained model
    check_pretrained

    # Check/generate data
    if ! check_data || [ "$GENERATE_DATA" = true ]; then
        generate_data
    fi

    # Quick mode info
    if [ "$QUICK_MODE" = true ]; then
        log_info "Quick mode: Running $EPOCHS epochs for pipeline validation"
    fi

    # Start tensorboard in background
    if command -v tensorboard &> /dev/null; then
        start_tensorboard
    fi

    # Run training
    run_training

    # Print summary
    print_summary
}

# Run
main "$@"
