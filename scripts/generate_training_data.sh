#!/bin/bash
# =============================================================================
# Generate Training Data for OccWorld
# =============================================================================
# Generates ~21,000 frames with mixed trajectory patterns for training.
#
# Usage:
#   ./scripts/generate_training_data.sh                    # Default: 21,000 frames
#   ./scripts/generate_training_data.sh --frames 10000     # Custom frame count
#   ./scripts/generate_training_data.sh --output /path     # Custom output dir
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Defaults
TOTAL_FRAMES=21000
OUTPUT_DIR="data/tokyo_gazebo"
FRAMES_PER_SESSION=300
NUM_WORKERS=""  # Empty = auto-detect CPU count

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --frames|-f)    TOTAL_FRAMES="$2"; shift 2 ;;
        --output|-o)    OUTPUT_DIR="$2"; shift 2 ;;
        --workers|-w)   NUM_WORKERS="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "  --frames, -f   Total frames to generate (default: 21000)"
            echo "  --output, -o   Output directory (default: data/tokyo_gazebo)"
            echo "  --workers, -w  Parallel workers (default: CPU count)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Build workers arg
WORKERS_ARG=""
if [ -n "$NUM_WORKERS" ]; then
    WORKERS_ARG="--workers $NUM_WORKERS"
fi

# Calculate sessions per pattern (60% random, 20% survey, 20% orbit)
TOTAL_SESSIONS=$((TOTAL_FRAMES / FRAMES_PER_SESSION))
RANDOM_SESSIONS=$((TOTAL_SESSIONS * 60 / 100))
SURVEY_SESSIONS=$((TOTAL_SESSIONS * 20 / 100))
ORBIT_SESSIONS=$((TOTAL_SESSIONS - RANDOM_SESSIONS - SURVEY_SESSIONS))

echo ""
echo "=============================================="
echo "  OccWorld Training Data Generator"
echo "=============================================="
echo ""
log_info "Target frames: $TOTAL_FRAMES"
log_info "Output: $OUTPUT_DIR"
log_info "Frames per session: $FRAMES_PER_SESSION"
echo ""
log_info "Pattern distribution:"
echo "  - Random: $RANDOM_SESSIONS sessions ($((RANDOM_SESSIONS * FRAMES_PER_SESSION)) frames)"
echo "  - Survey: $SURVEY_SESSIONS sessions ($((SURVEY_SESSIONS * FRAMES_PER_SESSION)) frames)"
echo "  - Orbit:  $ORBIT_SESSIONS sessions ($((ORBIT_SESSIONS * FRAMES_PER_SESSION)) frames)"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate random pattern sessions
if [ $RANDOM_SESSIONS -gt 0 ]; then
    log_info "Generating $RANDOM_SESSIONS random pattern sessions..."
    python scripts/gazebo_data_collector.py \
        --output "$OUTPUT_DIR" \
        --frames $FRAMES_PER_SESSION \
        --sessions $RANDOM_SESSIONS \
        --pattern random \
        $WORKERS_ARG
fi

# Generate survey pattern sessions
if [ $SURVEY_SESSIONS -gt 0 ]; then
    log_info "Generating $SURVEY_SESSIONS survey pattern sessions..."
    python scripts/gazebo_data_collector.py \
        --output "$OUTPUT_DIR" \
        --frames $FRAMES_PER_SESSION \
        --sessions $SURVEY_SESSIONS \
        --pattern survey \
        $WORKERS_ARG
fi

# Generate orbit pattern sessions
if [ $ORBIT_SESSIONS -gt 0 ]; then
    log_info "Generating $ORBIT_SESSIONS orbit pattern sessions..."
    python scripts/gazebo_data_collector.py \
        --output "$OUTPUT_DIR" \
        --frames $FRAMES_PER_SESSION \
        --sessions $ORBIT_SESSIONS \
        --pattern orbit \
        $WORKERS_ARG
fi

# Count actual frames generated
ACTUAL_FRAMES=$(find "$OUTPUT_DIR" -name "*_occupancy.npz" 2>/dev/null | wc -l)

echo ""
echo "=============================================="
log_success "Data generation complete!"
echo "=============================================="
echo ""
echo "  Output directory: $OUTPUT_DIR"
echo "  Total frames: $ACTUAL_FRAMES"
echo "  Sessions: $TOTAL_SESSIONS"
echo ""
echo "Next step - start training:"
echo "  python train.py --config config/finetune_tokyo.py --wandb --from-scratch"
echo ""
