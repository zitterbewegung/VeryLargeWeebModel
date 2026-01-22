#!/bin/bash
# =============================================================================
# Deploy VeryLargeWeebModel to Vast.ai
# =============================================================================
# Automates deployment from LOCAL machine to a Vast.ai GPU instance.
#
# Features:
#   - Finds and connects to Vast.ai instances via vastai CLI or SSH
#   - Uploads project files via rsync with smart exclusions
#   - Optionally uploads pretrained models and training data
#   - Runs setup script on remote instance
#   - Starts training in screen session
#   - Supports instance management (list, start, stop)
#
# Usage:
#   # Deploy to instance by IP (SSH):
#   ./scripts/vastai_deploy.sh <IP:PORT> [OPTIONS]
#
#   # Deploy using vastai CLI (recommended):
#   ./scripts/vastai_deploy.sh --instance <INSTANCE_ID> [OPTIONS]
#
#   # List available instances:
#   ./scripts/vastai_deploy.sh --list
#
# Options:
#   --instance ID       Deploy to Vast.ai instance ID (uses vastai CLI)
#   --ip IP:PORT        SSH connection string (e.g., root@192.0.2.1:22)
#   --key PATH          SSH key path (default: ~/.ssh/id_rsa)
#   --upload-data       Also upload local training data (~5GB)
#   --upload-models     Also upload pretrained models (~721MB)
#   --upload-all        Upload everything (data + models)
#   --start-train       Start training after deployment
#   --setup-only        Only run setup, don't start training
#   --list              List your Vast.ai instances
#   --sync-only         Only sync files, skip setup
#   --dry-run           Show what would be uploaded without doing it
#   --help              Show this help
#
# Prerequisites:
#   - vastai CLI installed (pip install vastai) OR SSH access
#   - SSH key registered with Vast.ai
#   - Running Vast.ai instance
#
# Examples:
#   # Deploy using vastai CLI
#   ./scripts/vastai_deploy.sh --instance 12345 --upload-models --start-train
#
#   # Deploy via SSH
#   ./scripts/vastai_deploy.sh root@192.168.1.100:22222 --key ~/.ssh/vastai_key
#
#   # Upload everything and start training
#   ./scripts/vastai_deploy.sh --instance 12345 --upload-all --start-train
#
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

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
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
log_step()    {
    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Default options
SSH_KEY="${HOME}/.ssh/id_rsa"
INSTANCE_ID=""
SSH_TARGET=""
UPLOAD_DATA=false
UPLOAD_MODELS=false
START_TRAINING=false
SETUP_ONLY=false
SYNC_ONLY=false
DRY_RUN=false
LIST_INSTANCES=false

# Remote paths
REMOTE_USER="root"
REMOTE_WORK_DIR="/workspace"
REMOTE_PROJECT="${REMOTE_WORK_DIR}/VeryLargeWeebModel"
REMOTE_CHECKPOINTS="${REMOTE_WORK_DIR}/checkpoints"

# =============================================================================
# Banner
# =============================================================================
show_banner() {
    echo ""
    echo -e "${CYAN}${BOLD}"
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                            ║"
    echo "║   ██╗   ██╗██╗    ██╗    ██╗███╗   ███╗  VAST.AI DEPLOY                   ║"
    echo "║   ██║   ██║██║    ██║    ██║████╗ ████║  ═══════════════                   ║"
    echo "║   ██║   ██║██║    ██║ █╗ ██║██╔████╔██║  Local → Remote                   ║"
    echo "║   ╚██╗ ██╔╝██║    ██║███╗██║██║╚██╔╝██║  Deployment Tool                  ║"
    echo "║    ╚████╔╝ ███████╗╚███╔███╔╝██║ ╚═╝ ██║                                   ║"
    echo "║     ╚═══╝  ╚══════╝ ╚══╝╚══╝ ╚═╝     ╚═╝                                   ║"
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
            --instance)     INSTANCE_ID="$2"; shift 2 ;;
            --ip)           SSH_TARGET="$2"; shift 2 ;;
            --key)          SSH_KEY="$2"; shift 2 ;;
            --upload-data)  UPLOAD_DATA=true; shift ;;
            --upload-models) UPLOAD_MODELS=true; shift ;;
            --upload-all)   UPLOAD_DATA=true; UPLOAD_MODELS=true; shift ;;
            --start-train)  START_TRAINING=true; shift ;;
            --setup-only)   SETUP_ONLY=true; shift ;;
            --sync-only)    SYNC_ONLY=true; shift ;;
            --dry-run)      DRY_RUN=true; shift ;;
            --list)         LIST_INSTANCES=true; shift ;;
            --help|-h)      head -55 "$0" | tail -50; exit 0 ;;
            -*)             log_error "Unknown option: $1" ;;
            *)
                # Assume it's an IP:PORT if not an option
                if [ -z "$SSH_TARGET" ]; then
                    SSH_TARGET="$1"
                fi
                shift
                ;;
        esac
    done
}

# =============================================================================
# Check Dependencies
# =============================================================================
check_dependencies() {
    log_step "Checking Dependencies"

    # Check SSH key
    if [ ! -f "$SSH_KEY" ]; then
        log_error "SSH key not found: $SSH_KEY"
    fi
    log_success "SSH key: $SSH_KEY"

    # Check rsync
    if ! command -v rsync &> /dev/null; then
        log_error "rsync not found. Install with: brew install rsync (macOS) or apt install rsync"
    fi
    log_success "rsync: installed"

    # Check vastai CLI (optional)
    if command -v vastai &> /dev/null; then
        VASTAI_CLI=true
        log_success "vastai CLI: installed"
    else
        VASTAI_CLI=false
        log_warn "vastai CLI not installed (pip install vastai)"
        log_info "Using SSH directly"
    fi

    # Check project exists
    if [ ! -d "$PROJECT_DIR" ]; then
        log_error "Project not found: $PROJECT_DIR"
    fi
    log_success "Project: $PROJECT_DIR"
}

# =============================================================================
# List Vast.ai Instances
# =============================================================================
list_instances() {
    log_step "Your Vast.ai Instances"

    if [ "$VASTAI_CLI" != true ]; then
        log_error "vastai CLI required for listing instances"
    fi

    echo ""
    vastai show instances --raw 2>/dev/null | \
        python3 -c "
import json, sys
data = json.load(sys.stdin)
if not data:
    print('  No instances found.')
    sys.exit(0)

print(f'  {'ID':<10} {'Status':<12} {'GPU':<20} {'$/hr':<8} {'SSH Command'}')
print(f'  {'-'*10} {'-'*12} {'-'*20} {'-'*8} {'-'*40}')

for inst in data:
    iid = inst.get('id', 'N/A')
    status = inst.get('actual_status', 'unknown')
    gpu = inst.get('gpu_name', 'N/A')[:18]
    price = inst.get('dph_total', 0)
    ssh_host = inst.get('ssh_host', '')
    ssh_port = inst.get('ssh_port', '')
    ssh_cmd = f'ssh -p {ssh_port} root@{ssh_host}' if ssh_host else 'N/A'
    print(f'  {iid:<10} {status:<12} {gpu:<20} \${price:<7.3f} {ssh_cmd}')
" 2>/dev/null || {
        log_warn "Could not parse instance list"
        vastai show instances
    }

    echo ""
    log_info "Deploy with: ./scripts/vastai_deploy.sh --instance <ID>"
}

# =============================================================================
# Get SSH Connection from Instance ID
# =============================================================================
get_ssh_from_instance() {
    if [ -z "$INSTANCE_ID" ]; then
        return 1
    fi

    log_info "Getting SSH info for instance $INSTANCE_ID..."

    if [ "$VASTAI_CLI" != true ]; then
        log_error "vastai CLI required for instance lookup"
    fi

    # Get instance SSH info
    SSH_INFO=$(vastai show instances --raw 2>/dev/null | \
        python3 -c "
import json, sys
data = json.load(sys.stdin)
for inst in data:
    if str(inst.get('id')) == '$INSTANCE_ID':
        host = inst.get('ssh_host', '')
        port = inst.get('ssh_port', '')
        status = inst.get('actual_status', 'unknown')
        if host and port:
            print(f'{host}:{port}:{status}')
            sys.exit(0)
sys.exit(1)
" 2>/dev/null) || log_error "Instance $INSTANCE_ID not found"

    SSH_HOST=$(echo "$SSH_INFO" | cut -d: -f1)
    SSH_PORT=$(echo "$SSH_INFO" | cut -d: -f2)
    INSTANCE_STATUS=$(echo "$SSH_INFO" | cut -d: -f3)

    if [ "$INSTANCE_STATUS" != "running" ]; then
        log_error "Instance $INSTANCE_ID is not running (status: $INSTANCE_STATUS)"
    fi

    SSH_TARGET="${REMOTE_USER}@${SSH_HOST}:${SSH_PORT}"
    log_success "Instance $INSTANCE_ID: $SSH_TARGET"
}

# =============================================================================
# Parse SSH Target
# =============================================================================
parse_ssh_target() {
    # Parse formats: user@host:port, host:port, user@host
    if [[ "$SSH_TARGET" == *"@"* ]]; then
        REMOTE_USER=$(echo "$SSH_TARGET" | cut -d@ -f1)
        HOST_PORT=$(echo "$SSH_TARGET" | cut -d@ -f2)
    else
        HOST_PORT="$SSH_TARGET"
    fi

    if [[ "$HOST_PORT" == *":"* ]]; then
        SSH_HOST=$(echo "$HOST_PORT" | cut -d: -f1)
        SSH_PORT=$(echo "$HOST_PORT" | cut -d: -f2)
    else
        SSH_HOST="$HOST_PORT"
        SSH_PORT="22"
    fi

    log_info "SSH connection: ${REMOTE_USER}@${SSH_HOST}:${SSH_PORT}"
}

# =============================================================================
# Test Connection
# =============================================================================
test_connection() {
    log_step "Testing SSH Connection"

    SSH_CMD="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT ${REMOTE_USER}@${SSH_HOST}"

    log_info "Connecting to ${REMOTE_USER}@${SSH_HOST}:${SSH_PORT}..."

    if ! $SSH_CMD "echo 'Connection successful'" 2>/dev/null; then
        log_error "Cannot connect to instance. Check:"
        echo "  - Instance is running"
        echo "  - SSH key is correct: $SSH_KEY"
        echo "  - Port forwarding in Vast.ai console"
    fi

    log_success "Connected to Vast.ai instance"

    # Get instance info
    log_info "Instance info:"
    $SSH_CMD "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null" || true
    echo ""
    $SSH_CMD "df -h /workspace 2>/dev/null | tail -1" || true
}

# =============================================================================
# Upload Project Files
# =============================================================================
upload_project() {
    log_step "Uploading Project Files"

    # Build rsync command
    RSYNC_OPTS="-avz --progress --delete"
    RSYNC_SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -p $SSH_PORT"

    # Exclusions
    EXCLUDES=(
        --exclude '.git'
        --exclude '__pycache__'
        --exclude '*.pyc'
        --exclude '.pytest_cache'
        --exclude '.mypy_cache'
        --exclude '*.egg-info'
        --exclude 'out/*'
        --exclude 'wandb/*'
        --exclude '.env'
        --exclude '*.log'
    )

    # Exclude data unless uploading
    if [ "$UPLOAD_DATA" = false ]; then
        EXCLUDES+=(
            --exclude 'data/plateau/raw/*'
            --exclude 'data/tokyo_gazebo/*'
            --exclude 'data/nuscenes/*'
            --exclude '*.zip'
            --exclude '*.tar.gz'
            --exclude '*.tgz'
        )
    fi

    # Exclude models unless uploading
    if [ "$UPLOAD_MODELS" = false ]; then
        EXCLUDES+=(
            --exclude 'pretrained/*.pth'
            --exclude 'pretrained/*/*.pth'
        )
    fi

    # Dry run mode
    if [ "$DRY_RUN" = true ]; then
        RSYNC_OPTS="$RSYNC_OPTS --dry-run"
        log_warn "DRY RUN - no files will be transferred"
    fi

    # Calculate local size
    LOCAL_SIZE=$(du -sh "$PROJECT_DIR" 2>/dev/null | cut -f1)
    log_info "Local project size: $LOCAL_SIZE"

    # Upload
    log_info "Uploading to ${REMOTE_USER}@${SSH_HOST}:${REMOTE_PROJECT}..."

    rsync $RSYNC_OPTS \
        -e "$RSYNC_SSH" \
        "${EXCLUDES[@]}" \
        "$PROJECT_DIR/" \
        "${REMOTE_USER}@${SSH_HOST}:${REMOTE_PROJECT}/"

    if [ "$DRY_RUN" = false ]; then
        log_success "Project files uploaded"
    fi
}

# =============================================================================
# Upload Pretrained Models (if requested)
# =============================================================================
upload_models() {
    if [ "$UPLOAD_MODELS" = false ]; then
        return 0
    fi

    log_step "Uploading Pretrained Models"

    RSYNC_SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -p $SSH_PORT"

    # OccWorld checkpoint
    if [ -f "$PROJECT_DIR/pretrained/occworld/latest.pth" ]; then
        log_info "Uploading OccWorld checkpoint..."
        rsync -avz --progress \
            -e "$RSYNC_SSH" \
            "$PROJECT_DIR/pretrained/occworld/latest.pth" \
            "${REMOTE_USER}@${SSH_HOST}:${REMOTE_PROJECT}/pretrained/occworld/"
        log_success "OccWorld checkpoint uploaded"
    else
        log_warn "OccWorld checkpoint not found locally"
    fi

    # VQVAE checkpoint
    if [ -f "$PROJECT_DIR/pretrained/vqvae/epoch_125.pth" ]; then
        log_info "Uploading VQVAE checkpoint..."
        rsync -avz --progress \
            -e "$RSYNC_SSH" \
            "$PROJECT_DIR/pretrained/vqvae/epoch_125.pth" \
            "${REMOTE_USER}@${SSH_HOST}:${REMOTE_PROJECT}/pretrained/vqvae/"
        log_success "VQVAE checkpoint uploaded"
    fi
}

# =============================================================================
# Upload Training Data (if requested)
# =============================================================================
upload_data() {
    if [ "$UPLOAD_DATA" = false ]; then
        return 0
    fi

    log_step "Uploading Training Data"

    RSYNC_SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -p $SSH_PORT"

    if [ -d "$PROJECT_DIR/data/tokyo_gazebo" ]; then
        DATA_SIZE=$(du -sh "$PROJECT_DIR/data/tokyo_gazebo" 2>/dev/null | cut -f1)
        log_info "Uploading training data ($DATA_SIZE)..."
        log_warn "This may take a while..."

        rsync -avz --progress \
            -e "$RSYNC_SSH" \
            "$PROJECT_DIR/data/tokyo_gazebo/" \
            "${REMOTE_USER}@${SSH_HOST}:${REMOTE_PROJECT}/data/tokyo_gazebo/"

        log_success "Training data uploaded"
    else
        log_warn "No local training data found at $PROJECT_DIR/data/tokyo_gazebo"
    fi
}

# =============================================================================
# Run Setup Script
# =============================================================================
run_setup() {
    if [ "$SYNC_ONLY" = true ]; then
        log_info "Sync-only mode, skipping setup"
        return 0
    fi

    log_step "Running Setup on Remote Instance"

    SSH_CMD="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -p $SSH_PORT ${REMOTE_USER}@${SSH_HOST}"

    log_info "Running vastai_setup.sh..."

    $SSH_CMD << 'REMOTE_SCRIPT'
cd /workspace/VeryLargeWeebModel

# Make scripts executable
chmod +x scripts/*.sh

# Run setup with minimal mode (dependencies installed, skip data download)
./scripts/vastai_setup.sh --minimal --skip-data

echo ""
echo "Setup complete on remote instance"
REMOTE_SCRIPT

    log_success "Remote setup complete"
}

# =============================================================================
# Start Training
# =============================================================================
start_training() {
    if [ "$START_TRAINING" = false ] || [ "$SETUP_ONLY" = true ]; then
        return 0
    fi

    log_step "Starting Training on Remote Instance"

    SSH_CMD="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -p $SSH_PORT ${REMOTE_USER}@${SSH_HOST}"

    log_info "Starting training in screen session..."

    $SSH_CMD << 'REMOTE_SCRIPT'
source ~/.bashrc 2>/dev/null || true

cd /workspace/VeryLargeWeebModel

# Check for pretrained model
if [ ! -f pretrained/occworld/latest.pth ]; then
    echo "WARNING: Pretrained model not found!"
    echo "Training will start from scratch (slower)"
fi

# Check for training data
DATA_SESSIONS=$(ls -d data/tokyo_gazebo/drone_* data/tokyo_gazebo/rover_* 2>/dev/null | wc -l)
if [ "$DATA_SESSIONS" -lt 5 ]; then
    echo "WARNING: Limited training data ($DATA_SESSIONS sessions)"
    echo "Consider generating more data or uploading with --upload-data"
fi

# Kill any existing training
pkill -f "train.py" 2>/dev/null || true
screen -X -S training quit 2>/dev/null || true

# Start training in screen
screen -dmS training bash -c "
    source ~/.bashrc 2>/dev/null || true
    cd /workspace/VeryLargeWeebModel
    export CUDA_LAUNCH_BLOCKING=0
    export NVIDIA_TF32_OVERRIDE=1

    echo ''
    echo '=========================================='
    echo 'Starting VeryLargeWeebModel Training'
    echo 'Time: $(date)'
    echo '=========================================='
    echo ''

    python train.py \
        --config config/finetune_tokyo.py \
        --work-dir /workspace/checkpoints \
        2>&1 | tee /workspace/checkpoints/training_\$(date +%Y%m%d_%H%M%S).log

    echo ''
    echo 'Training completed at $(date)'
    echo 'Press Enter to exit...'
    read
"

sleep 2

if screen -list | grep -q "training"; then
    echo "Training started in screen session 'training'"
else
    echo "ERROR: Failed to start training"
fi
REMOTE_SCRIPT

    log_success "Training started"
}

# =============================================================================
# Print Summary
# =============================================================================
print_summary() {
    echo ""
    echo -e "${GREEN}${BOLD}"
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                      DEPLOYMENT COMPLETE!                                  ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo "  Instance: ${SSH_HOST}:${SSH_PORT}"
    echo "  Project:  ${REMOTE_PROJECT}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Connect to instance:"
    echo "    ssh -i $SSH_KEY -p $SSH_PORT ${REMOTE_USER}@${SSH_HOST}"
    echo ""

    if [ "$START_TRAINING" = true ]; then
        echo "  Training is running in screen. Monitor with:"
        echo "    ssh -i $SSH_KEY -p $SSH_PORT ${REMOTE_USER}@${SSH_HOST} 'screen -r training'"
        echo ""
    else
        echo "  Start training manually:"
        echo "    ssh -i $SSH_KEY -p $SSH_PORT ${REMOTE_USER}@${SSH_HOST}"
        echo "    screen -S training"
        echo "    cd ${REMOTE_PROJECT}"
        echo "    python train.py --config config/finetune_tokyo.py --work-dir ${REMOTE_CHECKPOINTS}"
        echo ""
    fi

    echo "  Monitor GPU:"
    echo "    ssh -i $SSH_KEY -p $SSH_PORT ${REMOTE_USER}@${SSH_HOST} 'nvidia-smi -l 1'"
    echo ""
    echo "  TensorBoard (local port forwarding):"
    echo "    ssh -L 6006:localhost:6006 -i $SSH_KEY -p $SSH_PORT ${REMOTE_USER}@${SSH_HOST} 'tensorboard --logdir ${REMOTE_CHECKPOINTS} --port 6006'"
    echo "    Then open: http://localhost:6006"
    echo ""
    echo "  Download checkpoints:"
    echo "    rsync -avz -e 'ssh -i $SSH_KEY -p $SSH_PORT' \\"
    echo "      ${REMOTE_USER}@${SSH_HOST}:${REMOTE_CHECKPOINTS}/ ./checkpoints/"
    echo ""
    echo "  IMPORTANT: Download checkpoints before destroying the instance!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# =============================================================================
# Main
# =============================================================================
main() {
    show_banner
    parse_args "$@"

    # Check dependencies first
    check_dependencies

    # List instances mode
    if [ "$LIST_INSTANCES" = true ]; then
        list_instances
        exit 0
    fi

    # Get SSH info from instance ID
    if [ -n "$INSTANCE_ID" ]; then
        get_ssh_from_instance
    fi

    # Validate we have a target
    if [ -z "$SSH_TARGET" ]; then
        echo ""
        log_error "No target specified. Use one of:"
        echo "  --instance <ID>   Deploy to Vast.ai instance ID"
        echo "  --ip <IP:PORT>    Deploy via SSH"
        echo "  --list            List your instances"
        echo ""
        echo "Run '$0 --help' for more options"
        exit 1
    fi

    # Parse SSH target
    parse_ssh_target

    # Test connection
    test_connection

    # Upload files
    upload_project
    upload_models
    upload_data

    # Run setup
    run_setup

    # Start training
    start_training

    # Print summary
    print_summary
}

# Run main
main "$@"
