#!/bin/bash
# =============================================================================
# Deploy VeryLargeWeebModel Training to Lambda Cloud
# =============================================================================
# Run this script from your LOCAL machine to deploy to a Lambda instance.
#
# Usage:
#   ./scripts/deploy_to_lambda.sh <INSTANCE_IP> [OPTIONS]
#
# Options:
#   --key PATH      SSH key path (default: ~/.ssh/id_rsa)
#   --upload-data   Also upload local data directory
#   --upload-models Also upload local pretrained models
#   --start-train   Start training after deployment
#   --help          Show this help
#
# Examples:
#   ./scripts/deploy_to_lambda.sh 192.0.2.1
#   ./scripts/deploy_to_lambda.sh 192.0.2.1 --key ~/.ssh/lambda_key --upload-models
#   ./scripts/deploy_to_lambda.sh 192.0.2.1 --start-train
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Default options
SSH_KEY="$HOME/.ssh/id_rsa"
UPLOAD_DATA=false
UPLOAD_MODELS=false
START_TRAINING=false
INSTANCE_IP=""

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --key)          SSH_KEY="$2"; shift 2 ;;
        --upload-data)  UPLOAD_DATA=true; shift ;;
        --upload-models) UPLOAD_MODELS=true; shift ;;
        --start-train)  START_TRAINING=true; shift ;;
        --help|-h)      head -25 "$0" | tail -20; exit 0 ;;
        -*)             log_error "Unknown option: $1" ;;
        *)              INSTANCE_IP="$1"; shift ;;
    esac
done

# Validate
if [ -z "$INSTANCE_IP" ]; then
    echo "Usage: $0 <INSTANCE_IP> [OPTIONS]"
    echo "Run '$0 --help' for more information."
    exit 1
fi

if [ ! -f "$SSH_KEY" ]; then
    log_error "SSH key not found: $SSH_KEY"
fi

# SSH/SCP commands
SSH_CMD="ssh -i $SSH_KEY -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP"
SCP_CMD="scp -i $SSH_KEY -o StrictHostKeyChecking=no"
RSYNC_CMD="rsync -avz --progress -e 'ssh -i $SSH_KEY -o StrictHostKeyChecking=no'"

echo ""
echo "=============================================="
echo "    Deploy VeryLargeWeebModel to Lambda Cloud          "
echo "=============================================="
echo ""
log_info "Instance IP: $INSTANCE_IP"
log_info "SSH Key: $SSH_KEY"
log_info "Project: $PROJECT_DIR"
echo ""

# =============================================================================
# Test Connection
# =============================================================================
log_info "Testing SSH connection..."
$SSH_CMD "echo 'Connection successful'" || log_error "Cannot connect to instance"
log_success "Connected to Lambda instance"

# Get instance info
log_info "Instance info:"
$SSH_CMD "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" || true

# =============================================================================
# Upload Project
# =============================================================================
log_info "Uploading project files..."

# Create exclude list
EXCLUDE_ARGS=(
    --exclude '.git'
    --exclude '__pycache__'
    --exclude '*.pyc'
    --exclude '.pytest_cache'
    --exclude 'data/plateau/raw/*'
    --exclude 'data/tokyo_gazebo/*'
    --exclude '*.zip'
    --exclude '*.tar.gz'
    --exclude 'out/*'
    --exclude 'checkpoints/*'
)

if [ "$UPLOAD_DATA" = false ]; then
    EXCLUDE_ARGS+=(--exclude 'data/*')
fi

if [ "$UPLOAD_MODELS" = false ]; then
    EXCLUDE_ARGS+=(--exclude 'pretrained/*.pth')
fi

# Upload
eval $RSYNC_CMD "${EXCLUDE_ARGS[@]}" "$PROJECT_DIR/" "ubuntu@$INSTANCE_IP:~/VeryLargeWeebModel/"

log_success "Project uploaded"

# =============================================================================
# Upload Pretrained Models (if requested)
# =============================================================================
if [ "$UPLOAD_MODELS" = true ]; then
    log_info "Uploading pretrained models..."

    # VQVAE
    if [ -f "$PROJECT_DIR/pretrained/vqvae/epoch_125.pth" ]; then
        $SCP_CMD "$PROJECT_DIR/pretrained/vqvae/epoch_125.pth" \
            "ubuntu@$INSTANCE_IP:~/VeryLargeWeebModel/pretrained/vqvae/"
        log_success "Uploaded VQVAE checkpoint"
    fi

    # VeryLargeWeebModel
    if [ -f "$PROJECT_DIR/pretrained/occworld/latest.pth" ]; then
        $SCP_CMD "$PROJECT_DIR/pretrained/occworld/latest.pth" \
            "ubuntu@$INSTANCE_IP:~/VeryLargeWeebModel/pretrained/occworld/"
        log_success "Uploaded VeryLargeWeebModel checkpoint"
    fi
fi

# =============================================================================
# Upload Data (if requested)
# =============================================================================
if [ "$UPLOAD_DATA" = true ]; then
    log_info "Uploading data..."

    if [ -d "$PROJECT_DIR/data/tokyo_gazebo" ]; then
        eval $RSYNC_CMD "$PROJECT_DIR/data/tokyo_gazebo/" \
            "ubuntu@$INSTANCE_IP:~/VeryLargeWeebModel/data/tokyo_gazebo/"
        log_success "Uploaded Tokyo Gazebo data"
    else
        log_warn "No data found at $PROJECT_DIR/data/tokyo_gazebo"
    fi
fi

# =============================================================================
# Run Setup Script
# =============================================================================
log_info "Running setup script on instance..."

$SSH_CMD << 'REMOTE_SCRIPT'
cd ~/VeryLargeWeebModel
chmod +x scripts/lambda_setup.sh
./scripts/lambda_setup.sh
REMOTE_SCRIPT

log_success "Setup complete"

# =============================================================================
# Start Training (if requested)
# =============================================================================
if [ "$START_TRAINING" = true ]; then
    log_info "Starting training..."

    $SSH_CMD << 'REMOTE_SCRIPT'
source ~/.bashrc
conda activate occworld

# Check if pretrained models exist
if [ ! -f ~/VeryLargeWeebModel/pretrained/occworld/latest.pth ]; then
    echo "ERROR: Pretrained model not found!"
    echo "Please upload pretrained/occworld/latest.pth first"
    exit 1
fi

# Start training in tmux
tmux new-session -d -s training "
    source ~/.bashrc
    conda activate occworld
    cd ~/VeryLargeWeebModel
    python train.py \
        --config config/finetune_tokyo.py \
        --work-dir ~/checkpoints/occworld_tokyo \
        2>&1 | tee ~/training.log
"

echo "Training started in tmux session 'training'"
REMOTE_SCRIPT

    log_success "Training started"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "           Deployment Complete               "
echo "=============================================="
echo ""
echo "Connect to instance:"
echo "  ssh -i $SSH_KEY ubuntu@$INSTANCE_IP"
echo ""
echo "Start training (if not already started):"
echo "  tmux new -s training"
echo "  conda activate occworld"
echo "  cd ~/VeryLargeWeebModel"
echo "  python train.py --config config/finetune_tokyo.py --work-dir ~/checkpoints/occworld_tokyo"
echo ""
echo "Monitor training:"
echo "  tmux attach -t training     # View training"
echo "  nvtop                       # GPU usage"
echo "  tensorboard --logdir ~/checkpoints --port 6006 --bind_all"
echo ""
echo "Download checkpoints:"
echo "  scp -i $SSH_KEY -r ubuntu@$INSTANCE_IP:~/checkpoints/ ./lambda_checkpoints/"
echo ""
echo "IMPORTANT: Terminate instance when done to stop billing!"
echo ""
echo "=============================================="
