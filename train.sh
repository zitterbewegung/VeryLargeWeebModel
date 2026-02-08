#!/usr/bin/env bash
set -euo pipefail

# AerialWorld training launcher
# Usage: ./train.sh [CONFIG] [EXTRA_ARGS...]
# Examples:
#   ./train.sh                                    # Tokyo PLATEAU (default)
#   ./train.sh finetune_uavscenes.py              # UAVScenes
#   ./train.sh finetune_6dof.py --epochs 100      # 6DoF with extra args
#   ./train.sh test_local.py --epochs 2           # Quick local test

CONFIG="${1:-finetune_tokyo.py}"
shift 2>/dev/null || true

# Ensure setup
python scripts/vlwm_cli.py setup

# Run inside screen so training survives disconnects
if [ -z "${STY:-}" ] && command -v screen &>/dev/null; then
    echo "Tip: run inside 'screen -S training' to survive disconnects"
fi

exec python train.py \
    --config "config/${CONFIG}" \
    --work-dir ./checkpoints \
    --amp \
    "$@"
