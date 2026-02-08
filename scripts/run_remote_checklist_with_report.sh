#!/usr/bin/env bash
set -euo pipefail

# Remote-ready checklist runner with markdown diagnosis report.
#
# Usage:
#   cd /path/to/VeryLargeWeebModel
#   bash scripts/run_remote_checklist_with_report.sh
#
# Optional environment variables:
#   ROOT=/path/to/VeryLargeWeebModel
#   VENV=$ROOT/.venv-review
#   PY=$VENV/bin/python
#   AUTO_INSTALL=1
#   NUM_WORKERS=0
#   UAV_DATA=$ROOT/data/uavscenes
#   UAV_SCENES=AMtown,AMvalley
#   UAV_INTERVAL=1
#   UAV_SAMPLES=200
#   TEST_CONFIG=$ROOT/config/test_local.py
#   WORK_DIR=/tmp/vlwm-collapse-check
#   OCCWORLD_PATH_CHECK=/path/to/OccWorld
#   CHECKPOINT=/path/to/checkpoint.pth
#   SOURCE_CONFIG=$ROOT/config/finetune_nuscenes_6dof.py
#   TARGET_CONFIG=$ROOT/config/finetune_uavscenes.py
#   LOG_DIR=$ROOT/out/remote_checklist_YYYYmmdd_HHMMSS

ROOT="${ROOT:-$(pwd)}"
VENV="${VENV:-$ROOT/.venv-review}"
PY="${PY:-$VENV/bin/python}"
AUTO_INSTALL="${AUTO_INSTALL:-1}"

UAV_DATA="${UAV_DATA:-$ROOT/data/uavscenes}"
UAV_SCENES="${UAV_SCENES:-AMtown}"
UAV_INTERVAL="${UAV_INTERVAL:-1}"
UAV_SAMPLES="${UAV_SAMPLES:-200}"
TEST_CONFIG="${TEST_CONFIG:-$ROOT/config/test_local.py}"
NUM_WORKERS="${NUM_WORKERS:-0}"
WORK_DIR="${WORK_DIR:-/tmp/vlwm-collapse-check}"

OCCWORLD_PATH_CHECK="${OCCWORLD_PATH_CHECK:-}"
CHECKPOINT="${CHECKPOINT:-}"
SOURCE_CONFIG="${SOURCE_CONFIG:-}"
TARGET_CONFIG="${TARGET_CONFIG:-}"

LOG_DIR="${LOG_DIR:-$ROOT/out/remote_checklist_$(date +%Y%m%d_%H%M%S)}"
STATUS_FILE="$LOG_DIR/status.tsv"
REPORT_MD="$LOG_DIR/report.md"
SUMMARY_TXT="$LOG_DIR/summary.txt"

mkdir -p "$LOG_DIR"
: > "$STATUS_FILE"

ensure_env() {
  if [[ ! -x "$PY" ]]; then
    echo "Creating venv at $VENV"
    python3 -m venv "$VENV"
    PY="$VENV/bin/python"
  fi

  if [[ "$AUTO_INSTALL" != "1" ]]; then
    return
  fi

  local missing=0
  "$PY" - <<'PY' >/dev/null 2>&1 || missing=1
import importlib
mods = ["pytest", "torch", "numpy", "tensorboard", "cv2"]
for m in mods:
    importlib.import_module(m)
PY
  if [[ "$missing" -eq 1 ]]; then
    echo "Installing dependencies into $VENV"
    "$PY" -m pip install -U pip
    "$PY" -m pip install pytest torch numpy tensorboard opencv-python
  fi
}

record_status() {
  local step="$1"
  local status="$2"
  local log="$3"
  printf '%s\t%s\t%s\n' "$step" "$status" "$log" >> "$STATUS_FILE"
}

run_step() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo ">>> $name"
  set +e
  "$@" >"$log" 2>&1
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    echo "[PASS] $name"
    record_status "$name" "PASS" "$log"
  else
    echo "[FAIL] $name (see $log)"
    record_status "$name" "FAIL" "$log"
  fi
}

skip_step() {
  local name="$1"
  local reason="$2"
  local log="$LOG_DIR/${name}.log"
  echo "[SKIP] $name - $reason"
  echo "$reason" > "$log"
  record_status "$name" "SKIP" "$log"
}

main() {
  cd "$ROOT"
  ensure_env

  # Step 0: baseline health
  run_step "step0_pytest" "$PY" -m pytest "$ROOT/tests" -q
  run_step "step0_sanity" "$PY" "$ROOT/scripts/vlwm_cli.py" sanity --quick

  # Step 1: occupancy/geometry sanity
  IFS=',' read -r -a SCENE_ARR <<< "$UAV_SCENES"
  run_step "step1_uav_occupancy" \
    "$PY" "$ROOT/scripts/verify_uavscenes_occupancy.py" \
    --data "$UAV_DATA" --interval "$UAV_INTERVAL" --samples "$UAV_SAMPLES" \
    --scenes "${SCENE_ARR[@]}"

  # Step 2: collapse check
  run_step "step2_collapse_check" \
    "$PY" "$ROOT/train.py" \
    --config "$TEST_CONFIG" \
    --model-type 6dof \
    --epochs 2 \
    --debug-freq 10 \
    --num-workers "$NUM_WORKERS" \
    --skip-validation \
    --work-dir "$WORK_DIR"

  # Step 3: pose channel sanity
  local pose_script="$LOG_DIR/step3_pose_sanity.py"
  cat > "$pose_script" <<'PY'
import argparse
import torch
from dataset.uavscenes_dataset import UAVScenesDataset, UAVScenesConfig

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--scenes", required=True)
ap.add_argument("--interval", type=int, default=1)
args = ap.parse_args()

scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
cfg = UAVScenesConfig(scenes=scenes, interval=args.interval, split="train")
ds = UAVScenesDataset(args.data, cfg)
if len(ds) == 0:
    raise SystemExit("No UAVScenes samples found")

s = ds[0]
p = s["history_poses"]
quat_norm = torch.linalg.norm(p[:, 3:7], dim=-1).mean().item()
vel_std = p[:, 7:13].std().item()

print("pose_shape:", tuple(p.shape))
print("quat_norm_mean:", quat_norm)
print("vel_std:", vel_std)
print("pose_min:", float(p.min()))
print("pose_max:", float(p.max()))

if not (0.8 <= quat_norm <= 1.2):
    raise SystemExit(f"Quaternion norm out of range: {quat_norm}")
if vel_std < 1e-6:
    raise SystemExit(f"Velocity std too small: {vel_std}")
PY
  run_step "step3_pose_sanity" \
    "$PY" "$pose_script" --data "$UAV_DATA" --scenes "$UAV_SCENES" --interval "$UAV_INTERVAL"

  # Step 4: OccWorld import path check (optional)
  if [[ -n "$OCCWORLD_PATH_CHECK" ]]; then
    run_step "step4_occworld_import" \
      env OCCWORLD_PATH="$OCCWORLD_PATH_CHECK" "$PY" -c \
      "import train; ok, cls = train.try_import_occworld(); print(ok, getattr(cls,'__name__',None)); raise SystemExit(0 if ok else 1)"
  else
    skip_step "step4_occworld_import" "OCCWORLD_PATH_CHECK not set"
  fi

  # Step 5: checkpoint key-style check (optional)
  if [[ -n "$CHECKPOINT" ]]; then
    run_step "step5_checkpoint_keys" \
      "$PY" -c \
      "import torch; c=torch.load('$CHECKPOINT', map_location='cpu'); sd=c.get('state_dict', c); print('total_keys', len(sd)); print('module_prefixed', sum(k.startswith('module.') for k in sd))"
  else
    skip_step "step5_checkpoint_keys" "CHECKPOINT not set"
  fi

  # Step 6: domain-gap eval (optional)
  if [[ -n "$CHECKPOINT" && -n "$SOURCE_CONFIG" && -n "$TARGET_CONFIG" ]]; then
    run_step "step6_eval_source" \
      "$PY" "$ROOT/train.py" \
      --config "$SOURCE_CONFIG" \
      --eval-only \
      --resume-from "$CHECKPOINT" \
      --num-workers "$NUM_WORKERS" \
      --skip-validation

    run_step "step6_eval_target" \
      "$PY" "$ROOT/train.py" \
      --config "$TARGET_CONFIG" \
      --eval-only \
      --resume-from "$CHECKPOINT" \
      --num-workers "$NUM_WORKERS" \
      --skip-validation
  else
    skip_step "step6_eval_source" "Need CHECKPOINT + SOURCE_CONFIG + TARGET_CONFIG"
    skip_step "step6_eval_target" "Need CHECKPOINT + SOURCE_CONFIG + TARGET_CONFIG"
  fi

  "$PY" - "$STATUS_FILE" "$REPORT_MD" "$SUMMARY_TXT" <<'PY'
import json
import re
import sys
from pathlib import Path

status_file = Path(sys.argv[1])
report_md = Path(sys.argv[2])
summary_txt = Path(sys.argv[3])

rows = []
for line in status_file.read_text().splitlines():
    if not line.strip():
        continue
    step, status, log = line.split("\t", 2)
    rows.append({"step": step, "status": status, "log": log})

status_map = {r["step"]: r["status"] for r in rows}
log_map = {r["step"]: Path(r["log"]) for r in rows}

def read(step):
    p = log_map.get(step)
    if not p or not p.exists():
        return ""
    return p.read_text(errors="ignore")

metrics = {}

# Step 1 metrics
log1 = read("step1_uav_occupancy")
m = re.search(r"Zero-occupancy samples:\s*(\d+)", log1)
metrics["zero_occupancy_samples"] = int(m.group(1)) if m else None
m = re.search(r"Mean occupancy rate:\s*([0-9.eE+-]+)", log1)
metrics["mean_occupancy_rate"] = float(m.group(1)) if m else None

# Step 2 metrics
log2 = read("step2_collapse_check")
pred_means = [float(x) for x in re.findall(r"Pred mean:\s*([0-9.]+)", log2)]
metrics["collapse_pred_mean_first"] = pred_means[0] if pred_means else None
metrics["collapse_pred_mean_last"] = pred_means[-1] if pred_means else None
metrics["non_finite_loss_seen"] = ("Non-finite loss" in log2)

# Step 3 metrics
log3 = read("step3_pose_sanity")
m = re.search(r"quat_norm_mean:\s*([0-9.eE+-]+)", log3)
metrics["quat_norm_mean"] = float(m.group(1)) if m else None
m = re.search(r"vel_std:\s*([0-9.eE+-]+)", log3)
metrics["vel_std"] = float(m.group(1)) if m else None

# Step 5 metrics
log5 = read("step5_checkpoint_keys")
m = re.search(r"total_keys\s+(\d+)", log5)
metrics["ckpt_total_keys"] = int(m.group(1)) if m else None
m = re.search(r"module_prefixed\s+(\d+)", log5)
metrics["ckpt_module_prefixed"] = int(m.group(1)) if m else None

# Step 6 metrics
src = read("step6_eval_source")
tgt = read("step6_eval_target")
ms = re.search(r"Eval-only:\s*Val Loss\s*=\s*([0-9.eE+-]+)", src)
mt = re.search(r"Eval-only:\s*Val Loss\s*=\s*([0-9.eE+-]+)", tgt)
metrics["source_val_loss"] = float(ms.group(1)) if ms else None
metrics["target_val_loss"] = float(mt.group(1)) if mt else None
if metrics["source_val_loss"] and metrics["target_val_loss"]:
    metrics["target_source_loss_ratio"] = metrics["target_val_loss"] / max(metrics["source_val_loss"], 1e-12)
else:
    metrics["target_source_loss_ratio"] = None

root_causes = []
def add_cause(name, score, evidence, why, action):
    root_causes.append({
        "name": name,
        "score": score,
        "evidence": evidence,
        "why": why,
        "action": action,
    })

# 1) Data/geometry
score = 0
evidence = []
if status_map.get("step1_uav_occupancy") == "FAIL":
    score += 100
    evidence.append("step1 failed")
if metrics["mean_occupancy_rate"] is not None and metrics["mean_occupancy_rate"] < 5e-4:
    score += 80
    evidence.append(f"mean occupancy low: {metrics['mean_occupancy_rate']:.6f}")
if metrics["zero_occupancy_samples"] is not None and metrics["zero_occupancy_samples"] > 0:
    score += min(60, metrics["zero_occupancy_samples"])
    evidence.append(f"zero occupancy samples: {metrics['zero_occupancy_samples']}")
if status_map.get("step3_pose_sanity") == "FAIL":
    score += 90
    evidence.append("step3 failed")
if score:
    add_cause(
        "Data / Geometry Alignment",
        score,
        evidence,
        "If transforms or voxel ranges are off, the model receives inconsistent supervision and cannot learn stable occupancy dynamics.",
        "Fix frame conventions, point-cloud range, and pose parsing before hyperparameter tuning."
    )

# 2) Collapse / objective
score = 0
evidence = []
if status_map.get("step2_collapse_check") == "FAIL":
    score += 100
    evidence.append("step2 failed")
pm_last = metrics["collapse_pred_mean_last"]
if pm_last is not None and pm_last < 1e-3:
    score += 85
    evidence.append(f"pred mean collapsed: {pm_last:.6f}")
if metrics["non_finite_loss_seen"]:
    score += 60
    evidence.append("non-finite loss seen")
if score:
    add_cause(
        "Loss / Class-Imbalance Collapse",
        score,
        evidence,
        "Extreme occupancy sparsity can drive trivial all-empty predictions or unstable gradients if safeguards are insufficient.",
        "Tune focal/mean-matching balance and inspect target occupancy distribution per batch."
    )

# 3) OccWorld integration
score = 0
evidence = []
if status_map.get("step4_occworld_import") == "FAIL":
    score += 85
    evidence.append("step4 failed")
if score:
    add_cause(
        "OccWorld Integration Path",
        score,
        evidence,
        "Training may be running a fallback model path instead of the intended external OccWorld implementation.",
        "Fix import path/package loading for OCCWORLD_PATH and retest import explicitly."
    )

# 4) Checkpoint compatibility
score = 0
evidence = []
if status_map.get("step5_checkpoint_keys") == "FAIL":
    score += 75
    evidence.append("step5 failed")
elif status_map.get("step5_checkpoint_keys") == "PASS":
    mk = metrics["ckpt_module_prefixed"]
    tk = metrics["ckpt_total_keys"]
    if tk and mk is not None and mk == 0:
        score += 35
        evidence.append("checkpoint keys are non-DataParallel style")
if score:
    add_cause(
        "Checkpoint / Parallelism Mismatch",
        score,
        evidence,
        "If checkpoint key prefix style does not match runtime wrapping, loads may be partial or silently skipped.",
        "Match `module.` key style with runtime and load strictly during diagnosis."
    )

# 5) Domain gap
score = 0
evidence = []
ratio = metrics["target_source_loss_ratio"]
if ratio is not None and ratio >= 1.5:
    score += min(90, int(ratio * 25))
    evidence.append(f"target/source val loss ratio: {ratio:.2f}")
if score:
    add_cause(
        "Domain Shift (Sim/Real or Cross-Dataset)",
        score,
        evidence,
        "Model generalization may be limited by sensor and motion distribution shift between source and target domains.",
        "Use adaptation/fine-tuning curriculum and validate per-domain occupancy statistics."
    )

root_causes.sort(key=lambda x: x["score"], reverse=True)

pass_count = sum(1 for r in rows if r["status"] == "PASS")
fail_count = sum(1 for r in rows if r["status"] == "FAIL")
skip_count = sum(1 for r in rows if r["status"] == "SKIP")

with summary_txt.open("w") as f:
    f.write(f"PASS={pass_count}\n")
    f.write(f"FAIL={fail_count}\n")
    f.write(f"SKIP={skip_count}\n")
    f.write(f"REPORT={report_md}\n")

with report_md.open("w") as f:
    f.write("# Remote Checklist Report\n\n")
    f.write("## Overall\n\n")
    f.write(f"- PASS: {pass_count}\n")
    f.write(f"- FAIL: {fail_count}\n")
    f.write(f"- SKIP: {skip_count}\n\n")

    f.write("## Step Results\n\n")
    f.write("| Step | Status | Log |\n")
    f.write("|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['step']} | {r['status']} | `{r['log']}` |\n")
    f.write("\n")

    f.write("## Key Metrics\n\n")
    f.write("```json\n")
    f.write(json.dumps(metrics, indent=2))
    f.write("\n```\n\n")

    f.write("## Ranked Likely Root Causes\n\n")
    if not root_causes:
        f.write("No strong failure signals detected from this checklist run.\n\n")
    else:
        for i, rc in enumerate(root_causes, 1):
            f.write(f"{i}. **{rc['name']}** (score: {rc['score']})\n")
            f.write(f"   - Evidence: {', '.join(rc['evidence'])}\n")
            f.write(f"   - Why this blocks learning: {rc['why']}\n")
            f.write(f"   - Next action: {rc['action']}\n\n")

    f.write("## Notes\n\n")
    f.write("- This ranking is heuristic and evidence-driven from current logs.\n")
    f.write("- Re-run after each major fix to confirm root-cause score shifts.\n")
PY

  echo
  echo "Checklist done."
  echo "Summary: $SUMMARY_TXT"
  echo "Report:  $REPORT_MD"
}

main "$@"
