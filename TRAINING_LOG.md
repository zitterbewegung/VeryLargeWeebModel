# Training Log

## Problem: Loss Collapse (2024-01-18)

### Symptom
Loss drops from ~1.8 to 0.0001 within 500 batches of the first epoch and stays stuck:
```
Epoch 0 [0/17663] Loss: 1.8373
Epoch 0 [100/17663] Loss: 0.1035
Epoch 0 [500/17663] Loss: 0.0036
Epoch 0 [3750/17663] Loss: 0.0001  ← stuck
```

### Root Cause
**Extreme occupancy sparsity**: Only ~0.83% of voxels are occupied (99.17% empty).

With a 200×200×121 grid (4.84 million voxels), the model learns that predicting **all zeros** minimizes loss:
- Weighted BCE with pos_weight=10 is insufficient for 99% class imbalance
- Dice loss collapses: when pred≈0 and target≈0, dice_loss → 0
- Model takes the "lazy path" of outputting zeros everywhere

### Diagnosis
Added debug logging to track predictions during training:
```python
DEBUG [0]: Occ: 0.83%, Pred mean: 0.5000, min: 0.20, max: 0.80   # start
DEBUG [100]: Occ: 0.82%, Pred mean: 0.0100, min: 0.00, max: 0.05  # collapsing
DEBUG [500]: Occ: 0.83%, Pred mean: 0.0001, min: 0.00, max: 0.01  # collapsed
```

### Fixes Applied

#### 1. Replaced Weighted BCE with Focal Loss
Focal Loss down-weights easy examples (empty voxels) and focuses on hard examples:
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```
- `focal_alpha=0.99`: 99% weight on occupied voxels
- `focal_gamma=2.0`: standard focusing parameter

**Result**: Still collapsed, but slower.

#### 2. Added Mean-Matching Regularization
Forces the model's average prediction to match target average:
```python
mean_loss = MSE(pred.mean(), target.mean())
```
With `mean_weight=10.0`, the model **cannot** predict all zeros because that would give:
- pred_mean ≈ 0
- target_mean ≈ 0.008
- mean_loss = 10 * (0 - 0.008)² = high penalty

#### 3. Periodic Debug Logging
Log every 100 batches to catch collapse early:
```
DEBUG [0]: Occ: 0.83%, Pred mean: 0.5000, min: 0.20, max: 0.80
DEBUG [100]: Occ: 0.82%, Pred mean: 0.0082, min: 0.00, max: 0.15  # healthy
```

### Current Loss Function
```python
OccupancyLoss(
    focal_alpha=0.99,   # 99% weight on occupied
    focal_gamma=2.0,    # focus on hard examples
    dice_weight=1.0,    # overlap optimization
    mean_weight=10.0,   # prevent all-zero collapse
)
```

### Expected Behavior After Fix
- Loss should stabilize around 0.1-0.5, not drop to 0.0001
- `Pred mean` should stay ~0.008 (matching target occupancy)
- `Pred max` should remain >0.5 (confident positive predictions)

### Configuration
- GPU: A100-40GB
- Batch size: 3
- Grid size: 200×200×121 (4.84M voxels)
- Occupancy rate: ~0.83%
- Learning rate: 1e-4 with cosine annealing

---

## Weights & Biases Integration

### Setup
```bash
pip install wandb
wandb login
```

### Usage
```bash
# Basic usage
python train.py --config config/finetune_tokyo.py --wandb

# With custom project/run name
python train.py --config config/finetune_tokyo.py \
    --wandb \
    --wandb-project occworld-tokyo \
    --wandb-run-name experiment-v1 \
    --wandb-tags focal-loss baseline
```

### CLI Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--wandb` | False | Enable W&B logging |
| `--wandb-project` | `occworld-tokyo` | W&B project name |
| `--wandb-entity` | None | W&B team/username |
| `--wandb-run-name` | auto | Run name (auto-generated if not provided) |
| `--wandb-tags` | [] | Tags for the run |

### What Gets Logged

**Metrics (per step):**
- `train/loss` - Training loss
- `pred/mean`, `pred/min`, `pred/max` - Prediction statistics
- `pred/occupancy_rate` - Target occupancy rate

**Metrics (per epoch):**
- `epoch/train_loss`, `epoch/val_loss` - Epoch losses
- `epoch/lr` - Learning rate
- `epoch/time_seconds` - Epoch duration

**Config:**
- Dataset parameters (grid size, voxel size, etc.)
- Model architecture and parameter count
- Training hyperparameters (lr, batch size, etc.)
- Loss function settings

**Artifacts:**
- `model-checkpoint-epochN` - Periodic checkpoints
- `model-best` - Best model (lowest val loss)

### Monitoring Training Health

Watch for loss collapse in the W&B dashboard:
- **Healthy**: `pred/mean` stays around 0.008-0.02 (matching target occupancy)
- **Collapsing**: `pred/mean` drops toward 0.0001

The mean-matching regularization should prevent collapse, but monitor the `pred/*` metrics to verify.

---

## UAVScenes Voxelization Notes (2025-01-22)

### Symptom
Most UAVScenes samples had zero occupied voxels when using pose-only ego-frame alignment.

### Fix (original)
Keep `min_in_range_ratio=0.01` and enable the lidar-centering fallback for UAVScenes.

### Fix (2026-02-08, superseded)
Set `ego_frame=False` — UAVScenes LiDAR is already in sensor-local frame (standard
for LiDAR hardware). The ego transform was applying `(sensor_local - world_pos) @ R.T`,
which always produced out-of-range values, triggering 100% fallback. See entry below.

### Quick Check
```bash
python scripts/verify_uavscenes_occupancy.py --data data/uavscenes --scenes AMtown --samples 200
```

---

## UAVScenes ego_frame=False Fix (2026-02-08)

### Symptom
100% fallback rate in `_align_points` — every single call triggers the LiDAR-center
centering fallback, producing 5 warnings per dataset instantiation:
```
UAVScenes _align_points: ego-frame alignment fallback activated (1/1 calls).
UAVScenes _align_points: ego-frame alignment fallback activated (2/2 calls).
...
```

### Root Cause
UAVScenes LiDAR point clouds are in **sensor-local frame** (standard for LiDAR hardware —
the sensor reports distances relative to itself). The `_transform_points_to_ego` function
assumes **world-frame** input and computes `(sensor_local_points - world_position) @ R.T`,
which produces values far outside the [-40, 40] range → `_in_range_ratio` ≈ 0 → fallback.

The centering fallback was near-no-op (sensor-local points are already near origin), so
training worked despite this, but it was wasteful and misleading.

### Fix
- Set `ego_frame=False` as default in `UAVScenesConfig`
- Also fixed in `config/finetune_uavscenes.py`, `config/test_local.py`, and `train.py`
  fallback default (all three were hardcoding `ego_frame=True`, overriding the dataclass)
- Added early return in `_align_points` when `ego_frame=False` to skip both the transform
  and the centering fallback

### Lesson
Changing a dataclass default is not enough — config files and constructor callsites can
override it. Always grep for all usages of the parameter.

---

## Uncertainty Loss Spikes from GPS Velocity Noise (2026-02-08)

### Symptom
Periodic loss spikes dominated by uncertainty component (80-95% of total loss):
```
[SPIKE] Epoch 0 batch 94:  loss 121.34 is 21.6x running avg — uncertainty=114.95 (94.7%)
[SPIKE] Epoch 0 batch 334: loss 50.32  is  9.4x running avg — uncertainty=44.86  (89.1%)
[SPIKE] Epoch 0 batch 395: loss 31.43  is  5.6x running avg — uncertainty=25.04  (79.7%)
```

### Root Cause
GPS/GNSS noise in UAVScenes pose data causes **extreme velocity outliers** when computed
via finite-difference: `vel = (pos_curr - pos_prev) / dt`. Diagnostic output revealed:
```
vel: max=453.43   ← 453 m/s = ~1630 km/h, physically impossible for a UAV
```

The same `453.43` value appeared in multiple spikes, confirming specific bad frames rather
than random noise. The Gaussian NLL uncertainty loss (`error²/σ² + log(σ²)`) explodes when
the pose prediction error is driven by these impossible velocities.

### Fix
Added velocity clamping in `UAVScenesConfig` with physical UAV limits:
- `max_linear_velocity=20.0` m/s (~72 km/h, generous for multi-rotor UAVs)
- `max_angular_velocity=6.28` rad/s (~1 revolution/s)

Clamping applied in both velocity sources:
1. `_compute_velocity()` — finite-difference estimation
2. `_parse_pose_json()` — velocities loaded from JSON sampleinfos

### Diagnosis Tool
EMA-based loss spike detection added to `train_epoch()`:
```
[SPIKE] Epoch 0 batch 334: loss 50.32 is 9.4x running avg (5.34)
[SPIKE]   Dominant component: uncertainty=44.8557 (89.1% of total)
[SPIKE]   All components: uncertainty=44.8557, occ=2.8107, pose=2.3480, ...
[SPIKE]   Occ rate: 0.020%, pose_pos: min=0.04 max=85.05, vel: max=453.43
```

Triggers when any batch loss exceeds 5x the running EMA (α=0.02). Reports dominant
loss component, all components sorted, pose statistics, and scene names.
