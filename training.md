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

### Fix
Keep `min_in_range_ratio=0.01` and enable the lidar-centering fallback for UAVScenes:
- `ego_frame=True`
- `fallback_to_lidar_center=True`
- `min_in_range_ratio=0.01`

This ensures occupancy grids stay non-empty even when pose alignment is off.

### Quick Check
```bash
python scripts/verify_uavscenes_occupancy.py --data data/uavscenes --scenes AMtown --samples 200
```
