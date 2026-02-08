# 3D Occupancy Data Augmentation Transforms

This document describes the data augmentation transforms added to the VeryLargeWeebModel project for training with 3D occupancy grids.

## Overview

The `dataset/transforms.py` module provides geometric augmentation transforms that maintain physical consistency across temporal sequences. These transforms are designed to work with 3D occupancy grids and 6DoF pose vectors.

## Transform Classes

### RandomFlip3D
Randomly flips occupancy grids and corresponding poses along X or Y axis.

**Parameters:**
- `p` (float): Probability of applying flip for each axis (default: 0.5)
- `axes` (tuple): Axes to consider for flipping, e.g., `('x', 'y')`

**Physics:**
- X-axis flip: Negates x position and vx velocity
- Y-axis flip: Negates y position and vy velocity
- Quaternions remain unchanged (rotation symmetry)

**Example:**
```python
flip = RandomFlip3D(p=0.5, axes=('x', 'y'))
result = flip(sample)
```

### RandomRotate90
Randomly rotates occupancy and poses by 90° multiples around Z axis.

**Parameters:**
- `p` (float): Probability of applying rotation (default: 0.5)

**Physics:**
- Rotates occupancy grid in XY plane
- Applies 2D rotation to position (x, y) and velocity (vx, vy)
- Updates quaternion by multiplying with Z-axis rotation quaternion
- Maintains quaternion normalization

**Example:**
```python
rotate = RandomRotate90(p=0.5)
result = rotate(sample)
```

### OccupancyDropout
Randomly zeros out occupied voxels to simulate sensor noise.

**Parameters:**
- `p` (float): Probability of dropping each occupied voxel (default: 0.1)

**Behavior:**
- Only applies to `history_occupancy` (observations)
- Does NOT apply to `future_occupancy` (ground truth)
- Simulates LiDAR occlusion, sparse point clouds, and measurement dropout

**Example:**
```python
dropout = OccupancyDropout(p=0.1)
result = dropout(sample)
```

### Compose
Standard composition of multiple transforms.

**Parameters:**
- `transforms` (list): List of transform callables

**Example:**
```python
transforms = Compose([
    RandomFlip3D(p=0.5),
    RandomRotate90(p=0.5),
    OccupancyDropout(p=0.1),
])
result = transforms(sample)
```

## Factory Function

### get_train_transforms
Returns a pre-configured transform pipeline for training.

**Parameters:**
- `flip_p` (float): Probability of flipping (default: 0.5)
- `rotate_p` (float): Probability of rotation (default: 0.5)
- `dropout_p` (float): Probability of voxel dropout (default: 0.1)

**Returns:**
- `Compose` object with the full augmentation pipeline

**Example:**
```python
from dataset.transforms import get_train_transforms

transforms = get_train_transforms(flip_p=0.5, rotate_p=0.5, dropout_p=0.1)
```

## Dataset Integration

All dataset classes now support an optional `transform` parameter:

### UAVScenesDataset
```python
from dataset.uavscenes_dataset import UAVScenesDataset, UAVScenesConfig
from dataset.transforms import get_train_transforms

config = UAVScenesConfig(split='train')
transforms = get_train_transforms()
dataset = UAVScenesDataset('data/uavscenes', config, transform=transforms)
```

### NuScenes6DoFDataset
```python
from dataset.nuscenes_6dof_dataset import NuScenes6DoFDataset, NuScenes6DoFConfig
from dataset.transforms import get_train_transforms

config = NuScenes6DoFConfig(split='train')
transforms = get_train_transforms()
dataset = NuScenes6DoFDataset('data/nuscenes', config, transform=transforms)
```

### NuScenesOccWorldDataset
```python
from dataset.nuscenes_occworld_dataset import NuScenesOccWorldDataset, NuScenesConfig
from dataset.transforms import get_train_transforms

config = NuScenesConfig(split='train')
transforms = get_train_transforms()
dataset = NuScenesOccWorldDataset('data/nuscenes', config, transform=transforms)
```

### GazeboOccWorldDataset
```python
from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset, DatasetConfig
from dataset.transforms import get_train_transforms

config = DatasetConfig(split='train')
transforms = get_train_transforms()
dataset = GazeboOccWorldDataset('data/gazebo', config, transform=transforms)
```

### TartanAirDataset
```python
from dataset.tartanair_dataset import TartanAirDataset, TartanAirConfig
from dataset.transforms import get_train_transforms

config = TartanAirConfig(split='train')
transforms = get_train_transforms()
dataset = TartanAirDataset('data/tartanair', config, transform=transforms)
```

### MidAirDataset
```python
from dataset.midair_dataset import MidAirDataset, MidAirConfig
from dataset.transforms import get_train_transforms

config = MidAirConfig(split='train')
transforms = get_train_transforms()
dataset = MidAirDataset('data/midair', config, transform=transforms)
```

## Data Format

Transforms expect samples with the following keys:

```python
{
    'history_occupancy': torch.Tensor,  # [T_h, X, Y, Z]
    'future_occupancy': torch.Tensor,   # [T_f, X, Y, Z]
    'history_poses': torch.Tensor,      # [T_h, 13]
    'future_poses': torch.Tensor,       # [T_f, 13]
    # ... other keys passed through unchanged
}
```

### Pose Format (13D)
The 13D pose vector contains:
- Indices 0-2: Position (x, y, z)
- Indices 3-6: Quaternion (qw, qx, qy, qz) in [w, x, y, z] format
- Indices 7-9: Linear velocity (vx, vy, vz)
- Indices 10-12: Angular velocity (wx, wy, wz)

## Training vs Validation

**Training:** Use transforms to improve generalization
```python
train_dataset = MyDataset(data_root, config, transform=get_train_transforms())
```

**Validation/Testing:** Omit transforms for accurate evaluation
```python
val_dataset = MyDataset(data_root, config, transform=None)
```

## Example Usage

See `examples/use_transforms.py` for comprehensive examples:

```bash
python examples/use_transforms.py
```

## Testing

The transforms have been tested for:
- Shape preservation
- Physical correctness (flips, rotations)
- Quaternion normalization
- Dropout behavior (history only)

Run tests:
```bash
python -m pytest tests/test_datasets.py -v
```

## Files Modified

- **New file:** `dataset/transforms.py` - Transform implementations
- **New file:** `examples/use_transforms.py` - Usage examples
- **Modified:** All dataset `__init__` methods to accept `transform` parameter
- **Modified:** All dataset `__getitem__` methods to apply transforms

### Datasets with Transform Support
1. `dataset/uavscenes_dataset.py` - UAVScenesDataset
2. `dataset/nuscenes_6dof_dataset.py` - NuScenes6DoFDataset
3. `dataset/nuscenes_occworld_dataset.py` - NuScenesOccWorldDataset
4. `dataset/gazebo_occworld_dataset.py` - GazeboOccWorldDataset (already had support)
5. `dataset/tartanair_dataset.py` - TartanAirDataset
6. `dataset/midair_dataset.py` - MidAirDataset

## Implementation Notes

1. **Physical Consistency:** All transforms maintain physical consistency between occupancy grids and poses
2. **Temporal Consistency:** Transforms are applied to entire sequences, not per-frame
3. **Non-destructive:** Original sample data is not modified (uses `.clone()`)
4. **Quaternion Operations:** Uses proper quaternion multiplication and normalization
5. **Ground Truth Preservation:** Future occupancy is not affected by dropout

## Performance Considerations

- Transforms are applied on-the-fly during data loading
- Minimal computational overhead (mostly tensor operations)
- No additional memory required (compared to pre-augmented datasets)
- Random seed can be set for reproducibility during debugging
