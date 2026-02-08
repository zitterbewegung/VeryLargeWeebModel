#!/usr/bin/env python3
"""
Example: Using Data Augmentation Transforms

Demonstrates how to use the 3D occupancy transforms with datasets for training.

Usage:
    python examples/use_transforms.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from dataset.transforms import (
    RandomFlip3D,
    RandomRotate90,
    OccupancyDropout,
    Compose,
    get_train_transforms,
)


def example_basic_transforms():
    """Example 1: Using individual transforms."""
    print("=" * 70)
    print("Example 1: Basic Transform Usage")
    print("=" * 70)

    # Create sample data (would come from dataset)
    sample = {
        'history_occupancy': torch.rand(4, 200, 200, 121) > 0.5,  # Binary occupancy
        'future_occupancy': torch.rand(6, 200, 200, 121) > 0.5,
        'history_poses': torch.rand(4, 13),
        'future_poses': torch.rand(6, 13),
        'agent_type': torch.tensor(1),
        'scene': 'test_scene',
    }
    sample['history_occupancy'] = sample['history_occupancy'].float()
    sample['future_occupancy'] = sample['future_occupancy'].float()

    print(f"Original history occupancy: {sample['history_occupancy'].sum().item()} occupied voxels")

    # Apply flip transform
    flip = RandomFlip3D(p=1.0, axes=('x', 'y'))
    flipped = flip(sample)
    print(f"After flip: {flipped['history_occupancy'].sum().item()} occupied voxels (should be same)")

    # Apply rotation transform
    rotate = RandomRotate90(p=1.0)
    rotated = rotate(sample)
    print(f"After 90° rotation: {rotated['history_occupancy'].sum().item()} occupied voxels")

    # Apply dropout transform
    dropout = OccupancyDropout(p=0.2)
    dropped = dropout(sample)
    print(f"After dropout (p=0.2): {dropped['history_occupancy'].sum().item()} occupied voxels")
    print(f"Future occupancy unchanged: {dropped['future_occupancy'].sum().item()} voxels")
    print()


def example_composed_transforms():
    """Example 2: Using composed transforms (recommended)."""
    print("=" * 70)
    print("Example 2: Composed Transform Pipeline")
    print("=" * 70)

    # Create sample data
    sample = {
        'history_occupancy': torch.rand(4, 200, 200, 121) > 0.5,
        'future_occupancy': torch.rand(6, 200, 200, 121) > 0.5,
        'history_poses': torch.rand(4, 13),
        'future_poses': torch.rand(6, 13),
        'agent_type': torch.tensor(1),
        'scene': 'test_scene',
    }
    sample['history_occupancy'] = sample['history_occupancy'].float()
    sample['future_occupancy'] = sample['future_occupancy'].float()

    # Use the default training augmentation pipeline
    transforms = get_train_transforms(
        flip_p=0.5,      # 50% chance of flip per axis
        rotate_p=0.5,    # 50% chance of rotation
        dropout_p=0.1,   # Drop 10% of occupied voxels
    )

    print("Applying transform pipeline 5 times:")
    for i in range(5):
        result = transforms(sample)
        print(f"  Run {i+1}: {result['history_occupancy'].sum().item():.0f} occupied voxels")
    print()


def example_custom_transforms():
    """Example 3: Custom transform pipeline."""
    print("=" * 70)
    print("Example 3: Custom Transform Pipeline")
    print("=" * 70)

    # Create custom pipeline
    custom_transforms = Compose([
        RandomFlip3D(p=0.8, axes=('x',)),      # Aggressive X-axis flips only
        RandomRotate90(p=0.3),                  # Rare rotations
        OccupancyDropout(p=0.05),              # Minimal dropout
    ])

    sample = {
        'history_occupancy': torch.rand(4, 200, 200, 121) > 0.5,
        'future_occupancy': torch.rand(6, 200, 200, 121) > 0.5,
        'history_poses': torch.rand(4, 13),
        'future_poses': torch.rand(6, 13),
        'agent_type': torch.tensor(1),
    }
    sample['history_occupancy'] = sample['history_occupancy'].float()
    sample['future_occupancy'] = sample['future_occupancy'].float()

    result = custom_transforms(sample)
    print(f"Original: {sample['history_occupancy'].sum().item():.0f} occupied voxels")
    print(f"Transformed: {result['history_occupancy'].sum().item():.0f} occupied voxels")
    print()


def example_dataset_integration():
    """Example 4: Using transforms with datasets."""
    print("=" * 70)
    print("Example 4: Dataset Integration")
    print("=" * 70)

    from dataset.gazebo_occworld_dataset import DatasetConfig
    from dataset.uavscenes_dataset import UAVScenesConfig
    from dataset.nuscenes_6dof_dataset import NuScenes6DoFConfig

    # Get augmentation pipeline
    train_transforms = get_train_transforms(flip_p=0.5, rotate_p=0.5, dropout_p=0.1)

    print("Example dataset initialization with transforms:")
    print()

    # Example 1: Gazebo dataset
    print("# Gazebo dataset")
    print("from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset, DatasetConfig")
    print("from dataset.transforms import get_train_transforms")
    print()
    print("config = DatasetConfig(split='train')")
    print("transforms = get_train_transforms(flip_p=0.5, rotate_p=0.5, dropout_p=0.1)")
    print("dataset = GazeboOccWorldDataset('data/gazebo', config, transform=transforms)")
    print()

    # Example 2: UAVScenes dataset
    print("# UAVScenes dataset")
    print("from dataset.uavscenes_dataset import UAVScenesDataset, UAVScenesConfig")
    print()
    print("config = UAVScenesConfig(split='train')")
    print("transforms = get_train_transforms(flip_p=0.5, rotate_p=0.5, dropout_p=0.1)")
    print("dataset = UAVScenesDataset('data/uavscenes', config, transform=transforms)")
    print()

    # Example 3: NuScenes 6DoF dataset
    print("# NuScenes 6DoF dataset")
    print("from dataset.nuscenes_6dof_dataset import NuScenes6DoFDataset, NuScenes6DoFConfig")
    print()
    print("config = NuScenes6DoFConfig(split='train')")
    print("transforms = get_train_transforms(flip_p=0.5, rotate_p=0.5, dropout_p=0.1)")
    print("dataset = NuScenes6DoFDataset('data/nuscenes', config, transform=transforms)")
    print()

    print("Note: The transform is applied after data loading in __getitem__()")
    print()


def example_no_transforms():
    """Example 5: Disabling transforms for validation."""
    print("=" * 70)
    print("Example 5: Validation Without Transforms")
    print("=" * 70)

    print("For validation/testing, omit transforms or pass transform=None:")
    print()
    print("# Training")
    print("train_dataset = MyDataset(data_root, config, transform=get_train_transforms())")
    print()
    print("# Validation (no augmentation)")
    print("val_dataset = MyDataset(data_root, config, transform=None)")
    print()
    print("This ensures evaluation metrics reflect true model performance.")
    print()


def main():
    """Run all examples."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "3D Occupancy Transform Examples" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    example_basic_transforms()
    example_composed_transforms()
    example_custom_transforms()
    example_dataset_integration()
    example_no_transforms()

    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
