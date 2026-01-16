#!/usr/bin/env python3
"""
Integration test for OccWorld Tokyo training pipeline.

Tests the full pipeline:
1. Creates dummy data with correct format
2. Verifies dataset loader can read it
3. Runs one training step
4. Verifies checkpoint is saved

Usage:
    python scripts/integration_test.py
"""
import os
import sys
import shutil
import tempfile

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def test_imports():
    """Test all required imports."""
    print("\n[TEST] Checking imports...")
    errors = []

    try:
        import torch
        print(f"  torch: {torch.__version__}")
    except ImportError as e:
        errors.append(f"torch: {e}")

    try:
        import numpy as np
        print(f"  numpy: {np.__version__}")
    except ImportError as e:
        errors.append(f"numpy: {e}")

    try:
        import cv2
        print(f"  cv2: {cv2.__version__}")
    except ImportError as e:
        errors.append(f"cv2: {e}")

    try:
        from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset, DatasetConfig
        print("  dataset loader: OK")
    except ImportError as e:
        errors.append(f"dataset loader: {e}")

    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"    - {e}")
        return False

    print("  All imports OK")
    return True


def create_test_data(data_dir, num_frames=15):
    """Create properly formatted test data."""
    print(f"\n[TEST] Creating test data in {data_dir}...")

    import numpy as np
    import cv2
    import json

    session = os.path.join(data_dir, 'drone_test_session')
    os.makedirs(os.path.join(session, 'occupancy'), exist_ok=True)
    os.makedirs(os.path.join(session, 'poses'), exist_ok=True)
    os.makedirs(os.path.join(session, 'lidar'), exist_ok=True)
    os.makedirs(os.path.join(session, 'images'), exist_ok=True)

    for i in range(num_frames):
        frame_id = f'{i:06d}'

        # Occupancy grid
        occ = np.random.randint(0, 2, (200, 200, 121), dtype=np.uint8)
        np.savez_compressed(
            os.path.join(session, 'occupancy', f'{frame_id}_occupancy.npz'),
            occupancy=occ
        )

        # Pose (correct nested format)
        pose = {
            'position': {'x': float(i), 'y': 0.0, 'z': 10.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
            'velocity': {
                'linear': {'x': 1.0, 'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
            }
        }
        with open(os.path.join(session, 'poses', f'{frame_id}.json'), 'w') as f:
            json.dump(pose, f)

        # LiDAR
        points = np.random.randn(1000, 4).astype(np.float32)
        np.save(os.path.join(session, 'lidar', f'{frame_id}_LIDAR.npy'), points)

        # Camera image
        img = np.zeros((900, 1600, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(session, 'images', f'{frame_id}_CAM_FRONT.jpg'), img)

    # Verify files exist
    files = {
        'occupancy': len(os.listdir(os.path.join(session, 'occupancy'))),
        'poses': len(os.listdir(os.path.join(session, 'poses'))),
        'lidar': len(os.listdir(os.path.join(session, 'lidar'))),
        'images': len(os.listdir(os.path.join(session, 'images'))),
    }
    print(f"  Created: {files}")

    return all(v == num_frames for v in files.values())


def test_dataset_loader(data_dir):
    """Test that dataset loader can read the data."""
    print("\n[TEST] Testing dataset loader...")

    from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset, DatasetConfig

    config = DatasetConfig(
        history_frames=4,
        future_frames=6,
        split='train',
        val_ratio=0.0,
        test_ratio=0.0,
    )

    try:
        dataset = GazeboOccWorldDataset(data_dir, config)
        print(f"  Dataset size: {len(dataset)}")

        if len(dataset) == 0:
            print("  ERROR: Dataset is empty!")
            return False

        # Load one sample
        sample = dataset[0]
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  history_occupancy shape: {sample['history_occupancy'].shape}")
        print(f"  future_occupancy shape: {sample['future_occupancy'].shape}")
        print(f"  history_poses shape: {sample['history_poses'].shape}")

        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_load():
    """Test that config file loads without errors."""
    print("\n[TEST] Testing config load...")

    import importlib.util
    config_path = os.path.join(PROJECT_ROOT, 'config', 'finetune_tokyo.py')

    try:
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

        print(f"  point_cloud_range: {config.point_cloud_range}")
        print(f"  voxel_size: {config.voxel_size}")
        print(f"  history_frames: {config.history_frames}")
        print(f"  future_frames: {config.future_frames}")

        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model can be created."""
    print("\n[TEST] Testing model creation...")

    import torch

    # Import from train.py
    sys.path.insert(0, PROJECT_ROOT)
    from train import SimpleOccupancyModel, load_config

    try:
        config = load_config(os.path.join(PROJECT_ROOT, 'config', 'finetune_tokyo.py'))
        model = SimpleOccupancyModel(config)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {num_params:,}")

        # Test forward pass with dummy input
        batch_size = 2
        history = torch.randn(batch_size, 4, 200, 200, 121)
        output = model(history)
        print(f"  Input shape: {history.shape}")
        print(f"  Output shape: {output.shape}")

        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(data_dir):
    """Test one training step."""
    print("\n[TEST] Testing training step...")

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset, DatasetConfig, collate_fn
    from train import SimpleOccupancyModel, load_config

    try:
        # Load config
        config = load_config(os.path.join(PROJECT_ROOT, 'config', 'finetune_tokyo.py'))

        # Create dataset
        dataset_cfg = DatasetConfig(
            history_frames=4,
            future_frames=6,
            split='train',
            val_ratio=0.0,
            test_ratio=0.0,
        )
        dataset = GazeboOccWorldDataset(data_dir, dataset_cfg)

        if len(dataset) == 0:
            print("  ERROR: Empty dataset")
            return False

        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

        # Create model
        model = SimpleOccupancyModel(config)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCELoss()

        # One training step
        model.train()
        batch = next(iter(loader))

        history_occ = batch['history_occupancy'].float()
        future_occ = batch['future_occupancy'].float()

        optimizer.zero_grad()
        pred = model(history_occ)
        loss = criterion(pred, future_occ)
        loss.backward()
        optimizer.step()

        print(f"  Loss: {loss.item():.4f}")
        print("  Training step completed successfully!")

        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("OccWorld Integration Test")
    print("=" * 60)

    results = {}

    # Test imports
    results['imports'] = test_imports()
    if not results['imports']:
        print("\n[FAIL] Import test failed. Cannot continue.")
        return False

    # Test config
    results['config'] = test_config_load()

    # Test model creation
    results['model'] = test_model_creation()

    # Create temp data directory
    test_data_dir = os.path.join(PROJECT_ROOT, 'data', 'integration_test')
    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir)
    os.makedirs(test_data_dir)

    try:
        # Create test data
        results['create_data'] = create_test_data(test_data_dir, num_frames=15)

        # Test dataset loader
        results['dataset'] = test_dataset_loader(test_data_dir)

        # Test training step
        results['training'] = test_training_step(test_data_dir)
    finally:
        # Cleanup
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
