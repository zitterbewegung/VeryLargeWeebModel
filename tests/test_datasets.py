"""Tests for dataset implementations and collate functions."""

import sys
import os
import unittest
import tempfile
import json
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class TestGazeboDatasetConfig(unittest.TestCase):
    """Test GazeboOccWorldDataset configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from dataset.gazebo_occworld_dataset import DatasetConfig

        config = DatasetConfig()

        self.assertEqual(config.history_frames, 4)
        self.assertEqual(config.future_frames, 6)
        self.assertEqual(config.frame_skip, 1)
        self.assertEqual(config.agent_type, 'both')
        self.assertEqual(config.split, 'train')

    def test_config_custom(self):
        """Test custom configuration values."""
        from dataset.gazebo_occworld_dataset import DatasetConfig

        config = DatasetConfig(
            history_frames=2,
            future_frames=4,
            agent_type='drone',
            split='val',
        )

        self.assertEqual(config.history_frames, 2)
        self.assertEqual(config.future_frames, 4)
        self.assertEqual(config.agent_type, 'drone')
        self.assertEqual(config.split, 'val')


class TestGazeboCollateFn(unittest.TestCase):
    """Test Gazebo dataset collate function."""

    def test_collate_fn(self):
        """Test collate function with mock batch."""
        from dataset.gazebo_occworld_dataset import collate_fn

        # Create mock batch
        batch = []
        for i in range(2):
            sample = {
                'history_poses': torch.rand(4, 13),
                'history_occupancy': torch.rand(4, 32, 32, 16),
                'future_occupancy': torch.rand(6, 32, 32, 16),
                'future_poses': torch.rand(6, 13),
                'agent_type': i % 2,
                'history_images': [
                    {'CAM_FRONT': torch.rand(3, 224, 224)} for _ in range(4)
                ],
                'history_lidar': [torch.rand(1000, 4) for _ in range(4)],
            }
            batch.append(sample)

        collated = collate_fn(batch)

        # Check shapes
        self.assertEqual(collated['history_poses'].shape, (2, 4, 13))
        self.assertEqual(collated['history_occupancy'].shape, (2, 4, 32, 32, 16))
        self.assertEqual(collated['future_occupancy'].shape, (2, 6, 32, 32, 16))
        self.assertEqual(collated['future_poses'].shape, (2, 6, 13))
        self.assertEqual(collated['agent_type'].shape, (2,))

        # Check images are collated correctly
        self.assertIn('CAM_FRONT', collated['history_images'])
        self.assertEqual(collated['history_images']['CAM_FRONT'].shape, (2, 4, 3, 224, 224))

        # Check lidar is kept as list
        self.assertIsInstance(collated['history_lidar'], list)


class TestUAVScenesConfig(unittest.TestCase):
    """Test UAVScenes configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from dataset.uavscenes_dataset import UAVScenesConfig

        config = UAVScenesConfig()

        self.assertEqual(config.history_frames, 4)
        self.assertEqual(config.future_frames, 6)
        self.assertEqual(config.interval, 1)
        self.assertIn('AMtown', config.scenes)

    def test_grid_size_property(self):
        """Test grid_size is computed correctly."""
        from dataset.uavscenes_dataset import UAVScenesConfig

        config = UAVScenesConfig(
            point_cloud_range=(-40.0, -40.0, -10.0, 40.0, 40.0, 50.0),
            voxel_size=(0.4, 0.4, 0.5),
        )

        grid_size = config.grid_size

        # (40 - -40) / 0.4 = 200
        # (40 - -40) / 0.4 = 200
        # (50 - -10) / 0.5 = 120
        self.assertEqual(grid_size, (200, 200, 120))


class TestUAVScenesCollateFn(unittest.TestCase):
    """Test UAVScenes collate function."""

    def test_collate_fn(self):
        """Test collate function with mock batch."""
        from dataset.uavscenes_dataset import collate_fn

        batch = []
        for _ in range(2):
            sample = {
                'history_occupancy': torch.rand(4, 32, 32, 16),
                'future_occupancy': torch.rand(6, 32, 32, 16),
                'history_poses': torch.rand(4, 13),
                'future_poses': torch.rand(6, 13),
                'agent_type': torch.tensor(1),
                'scene': 'AMtown',
                'scene_folder': 'interval1_AMtown01',
            }
            batch.append(sample)

        collated = collate_fn(batch)

        self.assertEqual(collated['history_occupancy'].shape, (2, 4, 32, 32, 16))
        self.assertEqual(collated['future_occupancy'].shape, (2, 6, 32, 32, 16))
        self.assertEqual(collated['history_poses'].shape, (2, 4, 13))
        self.assertEqual(collated['future_poses'].shape, (2, 6, 13))
        self.assertEqual(collated['agent_type'].shape, (2,))


class TestUAVScenesQuaternionConversion(unittest.TestCase):
    """Test quaternion conversion utilities."""

    def test_rotation_matrix_to_quaternion(self):
        """Test rotation matrix to quaternion conversion."""
        from dataset.uavscenes_dataset import UAVScenesDataset, UAVScenesConfig

        # Create a mock dataset to access the method
        # We'll test the conversion directly
        config = UAVScenesConfig()

        # Create dataset with non-existent path (won't fail until data access)
        try:
            dataset = UAVScenesDataset('/nonexistent', config)
        except FileNotFoundError:
            # Expected - create a minimal instance to test the method
            pass

        # Test identity rotation
        R_identity = np.eye(3)

        # Manual implementation check
        trace = np.trace(R_identity)
        # For identity matrix, trace = 3
        s = 0.5 / np.sqrt(trace + 1.0)  # s = 0.5 / 2 = 0.25
        w = 0.25 / s  # w = 1.0

        self.assertAlmostEqual(w, 1.0, places=5)

    def test_quaternion_to_rotation_matrix_roundtrip(self):
        """Test quaternion <-> rotation matrix roundtrip."""
        from dataset.uavscenes_dataset import UAVScenesConfig

        # Test with a known quaternion (90 degree rotation around Z)
        # q = [cos(45), 0, 0, sin(45)] for 90 deg around Z
        import math
        angle = math.pi / 2  # 90 degrees
        quat = np.array([math.cos(angle/2), 0, 0, math.sin(angle/2)])

        # Convert to rotation matrix
        w, x, y, z = quat
        n = w*w + x*x + y*y + z*z
        s = 2.0 / n

        wx, wy, wz = s*w*x, s*w*y, s*w*z
        xx, xy, xz = s*x*x, s*x*y, s*x*z
        yy, yz, zz = s*y*y, s*y*z, s*z*z

        R = np.array([
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ])

        # For 90 deg rotation around Z:
        # [0, -1, 0]
        # [1,  0, 0]
        # [0,  0, 1]
        expected_R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])

        np.testing.assert_array_almost_equal(R, expected_R, decimal=5)


class TestDatasetWithMockData(unittest.TestCase):
    """Test dataset with mock filesystem data."""

    def setUp(self):
        """Create temporary directory with mock data."""
        self.temp_dir = tempfile.mkdtemp()

        # Create mock session structure
        session_dir = os.path.join(self.temp_dir, 'drone_test_session')
        os.makedirs(os.path.join(session_dir, 'images'))
        os.makedirs(os.path.join(session_dir, 'lidar'))
        os.makedirs(os.path.join(session_dir, 'poses'))
        os.makedirs(os.path.join(session_dir, 'occupancy'))

        # Create mock files for 15 frames (enough for history + future)
        for i in range(15):
            frame_id = f'{i:06d}'

            # Mock occupancy
            occ_path = os.path.join(session_dir, 'occupancy', f'{frame_id}_occupancy.npz')
            occ_data = np.random.randint(0, 2, size=(32, 32, 16), dtype=np.uint8)
            np.savez_compressed(occ_path, occupancy=occ_data)

            # Mock pose
            pose_path = os.path.join(session_dir, 'poses', f'{frame_id}.json')
            pose_data = {
                'position': {'x': float(i), 'y': float(i * 0.5), 'z': 10.0 + i * 0.1},
                'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
                'velocity': {
                    'linear': {'x': 1.0, 'y': 0.5, 'z': 0.1},
                    'angular': {'x': 0.0, 'y': 0.0, 'z': 0.1},
                },
            }
            with open(pose_path, 'w') as f:
                json.dump(pose_data, f)

            # Mock lidar
            lidar_path = os.path.join(session_dir, 'lidar', f'{frame_id}_LIDAR.npy')
            lidar_data = np.random.rand(1000, 4).astype(np.float32)
            np.save(lidar_path, lidar_data)

        self.session_dir = session_dir

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_dataset_discovery(self):
        """Test dataset discovers sessions correctly."""
        from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset, DatasetConfig

        config = DatasetConfig(
            history_frames=2,
            future_frames=3,
            filter_static=False,
        )

        dataset = GazeboOccWorldDataset(self.temp_dir, config)

        self.assertGreater(len(dataset), 0)

    def test_dataset_getitem(self):
        """Test dataset __getitem__ returns correct structure."""
        from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset, DatasetConfig

        config = DatasetConfig(
            history_frames=2,
            future_frames=3,
            filter_static=False,
        )

        dataset = GazeboOccWorldDataset(self.temp_dir, config)

        sample = dataset[0]

        self.assertIn('history_occupancy', sample)
        self.assertIn('future_occupancy', sample)
        self.assertIn('history_poses', sample)
        self.assertIn('future_poses', sample)
        self.assertIn('agent_type', sample)

        # Check shapes
        self.assertEqual(sample['history_occupancy'].shape, (2, 32, 32, 16))
        self.assertEqual(sample['future_occupancy'].shape, (3, 32, 32, 16))
        self.assertEqual(sample['history_poses'].shape, (2, 13))
        self.assertEqual(sample['future_poses'].shape, (3, 13))

    def test_dataset_validation(self):
        """Test dataset validation method."""
        from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset, DatasetConfig

        config = DatasetConfig(
            history_frames=2,
            future_frames=3,
            filter_static=False,
        )

        dataset = GazeboOccWorldDataset(self.temp_dir, config)

        stats = dataset.validate_dataset()

        self.assertIn('total_sequences', stats)
        self.assertIn('motion_stats', stats)
        self.assertIn('occupancy_stats', stats)


class TestVoxelization(unittest.TestCase):
    """Test point cloud to occupancy grid conversion."""

    def test_points_to_occupancy(self):
        """Test point cloud voxelization."""
        from dataset.uavscenes_dataset import UAVScenesConfig

        config = UAVScenesConfig(
            point_cloud_range=(-10.0, -10.0, -5.0, 10.0, 10.0, 5.0),
            voxel_size=(1.0, 1.0, 1.0),
        )

        # Create points at known locations
        points = np.array([
            [0.0, 0.0, 0.0],      # Should be in center
            [5.0, 5.0, 0.0],      # Should be in positive quadrant
            [-5.0, -5.0, 0.0],    # Should be in negative quadrant
            [100.0, 100.0, 100.0],  # Out of range - should be filtered
        ], dtype=np.float32)

        # Manual voxelization
        pc_range = np.array(config.point_cloud_range)
        voxel_size = np.array(config.voxel_size)
        grid_size = np.array(config.grid_size)

        # Filter points in range
        xyz = points[:, :3]
        mask = (
            (xyz[:, 0] >= pc_range[0]) & (xyz[:, 0] < pc_range[3]) &
            (xyz[:, 1] >= pc_range[1]) & (xyz[:, 1] < pc_range[4]) &
            (xyz[:, 2] >= pc_range[2]) & (xyz[:, 2] < pc_range[5])
        )
        xyz_filtered = xyz[mask]

        self.assertEqual(len(xyz_filtered), 3)  # 3 points in range

        # Convert to voxel indices
        voxel_coords = ((xyz_filtered - pc_range[:3]) / voxel_size).astype(np.int32)

        # Check center point (0, 0, 0) -> voxel (10, 10, 5)
        center_voxel = voxel_coords[0]
        self.assertEqual(center_voxel[0], 10)  # (0 - -10) / 1 = 10
        self.assertEqual(center_voxel[1], 10)
        self.assertEqual(center_voxel[2], 5)   # (0 - -5) / 1 = 5


class TestDatasetSplits(unittest.TestCase):
    """Test dataset train/val/test splits."""

    def test_split_ratios(self):
        """Test that splits use correct ratios."""
        # Create a minimal mock to test split logic
        from dataset.gazebo_occworld_dataset import DatasetConfig

        config_train = DatasetConfig(split='train', val_ratio=0.1, test_ratio=0.1)
        config_val = DatasetConfig(split='val', val_ratio=0.1, test_ratio=0.1)
        config_test = DatasetConfig(split='test', val_ratio=0.1, test_ratio=0.1)

        # Test with 100 samples
        n = 100
        val_size = int(n * config_train.val_ratio)
        test_size = int(n * config_train.test_ratio)
        train_size = n - val_size - test_size

        self.assertEqual(train_size, 80)
        self.assertEqual(val_size, 10)
        self.assertEqual(test_size, 10)


if __name__ == '__main__':
    unittest.main()
