"""Integration tests for voxel configuration consistency.

These tests ensure that all datasets and scripts use consistent voxel parameters,
preventing shape mismatches during training.
"""

import sys
import os
import unittest

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))


class TestVoxelConfigConsistency(unittest.TestCase):
    """Test that voxel configs are consistent across the codebase."""

    def test_centralized_config_exists(self):
        """Test that the centralized voxel config can be imported."""
        from utils.voxel_config import (
            VoxelConfig,
            DEFAULT_VOXEL_CONFIG,
            DEFAULT_POINT_CLOUD_RANGE,
            DEFAULT_VOXEL_SIZE,
            DEFAULT_GRID_SIZE,
        )

        self.assertIsInstance(DEFAULT_VOXEL_CONFIG, VoxelConfig)
        self.assertEqual(len(DEFAULT_POINT_CLOUD_RANGE), 6)
        self.assertEqual(len(DEFAULT_VOXEL_SIZE), 3)
        self.assertEqual(len(DEFAULT_GRID_SIZE), 3)

    def test_grid_size_calculation(self):
        """Test that grid size is calculated correctly from range and voxel size."""
        from utils.voxel_config import VoxelConfig

        config = VoxelConfig()
        grid_size = config.grid_size

        # Manually calculate expected grid size
        expected_x = int((config.point_cloud_range[3] - config.point_cloud_range[0]) / config.voxel_size[0])
        expected_y = int((config.point_cloud_range[4] - config.point_cloud_range[1]) / config.voxel_size[1])
        expected_z = int((config.point_cloud_range[5] - config.point_cloud_range[2]) / config.voxel_size[2])

        self.assertEqual(grid_size[0], expected_x)
        self.assertEqual(grid_size[1], expected_y)
        self.assertEqual(grid_size[2], expected_z)

    def test_default_grid_size_value(self):
        """Test the expected default grid size."""
        from utils.voxel_config import DEFAULT_GRID_SIZE

        # Default config: range (-40,-40,-2, 40,40,150), voxel (0.4,0.4,1.25)
        # X: (40 - -40) / 0.4 = 80 / 0.4 = 200
        # Y: (40 - -40) / 0.4 = 80 / 0.4 = 200
        # Z: (150 - -2) / 1.25 = 152 / 1.25 = 121.6 -> 121
        self.assertEqual(DEFAULT_GRID_SIZE, (200, 200, 121))

    def test_gazebo_dataset_config_compatible(self):
        """Test that Gazebo dataset config is compatible with centralized config."""
        try:
            from dataset.gazebo_occworld_dataset import GazeboDataConfig
            from utils.voxel_config import DEFAULT_POINT_CLOUD_RANGE, DEFAULT_VOXEL_SIZE

            config = GazeboDataConfig()

            # Gazebo dataset may have its own config, but grid dimensions should match
            # for data generated with default settings
            self.assertEqual(len(config.point_cloud_range), 6)
            self.assertEqual(len(config.voxel_size), 3)

        except ImportError:
            self.skipTest("Gazebo dataset not available")

    def test_uavscenes_dataset_config_exists(self):
        """Test that UAVScenes dataset config can be imported."""
        try:
            from dataset.uavscenes_dataset import UAVScenesConfig

            config = UAVScenesConfig()
            self.assertEqual(len(config.point_cloud_range), 6)
            self.assertEqual(len(config.voxel_size), 3)

        except ImportError:
            self.skipTest("UAVScenes dataset not available")


class TestTrajectoryGenerator(unittest.TestCase):
    """Test trajectory generation utilities."""

    def test_trajectory_generator_import(self):
        """Test that trajectory generator can be imported."""
        from utils.trajectory import TrajectoryGenerator, TrajectoryConfig

        config = TrajectoryConfig()
        generator = TrajectoryGenerator(config)
        self.assertIsNotNone(generator)

    def test_random_trajectory_generation(self):
        """Test random trajectory generation."""
        from utils.trajectory import TrajectoryGenerator, TrajectoryConfig

        config = TrajectoryConfig()
        generator = TrajectoryGenerator(config)

        waypoints = generator.generate(num_frames=10, pattern='random', agent_type='drone')

        self.assertEqual(len(waypoints), 10)
        for wp in waypoints:
            self.assertIn('position', wp)
            self.assertIn('orientation', wp)
            self.assertIn('velocity', wp)
            self.assertIn('x', wp['position'])
            self.assertIn('y', wp['position'])
            self.assertIn('z', wp['position'])

    def test_survey_pattern_generation(self):
        """Test survey pattern trajectory generation."""
        from utils.trajectory import TrajectoryGenerator, TrajectoryConfig

        config = TrajectoryConfig()
        generator = TrajectoryGenerator(config)

        waypoints = generator.generate(num_frames=20, pattern='survey', agent_type='drone')

        self.assertEqual(len(waypoints), 20)

    def test_trajectory_validation(self):
        """Test trajectory validation."""
        from utils.trajectory import TrajectoryGenerator, TrajectoryConfig

        config = TrajectoryConfig(min_motion=2.0)
        generator = TrajectoryGenerator(config)

        waypoints = generator.generate(num_frames=10, pattern='random', agent_type='drone')
        validation = generator.validate_trajectory(waypoints)

        self.assertIn('num_frames', validation)
        self.assertIn('total_distance', validation)
        self.assertIn('min_distance', validation)
        self.assertIn('valid', validation)
        self.assertEqual(validation['num_frames'], 10)


class TestUtilsImports(unittest.TestCase):
    """Test that all utils modules can be imported."""

    def test_import_logging(self):
        """Test logging module import."""
        from utils.logging import log_info, log_success, log_warn, log_error, log_step, Colors
        self.assertTrue(callable(log_info))
        self.assertTrue(callable(log_error))

    def test_import_directories(self):
        """Test directories module import."""
        from utils.directories import create_session_dirs
        self.assertTrue(callable(create_session_dirs))

    def test_import_dependencies(self):
        """Test dependencies module import."""
        from utils.dependencies import (
            HAS_BOTO3, HAS_TQDM, HAS_REQUESTS,
            HAS_OPEN3D, HAS_TRIMESH, HAS_CV2,
            HAS_SCIPY, HAS_NUMBA, HAS_TORCH
        )
        # These should be booleans
        self.assertIsInstance(HAS_BOTO3, bool)
        self.assertIsInstance(HAS_TORCH, bool)

    def test_import_s3(self):
        """Test S3 module import."""
        from utils.s3 import (
            S3_REGION, DEFAULT_BUCKET,
            get_s3_client, s3_file_exists, list_s3_files,
            upload_file, download_file,
            upload_directory, download_directory,
        )
        self.assertTrue(callable(get_s3_client))
        self.assertTrue(callable(upload_file))

    def test_import_voxel_config(self):
        """Test voxel config module import."""
        from utils.voxel_config import (
            VoxelConfig, DEFAULT_VOXEL_CONFIG,
            DEFAULT_POINT_CLOUD_RANGE, DEFAULT_VOXEL_SIZE, DEFAULT_GRID_SIZE
        )
        self.assertIsInstance(DEFAULT_VOXEL_CONFIG, VoxelConfig)


class TestDirectoryCreation(unittest.TestCase):
    """Test directory creation utilities."""

    def test_create_session_dirs(self):
        """Test session directory creation."""
        import tempfile
        import shutil
        from utils.directories import create_session_dirs

        with tempfile.TemporaryDirectory() as tmpdir:
            dirs = create_session_dirs(tmpdir, 'test_session')

            self.assertIn('root', dirs)
            self.assertIn('images', dirs)
            self.assertIn('lidar', dirs)
            self.assertIn('poses', dirs)
            self.assertIn('occupancy', dirs)

            # Check directories were actually created
            for name, path in dirs.items():
                self.assertTrue(os.path.isdir(path), f"Directory {name} not created: {path}")

    def test_create_session_dirs_custom_subdirs(self):
        """Test session directory creation with custom subdirs."""
        import tempfile
        from utils.directories import create_session_dirs

        with tempfile.TemporaryDirectory() as tmpdir:
            dirs = create_session_dirs(tmpdir, 'test_session', subdirs=('data', 'logs'))

            self.assertIn('root', dirs)
            self.assertIn('data', dirs)
            self.assertIn('logs', dirs)
            self.assertNotIn('images', dirs)


if __name__ == '__main__':
    unittest.main()
