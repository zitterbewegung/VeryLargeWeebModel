"""Tests for utility modules."""

import sys
import os
import unittest
import tempfile
import shutil
from io import StringIO
from unittest.mock import patch, MagicMock

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))


class TestLogging(unittest.TestCase):
    """Test logging utilities."""

    def test_log_info_output(self):
        """Test log_info produces correct output."""
        from utils.logging import log_info, Colors

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            log_info("test message")
            output = mock_stdout.getvalue()

        self.assertIn("[INFO]", output)
        self.assertIn("test message", output)

    def test_log_error_output(self):
        """Test log_error produces correct output."""
        from utils.logging import log_error, Colors

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            log_error("error message")
            output = mock_stdout.getvalue()

        self.assertIn("[ERROR]", output)
        self.assertIn("error message", output)

    def test_log_success_output(self):
        """Test log_success produces correct output."""
        from utils.logging import log_success

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            log_success("success message")
            output = mock_stdout.getvalue()

        self.assertIn("[OK]", output)
        self.assertIn("success message", output)

    def test_log_warn_output(self):
        """Test log_warn produces correct output."""
        from utils.logging import log_warn

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            log_warn("warning message")
            output = mock_stdout.getvalue()

        self.assertIn("[WARN]", output)
        self.assertIn("warning message", output)

    def test_log_step_output(self):
        """Test log_step produces correct output."""
        from utils.logging import log_step

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            log_step("step message")
            output = mock_stdout.getvalue()

        self.assertIn("==>", output)
        self.assertIn("step message", output)

    def test_colors_class_has_all_colors(self):
        """Test Colors class has all expected color codes."""
        from utils.logging import Colors

        self.assertTrue(hasattr(Colors, 'RED'))
        self.assertTrue(hasattr(Colors, 'GREEN'))
        self.assertTrue(hasattr(Colors, 'YELLOW'))
        self.assertTrue(hasattr(Colors, 'BLUE'))
        self.assertTrue(hasattr(Colors, 'CYAN'))
        self.assertTrue(hasattr(Colors, 'BOLD'))
        self.assertTrue(hasattr(Colors, 'NC'))

        # Verify they are ANSI escape codes
        self.assertTrue(Colors.RED.startswith('\033['))
        self.assertTrue(Colors.NC.startswith('\033['))


class TestS3Utilities(unittest.TestCase):
    """Test S3 utilities with mocks."""

    def test_s3_file_exists_true(self):
        """Test s3_file_exists returns True when file exists."""
        from utils.s3 import s3_file_exists

        mock_client = MagicMock()
        mock_client.head_object.return_value = {'ContentLength': 1000}

        result = s3_file_exists(mock_client, 'bucket', 'key')
        self.assertTrue(result)
        mock_client.head_object.assert_called_once_with(Bucket='bucket', Key='key')

    def test_s3_file_exists_false(self):
        """Test s3_file_exists returns False when file doesn't exist."""
        from utils.s3 import s3_file_exists, ClientError

        mock_client = MagicMock()
        mock_client.head_object.side_effect = ClientError(
            {'Error': {'Code': '404'}}, 'HeadObject'
        )

        result = s3_file_exists(mock_client, 'bucket', 'key')
        self.assertFalse(result)

    def test_s3_file_exists_size_match(self):
        """Test s3_file_exists checks size when provided."""
        from utils.s3 import s3_file_exists

        mock_client = MagicMock()
        mock_client.head_object.return_value = {'ContentLength': 1000}

        # Size matches
        result = s3_file_exists(mock_client, 'bucket', 'key', local_size=1000)
        self.assertTrue(result)

        # Size doesn't match
        result = s3_file_exists(mock_client, 'bucket', 'key', local_size=500)
        self.assertFalse(result)

    def test_upload_file_dry_run(self):
        """Test upload_file in dry run mode."""
        from utils.s3 import upload_file

        mock_client = MagicMock()
        result = upload_file(mock_client, '/path/to/file', 'bucket', 'key', dry_run=True)

        self.assertTrue(result)
        mock_client.upload_file.assert_not_called()

    def test_download_file_dry_run(self):
        """Test download_file in dry run mode."""
        from utils.s3 import download_file

        mock_client = MagicMock()
        result = download_file(mock_client, 'bucket', 'key', '/path/to/file', dry_run=True)

        self.assertTrue(result)
        mock_client.download_file.assert_not_called()

    def test_list_s3_files(self):
        """Test list_s3_files returns correct format."""
        from utils.s3 import list_s3_files
        from datetime import datetime

        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                'Contents': [
                    {'Key': 'file1.txt', 'Size': 100, 'LastModified': datetime.now()},
                    {'Key': 'file2.txt', 'Size': 200, 'LastModified': datetime.now()},
                ]
            }
        ]

        files = list_s3_files(mock_client, 'bucket', 'prefix/')

        self.assertEqual(len(files), 2)
        self.assertEqual(files[0]['key'], 'file1.txt')
        self.assertEqual(files[0]['size'], 100)
        self.assertEqual(files[1]['key'], 'file2.txt')

    def test_get_s3_client_without_boto3(self):
        """Test get_s3_client raises ImportError when boto3 not available."""
        from utils import s3

        # Temporarily set HAS_BOTO3 to False
        original = s3.HAS_BOTO3
        s3.HAS_BOTO3 = False

        try:
            with self.assertRaises(ImportError):
                s3.get_s3_client()
        finally:
            s3.HAS_BOTO3 = original


class TestTrajectoryPatterns(unittest.TestCase):
    """Test all trajectory patterns."""

    def test_orbit_pattern(self):
        """Test orbit pattern generation."""
        from utils.trajectory import TrajectoryGenerator, TrajectoryConfig

        config = TrajectoryConfig()
        generator = TrajectoryGenerator(config)

        waypoints = generator.generate(num_frames=12, pattern='orbit', agent_type='drone')

        self.assertEqual(len(waypoints), 12)

        # Check positions form roughly circular pattern
        positions = [(w['position']['x'], w['position']['y']) for w in waypoints]
        center_x = sum(p[0] for p in positions) / len(positions)
        center_y = sum(p[1] for p in positions) / len(positions)

        # Center should be near bounds center
        expected_center = (
            (config.bounds[0] + config.bounds[2]) / 2,
            (config.bounds[1] + config.bounds[3]) / 2
        )
        self.assertAlmostEqual(center_x, expected_center[0], delta=50)
        self.assertAlmostEqual(center_y, expected_center[1], delta=50)

    def test_figure8_pattern(self):
        """Test figure-8 pattern generation."""
        from utils.trajectory import TrajectoryGenerator, TrajectoryConfig

        config = TrajectoryConfig()
        generator = TrajectoryGenerator(config)

        waypoints = generator.generate(num_frames=20, pattern='figure8', agent_type='drone')

        self.assertEqual(len(waypoints), 20)

        # All waypoints should have valid positions
        for wp in waypoints:
            self.assertIsInstance(wp['position']['x'], float)
            self.assertIsInstance(wp['position']['y'], float)
            self.assertIsInstance(wp['position']['z'], float)

    def test_rover_trajectory(self):
        """Test rover trajectory stays at ground level."""
        from utils.trajectory import TrajectoryGenerator, TrajectoryConfig

        config = TrajectoryConfig(ground_height=1.5)
        generator = TrajectoryGenerator(config)

        waypoints = generator.generate(num_frames=10, pattern='random', agent_type='rover')

        for wp in waypoints:
            self.assertEqual(wp['position']['z'], config.ground_height)
            self.assertEqual(wp['agent_type'], 'rover')

    def test_drone_trajectory_altitude_range(self):
        """Test drone trajectory stays within altitude range."""
        from utils.trajectory import TrajectoryGenerator, TrajectoryConfig

        config = TrajectoryConfig(altitude_range=(30.0, 100.0))
        generator = TrajectoryGenerator(config)

        waypoints = generator.generate(num_frames=50, pattern='random', agent_type='drone')

        for wp in waypoints:
            z = wp['position']['z']
            self.assertGreaterEqual(z, config.altitude_range[0])
            self.assertLessEqual(z, config.altitude_range[1])

    def test_trajectory_bounds_respected(self):
        """Test trajectory stays within bounds."""
        from utils.trajectory import TrajectoryGenerator, TrajectoryConfig

        config = TrajectoryConfig(bounds=(-100, -100, 100, 100))
        generator = TrajectoryGenerator(config)

        waypoints = generator.generate(num_frames=100, pattern='random', agent_type='drone')

        for wp in waypoints:
            x = wp['position']['x']
            y = wp['position']['y']
            # Allow small buffer for boundary reflection
            self.assertGreaterEqual(x, config.bounds[0] - 10)
            self.assertLessEqual(x, config.bounds[2] + 10)
            self.assertGreaterEqual(y, config.bounds[1] - 10)
            self.assertLessEqual(y, config.bounds[3] + 10)

    def test_waypoint_orientation_quaternion(self):
        """Test waypoint orientations are valid quaternions."""
        from utils.trajectory import TrajectoryGenerator, TrajectoryConfig
        import math

        config = TrajectoryConfig()
        generator = TrajectoryGenerator(config)

        waypoints = generator.generate(num_frames=10, pattern='random', agent_type='drone')

        for wp in waypoints:
            q = wp['orientation']
            # Check quaternion is roughly normalized (x,y should be 0 for yaw-only rotation)
            self.assertEqual(q['x'], 0.0)
            self.assertEqual(q['y'], 0.0)
            magnitude = math.sqrt(q['z']**2 + q['w']**2)
            self.assertAlmostEqual(magnitude, 1.0, places=5)


class TestVoxelConfigEdgeCases(unittest.TestCase):
    """Test voxel configuration edge cases."""

    def test_custom_voxel_config(self):
        """Test creating custom voxel configuration."""
        from utils.voxel_config import VoxelConfig

        config = VoxelConfig(
            point_cloud_range=(-20.0, -20.0, -5.0, 20.0, 20.0, 35.0),
            voxel_size=(0.2, 0.2, 0.5)
        )

        grid_size = config.grid_size

        self.assertEqual(grid_size[0], 200)  # (20 - -20) / 0.2
        self.assertEqual(grid_size[1], 200)  # (20 - -20) / 0.2
        self.assertEqual(grid_size[2], 80)   # (35 - -5) / 0.5

    def test_expected_shape_alias(self):
        """Test expected_shape is alias for grid_size."""
        from utils.voxel_config import VoxelConfig

        config = VoxelConfig()
        self.assertEqual(config.expected_shape, config.grid_size)

    def test_voxel_config_immutable_defaults(self):
        """Test that default config values don't change."""
        from utils.voxel_config import DEFAULT_POINT_CLOUD_RANGE, DEFAULT_VOXEL_SIZE

        self.assertEqual(DEFAULT_POINT_CLOUD_RANGE, (-40.0, -40.0, -2.0, 40.0, 40.0, 150.0))
        self.assertEqual(DEFAULT_VOXEL_SIZE, (0.4, 0.4, 1.25))


class TestDependencyFlags(unittest.TestCase):
    """Test dependency flag behavior."""

    def test_dependency_flags_are_booleans(self):
        """Test all dependency flags are boolean."""
        from utils.dependencies import (
            HAS_BOTO3, HAS_TQDM, HAS_REQUESTS,
            HAS_OPEN3D, HAS_TRIMESH, HAS_CV2,
            HAS_SCIPY, HAS_NUMBA, HAS_TORCH
        )

        flags = [
            HAS_BOTO3, HAS_TQDM, HAS_REQUESTS,
            HAS_OPEN3D, HAS_TRIMESH, HAS_CV2,
            HAS_SCIPY, HAS_NUMBA, HAS_TORCH
        ]

        for flag in flags:
            self.assertIsInstance(flag, bool)

    def test_module_references_match_flags(self):
        """Test module references are None when flag is False."""
        from utils import dependencies

        if not dependencies.HAS_BOTO3:
            self.assertIsNone(dependencies.boto3)

        if not dependencies.HAS_TQDM:
            self.assertIsNone(dependencies.tqdm)

        if not dependencies.HAS_OPEN3D:
            self.assertIsNone(dependencies.o3d)


class TestDirectoriesEdgeCases(unittest.TestCase):
    """Test directory utilities edge cases."""

    def test_create_session_dirs_empty_subdirs(self):
        """Test creating session with no subdirs."""
        from utils.directories import create_session_dirs

        with tempfile.TemporaryDirectory() as tmpdir:
            dirs = create_session_dirs(tmpdir, 'test_session', subdirs=())

            self.assertEqual(len(dirs), 1)
            self.assertIn('root', dirs)
            self.assertTrue(os.path.isdir(dirs['root']))

    def test_create_session_dirs_nested_path(self):
        """Test creating session in nested path."""
        from utils.directories import create_session_dirs

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_base = os.path.join(tmpdir, 'level1', 'level2')
            dirs = create_session_dirs(nested_base, 'test_session')

            self.assertTrue(os.path.isdir(dirs['root']))
            self.assertTrue(os.path.isdir(dirs['images']))

    def test_create_session_dirs_idempotent(self):
        """Test creating session dirs twice doesn't fail."""
        from utils.directories import create_session_dirs

        with tempfile.TemporaryDirectory() as tmpdir:
            dirs1 = create_session_dirs(tmpdir, 'test_session')
            dirs2 = create_session_dirs(tmpdir, 'test_session')

            self.assertEqual(dirs1, dirs2)


if __name__ == '__main__':
    unittest.main()
