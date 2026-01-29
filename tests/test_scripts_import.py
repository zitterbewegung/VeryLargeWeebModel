"""Tests to verify all scripts can be imported without errors.

These tests catch import-time issues like:
- Missing dependencies
- Circular imports
- Syntax errors
- Missing modules
"""

import sys
import os
import unittest
import importlib.util

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_DIR)


def can_import_module(module_path: str) -> tuple:
    """Try to import a module and return (success, error_message)."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        if spec is None or spec.loader is None:
            return False, "Could not load spec"
        module = importlib.util.module_from_spec(spec)
        # Don't execute - just check it can be loaded
        return True, None
    except Exception as e:
        return False, str(e)


class TestUtilsModulesImport(unittest.TestCase):
    """Test that all utils modules can be imported."""

    def test_import_utils_package(self):
        """Test importing the utils package."""
        import utils
        self.assertTrue(hasattr(utils, 'log_info'))
        self.assertTrue(hasattr(utils, 'VoxelConfig'))
        self.assertTrue(hasattr(utils, 'TrajectoryGenerator'))

    def test_import_utils_logging(self):
        """Test importing utils.logging."""
        from utils import logging
        self.assertTrue(hasattr(logging, 'Colors'))
        self.assertTrue(hasattr(logging, 'log_info'))

    def test_import_utils_s3(self):
        """Test importing utils.s3."""
        from utils import s3
        self.assertTrue(hasattr(s3, 'get_s3_client'))
        self.assertTrue(hasattr(s3, 'upload_file'))
        self.assertTrue(hasattr(s3, 'download_directory'))

    def test_import_utils_trajectory(self):
        """Test importing utils.trajectory."""
        from utils import trajectory
        self.assertTrue(hasattr(trajectory, 'TrajectoryGenerator'))
        self.assertTrue(hasattr(trajectory, 'TrajectoryConfig'))

    def test_import_utils_voxel_config(self):
        """Test importing utils.voxel_config."""
        from utils import voxel_config
        self.assertTrue(hasattr(voxel_config, 'VoxelConfig'))
        self.assertTrue(hasattr(voxel_config, 'DEFAULT_VOXEL_CONFIG'))

    def test_import_utils_directories(self):
        """Test importing utils.directories."""
        from utils import directories
        self.assertTrue(hasattr(directories, 'create_session_dirs'))

    def test_import_utils_dependencies(self):
        """Test importing utils.dependencies."""
        from utils import dependencies
        self.assertTrue(hasattr(dependencies, 'HAS_BOTO3'))
        self.assertTrue(hasattr(dependencies, 'HAS_TORCH'))


class TestDatasetModulesImport(unittest.TestCase):
    """Test that dataset modules can be imported."""

    def test_import_dataset_package(self):
        """Test importing the dataset package."""
        try:
            import dataset
            # Check it has expected exports
            self.assertTrue(hasattr(dataset, '__all__'))
        except ImportError as e:
            # Some dependencies may be missing
            if 'torch' in str(e).lower() or 'cv2' in str(e).lower():
                self.skipTest(f"Optional dependency missing: {e}")
            raise

    def test_dataset_package_exports(self):
        """Test that dataset package exports expected classes."""
        try:
            from dataset import (
                GazeboOccWorldDataset,
                UAVScenesDataset,
                NuScenesOccWorldDataset,
            )
        except ImportError as e:
            if 'torch' in str(e).lower() or 'cv2' in str(e).lower():
                self.skipTest(f"Optional dependency missing: {e}")
            raise


class TestNoCircularImports(unittest.TestCase):
    """Test that there are no circular import issues."""

    def test_utils_no_circular_imports(self):
        """Test utils modules don't have circular imports."""
        # Import in various orders to catch circular dependencies
        from utils.logging import log_info
        from utils.dependencies import HAS_BOTO3
        from utils.s3 import get_s3_client
        from utils.voxel_config import VoxelConfig
        from utils.trajectory import TrajectoryGenerator
        from utils.directories import create_session_dirs

        # Re-import in different order
        from utils.directories import create_session_dirs
        from utils.trajectory import TrajectoryConfig
        from utils.voxel_config import DEFAULT_GRID_SIZE
        from utils.s3 import S3_REGION
        from utils.dependencies import HAS_TORCH
        from utils.logging import log_error

        # If we get here, no circular imports
        self.assertTrue(True)

    def test_reimport_utils_package(self):
        """Test reimporting utils package works."""
        import utils
        import importlib

        # Reload should work without issues
        importlib.reload(utils)
        self.assertTrue(hasattr(utils, 'log_info'))


class TestScriptsSyntax(unittest.TestCase):
    """Test that scripts have valid Python syntax."""

    def _check_script_syntax(self, script_path: str):
        """Check a script has valid syntax by compiling it."""
        with open(script_path, 'r') as f:
            source = f.read()
        try:
            compile(source, script_path, 'exec')
            return True
        except SyntaxError as e:
            self.fail(f"Syntax error in {script_path}: {e}")

    def test_utils_scripts_syntax(self):
        """Test all utils scripts have valid syntax."""
        utils_dir = os.path.join(SCRIPTS_DIR, 'utils')
        for filename in os.listdir(utils_dir):
            if filename.endswith('.py'):
                script_path = os.path.join(utils_dir, filename)
                self._check_script_syntax(script_path)

    def test_main_scripts_syntax(self):
        """Test main scripts have valid syntax."""
        scripts_to_check = [
            'gazebo_data_collector.py',
            'plateau_to_occworld.py',
            'uavscenes_s3.py',
            'download_pretrained.py',
            'sanity_check_data.py',
        ]

        for script_name in scripts_to_check:
            script_path = os.path.join(SCRIPTS_DIR, script_name)
            if os.path.exists(script_path):
                self._check_script_syntax(script_path)


class TestConfigConsistency(unittest.TestCase):
    """Test configuration consistency across modules."""

    def test_s3_config_values(self):
        """Test S3 configuration values are valid."""
        from utils.s3 import S3_REGION, DEFAULT_BUCKET

        self.assertIsInstance(S3_REGION, str)
        self.assertIsInstance(DEFAULT_BUCKET, str)
        self.assertTrue(len(S3_REGION) > 0)
        self.assertTrue(len(DEFAULT_BUCKET) > 0)

    def test_voxel_config_values_valid(self):
        """Test voxel config values are physically valid."""
        from utils.voxel_config import DEFAULT_VOXEL_CONFIG

        config = DEFAULT_VOXEL_CONFIG

        # Range should have min < max for all dimensions
        self.assertLess(config.point_cloud_range[0], config.point_cloud_range[3])  # x
        self.assertLess(config.point_cloud_range[1], config.point_cloud_range[4])  # y
        self.assertLess(config.point_cloud_range[2], config.point_cloud_range[5])  # z

        # Voxel sizes should be positive
        for voxel_dim in config.voxel_size:
            self.assertGreater(voxel_dim, 0)

        # Grid size should be positive
        for grid_dim in config.grid_size:
            self.assertGreater(grid_dim, 0)

    def test_trajectory_config_values_valid(self):
        """Test trajectory config values are physically valid."""
        from utils.trajectory import TrajectoryConfig

        config = TrajectoryConfig()

        # Bounds should have min < max
        self.assertLess(config.bounds[0], config.bounds[2])  # x
        self.assertLess(config.bounds[1], config.bounds[3])  # y

        # Altitude range should have min < max
        self.assertLess(config.altitude_range[0], config.altitude_range[1])

        # Speed and min_motion should be positive
        self.assertGreater(config.speed, 0)
        self.assertGreater(config.min_motion, 0)


if __name__ == '__main__':
    unittest.main()
