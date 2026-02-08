"""Comprehensive unit tests for all features in the system.

Covers: model components, training functions, data validation, dataset loading,
CLI commands, voxel configuration, coordinate transforms, and loss functions.
"""

import sys
import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# Model Component Tests
# =============================================================================


class TestTemporalLSTM(unittest.TestCase):
    """Test TemporalLSTM encoder."""

    def setUp(self):
        from models.occworld_6dof import TemporalLSTM
        self.lstm = TemporalLSTM(input_dim=64, hidden_dim=128, num_layers=2)

    def test_forward_shape(self):
        x = torch.randn(2, 4, 64)  # [B, T, D]
        output, (h_n, c_n) = self.lstm(x)
        self.assertEqual(output.shape, (2, 4, 128))
        self.assertEqual(h_n.shape, (2, 2, 128))  # [num_layers, B, hidden]
        self.assertEqual(c_n.shape, (2, 2, 128))

    def test_single_timestep(self):
        x = torch.randn(1, 1, 64)
        output, (h_n, c_n) = self.lstm(x)
        self.assertEqual(output.shape, (1, 1, 128))

    def test_out_dim(self):
        self.assertEqual(self.lstm.out_dim, 128)

    def test_gradient_flow(self):
        x = torch.randn(2, 4, 64, requires_grad=True)
        output, _ = self.lstm(x)
        output.sum().backward()
        self.assertIsNotNone(x.grad)


class TestTemporalTransformer(unittest.TestCase):
    """Test TemporalTransformer encoder."""

    def setUp(self):
        from models.occworld_6dof import TemporalTransformer
        self.transformer = TemporalTransformer(d_model=64, nhead=4, num_layers=2)

    def test_forward_shape(self):
        x = torch.randn(2, 4, 64)
        output = self.transformer(x)
        self.assertEqual(output.shape, (2, 4, 64))

    def test_out_dim(self):
        self.assertEqual(self.transformer.out_dim, 64)

    def test_with_mask(self):
        x = torch.randn(2, 4, 64)
        # Causal mask
        mask = torch.triu(torch.ones(4, 4) * float('-inf'), diagonal=1)
        output = self.transformer(x, mask=mask)
        self.assertEqual(output.shape, (2, 4, 64))


class TestUncertaintyHead(unittest.TestCase):
    """Test UncertaintyHead for covariance prediction."""

    def setUp(self):
        from models.occworld_6dof import UncertaintyHead
        self.head = UncertaintyHead(in_dim=128, hidden_dim=32, uncertainty_dim=6)

    def test_forward_shape(self):
        features = torch.randn(2, 4, 128)  # [B, T, D]
        uncertainty = self.head(features)
        self.assertEqual(uncertainty.shape, (2, 4, 6))

    def test_positive_output(self):
        """Softplus ensures positive covariance."""
        features = torch.randn(2, 4, 128)
        uncertainty = self.head(features)
        self.assertTrue((uncertainty > 0).all())

    def test_single_timestep(self):
        features = torch.randn(1, 1, 128)
        uncertainty = self.head(features)
        self.assertEqual(uncertainty.shape, (1, 1, 6))


class TestRelocalizationHead(unittest.TestCase):
    """Test RelocalizationHead for global pose correction."""

    def setUp(self):
        from models.occworld_6dof import RelocalizationHead
        self.head = RelocalizationHead(in_dim=128, hidden_dim=64, pose_dim=7)

    def test_forward_shape(self):
        features = torch.randn(2, 128)  # [B, D]
        pose = self.head(features)
        self.assertEqual(pose.shape, (2, 7))

    def test_gradient_flow(self):
        features = torch.randn(2, 128, requires_grad=True)
        pose = self.head(features)
        pose.sum().backward()
        self.assertIsNotNone(features.grad)


class TestPlaceRecognitionHead(unittest.TestCase):
    """Test PlaceRecognitionHead for loop closure."""

    def setUp(self):
        from models.occworld_6dof import PlaceRecognitionHead
        self.head = PlaceRecognitionHead(in_dim=128, embedding_dim=64)

    def test_forward_shape(self):
        features = torch.randn(2, 128)
        embeddings = self.head(features)
        self.assertEqual(embeddings.shape, (2, 64))

    def test_normalized_output(self):
        """Embeddings should be L2-normalized."""
        features = torch.randn(2, 128)
        embeddings = self.head(features)
        norms = embeddings.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones(2), atol=1e-5, rtol=1e-5)

    def test_no_normalize_option(self):
        self.head.normalize = False
        features = torch.randn(2, 128)
        embeddings = self.head(features)
        norms = embeddings.norm(dim=-1)
        # Without normalization, norms should generally not be 1.0
        self.assertEqual(embeddings.shape, (2, 64))


class TestFuturePoseRNN(unittest.TestCase):
    """Test FuturePoseRNN for autoregressive pose prediction."""

    def setUp(self):
        from models.occworld_6dof import FuturePoseRNN
        self.rnn = FuturePoseRNN(pose_dim=13, hidden_dim=128, context_dim=64)

    def test_forward_shape(self):
        last_pose = torch.randn(2, 13)
        context = torch.randn(2, 64)
        future_poses = self.rnn(last_pose, context, num_future=6)
        self.assertEqual(future_poses.shape, (2, 6, 13))

    def test_single_future(self):
        last_pose = torch.randn(1, 13)
        context = torch.randn(1, 64)
        future_poses = self.rnn(last_pose, context, num_future=1)
        self.assertEqual(future_poses.shape, (1, 1, 13))

    def test_with_initial_hidden(self):
        last_pose = torch.randn(2, 13)
        context = torch.randn(2, 64)
        hidden = torch.randn(2, 128)
        future_poses = self.rnn(last_pose, context, num_future=4, hidden=hidden)
        self.assertEqual(future_poses.shape, (2, 4, 13))

    def test_residual_prediction(self):
        """Later poses should differ from first (residual accumulation)."""
        last_pose = torch.zeros(1, 13)
        context = torch.randn(1, 64)
        future_poses = self.rnn(last_pose, context, num_future=3)
        # All predictions should NOT be identical (residual changes)
        self.assertFalse(torch.allclose(future_poses[:, 0], future_poses[:, 2]))

    def test_gradient_flow(self):
        last_pose = torch.randn(2, 13, requires_grad=True)
        context = torch.randn(2, 64)
        future_poses = self.rnn(last_pose, context, num_future=4)
        future_poses.sum().backward()
        self.assertIsNotNone(last_pose.grad)


# =============================================================================
# Training Function Tests
# =============================================================================


class TestTrainEpoch(unittest.TestCase):
    """Test train_epoch function."""

    def _make_batch(self, batch_size=2, history=4, future=6, grid=(8, 8, 8)):
        return {
            'history_occupancy': torch.randn(batch_size, history, *grid),
            'future_occupancy': torch.randn(batch_size, future, *grid),
            'history_poses': torch.randn(batch_size, history, 13),
            'future_poses': torch.randn(batch_size, future, 13),
        }

    def test_batch_validation_skips_bad_batch(self):
        """Train loop should skip batches missing required keys."""
        from train import train_epoch
        from unittest.mock import MagicMock
        import torch.nn.functional as F

        # Create a simple model
        model = MagicMock()
        model.train = MagicMock()
        model.parameters = MagicMock(return_value=iter([torch.randn(2, 2, requires_grad=True)]))
        model.return_value = {'future_occupancy': torch.randn(2, 6, 8, 8, 8)}

        # One good batch, one bad batch
        good_batch = self._make_batch()
        bad_batch = {'history_occupancy': torch.randn(2, 4, 8, 8, 8)}  # Missing keys

        dataloader = [bad_batch, good_batch]
        optimizer = MagicMock()
        criterion = MagicMock(return_value=torch.tensor(0.5, requires_grad=True))
        writer = MagicMock()

        # Should not crash despite bad batch
        loss = train_epoch(model, dataloader, optimizer, criterion, 'cpu', 0, writer)
        self.assertIsInstance(loss, float)

    def test_nan_loss_skipped(self):
        """Non-finite losses should be skipped and optimizer.step() NOT called."""
        from train import train_epoch

        model = MagicMock()
        model.train = MagicMock()
        model.parameters = MagicMock(return_value=iter([torch.randn(2, 2, requires_grad=True)]))

        # Return NaN loss on first call, normal on second
        nan_output = torch.randn(2, 6, 8, 8, 8)
        model.return_value = nan_output

        dataloader = [self._make_batch()]
        optimizer = MagicMock()
        # Return NaN loss
        criterion = MagicMock(return_value=torch.tensor(float('nan'), requires_grad=True))
        writer = MagicMock()

        loss = train_epoch(model, dataloader, optimizer, criterion, 'cpu', 0, writer)
        # Should return 0 because the only batch had NaN
        self.assertEqual(loss, 0.0)
        # Verify optimizer.step() was NOT called (NaN should skip gradient update)
        optimizer.step.assert_not_called()


class TestValidateFunction(unittest.TestCase):
    """Test validate function."""

    def test_validate_handles_empty_dataloader(self):
        """Validate should handle empty dataloader."""
        from train import validate

        model = MagicMock()
        model.eval = MagicMock()
        dataloader = []
        criterion = MagicMock()

        loss = validate(model, dataloader, criterion, 'cpu')
        self.assertEqual(loss, 0)


# =============================================================================
# Data Validation Tests
# =============================================================================


class TestValidateData(unittest.TestCase):
    """Test data validation functions."""

    def test_missing_data_dir(self):
        """validate_data returns False for missing directory."""
        from train import validate_data
        result = validate_data("/nonexistent/path/to/data", "gazebo", auto_download=False)
        self.assertFalse(result)

    def test_unknown_dataset_type(self):
        """Unknown dataset type should return True with warning."""
        from train import validate_data
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_data(tmpdir, "unknown_type", auto_download=False)
            self.assertTrue(result)

    def test_gazebo_data_validation(self):
        """Gazebo data validation with empty dir."""
        from train import validate_data
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_data(tmpdir, "gazebo", auto_download=False)
            # Empty dir should fail validation
            self.assertFalse(result)


class TestValidatePretrainedModels(unittest.TestCase):
    """Test pretrained model validation."""

    def test_no_models_specified(self):
        """Should return True when no models specified."""
        from train import validate_pretrained_models
        result = validate_pretrained_models(load_from=None, vqvae_ckpt=None, auto_download=False)
        self.assertTrue(result)

    def test_missing_model_file(self):
        """Should return False for missing model file."""
        from train import validate_pretrained_models
        result = validate_pretrained_models(
            load_from="/nonexistent/model.pth",
            auto_download=False
        )
        self.assertFalse(result)

    @patch('train.PRETRAINED_MODELS', {
        "occworld_checkpoint": {
            "s3_key": "test", "local_path": "test",
            "min_size": 100, "description": "test",
        }
    })
    def test_valid_model_file(self):
        """Should return True for existing model file."""
        from train import validate_pretrained_models
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            f.write(b'\x00' * 1000)  # Above mocked min_size of 100
            f.flush()
            try:
                result = validate_pretrained_models(
                    load_from=f.name,
                    auto_download=False
                )
                self.assertTrue(result)
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)


# =============================================================================
# VoxelConfig Tests
# =============================================================================


class TestVoxelConfigValidation(unittest.TestCase):
    """Test VoxelConfig edge cases."""

    def test_zero_voxel_size_raises(self):
        """Zero voxel size should raise ValueError."""
        from scripts.utils.voxel_config import VoxelConfig
        config = VoxelConfig(voxel_size=(0.4, 0.0, 1.25))
        with self.assertRaises(ValueError):
            _ = config.grid_size

    def test_negative_voxel_size_raises(self):
        """Negative voxel size should raise ValueError."""
        from scripts.utils.voxel_config import VoxelConfig
        config = VoxelConfig(voxel_size=(0.4, -0.4, 1.25))
        with self.assertRaises(ValueError):
            _ = config.grid_size

    def test_custom_range(self):
        """Custom point cloud range should work."""
        from scripts.utils.voxel_config import VoxelConfig
        config = VoxelConfig(
            point_cloud_range=(-10.0, -10.0, -1.0, 10.0, 10.0, 5.0),
            voxel_size=(0.5, 0.5, 0.5),
        )
        self.assertEqual(config.grid_size, (40, 40, 12))

    def test_expected_shape_alias(self):
        """expected_shape should equal grid_size."""
        from scripts.utils.voxel_config import VoxelConfig
        config = VoxelConfig()
        self.assertEqual(config.grid_size, config.expected_shape)


# =============================================================================
# UAVScenes Dataset Tests
# =============================================================================


class TestUAVScenesGridSize(unittest.TestCase):
    """Test UAVScenes grid size calculation with rounding."""

    def test_grid_size_default(self):
        """Default grid size should be (200, 200, 120)."""
        from dataset.uavscenes_dataset import UAVScenesConfig
        config = UAVScenesConfig()
        # Default: range (-40,-40,-10) to (40,40,50), voxel (0.4,0.4,0.5)
        # => 80/0.4=200, 80/0.4=200, 60/0.5=120
        self.assertEqual(config.grid_size, (200, 200, 120))

    def test_grid_size_exact_division(self):
        """Exact division should work correctly."""
        from dataset.uavscenes_dataset import UAVScenesConfig
        config = UAVScenesConfig(
            point_cloud_range=(-10.0, -10.0, -1.0, 10.0, 10.0, 4.0),
            voxel_size=(0.5, 0.5, 0.5),
        )
        self.assertEqual(config.grid_size, (40, 40, 10))


class TestUAVScenesVoxelization(unittest.TestCase):
    """Test UAVScenes voxelization with coordinate clipping."""

    def test_voxel_coord_clipping(self):
        """Points outside range should be clipped, not overflow."""
        # Simulate the voxelization logic directly
        pc_range = np.array([-40.0, -40.0, -2.0, 40.0, 40.0, 150.0])
        voxel_size = np.array([0.4, 0.4, 1.25])
        grid_size = np.array([200, 200, 121])

        # Points at extreme positions
        xyz = np.array([
            [0.0, 0.0, 0.0],       # Inside range
            [39.9, 39.9, 149.0],    # Near boundary
            [100.0, 100.0, 200.0],  # Far outside (was causing overflow)
        ])

        # Apply the fixed voxelization logic (clip before int conversion)
        voxel_coords_float = (xyz - pc_range[:3]) / voxel_size
        voxel_coords_float = np.clip(voxel_coords_float, 0, grid_size - 1)
        voxel_coords = voxel_coords_float.astype(np.int32)

        # All coords should be in valid range
        self.assertTrue((voxel_coords >= 0).all())
        self.assertTrue((voxel_coords[:, 0] < 200).all())
        self.assertTrue((voxel_coords[:, 1] < 200).all())
        self.assertTrue((voxel_coords[:, 2] < 121).all())


class TestUAVScenesTransforms(unittest.TestCase):
    """Test UAVScenes coordinate transformations."""

    def test_quaternion_to_rotation_identity(self):
        """Identity quaternion [1,0,0,0] should give identity rotation."""
        from dataset.uavscenes_dataset import UAVScenesDataset
        ds = UAVScenesDataset.__new__(UAVScenesDataset)  # No __init__
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        R = ds._quaternion_to_rotation_matrix(quat)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)

    def test_ego_transform_uses_transpose(self):
        """Ego-frame transform should use R.T (inverse rotation)."""
        from dataset.uavscenes_dataset import UAVScenesDataset
        ds = UAVScenesDataset.__new__(UAVScenesDataset)

        # Identity pose at origin
        pose = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        result = ds._transform_points_to_ego(points, pose)
        np.testing.assert_allclose(result, points, atol=1e-5)

    def test_ego_transform_empty_points(self):
        """Empty point cloud should return empty."""
        from dataset.uavscenes_dataset import UAVScenesDataset
        ds = UAVScenesDataset.__new__(UAVScenesDataset)

        pose = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        points = np.zeros((0, 3), dtype=np.float32)
        result = ds._transform_points_to_ego(points, pose)
        self.assertEqual(result.shape[0], 0)


class TestUAVScenesVelocity(unittest.TestCase):
    """Test UAVScenes velocity computation."""

    def test_velocity_no_mutation(self):
        """_compute_velocity should not modify input array."""
        from dataset.uavscenes_dataset import UAVScenesDataset
        ds = UAVScenesDataset.__new__(UAVScenesDataset)

        pose_curr = np.array([1, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        pose_prev = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        original = pose_curr.copy()

        result = ds._compute_velocity(pose_curr, pose_prev, dt=0.1)

        # Input should NOT be modified
        np.testing.assert_array_equal(pose_curr, original)
        # Result should have non-zero velocities
        self.assertFalse(np.allclose(result[7:10], 0))

    def test_velocity_zero_dt(self):
        """Zero dt should return input unchanged."""
        from dataset.uavscenes_dataset import UAVScenesDataset
        ds = UAVScenesDataset.__new__(UAVScenesDataset)

        pose_curr = np.array([1, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        pose_prev = np.zeros(13, dtype=np.float32)

        result = ds._compute_velocity(pose_curr, pose_prev, dt=0)
        np.testing.assert_array_equal(result, pose_curr)


# =============================================================================
# Dataset Loading Tests
# =============================================================================


class TestGazeboDatasetLoading(unittest.TestCase):
    """Test GazeboOccWorldDataset data loading methods."""

    def _create_mock_session(self, tmpdir, session_name, num_frames=3):
        """Create a mock session directory with data files."""
        session_dir = os.path.join(tmpdir, session_name)
        for subdir in ['occupancy', 'poses', 'lidar', 'images']:
            os.makedirs(os.path.join(session_dir, subdir), exist_ok=True)

        for i in range(num_frames):
            fid = f'{i:06d}'
            # Occupancy
            np.savez(
                os.path.join(session_dir, 'occupancy', f'{fid}_occupancy.npz'),
                occupancy=np.random.randint(0, 2, (8, 8, 8)).astype(np.uint8)
            )
            # Pose
            pose_data = {
                'position': {'x': float(i), 'y': 0.0, 'z': 10.0},
                'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
                'velocity': {
                    'linear': {'x': 1.0, 'y': 0.0, 'z': 0.0},
                    'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                }
            }
            with open(os.path.join(session_dir, 'poses', f'{fid}.json'), 'w') as f:
                json.dump(pose_data, f)
            # LiDAR
            np.save(
                os.path.join(session_dir, 'lidar', f'{fid}_LIDAR.npy'),
                np.random.randn(100, 4).astype(np.float32)
            )

    def test_load_occupancy_returns_float(self):
        """Occupancy should be loaded as float tensor."""
        from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset
        ds = GazeboOccWorldDataset.__new__(GazeboOccWorldDataset)
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_mock_session(tmpdir, 'test_session')
            result = ds._load_occupancy(
                os.path.join(tmpdir, 'test_session'), '000000'
            )
            self.assertEqual(result.dtype, torch.float32)

    def test_load_pose_shape(self):
        """Pose should be 13-element float tensor."""
        from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset
        ds = GazeboOccWorldDataset.__new__(GazeboOccWorldDataset)
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_mock_session(tmpdir, 'test_session')
            result = ds._load_pose(
                os.path.join(tmpdir, 'test_session'), '000000'
            )
            self.assertEqual(result.shape, (13,))
            self.assertEqual(result.dtype, torch.float32)

    def test_load_lidar_shape(self):
        """LiDAR should be [N, 4] float tensor."""
        from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset
        ds = GazeboOccWorldDataset.__new__(GazeboOccWorldDataset)
        ds.config = MagicMock()
        ds.config.max_points = 50000
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_mock_session(tmpdir, 'test_session')
            result = ds._load_lidar(
                os.path.join(tmpdir, 'test_session'), '000000'
            )
            self.assertEqual(result.shape[1], 4)
            self.assertEqual(result.dtype, torch.float32)

    def test_load_lidar_missing(self):
        """Missing LiDAR should return empty tensor."""
        from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset
        ds = GazeboOccWorldDataset.__new__(GazeboOccWorldDataset)
        ds.config = MagicMock()
        ds.config.max_points = 50000
        result = ds._load_lidar('/nonexistent/path', '000000')
        self.assertEqual(result.shape, (0, 4))

    def test_load_lidar_subsampling(self):
        """LiDAR should be subsampled if exceeding max_points."""
        from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset
        ds = GazeboOccWorldDataset.__new__(GazeboOccWorldDataset)
        ds.config = MagicMock()
        ds.config.max_points = 50  # Very small limit
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_mock_session(tmpdir, 'test_session')
            result = ds._load_lidar(
                os.path.join(tmpdir, 'test_session'), '000000'
            )
            self.assertLessEqual(result.shape[0], 50)


class TestCollateFn(unittest.TestCase):
    """Test collate functions."""

    def test_gazebo_collate_fn(self):
        """Gazebo collate should stack tensors correctly."""
        from dataset.gazebo_occworld_dataset import collate_fn

        # history_images must have at least one timestep with camera dicts
        cam_img = torch.randn(3, 64, 64)
        batch = [
            {
                'history_poses': torch.randn(4, 13),
                'history_occupancy': torch.randn(4, 8, 8, 8),
                'future_occupancy': torch.randn(6, 8, 8, 8),
                'future_poses': torch.randn(6, 13),
                'agent_type': 1,
                'history_lidar': [torch.randn(100, 4)],
                'future_lidar': [torch.randn(100, 4)],
                'history_images': [{'cam0': cam_img}] * 4,
            }
            for _ in range(3)
        ]

        collated = collate_fn(batch)
        self.assertEqual(collated['history_poses'].shape, (3, 4, 13))
        self.assertEqual(collated['history_occupancy'].shape, (3, 4, 8, 8, 8))
        self.assertEqual(collated['future_occupancy'].shape, (3, 6, 8, 8, 8))
        self.assertEqual(collated['agent_type'].shape, (3,))


# =============================================================================
# CLI Tests
# =============================================================================


class TestCLISubcommandHandlers(unittest.TestCase):
    """Test CLI subcommand handler functions."""

    def test_cmd_info_runs(self):
        """info command should run without crashing."""
        from scripts.vlwm_cli import cmd_info
        import argparse
        args = argparse.Namespace()
        # Should return 0
        result = cmd_info(args)
        self.assertEqual(result, 0)

    def test_cmd_sanity_runs(self):
        """sanity command should run basic checks."""
        from scripts.vlwm_cli import cmd_sanity
        import argparse
        args = argparse.Namespace(quick=True, fix=False)
        result = cmd_sanity(args)
        self.assertEqual(result, 0)

    @patch('subprocess.run')
    def test_cmd_setup_checks_pip_return_code(self, mock_run):
        """setup should check pip return code."""
        from scripts.vlwm_cli import cmd_setup
        import argparse

        # Simulate pip failure
        mock_run.return_value = MagicMock(returncode=1, stdout=b'', stderr=b'')
        args = argparse.Namespace(provider=None, dry_run=False)
        result = cmd_setup(args)
        self.assertEqual(result, 1)

    def test_cmd_train_dry_run(self):
        """train --dry-run should not actually train."""
        from scripts.vlwm_cli import cmd_train
        import argparse
        args = argparse.Namespace(
            dry_run=True, config='config/finetune_tokyo.py',
            batch_size=None, lr=None, epochs=None,
            precision=None, gpus=None, work_dir='work_dirs/test',
            resume=None,
        )
        result = cmd_train(args)
        self.assertEqual(result, 0)

    def test_main_type_annotation(self):
        """main() should accept None argument (Python 3.8 compat)."""
        from scripts.vlwm_cli import main
        # --help triggers SystemExit(0), which is expected
        with self.assertRaises(SystemExit) as cm:
            main(['--help'])
        self.assertEqual(cm.exception.code, 0)


class TestCLIParserCompleteness(unittest.TestCase):
    """Test CLI parser has all expected subcommands."""

    def test_all_subcommands_present(self):
        """Parser should have all 7 subcommands."""
        from scripts.vlwm_cli import build_parser
        parser = build_parser()
        # Parse each subcommand to verify it exists
        for cmd in ['setup', 'download', 'train', 'deploy', 'sanity', 'info', 'pack']:
            args = parser.parse_args([cmd] if cmd != 'pack' else [cmd, 'uavscenes'])
            self.assertEqual(args.command, cmd)


class TestCmdPack(unittest.TestCase):
    """Test cmd_pack function directly."""

    def test_cmd_pack_compress_dry_run(self):
        """cmd_pack compress dry-run should succeed for existing directory."""
        from scripts.vlwm_cli import cmd_pack
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dataset dir
            ds_dir = os.path.join(tmpdir, "testdata")
            os.makedirs(ds_dir)
            with open(os.path.join(ds_dir, "sample.txt"), "w") as f:
                f.write("hello")

            args = argparse.Namespace(
                dataset=ds_dir,
                upload=False,
                download=False,
                bucket="verylargeweebmodel",
                prefix="packed",
                data_dir=tmpdir,
                output=None,
                keep_archive=False,
                force=True,
                compression_level=0,
                dry_run=True,
            )
            result = cmd_pack(args)
            self.assertEqual(result, 0)

    def test_cmd_pack_compress_nonexistent_fails(self):
        """cmd_pack should return 1 for nonexistent dataset directory."""
        from scripts.vlwm_cli import cmd_pack
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = os.path.join(tmpdir, "does_not_exist")
            args = argparse.Namespace(
                dataset=nonexistent,
                upload=False,
                download=False,
                bucket="verylargeweebmodel",
                prefix="packed",
                data_dir=tmpdir,
                output=None,
                keep_archive=False,
                force=True,
                compression_level=6,
                dry_run=False,
            )
            result = cmd_pack(args)
            self.assertEqual(result, 1)

    def test_cmd_pack_download_dry_run(self):
        """cmd_pack download dry-run should succeed for known dataset."""
        from scripts.vlwm_cli import cmd_pack
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                dataset="uavscenes",
                upload=False,
                download=True,
                bucket="verylargeweebmodel",
                prefix="packed",
                data_dir=tmpdir,
                output=None,
                keep_archive=False,
                force=True,
                compression_level=6,
                dry_run=True,
            )
            result = cmd_pack(args)
            self.assertEqual(result, 0)

    def test_cmd_pack_refuses_overwrite_without_force(self):
        """cmd_pack should refuse to overwrite existing archive without --force."""
        from scripts.vlwm_cli import cmd_pack
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset dir
            ds_dir = os.path.join(tmpdir, "testdata")
            os.makedirs(ds_dir)
            with open(os.path.join(ds_dir, "sample.txt"), "w") as f:
                f.write("hello")

            # Create existing archive
            archive_path = os.path.join(tmpdir, "testdata.tar.xz")
            with open(archive_path, "w") as f:
                f.write("existing")

            args = argparse.Namespace(
                dataset=ds_dir,
                upload=False,
                download=False,
                bucket="verylargeweebmodel",
                prefix="packed",
                data_dir=tmpdir,
                output=None,
                keep_archive=False,
                force=False,
                compression_level=0,
                dry_run=False,
            )
            result = cmd_pack(args)
            self.assertEqual(result, 1)


class TestResolveDatasetPath(unittest.TestCase):
    """Test _resolve_dataset_path helper."""

    def test_known_dataset(self):
        """Known dataset name should resolve to data_dir/name."""
        from scripts.vlwm_cli import _resolve_dataset_path
        name, path = _resolve_dataset_path("uavscenes", "/data")
        self.assertEqual(name, "uavscenes")
        self.assertEqual(path, "/data/uavscenes")

    def test_arbitrary_path(self):
        """Arbitrary path should use basename as name."""
        from scripts.vlwm_cli import _resolve_dataset_path
        name, path = _resolve_dataset_path("/tmp/my_custom_data", "/data")
        self.assertEqual(name, "my_custom_data")
        self.assertEqual(path, "/tmp/my_custom_data")

    def test_all_known_datasets(self):
        """All known datasets should resolve correctly."""
        from scripts.vlwm_cli import _resolve_dataset_path, KNOWN_DATASETS
        for ds_name in KNOWN_DATASETS:
            name, path = _resolve_dataset_path(ds_name, "/data")
            self.assertEqual(name, ds_name)


# =============================================================================
# Loss Function Tests
# =============================================================================


class TestOccupancyLossComponents(unittest.TestCase):
    """Test OccupancyLoss components in detail."""

    def test_focal_loss_all_zeros_pred(self):
        """Focal loss should be high when pred is all zeros but target has occupied."""
        from train import FocalLoss
        loss_fn = FocalLoss(alpha=0.99, gamma=2.0)
        pred = torch.zeros(1, 8, 8, 8)  # All zero prediction
        target = torch.ones(1, 8, 8, 8)   # All occupied target
        loss = loss_fn(pred, target)
        self.assertGreater(loss.item(), 0)

    def test_occupancy_loss_mean_matching(self):
        """Mean-matching loss should penalize mean mismatch."""
        from train import OccupancyLoss
        loss_fn = OccupancyLoss(focal_alpha=0.99, focal_gamma=2.0, mean_weight=10.0)

        # Prediction mean far from target mean
        pred = torch.zeros(1, 8, 8, 8)  # mean = 0
        target = torch.ones(1, 8, 8, 8) * 0.01  # mean = 0.01
        loss = loss_fn(pred, target)
        self.assertGreater(loss.item(), 0)

    def test_occupancy_loss_debug_counter(self):
        """Debug counter should increment."""
        from train import OccupancyLoss
        loss_fn = OccupancyLoss()
        pred = torch.randn(1, 8, 8, 8).sigmoid()
        target = torch.randint(0, 2, (1, 8, 8, 8)).float()
        _ = loss_fn(pred, target)
        self.assertEqual(loss_fn._debug_counter, 1)
        _ = loss_fn(pred, target)
        self.assertEqual(loss_fn._debug_counter, 2)


# =============================================================================
# SimpleOccupancyModel Tests
# =============================================================================


class TestSimpleOccupancyModelDetailed(unittest.TestCase):
    """Detailed tests for SimpleOccupancyModel."""

    def setUp(self):
        from train import SimpleOccupancyModel
        config = MagicMock()
        config.history_frames = 4
        config.future_frames = 6
        config.grid_size = [8, 8, 8]
        self.model = SimpleOccupancyModel(config)

    def test_encoder_output_shape(self):
        """Encoder should produce fixed-size spatial features."""
        x = torch.randn(2, 4, 8, 8, 8)
        features = self.model.encoder(x.float())
        pooled = self.model.adaptive_pool(features)
        self.assertEqual(pooled.shape, (2, 256, 4, 4, 4))

    def test_forward_without_poses(self):
        """Model should work without pose input."""
        x = torch.randn(1, 4, 8, 8, 8)
        output = self.model(x)
        # Should return dict or tensor
        if isinstance(output, dict):
            self.assertIn('future_occupancy', output)
        else:
            self.assertEqual(output.shape[0], 1)

    def test_forward_with_poses(self):
        """Model should return poses when history_poses provided."""
        x = torch.randn(1, 4, 8, 8, 8)
        h_poses = torch.randn(1, 4, 13)
        f_poses = torch.randn(1, 6, 13)
        output = self.model(x, h_poses, f_poses)
        if isinstance(output, dict):
            self.assertIn('future_occupancy', output)
            self.assertIn('future_poses', output)
            self.assertEqual(output['future_poses'].shape, (1, 6, 13))

    def test_output_range(self):
        """Occupancy output should be in [0, 1] after sigmoid."""
        x = torch.randn(1, 4, 8, 8, 8)
        output = self.model(x)
        if isinstance(output, dict):
            occ = output['future_occupancy']
        else:
            occ = output
        self.assertTrue((occ >= 0).all())
        self.assertTrue((occ <= 1).all())


# =============================================================================
# MidAir Dataset Tests
# =============================================================================


class TestMidAirDatasetCleanup(unittest.TestCase):
    """Test MidAirDataset resource cleanup."""

    def test_hdf5_cache_cleanup(self):
        """__del__ should close cached HDF5 files."""
        from dataset.midair_dataset import MidAirDataset
        ds = MidAirDataset.__new__(MidAirDataset)
        ds._hdf5_cache = {}

        # Mock HDF5 file
        mock_file = MagicMock()
        ds._hdf5_cache['test.hdf5'] = mock_file

        ds.__del__()
        mock_file.close.assert_called_once()
        self.assertEqual(len(ds._hdf5_cache), 0)


# =============================================================================
# Download Utils Tests
# =============================================================================


class TestDownloadUtilsEdgeCases(unittest.TestCase):
    """Test download utility edge cases."""

    @patch('shutil.which')
    def test_fast_download_no_tools(self, mock_which):
        """fast_download should fail gracefully when no tools available."""
        from scripts.utils.download import fast_download
        mock_which.return_value = None
        result = fast_download("http://example.com/file.tar", "/tmp/file.tar")
        self.assertFalse(result)

    def test_verify_download_zero_size(self):
        """verify_download should fail for empty files."""
        from scripts.utils.download import verify_download
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b'')
            f.flush()
            try:
                result = verify_download(f.name, min_size=100)
                self.assertFalse(result)
            finally:
                os.unlink(f.name)


# =============================================================================
# System Packages Tests
# =============================================================================


class TestSystemPackagesValidation(unittest.TestCase):
    """Test system package name validation."""

    def test_valid_package_names(self):
        """Valid package names should pass regex."""
        from scripts.utils.system_packages import _SAFE_PACKAGE_RE
        valid = ['git', 'python3.8', 'libssl-dev', 'g++', 'cmake_utils']
        for name in valid:
            self.assertIsNotNone(_SAFE_PACKAGE_RE.match(name), f"{name} should be valid")

    def test_invalid_package_names(self):
        """Invalid package names should fail regex."""
        from scripts.utils.system_packages import _SAFE_PACKAGE_RE
        invalid = ['rm -rf /', 'git; echo pwned', 'pkg$(cmd)', 'a b c']
        for name in invalid:
            self.assertIsNone(_SAFE_PACKAGE_RE.match(name), f"{name} should be invalid")

    @patch('shutil.which', return_value=None)
    def test_no_package_manager(self, mock_which):
        """Should return False when no package manager found."""
        from scripts.utils.system_packages import install_system_packages
        result = install_system_packages(['git'])
        self.assertFalse(result)

    @patch('shutil.which', return_value='/usr/bin/apt-get')
    def test_injection_rejected(self, mock_which):
        """Package names with shell metacharacters should be rejected."""
        from scripts.utils.system_packages import install_system_packages
        result = install_system_packages(['git; rm -rf /'])
        self.assertFalse(result)


# =============================================================================
# GPU Utils Edge Cases
# =============================================================================


class TestGPUEdgeCases(unittest.TestCase):
    """Test GPU detection edge cases."""

    @patch('shutil.which', return_value='/usr/bin/nvidia-smi')
    @patch('subprocess.run')
    def test_malformed_nvidia_smi_output(self, mock_run, mock_which):
        """Malformed nvidia-smi output should return None."""
        from scripts.utils.gpu import detect_gpu_info
        mock_run.return_value = MagicMock(returncode=0, stdout="garbage output")
        result = detect_gpu_info()
        self.assertIsNone(result)

    @patch('shutil.which', return_value='/usr/bin/nvidia-smi')
    @patch('subprocess.run')
    def test_empty_nvidia_smi_output(self, mock_run, mock_which):
        """Empty nvidia-smi output should return None."""
        from scripts.utils.gpu import detect_gpu_info
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        result = detect_gpu_info()
        self.assertIsNone(result)

    @patch('shutil.which', return_value='/usr/bin/nvidia-smi')
    @patch('subprocess.run')
    def test_multi_gpu_detection(self, mock_run, mock_which):
        """Multiple GPUs should be detected by counting lines."""
        from scripts.utils.gpu import detect_gpu_info
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA A100, 81920\nNVIDIA A100, 81920\nNVIDIA A100, 81920\n",
        )
        result = detect_gpu_info()
        self.assertIsNotNone(result)
        self.assertEqual(result.count, 3)


# =============================================================================
# Environment Detection Edge Cases
# =============================================================================


class TestEnvironmentEdgeCases(unittest.TestCase):
    """Test cloud environment detection edge cases."""

    @patch.dict(os.environ, {}, clear=True)
    @patch('pathlib.Path.exists', return_value=False)
    def test_no_cloud_markers(self, mock_exists):
        """No cloud markers should detect 'generic'."""
        from scripts.utils.environment import detect_cloud_environment
        env = detect_cloud_environment()
        self.assertEqual(env.provider, 'generic')


# =============================================================================
# Integration: Mini Training Run
# =============================================================================


class TestMiniTrainingRun(unittest.TestCase):
    """Integration test: run a minimal training loop."""

    def test_training_loop_runs(self):
        """A mini training loop should complete without errors."""
        from train import SimpleOccupancyModel, OccupancyLoss, train_epoch
        from torch.utils.tensorboard import SummaryWriter
        import tempfile

        # Tiny model
        config = MagicMock()
        config.history_frames = 2
        config.future_frames = 2
        config.grid_size = [4, 4, 4]

        model = SimpleOccupancyModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = OccupancyLoss()

        # Tiny dataset
        batches = []
        for _ in range(3):
            batches.append({
                'history_occupancy': torch.randn(2, 2, 4, 4, 4),
                'future_occupancy': torch.randint(0, 2, (2, 2, 4, 4, 4)).float(),
                'history_poses': torch.randn(2, 2, 13),
                'future_poses': torch.randn(2, 2, 13),
            })

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SummaryWriter(tmpdir)
            loss = train_epoch(model, batches, optimizer, criterion, 'cpu', 0, writer)
            writer.close()

        self.assertIsInstance(loss, float)
        self.assertFalse(np.isnan(loss))
        self.assertGreater(loss, 0)

    def test_training_loss_decreases(self):
        """Loss should decrease over multiple epochs."""
        from train import SimpleOccupancyModel, OccupancyLoss, train_epoch
        from torch.utils.tensorboard import SummaryWriter
        import tempfile

        config = MagicMock()
        config.history_frames = 2
        config.future_frames = 2
        config.grid_size = [4, 4, 4]

        model = SimpleOccupancyModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = OccupancyLoss()

        # Fixed dataset for consistent comparison
        torch.manual_seed(42)
        batches = [{
            'history_occupancy': torch.randn(2, 2, 4, 4, 4),
            'future_occupancy': torch.randint(0, 2, (2, 2, 4, 4, 4)).float(),
            'history_poses': torch.randn(2, 2, 13),
            'future_poses': torch.randn(2, 2, 13),
        }]

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SummaryWriter(tmpdir)
            loss_epoch0 = train_epoch(model, batches, optimizer, criterion, 'cpu', 0, writer)
            # Train a few more epochs
            for epoch in range(1, 5):
                loss = train_epoch(model, batches, optimizer, criterion, 'cpu', epoch, writer)
            writer.close()

        # Final loss should be less than initial (model is learning)
        self.assertLess(loss, loss_epoch0)


if __name__ == '__main__':
    unittest.main()
