"""Tests for model implementations."""

import sys
import os
import unittest
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class TestOccWorld6DoFConfig(unittest.TestCase):
    """Test OccWorld6DoF configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from models.occworld_6dof import OccWorld6DoFConfig

        config = OccWorld6DoFConfig()

        self.assertEqual(config.grid_size, (200, 200, 121))
        self.assertEqual(config.history_frames, 4)
        self.assertEqual(config.future_frames, 6)
        self.assertEqual(config.pose_dim, 13)
        self.assertTrue(config.enable_uncertainty)
        self.assertTrue(config.enable_relocalization)
        self.assertTrue(config.enable_place_recognition)

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from models.occworld_6dof import OccWorld6DoFConfig

        config = OccWorld6DoFConfig(
            grid_size=(100, 100, 50),
            history_frames=2,
            future_frames=4,
            use_transformer=True,
            enable_uncertainty=False,
        )

        self.assertEqual(config.grid_size, (100, 100, 50))
        self.assertEqual(config.history_frames, 2)
        self.assertEqual(config.future_frames, 4)
        self.assertTrue(config.use_transformer)
        self.assertFalse(config.enable_uncertainty)


class TestOccWorld6DoFModel(unittest.TestCase):
    """Test OccWorld6DoF model forward pass."""

    def setUp(self):
        """Set up test fixtures."""
        from models.occworld_6dof import OccWorld6DoF, OccWorld6DoFConfig

        # Use small grid for fast tests
        self.config = OccWorld6DoFConfig(
            grid_size=(32, 32, 16),
            history_frames=2,
            future_frames=3,
            latent_dim=64,
            encoder_channels=(32, 64),
            enable_uncertainty=True,
            enable_relocalization=True,
            enable_place_recognition=True,
        )
        self.model = OccWorld6DoF(self.config)
        self.batch_size = 2

    def test_model_creation(self):
        """Test model can be created."""
        self.assertIsInstance(self.model, nn.Module)

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct output shapes."""
        B = self.batch_size
        T_h = self.config.history_frames
        T_f = self.config.future_frames
        X, Y, Z = self.config.grid_size

        # Create dummy inputs
        history_occ = torch.rand(B, T_h, X, Y, Z)
        history_poses = torch.rand(B, T_h, 13)

        # Forward pass
        outputs = self.model(history_occ, history_poses)

        # Check output keys
        self.assertIn('future_occupancy', outputs)
        self.assertIn('future_poses', outputs)
        self.assertIn('uncertainty', outputs)
        self.assertIn('global_pose', outputs)
        self.assertIn('place_embedding', outputs)

        # Check shapes
        self.assertEqual(outputs['future_occupancy'].shape, (B, T_f, X, Y, Z))
        self.assertEqual(outputs['future_poses'].shape, (B, T_f, 13))
        self.assertEqual(outputs['uncertainty'].shape, (B, T_f, 6))
        self.assertEqual(outputs['global_pose'].shape, (B, 7))
        self.assertEqual(outputs['place_embedding'].shape, (B, self.config.place_embedding_dim))

    def test_forward_pass_without_optional_heads(self):
        """Test forward pass with optional heads disabled."""
        from models.occworld_6dof import OccWorld6DoF, OccWorld6DoFConfig

        config = OccWorld6DoFConfig(
            grid_size=(32, 32, 16),
            history_frames=2,
            future_frames=3,
            latent_dim=64,
            encoder_channels=(32, 64),
            enable_uncertainty=False,
            enable_relocalization=False,
            enable_place_recognition=False,
        )
        model = OccWorld6DoF(config)

        history_occ = torch.rand(2, 2, 32, 32, 16)
        history_poses = torch.rand(2, 2, 13)

        outputs = model(history_occ, history_poses)

        self.assertIn('future_occupancy', outputs)
        self.assertIn('future_poses', outputs)
        self.assertNotIn('uncertainty', outputs)
        self.assertNotIn('global_pose', outputs)
        self.assertNotIn('place_embedding', outputs)

    def test_forward_pass_with_transformer(self):
        """Test forward pass using transformer temporal encoder."""
        from models.occworld_6dof import OccWorld6DoF, OccWorld6DoFConfig

        config = OccWorld6DoFConfig(
            grid_size=(32, 32, 16),
            history_frames=2,
            future_frames=3,
            latent_dim=64,
            encoder_channels=(32, 64),
            use_transformer=True,
            num_transformer_layers=2,
        )
        model = OccWorld6DoF(config)

        history_occ = torch.rand(2, 2, 32, 32, 16)
        history_poses = torch.rand(2, 2, 13)

        outputs = model(history_occ, history_poses)

        self.assertEqual(outputs['future_occupancy'].shape, (2, 3, 32, 32, 16))
        self.assertEqual(outputs['future_poses'].shape, (2, 3, 13))

    def test_output_ranges(self):
        """Test that outputs are in valid ranges."""
        history_occ = torch.rand(2, 2, 32, 32, 16)
        history_poses = torch.rand(2, 2, 13)

        outputs = self.model(history_occ, history_poses)

        # Occupancy should be in [0, 1] due to sigmoid
        self.assertTrue((outputs['future_occupancy'] >= 0).all())
        self.assertTrue((outputs['future_occupancy'] <= 1).all())

        # Place embeddings should be normalized (L2 norm ~ 1)
        norms = outputs['place_embedding'].norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


class TestSpatialEncoderDecoder(unittest.TestCase):
    """Test spatial encoder and decoder components."""

    def test_encoder_output_shape(self):
        """Test encoder produces expected output shape."""
        from models.occworld_6dof import SpatialEncoder3D

        encoder = SpatialEncoder3D(in_channels=4, channels=(32, 64))
        x = torch.rand(2, 4, 32, 32, 16)

        out = encoder(x)

        # After 2 conv layers with stride 2, spatial dims are divided by 4
        self.assertEqual(out.shape, (2, 64, 8, 8, 4))

    def test_decoder_output_shape(self):
        """Test decoder produces expected output shape."""
        from models.occworld_6dof import SpatialDecoder3D

        decoder = SpatialDecoder3D(
            in_channels=64,
            out_channels=6,
            channels=(64, 32),
            output_size=(32, 32, 16)
        )
        x = torch.rand(2, 64, 8, 8, 4)

        out = decoder(x)

        self.assertEqual(out.shape, (2, 6, 32, 32, 16))


class TestPoseEncoderDecoder(unittest.TestCase):
    """Test pose encoder and decoder."""

    def test_pose_encoder(self):
        """Test pose encoder."""
        from models.occworld_6dof import PoseEncoder

        encoder = PoseEncoder(pose_dim=13, hidden_dim=64, out_dim=128)
        poses = torch.rand(2, 4, 13)  # B, T, pose_dim

        features = encoder(poses)

        self.assertEqual(features.shape, (2, 4, 128))

    def test_pose_decoder(self):
        """Test pose decoder."""
        from models.occworld_6dof import PoseDecoder

        decoder = PoseDecoder(in_dim=128, hidden_dim=64, pose_dim=13)
        features = torch.rand(2, 6, 128)  # B, T, feature_dim

        poses = decoder(features)

        self.assertEqual(poses.shape, (2, 6, 13))


class TestCountParameters(unittest.TestCase):
    """Test parameter counting utility."""

    def test_count_parameters(self):
        """Test parameter counting."""
        from models.occworld_6dof import OccWorld6DoF, OccWorld6DoFConfig, count_parameters

        config = OccWorld6DoFConfig(
            grid_size=(32, 32, 16),
            latent_dim=64,
            encoder_channels=(32, 64),
        )
        model = OccWorld6DoF(config)

        counts = count_parameters(model)

        self.assertIn('total', counts)
        self.assertIn('spatial_encoder', counts)
        self.assertIn('pose_encoder', counts)
        self.assertGreater(counts['total'], 0)

        # Verify total is sum of all submodules
        submodule_sum = sum(v for k, v in counts.items() if k != 'total')
        self.assertEqual(counts['total'], submodule_sum)


class TestSimpleOccupancyModel(unittest.TestCase):
    """Test SimpleOccupancyModel from train.py."""

    def test_simple_model_forward(self):
        """Test simple occupancy model forward pass."""
        # Import from train.py
        import sys
        sys.path.insert(0, PROJECT_ROOT)

        # Create a mock config object
        class MockConfig:
            history_frames = 2
            future_frames = 3
            grid_size = [32, 32, 16]

        from train import SimpleOccupancyModel

        model = SimpleOccupancyModel(MockConfig())

        history_occ = torch.rand(2, 2, 32, 32, 16)
        history_poses = torch.rand(2, 2, 13)

        outputs = model(history_occ, history_poses)

        self.assertIn('future_occupancy', outputs)
        self.assertIn('future_poses', outputs)
        self.assertEqual(outputs['future_occupancy'].shape, (2, 3, 32, 32, 16))
        self.assertEqual(outputs['future_poses'].shape, (2, 3, 13))

    def test_simple_model_without_poses(self):
        """Test simple model works without pose input."""
        class MockConfig:
            history_frames = 2
            future_frames = 3
            grid_size = [32, 32, 16]

        from train import SimpleOccupancyModel

        model = SimpleOccupancyModel(MockConfig())

        history_occ = torch.rand(2, 2, 32, 32, 16)

        # Forward without poses
        output = model(history_occ, None, None)

        # Should return just tensor (not dict) when poses not provided
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (2, 3, 32, 32, 16))


if __name__ == '__main__':
    unittest.main()
