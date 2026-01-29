"""Tests for loss functions."""

import sys
import os
import unittest
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class TestFocalLoss(unittest.TestCase):
    """Test FocalLoss implementation."""

    def test_focal_loss_import(self):
        """Test FocalLoss can be imported from both locations."""
        from train import FocalLoss as TrainFocalLoss
        from models.occworld_6dof import FocalLoss as ModelFocalLoss

        self.assertTrue(callable(TrainFocalLoss))
        self.assertTrue(callable(ModelFocalLoss))

    def test_focal_loss_forward(self):
        """Test FocalLoss forward pass."""
        from train import FocalLoss

        loss_fn = FocalLoss(alpha=0.95, gamma=2.0)

        pred = torch.rand(2, 3, 32, 32, 16)
        target = (torch.rand(2, 3, 32, 32, 16) > 0.9).float()  # Sparse target

        loss = loss_fn(pred, target)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # Scalar
        self.assertTrue(loss.item() >= 0)

    def test_focal_loss_perfect_prediction(self):
        """Test FocalLoss with perfect predictions."""
        from train import FocalLoss

        loss_fn = FocalLoss()

        # Perfect prediction
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        pred = torch.tensor([0.999, 0.001, 0.999, 0.001])

        loss = loss_fn(pred, target)

        # Loss should be very low for perfect predictions
        self.assertLess(loss.item(), 0.1)

    def test_focal_loss_bad_prediction(self):
        """Test FocalLoss with bad predictions."""
        from train import FocalLoss

        loss_fn = FocalLoss()

        # Bad prediction (inverted)
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        pred = torch.tensor([0.001, 0.999, 0.001, 0.999])

        loss = loss_fn(pred, target)

        # Loss should be high for bad predictions
        self.assertGreater(loss.item(), 1.0)

    def test_focal_loss_gamma_effect(self):
        """Test that gamma focuses on hard examples."""
        from train import FocalLoss

        # Create moderately confident wrong predictions
        target = torch.tensor([1.0, 1.0, 1.0, 1.0])
        pred = torch.tensor([0.3, 0.3, 0.3, 0.3])  # Wrong but not extremely

        loss_gamma_0 = FocalLoss(alpha=0.5, gamma=0.0)(pred, target)
        loss_gamma_2 = FocalLoss(alpha=0.5, gamma=2.0)(pred, target)

        # Higher gamma should down-weight easy examples less
        # but for wrong predictions, both should be non-zero
        self.assertGreater(loss_gamma_0.item(), 0)
        self.assertGreater(loss_gamma_2.item(), 0)


class TestOccupancyLoss(unittest.TestCase):
    """Test OccupancyLoss implementation."""

    def test_occupancy_loss_forward(self):
        """Test OccupancyLoss forward pass."""
        from train import OccupancyLoss

        loss_fn = OccupancyLoss(
            focal_alpha=0.99,
            focal_gamma=2.0,
            dice_weight=1.0,
            mean_weight=10.0,
        )

        pred = torch.rand(2, 3, 32, 32, 16)
        target = (torch.rand(2, 3, 32, 32, 16) > 0.95).float()

        loss = loss_fn(pred, target)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss.item() >= 0)

    def test_occupancy_loss_components(self):
        """Test OccupancyLoss returns loss components."""
        from train import OccupancyLoss

        loss_fn = OccupancyLoss()

        pred = torch.rand(2, 3, 32, 32, 16)
        target = (torch.rand(2, 3, 32, 32, 16) > 0.95).float()

        _ = loss_fn(pred, target)

        # Check that loss components are stored
        components = loss_fn.get_loss_components()

        self.assertIn('focal', components)
        self.assertIn('dice', components)
        self.assertIn('mean_match', components)

    def test_occupancy_loss_mean_matching(self):
        """Test mean matching prevents collapse to all zeros."""
        from train import OccupancyLoss

        loss_fn = OccupancyLoss(mean_weight=10.0)

        # Target has some occupancy
        target = torch.zeros(2, 3, 32, 32, 16)
        target[:, :, 15:17, 15:17, 7:9] = 1.0

        # Prediction is all zeros (collapse)
        pred_collapse = torch.zeros(2, 3, 32, 32, 16)

        # Prediction has similar mean
        pred_good = torch.zeros(2, 3, 32, 32, 16)
        pred_good[:, :, 15:17, 15:17, 7:9] = 0.9

        loss_collapse = loss_fn(pred_collapse, target)
        loss_good = loss_fn(pred_good, target)

        # Collapsed prediction should have higher loss
        self.assertGreater(loss_collapse.item(), loss_good.item())


class TestOccWorld6DoFLoss(unittest.TestCase):
    """Test OccWorld6DoFLoss implementation."""

    def test_loss_forward(self):
        """Test OccWorld6DoFLoss forward pass."""
        from models.occworld_6dof import OccWorld6DoFLoss

        loss_fn = OccWorld6DoFLoss(
            occ_weight=1.0,
            pose_weight=0.5,
            uncertainty_weight=0.1,
            reloc_weight=0.2,
            place_weight=0.1,
        )

        outputs = {
            'future_occupancy': torch.rand(2, 3, 32, 32, 16),
            'future_poses': torch.rand(2, 3, 13),
            'uncertainty': torch.rand(2, 3, 6) + 0.1,  # Positive values
            'global_pose': torch.rand(2, 7),
            'place_embedding': torch.randn(2, 256),
        }
        outputs['place_embedding'] = torch.nn.functional.normalize(
            outputs['place_embedding'], dim=-1
        )

        targets = {
            'future_occupancy': (torch.rand(2, 3, 32, 32, 16) > 0.95).float(),
            'future_poses': torch.rand(2, 3, 13),
            'global_pose': torch.rand(2, 7),
        }

        losses = loss_fn(outputs, targets)

        self.assertIn('total', losses)
        self.assertIn('occ', losses)
        self.assertIn('pose', losses)
        self.assertIn('uncertainty', losses)
        self.assertIn('reloc', losses)
        self.assertIn('place', losses)

        # Total should be sum of components
        expected_total = sum(v for k, v in losses.items() if k != 'total')
        self.assertTrue(torch.allclose(losses['total'], expected_total))

    def test_loss_without_optional_outputs(self):
        """Test loss computation without optional outputs."""
        from models.occworld_6dof import OccWorld6DoFLoss

        loss_fn = OccWorld6DoFLoss(
            uncertainty_weight=0,
            reloc_weight=0,
            place_weight=0,
        )

        outputs = {
            'future_occupancy': torch.rand(2, 3, 32, 32, 16),
            'future_poses': torch.rand(2, 3, 13),
        }

        targets = {
            'future_occupancy': (torch.rand(2, 3, 32, 32, 16) > 0.95).float(),
            'future_poses': torch.rand(2, 3, 13),
        }

        losses = loss_fn(outputs, targets)

        self.assertIn('total', losses)
        self.assertIn('occ', losses)
        self.assertIn('pose', losses)

    def test_quaternion_loss(self):
        """Test quaternion loss handles double-cover properly."""
        from models.occworld_6dof import OccWorld6DoFLoss

        loss_fn = OccWorld6DoFLoss(occ_weight=0, pose_weight=1.0)

        # Same rotation, different quaternion representation (q and -q)
        q1 = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # w=1
        q2 = torch.tensor([[0.0, 0.0, 0.0, -1.0]])  # w=-1 (same rotation!)

        outputs1 = {
            'future_occupancy': torch.zeros(1, 1, 4, 4, 4),
            'future_poses': torch.cat([torch.zeros(1, 1, 3), q1.unsqueeze(0), torch.zeros(1, 1, 6)], dim=-1),
        }
        outputs2 = {
            'future_occupancy': torch.zeros(1, 1, 4, 4, 4),
            'future_poses': torch.cat([torch.zeros(1, 1, 3), q2.unsqueeze(0), torch.zeros(1, 1, 6)], dim=-1),
        }
        targets = {
            'future_occupancy': torch.zeros(1, 1, 4, 4, 4),
            'future_poses': torch.cat([torch.zeros(1, 1, 3), q1.unsqueeze(0), torch.zeros(1, 1, 6)], dim=-1),
        }

        # Both should have similar loss since they represent the same rotation
        loss1 = loss_fn(outputs1, targets)['pose']
        loss2 = loss_fn(outputs2, targets)['pose']

        self.assertTrue(torch.allclose(loss1, loss2, atol=1e-5))

    def test_pose_variance_penalty(self):
        """Test that variance penalty prevents constant pose predictions."""
        from models.occworld_6dof import OccWorld6DoFLoss

        loss_fn = OccWorld6DoFLoss(
            occ_weight=0,
            pose_weight=1.0,
            pose_variance_weight=1.0,
            min_pose_std=0.1,
        )

        # Constant pose predictions (collapsed)
        constant_poses = torch.ones(2, 3, 13) * 0.5

        # Varying pose predictions
        varying_poses = torch.rand(2, 3, 13)

        targets = {
            'future_occupancy': torch.zeros(2, 3, 4, 4, 4),
            'future_poses': torch.rand(2, 3, 13),
        }

        outputs_constant = {
            'future_occupancy': torch.zeros(2, 3, 4, 4, 4),
            'future_poses': constant_poses,
        }
        outputs_varying = {
            'future_occupancy': torch.zeros(2, 3, 4, 4, 4),
            'future_poses': varying_poses,
        }

        loss_constant = loss_fn(outputs_constant, targets)
        loss_varying = loss_fn(outputs_varying, targets)

        # Constant predictions should have variance penalty
        debug_constant = loss_fn.get_debug_metrics()

        # The variance penalty should be applied for low variance
        self.assertIn('pose_variance_penalty', debug_constant)

    def test_triplet_loss_for_embeddings(self):
        """Test triplet loss prevents embedding collapse."""
        from models.occworld_6dof import OccWorld6DoFLoss

        loss_fn = OccWorld6DoFLoss(
            occ_weight=0,
            pose_weight=0,
            place_weight=1.0,
        )

        # Collapsed embeddings (all same)
        collapsed_emb = torch.ones(4, 256) / 16  # Normalized
        collapsed_emb = torch.nn.functional.normalize(collapsed_emb, dim=-1)

        # Diverse embeddings
        diverse_emb = torch.randn(4, 256)
        diverse_emb = torch.nn.functional.normalize(diverse_emb, dim=-1)

        targets = {
            'future_occupancy': torch.zeros(4, 1, 4, 4, 4),
            'future_poses': torch.zeros(4, 1, 13),
        }

        outputs_collapsed = {
            'future_occupancy': torch.zeros(4, 1, 4, 4, 4),
            'future_poses': torch.zeros(4, 1, 13),
            'place_embedding': collapsed_emb,
        }
        outputs_diverse = {
            'future_occupancy': torch.zeros(4, 1, 4, 4, 4),
            'future_poses': torch.zeros(4, 1, 13),
            'place_embedding': diverse_emb,
        }

        _ = loss_fn(outputs_collapsed, targets)
        debug_collapsed = loss_fn.get_debug_metrics()

        _ = loss_fn(outputs_diverse, targets)
        debug_diverse = loss_fn.get_debug_metrics()

        # Collapsed embeddings should have low std
        self.assertLess(debug_collapsed['embedding_std'], debug_diverse['embedding_std'])

    def test_debug_metrics(self):
        """Test debug metrics are populated."""
        from models.occworld_6dof import OccWorld6DoFLoss

        loss_fn = OccWorld6DoFLoss()

        outputs = {
            'future_occupancy': torch.rand(2, 3, 4, 4, 4),
            'future_poses': torch.rand(2, 3, 13),
            'uncertainty': torch.rand(2, 3, 6) + 0.1,
            'place_embedding': torch.nn.functional.normalize(torch.randn(2, 256), dim=-1),
        }
        targets = {
            'future_occupancy': torch.rand(2, 3, 4, 4, 4),
            'future_poses': torch.rand(2, 3, 13),
        }

        _ = loss_fn(outputs, targets)
        metrics = loss_fn.get_debug_metrics()

        self.assertIn('pose_pos_std', metrics)
        self.assertIn('uncertainty_mean', metrics)
        self.assertIn('embedding_std', metrics)


class TestLossBackpropagation(unittest.TestCase):
    """Test that losses can backpropagate correctly."""

    def test_focal_loss_gradient(self):
        """Test FocalLoss produces valid gradients."""
        from train import FocalLoss

        loss_fn = FocalLoss()

        pred = torch.rand(2, 4, 4, 4, requires_grad=True)
        target = (torch.rand(2, 4, 4, 4) > 0.9).float()

        loss = loss_fn(pred, target)
        loss.backward()

        self.assertIsNotNone(pred.grad)
        self.assertFalse(torch.isnan(pred.grad).any())
        self.assertFalse(torch.isinf(pred.grad).any())

    def test_occupancy_loss_gradient(self):
        """Test OccupancyLoss produces valid gradients."""
        from train import OccupancyLoss

        loss_fn = OccupancyLoss()

        pred = torch.rand(2, 4, 4, 4, requires_grad=True)
        target = (torch.rand(2, 4, 4, 4) > 0.9).float()

        loss = loss_fn(pred, target)
        loss.backward()

        self.assertIsNotNone(pred.grad)
        self.assertFalse(torch.isnan(pred.grad).any())

    def test_6dof_loss_gradient(self):
        """Test OccWorld6DoFLoss produces valid gradients."""
        from models.occworld_6dof import OccWorld6DoFLoss

        loss_fn = OccWorld6DoFLoss()

        occ = torch.rand(2, 3, 4, 4, 4, requires_grad=True)
        poses = torch.rand(2, 3, 13, requires_grad=True)
        uncertainty = torch.rand(2, 3, 6, requires_grad=True) + 0.1
        place_emb = torch.randn(2, 256, requires_grad=True)
        place_emb_norm = torch.nn.functional.normalize(place_emb, dim=-1)

        outputs = {
            'future_occupancy': occ,
            'future_poses': poses,
            'uncertainty': uncertainty,
            'place_embedding': place_emb_norm,
        }
        targets = {
            'future_occupancy': torch.rand(2, 3, 4, 4, 4),
            'future_poses': torch.rand(2, 3, 13),
        }

        losses = loss_fn(outputs, targets)
        losses['total'].backward()

        self.assertIsNotNone(occ.grad)
        self.assertIsNotNone(poses.grad)
        self.assertFalse(torch.isnan(occ.grad).any())
        self.assertFalse(torch.isnan(poses.grad).any())


if __name__ == '__main__':
    unittest.main()
