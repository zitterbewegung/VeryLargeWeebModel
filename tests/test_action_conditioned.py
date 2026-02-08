"""
Tests for action-conditioned inference in OccWorld6DoF.

These tests verify that the planned_trajectory parameter works correctly
and produces different occupancy predictions while preserving the pose
predictor's output.
"""

import torch
import pytest
from models.occworld_6dof import OccWorld6DoF, OccWorld6DoFConfig


class TestActionConditionedInference:
    """Test action-conditioned inference feature."""

    @pytest.fixture
    def model(self):
        """Create a test model."""
        config = OccWorld6DoFConfig(
            grid_size=(200, 200, 121),
            history_frames=4,
            future_frames=6,
            use_transformer=False,
        )
        model = OccWorld6DoF(config)
        model.eval()
        return model

    @pytest.fixture
    def inputs(self):
        """Create test inputs."""
        B = 2
        return {
            'history_occupancy': torch.rand(B, 4, 200, 200, 121),
            'history_poses': torch.rand(B, 4, 13),
            'planned_trajectory': torch.rand(B, 6, 13),
        }

    def test_forward_without_planned_trajectory(self, model, inputs):
        """Test that forward works without planned_trajectory (backward compatibility)."""
        with torch.no_grad():
            outputs = model(
                inputs['history_occupancy'],
                inputs['history_poses']
            )

        assert 'future_occupancy' in outputs
        assert 'future_poses' in outputs
        assert outputs['future_occupancy'].shape == torch.Size([2, 6, 200, 200, 121])
        assert outputs['future_poses'].shape == torch.Size([2, 6, 13])

    def test_forward_with_planned_trajectory(self, model, inputs):
        """Test that forward works with planned_trajectory."""
        with torch.no_grad():
            outputs = model(
                inputs['history_occupancy'],
                inputs['history_poses'],
                planned_trajectory=inputs['planned_trajectory']
            )

        assert 'future_occupancy' in outputs
        assert 'future_poses' in outputs
        assert outputs['future_occupancy'].shape == torch.Size([2, 6, 200, 200, 121])
        assert outputs['future_poses'].shape == torch.Size([2, 6, 13])

    def test_planned_trajectory_affects_occupancy(self, model, inputs):
        """Test that planned_trajectory changes occupancy predictions."""
        with torch.no_grad():
            outputs_standard = model(
                inputs['history_occupancy'],
                inputs['history_poses']
            )
            outputs_planned = model(
                inputs['history_occupancy'],
                inputs['history_poses'],
                planned_trajectory=inputs['planned_trajectory']
            )

        # Occupancy should differ (conditioned on different trajectories)
        occ_diff = (outputs_standard['future_occupancy'] - outputs_planned['future_occupancy']).abs().max()
        assert occ_diff > 0, "Occupancy should differ when using planned trajectory"

    def test_planned_trajectory_does_not_affect_pose_prediction(self, model, inputs):
        """Test that planned_trajectory doesn't change the predicted poses."""
        with torch.no_grad():
            outputs_standard = model(
                inputs['history_occupancy'],
                inputs['history_poses']
            )
            outputs_planned = model(
                inputs['history_occupancy'],
                inputs['history_poses'],
                planned_trajectory=inputs['planned_trajectory']
            )

        # Predicted poses should be identical (both use same predictor)
        pose_diff = (outputs_standard['future_poses'] - outputs_planned['future_poses']).abs().max()
        assert pose_diff < 1e-5, "Predicted poses should be identical (both from predictor)"

    def test_different_planned_trajectories_produce_different_occupancy(self, model, inputs):
        """Test that different planned trajectories produce different occupancy."""
        planned_traj_1 = torch.rand(2, 6, 13)
        planned_traj_2 = torch.rand(2, 6, 13)

        with torch.no_grad():
            outputs_1 = model(
                inputs['history_occupancy'],
                inputs['history_poses'],
                planned_trajectory=planned_traj_1
            )
            outputs_2 = model(
                inputs['history_occupancy'],
                inputs['history_poses'],
                planned_trajectory=planned_traj_2
            )

        # Different planned trajectories should produce different occupancy
        occ_diff = (outputs_1['future_occupancy'] - outputs_2['future_occupancy']).abs().max()
        assert occ_diff > 0, "Different planned trajectories should produce different occupancy"

    def test_planned_trajectory_shape_validation(self, model, inputs):
        """Test that planned_trajectory with wrong shape is handled correctly."""
        # This should work - model will broadcast/handle it
        wrong_shape = torch.rand(2, 3, 13)  # Wrong number of timesteps

        # The forward pass should either work or raise a clear error
        # (depends on implementation details - just verify it doesn't crash silently)
        try:
            with torch.no_grad():
                outputs = model(
                    inputs['history_occupancy'],
                    inputs['history_poses'],
                    planned_trajectory=wrong_shape
                )
            # If it succeeds, check output shapes are still valid
            assert outputs['future_occupancy'].shape[1] == 6
        except (RuntimeError, AssertionError) as e:
            # Expected to fail with shape mismatch
            assert 'shape' in str(e).lower() or 'size' in str(e).lower()

    def test_gradients_flow_through_planned_trajectory(self, model, inputs):
        """Test that gradients can flow through the planned trajectory."""
        model.train()  # Enable training mode

        # Make planned trajectory require gradients
        planned_traj = inputs['planned_trajectory'].clone().requires_grad_(True)

        outputs = model(
            inputs['history_occupancy'],
            inputs['history_poses'],
            planned_trajectory=planned_traj
        )

        # Compute a dummy loss and backprop
        loss = outputs['future_occupancy'].mean()
        loss.backward()

        # Check that gradients exist for planned_trajectory
        assert planned_traj.grad is not None, "Gradients should flow through planned_trajectory"
        assert planned_traj.grad.abs().max() > 0, "Gradients should be non-zero"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
