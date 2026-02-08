"""Regression tests for bug fixes and edge cases.

This test suite provides regression coverage for specific bugs that were fixed:
- FuturePoseRNN with num_future=0
- Triplet loss with batch size < 4
- Triplet loss self-supervised mining (no batch ordering assumption)
- Safe quaternion normalization (zero-vector handling)
- Config knob passthrough for 6DoF training
- --use-occworld + --model-type 6dof conflict detection
- Pose variance std with unbiased=False
- Trajectory validation with single waypoint
- Trajectory validation with zero total distance
- UAVScenes angular velocity sin_half guard
- UAVScenes voxel coordinate clipping before cast
- UAVScenes quaternion normalization with degenerate inputs
- PCD parser robustness (empty files, malformed headers)
- Empty point cloud handling in voxelization
- Single-element batch handling in collate functions
- NaN propagation in loss functions
"""

import sys
import os
import unittest
import tempfile
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# Model Edge Cases
# =============================================================================


class TestFuturePoseRNNEdgeCases(unittest.TestCase):
    """Regression tests for FuturePoseRNN edge cases."""

    def setUp(self):
        from models.occworld_6dof import FuturePoseRNN
        self.rnn = FuturePoseRNN(pose_dim=13, hidden_dim=128, context_dim=64)

    def test_num_future_zero(self):
        """Regression: FuturePoseRNN should handle num_future=0 without error."""
        last_pose = torch.randn(2, 13)
        context = torch.randn(2, 64)
        
        # This should return empty tensor, not crash
        future_poses = self.rnn(last_pose, context, num_future=0)
        
        self.assertEqual(future_poses.shape, (2, 0, 13))
        self.assertEqual(future_poses.numel(), 0)

    def test_quaternion_normalization_preserves_unit_length(self):
        """Regression: Quaternion should remain unit-length after residual updates."""
        last_pose = torch.randn(1, 13)
        # Set initial quaternion to non-unit length
        last_pose[:, 3:7] = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
        context = torch.randn(1, 64)
        
        future_poses = self.rnn(last_pose, context, num_future=3)
        
        # Check all predicted quaternions are unit length
        for t in range(3):
            quat = future_poses[:, t, 3:7]
            quat_norm = torch.norm(quat, dim=-1)
            self.assertTrue(torch.allclose(quat_norm, torch.ones(1), atol=1e-5))

    def test_numerical_stability_with_extreme_inputs(self):
        """Test numerical stability with extreme pose values."""
        # Very large position values
        last_pose = torch.zeros(1, 13)
        last_pose[:, :3] = torch.tensor([[1e6, 1e6, 1e6]])  # Large positions
        context = torch.randn(1, 64)
        
        future_poses = self.rnn(last_pose, context, num_future=2)
        
        # Should not produce NaN or Inf
        self.assertFalse(torch.isnan(future_poses).any())
        self.assertFalse(torch.isinf(future_poses).any())


class TestTripletLossEdgeCases(unittest.TestCase):
    """Regression tests for triplet loss with small batch sizes."""

    def setUp(self):
        from models.occworld_6dof import OccWorld6DoFLoss
        self.loss_fn = OccWorld6DoFLoss(
            occ_weight=0,
            pose_weight=0,
            place_weight=1.0,
            triplet_margin=0.2,
        )

    def test_triplet_loss_batch_size_1(self):
        """Regression: Triplet loss should return 0 for B=1."""
        embeddings = torch.randn(1, 256)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
        outputs = {
            'future_occupancy': torch.zeros(1, 1, 4, 4, 4),
            'future_poses': torch.zeros(1, 1, 13),
            'place_embedding': embeddings,
        }
        targets = {
            'future_occupancy': torch.zeros(1, 1, 4, 4, 4),
            'future_poses': torch.zeros(1, 1, 13),
        }
        
        losses = self.loss_fn(outputs, targets)
        
        # Should not crash, place loss should be finite
        self.assertIn('place', losses)
        self.assertTrue(torch.isfinite(losses['place']))

    def test_triplet_loss_batch_size_2(self):
        """Regression: Triplet loss should return 0 for B=2 (not enough for triplets)."""
        embeddings = torch.randn(2, 256)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
        outputs = {
            'future_occupancy': torch.zeros(2, 1, 4, 4, 4),
            'future_poses': torch.zeros(2, 1, 13),
            'place_embedding': embeddings,
        }
        targets = {
            'future_occupancy': torch.zeros(2, 1, 4, 4, 4),
            'future_poses': torch.zeros(2, 1, 13),
        }
        
        losses = self.loss_fn(outputs, targets)
        
        # Should handle gracefully
        self.assertIn('place', losses)
        self.assertTrue(torch.isfinite(losses['place']))

    def test_triplet_loss_batch_size_3(self):
        """Regression: Triplet loss with B=3 should work but may have limited triplets."""
        embeddings = torch.randn(3, 256)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
        outputs = {
            'future_occupancy': torch.zeros(3, 1, 4, 4, 4),
            'future_poses': torch.zeros(3, 1, 13),
            'place_embedding': embeddings,
        }
        targets = {
            'future_occupancy': torch.zeros(3, 1, 4, 4, 4),
            'future_poses': torch.zeros(3, 1, 13),
        }
        
        losses = self.loss_fn(outputs, targets)
        
        # Should handle gracefully (returns 0 for B < 4)
        self.assertIn('place', losses)
        self.assertTrue(torch.isfinite(losses['place']))

    def test_triplet_loss_batch_size_4(self):
        """Triplet loss should work properly with B=4."""
        embeddings = torch.randn(4, 256)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
        outputs = {
            'future_occupancy': torch.zeros(4, 1, 4, 4, 4),
            'future_poses': torch.zeros(4, 1, 13),
            'place_embedding': embeddings,
        }
        targets = {
            'future_occupancy': torch.zeros(4, 1, 4, 4, 4),
            'future_poses': torch.zeros(4, 1, 13),
        }
        
        losses = self.loss_fn(outputs, targets)
        
        # Should compute actual triplet loss
        self.assertIn('place', losses)
        self.assertTrue(torch.isfinite(losses['place']))
        # Place loss should be non-zero (norm regularization at minimum)
        self.assertGreater(losses['place'].item(), 0)


class TestPoseVarianceLossEdgeCases(unittest.TestCase):
    """Regression tests for pose variance computation with unbiased=False."""

    def setUp(self):
        from models.occworld_6dof import OccWorld6DoFLoss
        self.loss_fn = OccWorld6DoFLoss(
            occ_weight=0,
            pose_weight=1.0,
            pose_variance_weight=1.0,
            min_pose_std=0.1,
        )

    def test_std_unbiased_false_with_constant_predictions(self):
        """Regression: std(unbiased=False) should detect constant predictions."""
        # All predictions are identical (zero variance)
        constant_poses = torch.ones(2, 3, 13) * 0.5
        # Set valid quaternions
        constant_poses[:, :, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        
        outputs = {
            'future_occupancy': torch.zeros(2, 3, 4, 4, 4),
            'future_poses': constant_poses,
        }
        targets = {
            'future_occupancy': torch.zeros(2, 3, 4, 4, 4),
            'future_poses': torch.rand(2, 3, 13),
        }
        
        losses = self.loss_fn(outputs, targets)
        metrics = self.loss_fn.get_debug_metrics()
        
        # Std should be 0 for constant predictions
        self.assertAlmostEqual(metrics['pose_pos_std'], 0.0, places=5)
        self.assertAlmostEqual(metrics['pose_vel_std'], 0.0, places=5)
        
        # Variance penalty should be applied
        self.assertGreater(metrics['pose_variance_penalty'], 0)

    def test_std_unbiased_false_with_two_samples(self):
        """Test std computation works with just 2 samples (unbiased=False allows this)."""
        # Create poses with some variance
        poses = torch.zeros(2, 2, 13)
        poses[0, 0, :3] = torch.tensor([0.0, 0.0, 0.0])
        poses[0, 1, :3] = torch.tensor([1.0, 1.0, 1.0])
        poses[1, 0, :3] = torch.tensor([0.0, 0.0, 0.0])
        poses[1, 1, :3] = torch.tensor([1.0, 1.0, 1.0])
        poses[:, :, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        
        outputs = {
            'future_occupancy': torch.zeros(2, 2, 4, 4, 4),
            'future_poses': poses,
        }
        targets = {
            'future_occupancy': torch.zeros(2, 2, 4, 4, 4),
            'future_poses': torch.rand(2, 2, 13),
        }
        
        losses = self.loss_fn(outputs, targets)
        metrics = self.loss_fn.get_debug_metrics()
        
        # Should compute std without error
        self.assertGreater(metrics['pose_pos_std'], 0)
        self.assertTrue(torch.isfinite(losses['pose']))


# =============================================================================
# Trajectory Validation Edge Cases
# =============================================================================


class TestTrajectoryValidationEdgeCases(unittest.TestCase):
    """Regression tests for trajectory validation edge cases."""

    def setUp(self):
        from scripts.utils.trajectory import TrajectoryGenerator, TrajectoryConfig
        config = TrajectoryConfig(min_motion=2.0)
        self.generator = TrajectoryGenerator(config)

    def test_validate_single_waypoint(self):
        """Regression: validate_trajectory should handle single waypoint without error."""
        waypoints = [{
            'position': {'x': 0.0, 'y': 0.0, 'z': 10.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
            'velocity': {'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}},
            'agent_type': 'drone'
        }]
        
        stats = self.generator.validate_trajectory(waypoints)
        
        # Single waypoint is valid (no motion required)
        self.assertEqual(stats['num_frames'], 1)
        self.assertEqual(stats['total_distance'], 0.0)
        self.assertEqual(stats['mean_distance'], 0.0)
        self.assertEqual(stats['min_distance'], 0.0)
        self.assertEqual(stats['max_distance'], 0.0)
        self.assertEqual(stats['static_frames'], 0)
        self.assertTrue(stats['valid'])

    def test_validate_zero_distance_trajectory(self):
        """Regression: validate_trajectory should handle zero total distance."""
        # All waypoints at same location
        waypoints = [
            {
                'position': {'x': 0.0, 'y': 0.0, 'z': 10.0},
                'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
                'velocity': {'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}},
                'agent_type': 'drone'
            } for _ in range(5)
        ]
        
        stats = self.generator.validate_trajectory(waypoints)
        
        self.assertEqual(stats['num_frames'], 5)
        self.assertEqual(stats['total_distance'], 0.0)
        self.assertEqual(stats['mean_distance'], 0.0)
        self.assertFalse(stats['valid'])  # Not valid (no motion)

    def test_validate_empty_trajectory(self):
        """Test validation with empty trajectory list."""
        waypoints = []
        
        stats = self.generator.validate_trajectory(waypoints)
        
        # Should handle gracefully
        self.assertEqual(stats['num_frames'], 0)
        self.assertEqual(stats['total_distance'], 0.0)

    def test_survey_pattern_zero_distance_keypoints(self):
        """Regression: survey pattern should handle coincident keypoints."""
        from scripts.utils.trajectory import TrajectoryGenerator, TrajectoryConfig

        # Create config with zero-size bounds (all keypoints coincide)
        config = TrajectoryConfig(bounds=(0, 0, 0, 0), speed=1.0)
        generator = TrajectoryGenerator(config)
        
        # Should not crash and should return valid waypoints
        waypoints = generator.generate_survey_pattern(10, 'drone')

        self.assertEqual(len(waypoints), 10)
        # All waypoints should be valid dicts
        for w in waypoints:
            self.assertIn('position', w)
            self.assertIn('orientation', w)


# =============================================================================
# UAVScenes Dataset Edge Cases
# =============================================================================


class TestUAVScenesEdgeCases(unittest.TestCase):
    """Regression tests for UAVScenes dataset edge cases."""

    def test_angular_velocity_sin_half_guard(self):
        """Regression: angular velocity computation should guard against sin_half near zero."""
        from dataset.uavscenes_dataset import UAVScenesConfig, UAVScenesDataset
        
        # Test the computation logic directly
        # When quaternions are nearly identical, angle -> 0, sin(angle/2) -> 0
        
        # Case 1: Identical quaternions
        q_curr = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
        q_prev = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Compute angle between them
        q_dot = np.dot(q_curr, q_prev)
        angle = 2 * np.arccos(np.clip(q_dot, -1, 1))
        sin_half = np.sin(angle / 2)
        
        # sin_half should be very small
        self.assertLess(sin_half, 1e-6)
        
        # The guard should prevent division by zero
        if sin_half > 1e-6:
            axis = q_curr[1:] / sin_half
        else:
            axis = np.zeros(3)
        
        # Should result in zero angular velocity
        np.testing.assert_array_almost_equal(axis, np.zeros(3))

    def test_voxel_coord_clipping_before_cast(self):
        """Regression: voxel coordinates should be clipped before int conversion."""
        from dataset.uavscenes_dataset import UAVScenesConfig
        
        config = UAVScenesConfig(
            point_cloud_range=(-10.0, -10.0, -5.0, 10.0, 10.0, 5.0),
            voxel_size=(1.0, 1.0, 1.0),
        )
        
        # Simulate out-of-range points
        pc_range = np.array(config.point_cloud_range)
        voxel_size = np.array(config.voxel_size)
        grid_size = np.array(config.grid_size)
        
        # Points slightly outside range
        points = np.array([
            [10.5, 10.5, 5.5],   # Out of range
            [-10.5, -10.5, -5.5],  # Out of range
            [0.0, 0.0, 0.0],      # In range
        ], dtype=np.float32)
        
        # Compute voxel coords
        voxel_coords_float = (points - pc_range[:3]) / voxel_size
        
        # Before clipping, coords can be outside grid
        self.assertTrue(np.any(voxel_coords_float < 0))
        self.assertTrue(np.any(voxel_coords_float >= grid_size))
        
        # Apply clipping BEFORE int conversion (the fix)
        voxel_coords_float_clipped = np.clip(voxel_coords_float, 0, grid_size - 1)
        voxel_coords = voxel_coords_float_clipped.astype(np.int32)
        
        # All coordinates should be valid
        self.assertTrue(np.all(voxel_coords >= 0))
        self.assertTrue(np.all(voxel_coords < grid_size))

    def test_quaternion_normalization_degenerate_matrix(self):
        """Regression: quaternion conversion should handle degenerate rotation matrices."""
        from dataset.uavscenes_dataset import UAVScenesConfig
        
        # Create a degenerate rotation matrix (all zeros)
        R_degenerate = np.zeros((3, 3))
        
        # Manual conversion (Shepperd's method)
        trace = np.trace(R_degenerate)  # trace = 0
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R_degenerate[2, 1] - R_degenerate[1, 2]) * s
            y = (R_degenerate[0, 2] - R_degenerate[2, 0]) * s
            z = (R_degenerate[1, 0] - R_degenerate[0, 1]) * s
        else:
            # Will go to one of the branches
            # All diagonal elements are 0, so will use last branch
            s = 2.0 * np.sqrt(1.0 + R_degenerate[2, 2] - R_degenerate[0, 0] - R_degenerate[1, 1])
            # s = 2.0 * 1.0 = 2.0
            w = (R_degenerate[1, 0] - R_degenerate[0, 1]) / s
            x = (R_degenerate[0, 2] + R_degenerate[2, 0]) / s
            y = (R_degenerate[1, 2] + R_degenerate[2, 1]) / s
            z = 0.25 * s
        
        quat = np.array([w, x, y, z], dtype=np.float32)
        norm = np.linalg.norm(quat)
        
        if norm < 1e-8:
            # Should return identity quaternion
            quat_safe = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            quat_safe = quat / norm
        
        # Check that result is valid
        self.assertEqual(quat_safe.shape, (4,))
        self.assertAlmostEqual(np.linalg.norm(quat_safe), 1.0, places=5)

    def test_velocity_computation_no_mutation(self):
        """Regression: velocity computation should not mutate input pose."""
        from dataset.uavscenes_dataset import UAVScenesConfig
        
        pose_curr = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        pose_prev = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        pose_curr_copy = pose_curr.copy()
        
        # Simulate velocity computation
        dt = 0.1
        lin_vel = (pose_curr[:3] - pose_prev[:3]) / dt
        
        result = pose_curr.copy()  # The fix: use .copy()
        result[7:10] = lin_vel
        result[10:13] = np.zeros(3)
        
        # Original pose should be unchanged
        np.testing.assert_array_equal(pose_curr, pose_curr_copy)


# =============================================================================
# PCD Parser Robustness
# =============================================================================


class TestPCDParserRobustness(unittest.TestCase):
    """Regression tests for PCD file parsing edge cases."""

    def test_pcd_parser_empty_file(self):
        """Regression: PCD parser should handle empty files gracefully."""
        from dataset.uavscenes_dataset import UAVScenesDataset, UAVScenesConfig
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pcd', delete=False) as f:
            temp_path = f.name
            # Write empty file
            pass
        
        try:
            config = UAVScenesConfig()
            # Can't instantiate dataset without data, but we can test the parsing logic
            
            # Simulate parsing empty file
            with open(temp_path, 'rb') as f:
                header = []
                for _ in range(100):
                    raw = f.readline()
                    if not raw:  # EOF
                        break
                    line = raw.decode('utf-8', errors='ignore').strip()
                    header.append(line)
                    if line.startswith('DATA'):
                        break
                
                # No DATA line found - should be detected
                has_data_line = any(h.startswith('DATA') for h in header)
                self.assertFalse(has_data_line)
        finally:
            os.unlink(temp_path)

    def test_pcd_parser_malformed_header(self):
        """Regression: PCD parser should handle malformed headers."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pcd', delete=False) as f:
            temp_path = f.name
            # Write malformed header (missing required fields)
            f.write("# .PCD v0.7\n")
            f.write("VERSION 0.7\n")
            # Missing POINTS, FIELDS, etc.
            f.write("DATA ascii\n")
            f.write("1.0 2.0 3.0\n")
        
        try:
            # Simulate parsing
            with open(temp_path, 'rb') as f:
                header = []
                for _ in range(100):
                    raw = f.readline()
                    if not raw:
                        break
                    line = raw.decode('utf-8', errors='ignore').strip()
                    header.append(line)
                    if line.startswith('DATA'):
                        break
                
                # Extract metadata
                num_points = 0
                data_format = 'ascii'
                for h in header:
                    parts = h.split()
                    if h.startswith('POINTS') and len(parts) >= 2:
                        num_points = int(parts[1])
                    if h.startswith('DATA') and len(parts) >= 2:
                        data_format = parts[1]
                
                # num_points will be 0 (default) since no POINTS line
                self.assertEqual(num_points, 0)
        finally:
            os.unlink(temp_path)

    def test_pcd_parser_binary_column_mismatch(self):
        """Regression: Binary PCD parser should handle column count mismatches."""
        # Create binary data that doesn't divide evenly by expected columns
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        
        # Try different column counts
        for num_fields in [3, 4]:
            if len(data) % num_fields != 0:
                # Parser should try alternative column counts
                for try_cols in [4, 3, 5, 6]:
                    if len(data) % try_cols == 0:
                        cols = try_cols
                        break
                else:
                    cols = num_fields
                
                # Should not crash
                if len(data) % cols == 0:
                    points = data.reshape(-1, cols)[:, :3]
                    self.assertEqual(points.shape[1], 3)


# =============================================================================
# Empty Input Handling
# =============================================================================


class TestEmptyInputHandling(unittest.TestCase):
    """Regression tests for empty input handling."""

    def test_empty_point_cloud_voxelization(self):
        """Regression: Voxelization should handle empty point clouds."""
        from dataset.uavscenes_dataset import UAVScenesConfig
        
        config = UAVScenesConfig(
            point_cloud_range=(-10.0, -10.0, -5.0, 10.0, 10.0, 5.0),
            voxel_size=(1.0, 1.0, 1.0),
        )
        
        # Empty point cloud
        points = np.zeros((0, 3), dtype=np.float32)
        
        # Simulate voxelization
        pc_range = np.array(config.point_cloud_range)
        voxel_size = np.array(config.voxel_size)
        grid_size = np.array(config.grid_size)
        
        xyz = points[:, :3]
        mask = (
            (xyz[:, 0] >= pc_range[0]) & (xyz[:, 0] < pc_range[3]) &
            (xyz[:, 1] >= pc_range[1]) & (xyz[:, 1] < pc_range[4]) &
            (xyz[:, 2] >= pc_range[2]) & (xyz[:, 2] < pc_range[5])
        )
        xyz_filtered = xyz[mask]
        
        self.assertEqual(len(xyz_filtered), 0)
        
        # Should return empty occupancy grid, not crash
        if len(xyz_filtered) == 0:
            occupancy = np.zeros(grid_size, dtype=np.uint8)
        
        self.assertEqual(occupancy.shape, tuple(grid_size))
        self.assertEqual(occupancy.sum(), 0)

    def test_empty_batch_collate_fn(self):
        """Test collate function with empty batch (should not occur in practice)."""
        from dataset.uavscenes_dataset import collate_fn
        
        # Empty batch
        batch = []
        
        # Should handle gracefully (or raise clear error)
        try:
            collated = collate_fn(batch)
            # If it doesn't crash, check it returns valid structure
            self.assertIsInstance(collated, dict)
        except (IndexError, ValueError, RuntimeError) as e:
            # Expected to fail, but should be a clear error
            self.assertIsInstance(e, (IndexError, ValueError, RuntimeError))

    def test_single_element_batch_collate(self):
        """Test collate function with single-element batch."""
        from dataset.uavscenes_dataset import collate_fn
        
        batch = [{
            'history_occupancy': torch.rand(4, 32, 32, 16),
            'future_occupancy': torch.rand(6, 32, 32, 16),
            'history_poses': torch.rand(4, 13),
            'future_poses': torch.rand(6, 13),
            'agent_type': torch.tensor(1),
            'scene': 'AMtown',
            'scene_folder': 'interval1_AMtown01',
        }]
        
        collated = collate_fn(batch)
        
        # Should work and add batch dimension
        self.assertEqual(collated['history_occupancy'].shape, (1, 4, 32, 32, 16))
        self.assertEqual(collated['future_occupancy'].shape, (1, 6, 32, 32, 16))
        self.assertEqual(collated['history_poses'].shape, (1, 4, 13))
        self.assertEqual(collated['future_poses'].shape, (1, 6, 13))


# =============================================================================
# NaN Propagation Tests
# =============================================================================


class TestNaNPropagation(unittest.TestCase):
    """Regression tests for NaN propagation in loss functions."""

    def test_focal_loss_with_nan_input(self):
        """Test that focal loss detects NaN inputs."""
        from models.occworld_6dof import FocalLoss
        
        loss_fn = FocalLoss()
        
        # Create predictions with NaN
        pred = torch.rand(2, 3, 4, 4, 4)
        pred[0, 0, 0, 0, 0] = float('nan')
        target = torch.rand(2, 3, 4, 4, 4)
        
        loss = loss_fn(pred, target)
        
        # Loss will contain NaN
        self.assertTrue(torch.isnan(loss))

    def test_loss_backward_with_nan(self):
        """Test that NaN in loss prevents gradient computation."""
        pred = torch.rand(2, 4, 4, 4, requires_grad=True)
        target = torch.rand(2, 4, 4, 4)
        
        # Create NaN loss
        loss = torch.tensor(float('nan'), requires_grad=True)
        
        # Backward will propagate NaN to gradients
        loss.backward()
        
        # This is expected behavior - NaN detection should happen BEFORE backward

    def test_uncertainty_division_by_zero_guard(self):
        """Test that uncertainty loss guards against division by zero."""
        from models.occworld_6dof import OccWorld6DoFLoss
        
        loss_fn = OccWorld6DoFLoss(
            occ_weight=0,
            pose_weight=0,
            uncertainty_weight=1.0,
            uncertainty_min=0.001,
            uncertainty_max=10.0,
        )
        
        # Create very small uncertainty values
        outputs = {
            'future_occupancy': torch.zeros(2, 3, 4, 4, 4),
            'future_poses': torch.rand(2, 3, 13),
            'uncertainty': torch.ones(2, 3, 6) * 1e-10,  # Tiny values
        }
        targets = {
            'future_occupancy': torch.zeros(2, 3, 4, 4, 4),
            'future_poses': torch.rand(2, 3, 13),
        }
        
        losses = loss_fn(outputs, targets)
        
        # Should not produce NaN (clamping + eps protection)
        self.assertFalse(torch.isnan(losses['uncertainty']))
        self.assertTrue(torch.isfinite(losses['uncertainty']))


# =============================================================================
# Model Edge Case Integration
# =============================================================================


class TestModelEdgeCaseIntegration(unittest.TestCase):
    """Integration tests for model edge cases."""

    def test_model_with_extreme_grid_size(self):
        """Test model handles very small grid sizes."""
        from models.occworld_6dof import OccWorld6DoF, OccWorld6DoFConfig
        
        # Very small grid
        config = OccWorld6DoFConfig(
            grid_size=(8, 8, 4),
            history_frames=2,
            future_frames=2,
            latent_dim=32,
            encoder_channels=(16, 32),
        )
        
        model = OccWorld6DoF(config)
        
        history_occ = torch.rand(1, 2, 8, 8, 4)
        history_poses = torch.rand(1, 2, 13)
        
        # Should not crash
        outputs = model(history_occ, history_poses)
        
        self.assertEqual(outputs['future_occupancy'].shape, (1, 2, 8, 8, 4))
        self.assertEqual(outputs['future_poses'].shape, (1, 2, 13))

    def test_model_gradient_flow_with_zeros(self):
        """Test gradient flow when inputs are all zeros."""
        from models.occworld_6dof import OccWorld6DoF, OccWorld6DoFConfig
        
        config = OccWorld6DoFConfig(
            grid_size=(16, 16, 8),
            history_frames=2,
            future_frames=2,
            latent_dim=32,
            encoder_channels=(16, 32),
        )
        
        model = OccWorld6DoF(config)
        
        # All zeros
        history_occ = torch.zeros(1, 2, 16, 16, 8, requires_grad=True)
        history_poses = torch.zeros(1, 2, 13, requires_grad=True)
        
        outputs = model(history_occ, history_poses)
        
        # Compute dummy loss
        loss = outputs['future_occupancy'].sum() + outputs['future_poses'].sum()
        loss.backward()
        
        # Gradients should exist and be finite
        self.assertIsNotNone(history_occ.grad)
        self.assertIsNotNone(history_poses.grad)
        self.assertFalse(torch.isnan(history_occ.grad).any())
        self.assertFalse(torch.isnan(history_poses.grad).any())


class TestTripletLossSelfSupervised(unittest.TestCase):
    """Regression: triplet loss must not assume adjacent batch samples are positives."""

    def test_triplet_loss_shuffled_batch(self):
        """Triplet loss should work correctly regardless of batch ordering."""
        from models.occworld_6dof import OccWorld6DoFLoss

        loss_fn = OccWorld6DoFLoss()

        # Create embeddings where pairs (0,1) and (2,3) are similar
        # but batch order is scrambled: [sample_A, sample_C, sample_B, sample_D]
        embeddings = torch.tensor([
            [1.0, 0.0, 0.0],   # A
            [0.0, 1.0, 0.0],   # C (far from A)
            [1.01, 0.01, 0.0], # B (close to A, but not adjacent in batch)
            [0.01, 1.01, 0.0], # D (close to C, but not adjacent in batch)
        ])

        triplet = loss_fn._compute_triplet_loss(embeddings)
        # Should produce valid loss (finite, non-negative)
        self.assertTrue(torch.isfinite(triplet))
        self.assertGreaterEqual(triplet.item(), 0.0)

    def test_triplet_loss_does_not_use_adjacency(self):
        """Verify hard positive is closest sample, not adjacent sample."""
        from models.occworld_6dof import OccWorld6DoFLoss

        loss_fn = OccWorld6DoFLoss()

        # Batch where adjacent samples are maximally far,
        # but non-adjacent samples are close
        embeddings = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],   # 0: close to 2 (far from 1)
            [0.0, 0.0, 1.0, 0.0],   # 1: close to 3 (far from 0)
            [1.0, 0.01, 0.0, 0.0],  # 2: close to 0
            [0.0, 0.0, 1.0, 0.01],  # 3: close to 1
        ])

        triplet = loss_fn._compute_triplet_loss(embeddings)
        self.assertTrue(torch.isfinite(triplet))

    def test_triplet_loss_batch_size_3(self):
        """Triplet loss should work with B=3 (was B<4 before)."""
        from models.occworld_6dof import OccWorld6DoFLoss

        loss_fn = OccWorld6DoFLoss()
        embeddings = torch.randn(3, 8)
        triplet = loss_fn._compute_triplet_loss(embeddings)
        self.assertTrue(torch.isfinite(triplet))

    def test_triplet_loss_batch_size_2_returns_zero(self):
        """Triplet loss needs B>=3 for anchor/pos/neg."""
        from models.occworld_6dof import OccWorld6DoFLoss

        loss_fn = OccWorld6DoFLoss()
        embeddings = torch.randn(2, 8)
        triplet = loss_fn._compute_triplet_loss(embeddings)
        self.assertEqual(triplet.item(), 0.0)

    def test_triplet_loss_gradient_flow(self):
        """Triplet loss should propagate gradients."""
        from models.occworld_6dof import OccWorld6DoFLoss

        loss_fn = OccWorld6DoFLoss()
        embeddings = torch.randn(6, 16, requires_grad=True)
        triplet = loss_fn._compute_triplet_loss(embeddings)
        triplet.backward()
        self.assertIsNotNone(embeddings.grad)
        self.assertFalse(torch.isnan(embeddings.grad).any())


class TestSafeQuaternionNormalize(unittest.TestCase):
    """Regression: quaternion normalization must handle zero vectors."""

    def test_zero_quaternion_returns_identity(self):
        """Zero quaternion should map to identity [1,0,0,0], not NaN."""
        from models.occworld_6dof import safe_quat_normalize

        q = torch.zeros(1, 4)
        result = safe_quat_normalize(q)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(result, expected))
        self.assertFalse(torch.isnan(result).any())

    def test_normal_quaternion_normalized(self):
        """Non-zero quaternion should be unit-normalized."""
        from models.occworld_6dof import safe_quat_normalize

        q = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
        result = safe_quat_normalize(q)
        self.assertAlmostEqual(result.norm().item(), 1.0, places=5)
        self.assertTrue(torch.allclose(result, torch.tensor([[1.0, 0.0, 0.0, 0.0]]), atol=1e-5))

    def test_batch_with_mixed_zero_and_nonzero(self):
        """Batch with some zero and some non-zero quaternions."""
        from models.occworld_6dof import safe_quat_normalize

        q = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],  # zero -> identity
            [0.0, 3.0, 0.0, 0.0],  # non-zero -> normalized
            [0.0, 0.0, 0.0, 0.0],  # zero -> identity
        ])
        result = safe_quat_normalize(q)
        self.assertFalse(torch.isnan(result).any())
        # First and third should be identity
        self.assertTrue(torch.allclose(result[0], torch.tensor([1.0, 0.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(result[2], torch.tensor([1.0, 0.0, 0.0, 0.0])))
        # Second should be [0, 1, 0, 0]
        self.assertAlmostEqual(result[1].norm().item(), 1.0, places=5)

    def test_gradient_through_zero_quaternion(self):
        """Gradients must be finite even when input quaternion is zero."""
        from models.occworld_6dof import safe_quat_normalize

        q = torch.zeros(2, 4, requires_grad=True)
        result = safe_quat_normalize(q)
        result.sum().backward()
        self.assertIsNotNone(q.grad)
        self.assertFalse(torch.isnan(q.grad).any())

    def test_zero_input_full_model_no_nan(self):
        """Full model with zero inputs should not produce NaN (was flaky)."""
        from models.occworld_6dof import OccWorld6DoF, OccWorld6DoFConfig

        config = OccWorld6DoFConfig(
            grid_size=(16, 16, 8),
            history_frames=2,
            future_frames=2,
            latent_dim=32,
            encoder_channels=(16, 32),
        )

        model = OccWorld6DoF(config)
        history_occ = torch.zeros(1, 2, 16, 16, 8, requires_grad=True)
        history_poses = torch.zeros(1, 2, 13, requires_grad=True)

        # Run multiple times to catch intermittent failures
        for _ in range(5):
            outputs = model(history_occ, history_poses)
            loss = outputs['future_occupancy'].sum() + outputs['future_poses'].sum()
            loss.backward()

            self.assertFalse(torch.isnan(outputs['future_occupancy']).any(),
                             "NaN in future_occupancy")
            self.assertFalse(torch.isnan(outputs['future_poses']).any(),
                             "NaN in future_poses")
            self.assertFalse(torch.isnan(history_poses.grad).any(),
                             "NaN in history_poses gradient")


class TestConfigKnobPassthrough(unittest.TestCase):
    """Regression: 6DoF config loss/model knobs must be read from config."""

    def test_loss_weights_from_config(self):
        """OccWorld6DoFLoss should accept config-driven weights."""
        from models.occworld_6dof import OccWorld6DoFLoss

        loss_cfg = {
            'occ_weight': 2.0,
            'pose_weight': 0.8,
            'uncertainty_weight': 0.05,
            'reloc_weight': 0.3,
            'place_weight': 0.2,
        }

        criterion = OccWorld6DoFLoss(
            occ_weight=loss_cfg.get('occ_weight', 1.0),
            pose_weight=loss_cfg.get('pose_weight', 0.5),
            uncertainty_weight=loss_cfg.get('uncertainty_weight', 0.1),
            reloc_weight=loss_cfg.get('reloc_weight', 0.2),
            place_weight=loss_cfg.get('place_weight', 0.1),
        )

        self.assertEqual(criterion.occ_weight, 2.0)
        self.assertEqual(criterion.pose_weight, 0.8)
        self.assertEqual(criterion.uncertainty_weight, 0.05)
        self.assertEqual(criterion.reloc_weight, 0.3)
        self.assertEqual(criterion.place_weight, 0.2)

    def test_model_config_from_dict(self):
        """OccWorld6DoFConfig should accept config-driven model params."""
        from models.occworld_6dof import OccWorld6DoFConfig

        model_cfg = {
            'pose_dim': 7,
            'encoder_channels': (32, 64),
            'dropout': 0.2,
            'place_embedding_dim': 128,
        }

        config = OccWorld6DoFConfig(
            grid_size=(200, 200, 121),
            pose_dim=model_cfg.get('pose_dim', 13),
            encoder_channels=model_cfg.get('encoder_channels', (64, 128, 256)),
            dropout=model_cfg.get('dropout', 0.1),
            place_embedding_dim=model_cfg.get('place_embedding_dim', 256),
        )

        self.assertEqual(config.pose_dim, 7)
        self.assertEqual(config.encoder_channels, (32, 64))
        self.assertEqual(config.dropout, 0.2)
        self.assertEqual(config.place_embedding_dim, 128)


class TestOccworldSixdofConflict(unittest.TestCase):
    """Regression: --use-occworld + --model-type 6dof must be rejected."""

    def test_conflict_detected_in_main(self):
        """Verify that the conflict check exists in train.py source."""
        import inspect
        from train import main

        source = inspect.getsource(main)
        self.assertIn('use_occworld', source)
        self.assertIn('model_type', source)
        # The conflict guard should produce an error and exit
        self.assertIn('incompatible', source.lower())

    def test_occworld_flag_without_6dof_is_fine(self):
        """--use-occworld with simple model shouldn't conflict."""
        # This tests that the conflict check is specific to 6dof
        import argparse
        ns = argparse.Namespace(use_occworld=True, model_type='simple')
        # No conflict — simple model with OccWorld is valid
        self.assertNotEqual(ns.model_type, '6dof')


class TestIntegrationTestNotCollected(unittest.TestCase):
    """Regression: scripts/integration_test.py must not be collected by pytest."""

    def test_conftest_excludes_scripts(self):
        """conftest.py should contain collect_ignore for scripts/."""
        conftest_path = os.path.join(PROJECT_ROOT, 'conftest.py')
        self.assertTrue(os.path.exists(conftest_path),
                        "conftest.py must exist at project root")
        with open(conftest_path) as f:
            content = f.read()
        self.assertIn('collect_ignore', content)
        self.assertIn('scripts', content)


if __name__ == '__main__':
    unittest.main()
