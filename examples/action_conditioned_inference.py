#!/usr/bin/env python
"""
Action-Conditioned Inference Demo

Demonstrates how to use OccWorld6DoF with planned trajectories for
"what if I fly along this path?" queries during inference.

Usage:
    python examples/action_conditioned_inference.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.occworld_6dof import OccWorld6DoF, OccWorld6DoFConfig


def create_example_trajectory(batch_size: int, num_frames: int) -> torch.Tensor:
    """Create an example planned trajectory.

    Args:
        batch_size: Batch size
        num_frames: Number of future frames

    Returns:
        Trajectory tensor [B, T_f, 13] containing:
            - position (3): x, y, z
            - orientation (4): quaternion [w, x, y, z]
            - linear velocity (3): vx, vy, vz
            - angular velocity (3): wx, wy, wz
    """
    # Example: straight line trajectory with constant velocity
    trajectory = torch.zeros(batch_size, num_frames, 13)

    for t in range(num_frames):
        # Linear motion along x-axis
        trajectory[:, t, 0] = t * 1.0  # x position
        trajectory[:, t, 1] = 0.0      # y position
        trajectory[:, t, 2] = 5.0      # z position (constant altitude)

        # Identity quaternion (no rotation)
        trajectory[:, t, 3] = 1.0      # qw
        trajectory[:, t, 4:7] = 0.0    # qx, qy, qz

        # Constant velocity
        trajectory[:, t, 7] = 1.0      # vx
        trajectory[:, t, 8:10] = 0.0   # vy, vz

        # No angular velocity
        trajectory[:, t, 10:13] = 0.0  # wx, wy, wz

    return trajectory


def main():
    print("Action-Conditioned Inference Demo")
    print("=" * 50)

    # Create model
    config = OccWorld6DoFConfig(
        grid_size=(200, 200, 121),
        history_frames=4,
        future_frames=6,
        use_transformer=False,
    )
    model = OccWorld6DoF(config)
    model.eval()
    print(f"Model created with config:")
    print(f"  Grid size: {config.grid_size}")
    print(f"  History frames: {config.history_frames}")
    print(f"  Future frames: {config.future_frames}")
    print()

    # Create dummy inputs
    B = 2
    history_occupancy = torch.rand(B, config.history_frames, *config.grid_size)
    history_poses = torch.rand(B, config.history_frames, 13)

    # Scenario 1: Standard prediction (let model predict trajectory)
    print("Scenario 1: Standard prediction")
    print("-" * 50)
    with torch.no_grad():
        outputs_standard = model(history_occupancy, history_poses)

    print(f"Predicted future poses:")
    print(f"  Shape: {outputs_standard['future_poses'].shape}")
    print(f"  First pose position: {outputs_standard['future_poses'][0, 0, :3]}")
    print(f"Predicted future occupancy:")
    print(f"  Shape: {outputs_standard['future_occupancy'].shape}")
    print(f"  Occupancy rate: {outputs_standard['future_occupancy'].mean():.4f}")
    print()

    # Scenario 2: Action-conditioned prediction (use planned trajectory)
    print("Scenario 2: Action-conditioned prediction")
    print("-" * 50)
    planned_trajectory = create_example_trajectory(B, config.future_frames)
    print(f"Planned trajectory:")
    print(f"  Shape: {planned_trajectory.shape}")
    print(f"  First pose position: {planned_trajectory[0, 0, :3]}")
    print(f"  Last pose position: {planned_trajectory[0, -1, :3]}")

    with torch.no_grad():
        outputs_planned = model(
            history_occupancy,
            history_poses,
            planned_trajectory=planned_trajectory
        )

    print(f"\nPredicted future poses (still from model predictor):")
    print(f"  Shape: {outputs_planned['future_poses'].shape}")
    print(f"  First pose position: {outputs_planned['future_poses'][0, 0, :3]}")
    print(f"Predicted future occupancy (conditioned on planned trajectory):")
    print(f"  Shape: {outputs_planned['future_occupancy'].shape}")
    print(f"  Occupancy rate: {outputs_planned['future_occupancy'].mean():.4f}")
    print()

    # Compare outputs
    print("Comparison:")
    print("-" * 50)
    pose_diff = (outputs_standard['future_poses'] - outputs_planned['future_poses']).abs().max()
    occ_diff = (outputs_standard['future_occupancy'] - outputs_planned['future_occupancy']).abs().max()

    print(f"Max difference in future_poses: {pose_diff:.6f}")
    print(f"  (Should be ~0 since both use same pose predictor)")
    print(f"Max difference in future_occupancy: {occ_diff:.6f}")
    print(f"  (Should be >0 since occupancy is conditioned differently)")
    print()

    # Use case explanation
    print("Use Cases:")
    print("-" * 50)
    print("1. Path planning: 'What obstacles will I encounter if I follow path A vs B?'")
    print("2. Safety verification: 'Is this planned trajectory collision-free?'")
    print("3. Multi-agent coordination: 'What will agent see if it follows commanded path?'")
    print("4. Interactive exploration: Query occupancy for user-defined trajectories")
    print()

    print("=" * 50)
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
