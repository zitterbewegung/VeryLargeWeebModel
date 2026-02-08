# Action-Conditioned Inference for OccWorld6DoF

## Overview

The `OccWorld6DoF` model now supports **action-conditioned inference**, enabling "what if I fly along trajectory X?" queries during inference. This allows the model to condition future occupancy predictions on a planned or commanded trajectory instead of just the predicted trajectory.

## Changes Made

### 1. Updated `forward()` Method Signature

Added an optional `planned_trajectory` parameter to the `OccWorld6DoF.forward()` method:

```python
def forward(
    self,
    history_occupancy: torch.Tensor,
    history_poses: torch.Tensor,
    future_poses: Optional[torch.Tensor] = None,
    planned_trajectory: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
```

**Parameters:**
- `history_occupancy`: [B, T_h, X, Y, Z] - Past occupancy grids
- `history_poses`: [B, T_h, 13] - Past poses
- `future_poses`: [B, T_f, 13] - Future poses (for training, optional)
- `planned_trajectory`: [B, T_f, 13] - Planned/commanded trajectory for action-conditioned generation (optional)

### 2. FiLM Conditioning Logic

The future occupancy decoder uses Feature-wise Linear Modulation (FiLM) to condition spatial features on pose information. The conditioning now uses the planned trajectory when provided:

```python
# Use planned trajectory for conditioning if provided (action-conditioned generation)
film_poses = planned_trajectory if planned_trajectory is not None else predicted_future_poses
pose_flat = film_poses.reshape(B, -1)  # [B, T_f * pose_dim]
film_params = self.pose_film(pose_flat)  # [B, latent_dim * 2]
```

### 3. Behavior

**Without `planned_trajectory` (standard mode):**
- Model predicts future poses using `FuturePoseRNN`
- Future occupancy is conditioned on these predicted poses
- Returns predicted poses and occupancy

**With `planned_trajectory` (action-conditioned mode):**
- Model still predicts future poses using `FuturePoseRNN` (returned in output)
- Future occupancy is conditioned on the **planned trajectory** instead
- Enables querying "what will I see if I follow this specific path?"
- Returns predicted poses (from model) and occupancy (conditioned on planned trajectory)

## Use Cases

### 1. Path Planning
Query multiple candidate trajectories to evaluate which paths are collision-free:

```python
# Evaluate candidate paths
for path_id, candidate_path in enumerate(candidate_paths):
    outputs = model(
        history_occupancy,
        history_poses,
        planned_trajectory=candidate_path
    )
    occupancy = outputs['future_occupancy']
    collision_risk = compute_collision_risk(occupancy)
    print(f"Path {path_id}: collision risk = {collision_risk}")
```

### 2. Safety Verification
Verify if a planned trajectory from a navigation planner is safe:

```python
# Check if planned trajectory is collision-free
outputs = model(
    history_occupancy,
    history_poses,
    planned_trajectory=navigation_planner_output
)
future_occupancy = outputs['future_occupancy']
is_safe = verify_collision_free(future_occupancy, navigation_planner_output)
```

### 3. Multi-Agent Coordination
Predict what each agent will observe when following commanded trajectories:

```python
# Coordinate multiple agents
for agent_id, commanded_trajectory in enumerate(agent_commands):
    outputs = model(
        agent_histories[agent_id],
        agent_poses[agent_id],
        planned_trajectory=commanded_trajectory
    )
    agent_predictions[agent_id] = outputs['future_occupancy']
```

### 4. Interactive Exploration
Allow users to interactively explore "what if" scenarios:

```python
# User draws a path on the map
user_trajectory = gui.get_user_trajectory()

# Show what the drone would see
outputs = model(
    current_occupancy,
    current_poses,
    planned_trajectory=user_trajectory
)
visualize_future_occupancy(outputs['future_occupancy'])
```

## Example Code

See `examples/action_conditioned_inference.py` for a complete working example.

```python
import torch
from models.occworld_6dof import OccWorld6DoF, OccWorld6DoFConfig

# Create model
config = OccWorld6DoFConfig(
    grid_size=(200, 200, 121),
    history_frames=4,
    future_frames=6,
)
model = OccWorld6DoF(config)
model.eval()

# Prepare inputs
history_occupancy = torch.rand(2, 4, 200, 200, 121)
history_poses = torch.rand(2, 4, 13)
planned_trajectory = create_trajectory(2, 6)  # Your trajectory

# Action-conditioned inference
with torch.no_grad():
    outputs = model(
        history_occupancy,
        history_poses,
        planned_trajectory=planned_trajectory
    )

print(f"Future occupancy shape: {outputs['future_occupancy'].shape}")
print(f"Predicted poses shape: {outputs['future_poses'].shape}")
```

## Technical Details

### Pose Format
Trajectories use 13-dimensional pose representation:
- **Position** (3): x, y, z coordinates
- **Orientation** (4): quaternion [w, x, y, z]
- **Linear velocity** (3): vx, vy, vz
- **Angular velocity** (3): wx, wy, wz

### FiLM Mechanism
Feature-wise Linear Modulation (FiLM) conditions the spatial features using scale (γ) and shift (β) parameters derived from the trajectory:

```
conditioned_features = features * (1 + γ) + β
```

Where γ and β are computed from the flattened trajectory:
```python
film_params = pose_film(trajectory.reshape(B, -1))
gamma = film_params[:, :latent_dim]
beta = film_params[:, latent_dim:]
```

### Backward Compatibility
The change is fully backward compatible:
- Existing code without `planned_trajectory` works unchanged
- All existing tests pass
- The pose predictor always runs and returns predicted poses

## Testing

Run the model-specific tests:
```bash
python -m pytest tests/ -k "TestOccWorld6DoFModel" -v
```

Run the demo:
```bash
python examples/action_conditioned_inference.py
```

## Implementation Files

- **Model**: `/Users/r2q2/Projects/VeryLargeWeebModel/models/occworld_6dof.py`
  - Line 597-603: Updated `forward()` signature
  - Line 611-615: Added `planned_trajectory` documentation
  - Line 680-684: Modified FiLM conditioning logic

- **Demo**: `/Users/r2q2/Projects/VeryLargeWeebModel/examples/action_conditioned_inference.py`
  - Complete working example
  - Demonstrates both standard and action-conditioned modes
  - Shows practical use cases

## Future Enhancements

Potential extensions to this feature:

1. **Multi-modal trajectories**: Support ensembles of candidate trajectories
2. **Trajectory optimization**: Gradient-based optimization of trajectories in latent space
3. **Interactive visualization**: Real-time visualization of occupancy for user-drawn paths
4. **Uncertainty propagation**: Propagate trajectory uncertainty through occupancy prediction
5. **Temporal consistency**: Enforce smoothness across trajectory-conditioned predictions

## References

- **FiLM**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (AAAI 2018)
- **OccWorld**: Original OccWorld architecture for occupancy prediction
- **AerialWorld**: Paper describing the full aerial navigation system
