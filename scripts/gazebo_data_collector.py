#!/usr/bin/env python3
"""
Gazebo simulation data collector for OccWorld training.

Generates synthetic training data with:
- LiDAR point clouds
- Ego poses (6DoF)
- Ground truth occupancy grids

Usage:
    python scripts/gazebo_data_collector.py --output data/tokyo_gazebo --frames 1000
    python scripts/gazebo_data_collector.py -o data/tokyo_gazebo -f 300 -s 70 -p random
    python scripts/gazebo_data_collector.py -o data/tokyo_gazebo -f 300 -s 70 -p random --workers 8
"""
import os
import sys
import time
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

# Import shared utilities
from utils import create_session_dirs, HAS_CV2


def generate_trajectory(num_frames: int, pattern: str = 'random') -> list:
    """Generate drone trajectory waypoints."""
    waypoints = []

    if pattern == 'survey':
        # Grid survey pattern
        grid_size = max(2, int(np.sqrt(num_frames)))
        spacing = 20.0  # meters
        altitude = 50.0

        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= num_frames:
                    break
                x = (i - grid_size/2) * spacing
                y = (j - grid_size/2) * spacing
                # Alternate direction each row
                if i % 2 == 1:
                    y = -y

                # Add some randomness
                x += np.random.uniform(-2, 2)
                y += np.random.uniform(-2, 2)
                alt = altitude + np.random.uniform(-5, 10)

                yaw = np.random.uniform(-np.pi, np.pi)
                waypoints.append({
                    'position': {'x': float(x), 'y': float(y), 'z': float(alt)},
                    'orientation': {'x': 0.0, 'y': 0.0, 'z': float(np.sin(yaw/2)), 'w': float(np.cos(yaw/2))},
                    'velocity': {
                        'linear': {'x': float(np.random.uniform(-5, 5)), 'y': float(np.random.uniform(-5, 5)), 'z': 0.0},
                        'angular': {'x': 0.0, 'y': 0.0, 'z': float(np.random.uniform(-0.2, 0.2))}
                    }
                })
                idx += 1
            if idx >= num_frames:
                break

        # Fill remaining if needed
        while len(waypoints) < num_frames:
            waypoints.append(waypoints[-1].copy())

    elif pattern == 'orbit':
        # Circular orbit around a point of interest
        radius = np.random.uniform(50, 150)
        altitude = np.random.uniform(40, 100)
        center_x = np.random.uniform(-100, 100)
        center_y = np.random.uniform(-100, 100)

        for i in range(num_frames):
            angle = 2 * np.pi * i / num_frames
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            z = altitude + np.random.uniform(-3, 3)

            # Yaw to face center
            yaw = angle + np.pi + np.random.uniform(-0.1, 0.1)

            waypoints.append({
                'position': {'x': float(x), 'y': float(y), 'z': float(z)},
                'orientation': {'x': 0.0, 'y': 0.0, 'z': float(np.sin(yaw/2)), 'w': float(np.cos(yaw/2))},
                'velocity': {
                    'linear': {'x': float(-radius * np.sin(angle) * 0.1), 'y': float(radius * np.cos(angle) * 0.1), 'z': 0.0},
                    'angular': {'x': 0.0, 'y': 0.0, 'z': 0.1}
                }
            })

    else:  # random
        # Random exploration with smooth transitions
        pos = np.array([
            np.random.uniform(-100, 100),
            np.random.uniform(-100, 100),
            np.random.uniform(30, 80)
        ])
        vel = np.array([0.0, 0.0, 0.0])
        yaw = np.random.uniform(-np.pi, np.pi)

        for i in range(num_frames):
            # Random acceleration
            acc = np.array([
                np.random.uniform(-2, 2),
                np.random.uniform(-2, 2),
                np.random.uniform(-0.5, 0.5)
            ])

            # Update velocity with damping
            vel = vel * 0.95 + acc * 0.1
            vel = np.clip(vel, -10, 10)

            # Update position
            pos = pos + vel

            # Keep in bounds
            pos[0] = np.clip(pos[0], -200, 200)
            pos[1] = np.clip(pos[1], -200, 200)
            pos[2] = np.clip(pos[2], 20, 120)

            # Random yaw changes
            yaw += np.random.uniform(-0.1, 0.1)

            waypoints.append({
                'position': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])},
                'orientation': {'x': 0.0, 'y': 0.0, 'z': float(np.sin(yaw/2)), 'w': float(np.cos(yaw/2))},
                'velocity': {
                    'linear': {'x': float(vel[0]), 'y': float(vel[1]), 'z': float(vel[2])},
                    'angular': {'x': 0.0, 'y': 0.0, 'z': float(np.random.uniform(-0.3, 0.3))}
                }
            })

    return waypoints


def simulate_occupancy(position: dict, grid_size: tuple = (200, 200, 121), seed: int = None) -> np.ndarray:
    """
    Simulate occupancy grid with procedural buildings.

    Creates a realistic urban occupancy pattern with:
    - Ground plane
    - Random buildings of varying sizes
    - Some structural coherence
    """
    if seed is not None:
        np.random.seed(seed)

    occ = np.zeros(grid_size, dtype=np.uint8)

    # Ground plane (first few layers)
    occ[:, :, 0:3] = 1

    # Generate buildings based on position (so same area has consistent buildings)
    area_seed = int(abs(position['x'] * 100 + position['y'] * 10)) % 10000
    np.random.seed(area_seed)

    num_buildings = np.random.randint(8, 25)

    for _ in range(num_buildings):
        # Building footprint
        bx = np.random.randint(10, grid_size[0] - 30)
        by = np.random.randint(10, grid_size[1] - 30)
        bw = np.random.randint(8, 30)
        bh = np.random.randint(8, 30)

        # Building height (in voxels, scaled to altitude)
        # Grid z covers -2 to 150m, so 121 voxels = 152m, ~1.25m per voxel
        max_height = np.random.randint(20, 100)  # 25-125m buildings

        # Fill building volume
        occ[bx:bx+bw, by:by+bh, 3:3+max_height] = 1

    # Add some random small structures (poles, signs, etc.)
    num_small = np.random.randint(5, 15)
    for _ in range(num_small):
        sx = np.random.randint(5, grid_size[0] - 5)
        sy = np.random.randint(5, grid_size[1] - 5)
        sh = np.random.randint(5, 20)
        occ[sx:sx+2, sy:sy+2, 3:3+sh] = 1

    return occ


def simulate_lidar(position: dict, occupancy: np.ndarray, num_points: int = 50000) -> np.ndarray:
    """
    Simulate LiDAR point cloud from occupancy grid.

    Samples points from occupied voxels visible from the sensor position.
    """
    grid_size = occupancy.shape

    # Voxel to world coordinates
    # Assuming grid covers -40 to 40 in x/y, -2 to 150 in z
    voxel_size = np.array([0.4, 0.4, 1.25])
    origin = np.array([-40, -40, -2])

    # Get occupied voxel indices
    occupied = np.argwhere(occupancy > 0)

    if len(occupied) == 0:
        # Return empty point cloud
        return np.zeros((0, 4), dtype=np.float32)

    # Sample points from occupied voxels
    num_samples = min(num_points, len(occupied) * 10)
    indices = np.random.choice(len(occupied), size=num_samples, replace=True)
    sampled_voxels = occupied[indices]

    # Convert to world coordinates with random offset within voxel
    points_world = (sampled_voxels + np.random.uniform(0, 1, sampled_voxels.shape)) * voxel_size + origin

    # Transform to sensor frame (subtract sensor position)
    sensor_pos = np.array([position['x'], position['y'], position['z']])
    points_local = points_world - sensor_pos

    # Filter by distance (LiDAR range)
    distances = np.linalg.norm(points_local, axis=1)
    mask = (distances > 1.0) & (distances < 100.0)
    points_local = points_local[mask]
    distances = distances[mask]

    # Limit to num_points
    if len(points_local) > num_points:
        indices = np.random.choice(len(points_local), size=num_points, replace=False)
        points_local = points_local[indices]
        distances = distances[indices]

    # Add intensity (inverse distance, normalized)
    if len(distances) > 0:
        intensity = 1.0 / (distances / distances.max() + 0.1)
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
    else:
        intensity = np.array([])

    # Add noise
    points_local += np.random.normal(0, 0.02, points_local.shape)

    # Stack with intensity
    if len(points_local) > 0:
        lidar_data = np.column_stack([points_local, intensity]).astype(np.float32)
    else:
        lidar_data = np.zeros((0, 4), dtype=np.float32)

    return lidar_data


def collect_session_worker(args):
    """Worker function for multiprocessing."""
    session_id, output_dir, num_frames, pattern = args
    return collect_data(output_dir, num_frames, pattern, session_id)


def collect_data(output_dir: str, num_frames: int, pattern: str = 'random', session_id: int = 0):
    """Main data collection loop for one session."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Add random suffix to avoid collisions in parallel execution
    rand_suffix = np.random.randint(1000, 9999)
    session_name = f'{pattern}_{session_id:03d}_{timestamp}_{rand_suffix}'
    dirs = create_session_dirs(output_dir, session_name)

    print(f"  [Worker {session_id}] Session: {session_name}")
    print(f"  [Worker {session_id}] Frames: {num_frames}, Pattern: {pattern}")

    # Generate trajectory
    waypoints = generate_trajectory(num_frames, pattern)

    for i, wp in enumerate(waypoints):
        frame_id = f'{i:06d}'

        # Generate occupancy first (needed for LiDAR simulation)
        occ = simulate_occupancy(wp['position'])

        # Generate LiDAR from occupancy
        lidar = simulate_lidar(wp['position'], occ)

        # Save pose (convert to 13D: pos(3) + quat(4) + lin_vel(3) + ang_vel(3))
        pose_13d = [
            wp['position']['x'], wp['position']['y'], wp['position']['z'],
            wp['orientation']['w'], wp['orientation']['x'], wp['orientation']['y'], wp['orientation']['z'],
            wp['velocity']['linear']['x'], wp['velocity']['linear']['y'], wp['velocity']['linear']['z'],
            wp['velocity']['angular']['x'], wp['velocity']['angular']['y'], wp['velocity']['angular']['z'],
        ]

        with open(os.path.join(dirs['poses'], f'{frame_id}.json'), 'w') as f:
            json.dump({'pose_13d': pose_13d, **wp}, f, indent=2)

        # Save LiDAR
        np.save(os.path.join(dirs['lidar'], f'{frame_id}_LIDAR.npy'), lidar)

        # Save occupancy
        np.savez_compressed(
            os.path.join(dirs['occupancy'], f'{frame_id}_occupancy.npz'),
            occupancy=occ
        )

        # Progress
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{num_frames} frames")

    return dirs['root']


def main():
    parser = argparse.ArgumentParser(description='Generate training data for OccWorld')
    parser.add_argument('--output', '-o', default='data/tokyo_gazebo', help='Output directory')
    parser.add_argument('--frames', '-f', type=int, default=300, help='Frames per session')
    parser.add_argument('--pattern', '-p', choices=['survey', 'orbit', 'random'], default='random',
                       help='Flight pattern')
    parser.add_argument('--sessions', '-s', type=int, default=1, help='Number of sessions')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    args = parser.parse_args()

    # Determine number of workers
    num_workers = args.workers if args.workers else min(cpu_count(), args.sessions)
    num_workers = max(1, min(num_workers, args.sessions))  # Clamp to valid range

    print("=" * 50)
    print("OccWorld Training Data Generator")
    print("=" * 50)
    print(f"Output: {args.output}")
    print(f"Sessions: {args.sessions}")
    print(f"Frames/session: {args.frames}")
    print(f"Pattern: {args.pattern}")
    print(f"Total frames: {args.sessions * args.frames}")
    print(f"Workers: {num_workers} (parallel)")
    print("=" * 50)

    os.makedirs(args.output, exist_ok=True)

    # Prepare work items
    work_items = [
        (s, args.output, args.frames, args.pattern)
        for s in range(args.sessions)
    ]

    start_time = time.time()

    if num_workers == 1:
        # Sequential execution
        for s in range(args.sessions):
            print(f"\n[{s+1}/{args.sessions}]")
            collect_data(args.output, args.frames, args.pattern, session_id=s)
    else:
        # Parallel execution
        print(f"\nStarting {num_workers} parallel workers...")
        with Pool(processes=num_workers) as pool:
            results = pool.map(collect_session_worker, work_items)
        print(f"\nAll {len(results)} sessions complete.")

    elapsed = time.time() - start_time

    print("\n" + "=" * 50)
    print("Data generation complete!")
    print(f"Total frames: {args.sessions * args.frames}")
    print(f"Output: {args.output}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Speed: {args.sessions * args.frames / elapsed:.1f} frames/sec")
    print("=" * 50)


if __name__ == '__main__':
    main()
