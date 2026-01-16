#!/usr/bin/env python3
"""
Create dummy training data for testing the OccWorld pipeline.

This generates synthetic occupancy grids, poses, and LiDAR data
to verify the training pipeline works before using real simulation data.

Usage:
    python scripts/create_dummy_data.py
    python scripts/create_dummy_data.py --frames 100 --sessions 3
"""
import os
import argparse
import numpy as np
import json

def create_dummy_session(session_path, num_frames=20):
    os.makedirs(f'{session_path}/occupancy', exist_ok=True)
    os.makedirs(f'{session_path}/poses', exist_ok=True)
    os.makedirs(f'{session_path}/lidar', exist_ok=True)

    for i in range(num_frames):
        occ = np.random.randint(0, 2, (200, 200, 121), dtype=np.uint8)
        np.savez_compressed(f'{session_path}/occupancy/{i:06d}_occupancy.npz', occupancy=occ)

        pose = {
            'position': [float(i), 0.0, 10.0 + i],
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'linear_velocity': [1.0, 0.0, 0.0],
            'angular_velocity': [0.0, 0.0, 0.0]
        }
        with open(f'{session_path}/poses/{i:06d}.json', 'w') as f:
            json.dump(pose, f)

        points = np.random.randn(1000, 4).astype(np.float32)
        np.save(f'{session_path}/lidar/{i:06d}_LIDAR.npy', points)

    print(f'Created {num_frames} frames in {session_path}')

def main():
    parser = argparse.ArgumentParser(description='Create dummy training data')
    parser.add_argument('--output', '-o', default='data/tokyo_gazebo', help='Output directory')
    parser.add_argument('--frames', '-f', type=int, default=20, help='Frames per session')
    parser.add_argument('--sessions', '-s', type=int, default=1, help='Number of sessions')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for s in range(args.sessions):
        session_name = f'drone_dummy_session_{s:02d}'
        session_path = os.path.join(args.output, session_name)
        create_dummy_session(session_path, args.frames)

    print(f'\nCreated {args.sessions} session(s) with {args.frames} frames each')
    print(f'Total: {args.sessions * args.frames} training samples')

if __name__ == '__main__':
    main()
