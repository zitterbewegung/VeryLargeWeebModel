#!/usr/bin/env python3
"""
Sanity check for training data compatibility.
Run this after generating data to verify everything works.

Usage:
    python scripts/sanity_check_data.py --data data/tokyo_gazebo
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

def check_session(session_dir: str) -> dict:
    """Check a single session for data integrity."""
    results = {
        'valid': True,
        'frames': 0,
        'errors': [],
        'warnings': [],
        'occupancy_stats': {},
    }

    session_path = Path(session_dir)

    # Check required directories
    required_dirs = ['occupancy', 'poses', 'lidar']
    for d in required_dirs:
        if not (session_path / d).exists():
            results['errors'].append(f"Missing directory: {d}")
            results['valid'] = False

    if not results['valid']:
        return results

    # Get frame list from occupancy files
    occ_files = sorted((session_path / 'occupancy').glob('*_occupancy.npz'))
    results['frames'] = len(occ_files)

    if len(occ_files) == 0:
        results['errors'].append("No occupancy files found")
        results['valid'] = False
        return results

    # Check a sample of frames
    sample_indices = [0, len(occ_files)//2, -1] if len(occ_files) > 2 else range(len(occ_files))

    total_occupied = 0
    total_voxels = 0
    grid_shapes = set()

    for idx in sample_indices:
        occ_file = occ_files[idx]
        frame_id = occ_file.stem.replace('_occupancy', '')

        # Check occupancy
        try:
            data = np.load(occ_file)
            occ = data['occupancy']
            grid_shapes.add(occ.shape)

            occupied = np.sum(occ > 0)
            total = occ.size
            total_occupied += occupied
            total_voxels += total

            if occupied == 0:
                results['warnings'].append(f"Frame {frame_id}: Zero occupied voxels")
            elif occupied / total > 0.5:
                results['warnings'].append(f"Frame {frame_id}: >50% occupied ({occupied/total*100:.1f}%)")

        except Exception as e:
            results['errors'].append(f"Frame {frame_id} occupancy error: {e}")
            results['valid'] = False

        # Check pose
        pose_file = session_path / 'poses' / f'{frame_id}.json'
        if pose_file.exists():
            try:
                with open(pose_file) as f:
                    pose = json.load(f)

                # Verify required fields
                required_fields = [
                    ('position', ['x', 'y', 'z']),
                    ('orientation', ['x', 'y', 'z', 'w']),
                    ('velocity', None),  # Has nested structure
                ]

                for field, subfields in required_fields:
                    if field not in pose:
                        results['errors'].append(f"Frame {frame_id} pose missing: {field}")
                        results['valid'] = False
                    elif subfields:
                        for sf in subfields:
                            if sf not in pose[field]:
                                results['errors'].append(f"Frame {frame_id} pose.{field} missing: {sf}")
                                results['valid'] = False

                # Check velocity structure
                if 'velocity' in pose:
                    vel = pose['velocity']
                    if 'linear' not in vel or 'angular' not in vel:
                        results['errors'].append(f"Frame {frame_id} velocity missing linear/angular")
                        results['valid'] = False
                    else:
                        for key in ['x', 'y', 'z']:
                            if key not in vel['linear'] or key not in vel['angular']:
                                results['errors'].append(f"Frame {frame_id} velocity.linear/angular missing {key}")
                                results['valid'] = False

            except Exception as e:
                results['errors'].append(f"Frame {frame_id} pose error: {e}")
                results['valid'] = False
        else:
            results['errors'].append(f"Frame {frame_id} missing pose file")
            results['valid'] = False

        # Check lidar
        lidar_file = session_path / 'lidar' / f'{frame_id}_LIDAR.npy'
        if lidar_file.exists():
            try:
                lidar = np.load(lidar_file)
                if lidar.shape[1] != 4:
                    results['warnings'].append(f"Frame {frame_id} lidar shape {lidar.shape}, expected (N, 4)")
            except Exception as e:
                results['errors'].append(f"Frame {frame_id} lidar error: {e}")
        else:
            results['warnings'].append(f"Frame {frame_id} missing lidar file")

    # Summary stats
    if total_voxels > 0:
        results['occupancy_stats'] = {
            'grid_shapes': list(grid_shapes),
            'avg_occupancy_rate': total_occupied / total_voxels,
            'expected_shape': (200, 200, 121),
        }

        # Check grid shape
        expected = (200, 200, 121)
        for shape in grid_shapes:
            if shape != expected:
                results['warnings'].append(f"Grid shape {shape} differs from expected {expected}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Sanity check training data')
    parser.add_argument('--data', '-d', default='data/tokyo_gazebo', help='Data directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    data_path = Path(args.data)

    print("=" * 60)
    print("Training Data Sanity Check")
    print("=" * 60)
    print(f"\nData directory: {data_path}")

    if not data_path.exists():
        print(f"\n❌ ERROR: Data directory does not exist!")
        print(f"   Run: ./scripts/onstart.sh --skip-nuscenes")
        sys.exit(1)

    # Find sessions
    sessions = []
    for pattern in ['drone_*', 'rover_*', '*_session_*']:
        sessions.extend(data_path.glob(pattern))
    sessions = [s for s in sessions if s.is_dir()]

    print(f"Found {len(sessions)} sessions")

    if len(sessions) == 0:
        print(f"\n❌ ERROR: No training sessions found!")
        print(f"   Expected directories like: drone_20240117_..., rover_...")
        sys.exit(1)

    # Check each session
    total_frames = 0
    all_valid = True
    all_errors = []
    all_warnings = []

    print(f"\n{'Session':<40} {'Frames':>8} {'Status':>10}")
    print("-" * 60)

    for session in sorted(sessions):
        result = check_session(str(session))
        total_frames += result['frames']

        status = "✓ OK" if result['valid'] else "❌ FAIL"
        if result['warnings'] and result['valid']:
            status = "⚠ WARN"

        print(f"{session.name:<40} {result['frames']:>8} {status:>10}")

        if args.verbose:
            for err in result['errors']:
                print(f"    ❌ {err}")
            for warn in result['warnings'][:3]:  # Limit warnings shown
                print(f"    ⚠ {warn}")
            if len(result['warnings']) > 3:
                print(f"    ... and {len(result['warnings'])-3} more warnings")

        if not result['valid']:
            all_valid = False
        all_errors.extend(result['errors'])
        all_warnings.extend(result['warnings'])

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total sessions: {len(sessions)}")
    print(f"  Total frames:   {total_frames}")
    print(f"  Errors:         {len(all_errors)}")
    print(f"  Warnings:       {len(all_warnings)}")

    # Check if enough for training
    min_frames = 50  # history_frames(4) + future_frames(6) + margin

    print("\n" + "=" * 60)
    print("Training Readiness")
    print("=" * 60)

    if not all_valid:
        print("❌ FAIL: Data has errors that will break training")
        for err in all_errors[:5]:
            print(f"   - {err}")
        sys.exit(1)

    if total_frames < min_frames:
        print(f"❌ FAIL: Not enough frames ({total_frames} < {min_frames} minimum)")
        sys.exit(1)

    # Estimate train/val split
    val_ratio = 0.1
    train_frames = int(total_frames * (1 - val_ratio - 0.1))
    val_frames = int(total_frames * val_ratio)

    print(f"✓ Data is valid for training!")
    print(f"  Estimated train samples: ~{train_frames}")
    print(f"  Estimated val samples:   ~{val_frames}")

    if val_frames < 10:
        print(f"  ⚠ Warning: Very few validation samples")

    print("\n" + "=" * 60)
    print("Ready to train!")
    print("=" * 60)
    print(f"\n  python train.py --config config/finetune_tokyo.py --work-dir out/tokyo")


if __name__ == '__main__':
    main()
