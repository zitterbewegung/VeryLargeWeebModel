#!/usr/bin/env python3
"""
Quick occupancy sanity check for UAVScenes.

Samples a subset of sequences and reports occupancy rates to validate
voxelization settings (range/voxel size) and ego-frame transforms.
"""

import argparse
import random
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dataset.uavscenes_dataset import UAVScenesConfig, UAVScenesDataset
from scripts.utils.voxel_config import DEFAULT_POINT_CLOUD_RANGE, DEFAULT_VOXEL_SIZE


def parse_args():
    parser = argparse.ArgumentParser(description='UAVScenes occupancy sanity check')
    parser.add_argument('--data', default='data/uavscenes', help='UAVScenes data root')
    parser.add_argument('--scenes', nargs='+', default=['AMtown'], help='Scenes to include')
    parser.add_argument('--interval', type=int, default=1, help='Interval (1=full, 5=keyframes)')
    parser.add_argument('--history', type=int, default=4, help='History frames')
    parser.add_argument('--future', type=int, default=6, help='Future frames')
    parser.add_argument('--frame-skip', type=int, default=1, help='Frame skip')
    parser.add_argument('--samples', type=int, default=50, help='Max samples to check')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-ego-frame', action='store_true', help='Disable ego-frame transform')
    parser.add_argument('--no-fallback', action='store_true', help='Disable lidar-centering fallback')
    parser.add_argument('--min-in-range', type=float, default=0.01,
                        help='Min in-range ratio before fallback triggers')
    parser.add_argument('--pc-range', nargs=6, type=float,
                        default=list(DEFAULT_POINT_CLOUD_RANGE),
                        help='Point cloud range: xmin ymin zmin xmax ymax zmax')
    parser.add_argument('--voxel-size', nargs=3, type=float,
                        default=list(DEFAULT_VOXEL_SIZE),
                        help='Voxel size: x y z')
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = UAVScenesConfig(
        scenes=args.scenes,
        interval=args.interval,
        history_frames=args.history,
        future_frames=args.future,
        frame_skip=args.frame_skip,
        point_cloud_range=tuple(args.pc_range),
        voxel_size=tuple(args.voxel_size),
        ego_frame=not args.no_ego_frame,
        fallback_to_lidar_center=not args.no_fallback,
        min_in_range_ratio=args.min_in_range,
        split='train',
    )

    dataset = UAVScenesDataset(args.data, config)
    if len(dataset) == 0:
        print('No samples found. Check data path and scene selection.')
        return

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:min(args.samples, len(indices))]

    rates = []
    zero_count = 0

    for idx in indices:
        sample = dataset[idx]
        occ = sample['history_occupancy'][0].numpy()
        rate = float((occ > 0).mean())
        rates.append(rate)
        if rate == 0.0:
            zero_count += 1

    print('=' * 60)
    print('UAVScenes Occupancy Sanity Check')
    print('=' * 60)
    print(f'Data root: {args.data}')
    print(f'Scenes: {args.scenes}')
    print(f'Interval: {args.interval}')
    print(f'Ego frame: {not args.no_ego_frame}')
    print(f'Fallback to lidar center: {not args.no_fallback}')
    print(f'Min in-range ratio: {args.min_in_range}')
    print(f'Samples checked: {len(rates)}')
    print(f'Zero-occupancy samples: {zero_count}')
    print(f'Min occupancy rate: {min(rates):.6f}')
    print(f'Mean occupancy rate: {sum(rates) / len(rates):.6f}')
    print(f'Max occupancy rate: {max(rates):.6f}')


if __name__ == '__main__':
    main()
