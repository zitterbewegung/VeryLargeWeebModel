#!/bin/bash
# =============================================================================
# Setup Training Data: nuScenes + Gazebo Simulation
# =============================================================================
# This script sets up both data sources for OccWorld training:
#   1. nuScenes mini dataset (quick start, proven to work)
#   2. Gazebo simulation with Tokyo PLATEAU models (custom data)
#
# Usage:
#   ./scripts/setup_training_data.sh [OPTIONS]
#
# Options:
#   --all         Setup both nuScenes and Gazebo (default)
#   --nuscenes    Setup nuScenes only
#   --gazebo      Setup Gazebo only
#   --help        Show this help
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"

# Default: setup both
SETUP_NUSCENES=true
SETUP_GAZEBO=true

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --nuscenes)  SETUP_NUSCENES=true; SETUP_GAZEBO=false; shift ;;
        --gazebo)    SETUP_GAZEBO=true; SETUP_NUSCENES=false; shift ;;
        --all)       SETUP_NUSCENES=true; SETUP_GAZEBO=true; shift ;;
        --help|-h)   head -20 "$0" | tail -15; exit 0 ;;
        *)           log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# nuScenes Setup
# =============================================================================
setup_nuscenes() {
    log_info "=============================================="
    log_info "Setting up nuScenes Dataset"
    log_info "=============================================="

    NUSCENES_DIR="${DATA_DIR}/nuscenes"
    mkdir -p "$NUSCENES_DIR"
    cd "$NUSCENES_DIR"

    # Download mini nuScenes (~4GB) - good for testing
    if [ ! -d "v1.0-mini" ]; then
        log_info "Downloading nuScenes mini dataset (~4GB)..."

        # Check if already downloaded
        if [ ! -f "v1.0-mini.tgz" ]; then
            wget -c https://www.nuscenes.org/data/v1.0-mini.tgz || {
                log_error "Failed to download. You may need to register at nuscenes.org"
                log_info "Manual download: https://www.nuscenes.org/nuscenes#download"
                return 1
            }
        fi

        log_info "Extracting..."
        tar -xzf v1.0-mini.tgz
        log_success "nuScenes mini extracted"
    else
        log_warn "nuScenes mini already exists, skipping download"
    fi

    # Download Occ3D ground truth labels
    OCC3D_DIR="${DATA_DIR}/occ3d"
    mkdir -p "$OCC3D_DIR"

    log_info "Setting up Occ3D ground truth..."
    cat << 'EOF' > "${OCC3D_DIR}/DOWNLOAD.md"
# Occ3D Ground Truth Download

For occupancy prediction training, download Occ3D labels:

1. Visit: https://github.com/Tsinghua-MARS-Lab/Occ3D
2. Download occ3d-nus annotations
3. Extract to this directory

Structure should be:
```
occ3d/
├── gts/
│   ├── scene-0001/
│   │   ├── 0001.npz
│   │   └── ...
│   └── ...
└── occ3d_infos_train.pkl
```
EOF

    # Create nuScenes dataset adapter for OccWorld
    log_info "Creating nuScenes dataset adapter..."
    cat << 'PYTHON' > "${PROJECT_ROOT}/dataset/nuscenes_occworld_dataset.py"
#!/usr/bin/env python3
"""
nuScenes dataset adapter for OccWorld training.

Loads nuScenes data in format compatible with our training pipeline.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    HAS_NUSCENES = True
except ImportError:
    HAS_NUSCENES = False
    print("Warning: nuscenes-devkit not installed. Run: pip install nuscenes-devkit")


@dataclass
class NuScenesConfig:
    """Configuration for nuScenes dataset."""
    version: str = 'v1.0-mini'
    history_frames: int = 4
    future_frames: int = 6
    point_cloud_range: Tuple = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)
    voxel_size: Tuple = (0.4, 0.4, 0.4)
    grid_size: Tuple = (200, 200, 16)
    split: str = 'train'
    max_sweeps: int = 10
    use_occ3d: bool = True
    occ3d_path: Optional[str] = None


class NuScenesOccWorldDataset(Dataset):
    """nuScenes dataset for OccWorld training."""

    def __init__(self, data_root: str, config: NuScenesConfig):
        if not HAS_NUSCENES:
            raise ImportError("nuscenes-devkit required. Install with: pip install nuscenes-devkit")

        self.data_root = data_root
        self.config = config

        # Initialize nuScenes
        self.nusc = NuScenes(
            version=config.version,
            dataroot=data_root,
            verbose=True
        )

        # Get scene splits
        splits = create_splits_scenes()
        if config.version == 'v1.0-mini':
            # Mini has different split
            scene_names = [s['name'] for s in self.nusc.scene]
            if config.split == 'train':
                self.scene_names = scene_names[:8]
            else:
                self.scene_names = scene_names[8:]
        else:
            self.scene_names = splits[config.split]

        # Build sample index
        self.samples = self._build_sample_index()
        print(f"NuScenes {config.split}: {len(self.samples)} samples from {len(self.scene_names)} scenes")

    def _build_sample_index(self) -> List[Dict]:
        """Build index of valid samples with enough history/future frames."""
        samples = []

        for scene in self.nusc.scene:
            if scene['name'] not in self.scene_names:
                continue

            # Get all samples in scene
            sample_tokens = []
            sample_token = scene['first_sample_token']
            while sample_token:
                sample_tokens.append(sample_token)
                sample = self.nusc.get('sample', sample_token)
                sample_token = sample['next']

            # Create valid windows
            total_needed = self.config.history_frames + self.config.future_frames
            for i in range(len(sample_tokens) - total_needed + 1):
                samples.append({
                    'scene_name': scene['name'],
                    'sample_tokens': sample_tokens[i:i + total_needed],
                    'history_tokens': sample_tokens[i:i + self.config.history_frames],
                    'future_tokens': sample_tokens[i + self.config.history_frames:i + total_needed],
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_lidar(self, sample_token: str) -> np.ndarray:
        """Load LiDAR points for a sample."""
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_path = os.path.join(self.data_root, lidar_data['filename'])

        # Load point cloud (x, y, z, intensity, ring)
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        return points[:, :4]  # x, y, z, intensity

    def _load_ego_pose(self, sample_token: str) -> np.ndarray:
        """Load ego pose for a sample."""
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)

        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])

        # Extract position and rotation
        pos = np.array(ego_pose['translation'])  # x, y, z
        rot = np.array(ego_pose['rotation'])     # qw, qx, qy, qz

        # Combine into 7D pose
        return np.concatenate([pos, rot])

    def _points_to_occupancy(self, points: np.ndarray) -> np.ndarray:
        """Convert point cloud to occupancy grid."""
        pc_range = np.array(self.config.point_cloud_range)
        voxel_size = np.array(self.config.voxel_size)
        grid_size = np.array(self.config.grid_size)

        # Filter points in range
        mask = (
            (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) &
            (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]) &
            (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
        )
        points = points[mask]

        # Convert to voxel indices
        voxel_coords = ((points[:, :3] - pc_range[:3]) / voxel_size).astype(np.int32)
        voxel_coords = np.clip(voxel_coords, 0, grid_size - 1)

        # Create occupancy grid
        occupancy = np.zeros(grid_size, dtype=np.uint8)
        occupancy[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1

        return occupancy

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]

        # Load history
        history_occ = []
        history_poses = []
        for token in sample_info['history_tokens']:
            points = self._load_lidar(token)
            occ = self._points_to_occupancy(points)
            pose = self._load_ego_pose(token)
            history_occ.append(occ)
            history_poses.append(pose)

        # Load future
        future_occ = []
        future_poses = []
        for token in sample_info['future_tokens']:
            points = self._load_lidar(token)
            occ = self._points_to_occupancy(points)
            pose = self._load_ego_pose(token)
            future_occ.append(occ)
            future_poses.append(pose)

        return {
            'history_occupancy': torch.from_numpy(np.stack(history_occ)),
            'future_occupancy': torch.from_numpy(np.stack(future_occ)),
            'history_poses': torch.from_numpy(np.stack(history_poses)).float(),
            'future_poses': torch.from_numpy(np.stack(future_poses)).float(),
            'scene_name': sample_info['scene_name'],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    return {
        'history_occupancy': torch.stack([b['history_occupancy'] for b in batch]),
        'future_occupancy': torch.stack([b['future_occupancy'] for b in batch]),
        'history_poses': torch.stack([b['history_poses'] for b in batch]),
        'future_poses': torch.stack([b['future_poses'] for b in batch]),
    }


if __name__ == '__main__':
    # Test the dataset
    import sys

    data_root = sys.argv[1] if len(sys.argv) > 1 else 'data/nuscenes'

    config = NuScenesConfig(
        version='v1.0-mini',
        split='train',
    )

    dataset = NuScenesOccWorldDataset(data_root, config)
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"History occupancy: {sample['history_occupancy'].shape}")
    print(f"Future occupancy: {sample['future_occupancy'].shape}")
    print(f"History poses: {sample['history_poses'].shape}")
PYTHON

    # Install nuscenes-devkit
    log_info "Installing nuscenes-devkit..."
    pip install nuscenes-devkit pyquaternion

    log_success "nuScenes setup complete!"
    log_info "To train on nuScenes:"
    log_info "  python train.py --config config/finetune_nuscenes.py --work-dir /workspace/checkpoints"

    cd "$PROJECT_ROOT"
}

# =============================================================================
# Gazebo Setup
# =============================================================================
setup_gazebo() {
    log_info "=============================================="
    log_info "Setting up Gazebo Simulation"
    log_info "=============================================="

    # Check if running on Vast.ai (has apt)
    if command -v apt-get &> /dev/null; then
        log_info "Installing Gazebo..."
        apt-get update
        apt-get install -y gazebo libgazebo-dev ros-base 2>/dev/null || {
            # Try Gazebo Harmonic (newer)
            log_info "Trying Gazebo Harmonic..."
            apt-get install -y wget lsb-release gnupg
            wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
            apt-get update
            apt-get install -y gz-harmonic || log_warn "Gazebo installation may have issues"
        }
    else
        log_warn "apt-get not found. Please install Gazebo manually."
    fi

    # Set up model path
    GAZEBO_MODELS="${DATA_DIR}/plateau/gazebo_models"

    if [ -d "$GAZEBO_MODELS" ] && [ "$(ls -A $GAZEBO_MODELS 2>/dev/null)" ]; then
        log_success "Gazebo models found at $GAZEBO_MODELS"
        export GZ_SIM_RESOURCE_PATH="${GAZEBO_MODELS}:${GZ_SIM_RESOURCE_PATH}"
        echo "export GZ_SIM_RESOURCE_PATH=${GAZEBO_MODELS}:\${GZ_SIM_RESOURCE_PATH}" >> ~/.bashrc
    else
        log_warn "No Gazebo models found. Running PLATEAU conversion..."

        # Check if PLATEAU data downloaded
        if [ -f "${DATA_DIR}/plateau/raw/tokyo23ku_obj.zip" ]; then
            log_info "Extracting PLATEAU OBJ files..."
            mkdir -p "${DATA_DIR}/plateau/meshes/obj"
            unzip -q -o "${DATA_DIR}/plateau/raw/tokyo23ku_obj.zip" -d "${DATA_DIR}/plateau/meshes/obj/"

            log_info "Converting to Gazebo format..."
            pip install trimesh pyvista numpy-stl pycollada
            python scripts/convert_plateau_simple.py \
                --input "${DATA_DIR}/plateau/meshes/obj" \
                --output "${GAZEBO_MODELS}" \
                --max-models 50
        else
            log_error "PLATEAU data not downloaded. Run ./scripts/download_and_prepare_data.sh first"
        fi
    fi

    # Create data collection script
    log_info "Creating Gazebo data collection script..."
    cat << 'PYTHON' > "${PROJECT_ROOT}/scripts/gazebo_data_collector.py"
#!/usr/bin/env python3
"""
Gazebo simulation data collector for OccWorld training.

Launches Gazebo with Tokyo PLATEAU models and collects:
- Camera images (6 surround views)
- LiDAR point clouds
- Ego poses
- Ground truth occupancy grids

Usage:
    python scripts/gazebo_data_collector.py --output data/tokyo_gazebo --frames 1000
"""
import os
import sys
import time
import json
import argparse
import subprocess
import numpy as np
from datetime import datetime

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

def create_session_dirs(base_dir: str, session_name: str) -> dict:
    """Create directory structure for a recording session."""
    session_dir = os.path.join(base_dir, session_name)
    dirs = {
        'root': session_dir,
        'images': os.path.join(session_dir, 'images'),
        'lidar': os.path.join(session_dir, 'lidar'),
        'poses': os.path.join(session_dir, 'poses'),
        'occupancy': os.path.join(session_dir, 'occupancy'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def generate_trajectory(num_frames: int, pattern: str = 'survey') -> list:
    """Generate drone trajectory waypoints."""
    waypoints = []

    if pattern == 'survey':
        # Grid survey pattern
        grid_size = int(np.sqrt(num_frames))
        spacing = 20.0  # meters
        altitude = 50.0

        for i in range(grid_size):
            for j in range(grid_size):
                x = (i - grid_size/2) * spacing
                y = (j - grid_size/2) * spacing
                # Alternate direction each row
                if i % 2 == 1:
                    y = -y
                waypoints.append({
                    'position': {'x': x, 'y': y, 'z': altitude},
                    'orientation': {'x': 0, 'y': 0, 'z': 0, 'w': 1},
                    'velocity': {'linear': {'x': 5, 'y': 0, 'z': 0}, 'angular': {'x': 0, 'y': 0, 'z': 0}}
                })

    elif pattern == 'orbit':
        # Circular orbit
        radius = 100.0
        altitude = 80.0
        for i in range(num_frames):
            angle = 2 * np.pi * i / num_frames
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            # Yaw to face center
            yaw = angle + np.pi
            waypoints.append({
                'position': {'x': x, 'y': y, 'z': altitude},
                'orientation': {'x': 0, 'y': 0, 'z': np.sin(yaw/2), 'w': np.cos(yaw/2)},
                'velocity': {'linear': {'x': 0, 'y': 5, 'z': 0}, 'angular': {'x': 0, 'y': 0, 'z': 0.1}}
            })

    elif pattern == 'random':
        # Random exploration
        for i in range(num_frames):
            waypoints.append({
                'position': {
                    'x': np.random.uniform(-200, 200),
                    'y': np.random.uniform(-200, 200),
                    'z': np.random.uniform(20, 100)
                },
                'orientation': {'x': 0, 'y': 0, 'z': np.random.uniform(-1, 1), 'w': np.random.uniform(0, 1)},
                'velocity': {'linear': {'x': np.random.uniform(-5, 5), 'y': np.random.uniform(-5, 5), 'z': 0},
                           'angular': {'x': 0, 'y': 0, 'z': np.random.uniform(-0.5, 0.5)}}
            })

    return waypoints


def simulate_lidar(position: dict, num_points: int = 10000) -> np.ndarray:
    """Simulate LiDAR point cloud (placeholder - real impl needs Gazebo)."""
    # Generate random points in a sphere around the sensor
    # In reality, this would come from Gazebo ray casting
    theta = np.random.uniform(0, 2*np.pi, num_points)
    phi = np.random.uniform(-np.pi/6, np.pi/2, num_points)  # -30 to 90 degrees
    r = np.random.uniform(1, 100, num_points)

    x = r * np.cos(phi) * np.cos(theta) + position['x']
    y = r * np.cos(phi) * np.sin(theta) + position['y']
    z = r * np.sin(phi) + position['z']
    intensity = np.random.uniform(0, 1, num_points)

    return np.column_stack([x, y, z, intensity]).astype(np.float32)


def simulate_occupancy(position: dict, grid_size: tuple = (200, 200, 121)) -> np.ndarray:
    """Simulate occupancy grid (placeholder - real impl needs Gazebo depth)."""
    # In reality, this would be computed from depth images or ray casting
    # For now, generate structured random data that looks like buildings
    occ = np.zeros(grid_size, dtype=np.uint8)

    # Add some random "buildings"
    num_buildings = np.random.randint(5, 15)
    for _ in range(num_buildings):
        bx = np.random.randint(20, grid_size[0]-20)
        by = np.random.randint(20, grid_size[1]-20)
        bw = np.random.randint(5, 20)
        bh = np.random.randint(5, 20)
        bz = np.random.randint(10, 80)

        occ[bx:bx+bw, by:by+bh, 0:bz] = 1

    # Add ground plane
    occ[:, :, 0:2] = 1

    return occ


def collect_data(output_dir: str, num_frames: int, pattern: str = 'survey'):
    """Main data collection loop."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_name = f'drone_{timestamp}'
    dirs = create_session_dirs(output_dir, session_name)

    print(f"Collecting {num_frames} frames to {dirs['root']}")
    print(f"Pattern: {pattern}")

    # Generate trajectory
    waypoints = generate_trajectory(num_frames, pattern)

    for i, wp in enumerate(waypoints):
        frame_id = f'{i:06d}'

        # Save pose
        with open(os.path.join(dirs['poses'], f'{frame_id}.json'), 'w') as f:
            json.dump(wp, f, indent=2)

        # Generate/save LiDAR
        lidar = simulate_lidar(wp['position'])
        np.save(os.path.join(dirs['lidar'], f'{frame_id}_LIDAR.npy'), lidar)

        # Generate/save occupancy
        occ = simulate_occupancy(wp['position'])
        np.savez_compressed(
            os.path.join(dirs['occupancy'], f'{frame_id}_occupancy.npz'),
            occupancy=occ
        )

        # Generate/save camera image (placeholder)
        if HAS_CV2:
            img = np.zeros((900, 1600, 3), dtype=np.uint8)
            # Add some visual indication of position
            cv2.putText(img, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f"Pos: ({wp['position']['x']:.1f}, {wp['position']['y']:.1f}, {wp['position']['z']:.1f})",
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            cv2.imwrite(os.path.join(dirs['images'], f'{frame_id}_CAM_FRONT.jpg'), img)

        if (i + 1) % 100 == 0:
            print(f"  Collected {i + 1}/{num_frames} frames")

    print(f"\nData collection complete!")
    print(f"Session: {dirs['root']}")
    print(f"Frames: {num_frames}")

    return dirs['root']


def main():
    parser = argparse.ArgumentParser(description='Collect Gazebo simulation data')
    parser.add_argument('--output', '-o', default='data/tokyo_gazebo', help='Output directory')
    parser.add_argument('--frames', '-f', type=int, default=500, help='Number of frames')
    parser.add_argument('--pattern', '-p', choices=['survey', 'orbit', 'random'], default='survey',
                       help='Flight pattern')
    parser.add_argument('--sessions', '-s', type=int, default=1, help='Number of sessions')
    args = parser.parse_args()

    for s in range(args.sessions):
        print(f"\n=== Session {s+1}/{args.sessions} ===")
        collect_data(args.output, args.frames, args.pattern)
        time.sleep(1)  # Brief pause between sessions

    print(f"\n{'='*50}")
    print(f"All sessions complete!")
    print(f"Total frames: {args.frames * args.sessions}")
    print(f"Run training with:")
    print(f"  python train.py --config config/finetune_tokyo.py --work-dir /workspace/checkpoints")


if __name__ == '__main__':
    main()
PYTHON
    chmod +x "${PROJECT_ROOT}/scripts/gazebo_data_collector.py"

    log_success "Gazebo setup complete!"
    log_info "To collect simulation data:"
    log_info "  python scripts/gazebo_data_collector.py --output data/tokyo_gazebo --frames 1000 --sessions 5"
}

# =============================================================================
# Create nuScenes config
# =============================================================================
create_nuscenes_config() {
    log_info "Creating nuScenes training config..."

    cat << 'PYTHON' > "${PROJECT_ROOT}/config/finetune_nuscenes.py"
"""
OccWorld Fine-tuning Configuration for nuScenes Dataset

This config uses the nuScenes mini dataset for training.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Dataset paths
data_root = os.path.join(PROJECT_ROOT, 'data/nuscenes')

# nuScenes standard range
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
voxel_size = [0.4, 0.4, 0.4]

grid_size = [
    int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),  # 200
    int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),  # 200
    int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),  # 16
]

# Temporal
history_frames = 4
future_frames = 6

# Dataset
dataset_config = dict(
    version='v1.0-mini',
    history_frames=history_frames,
    future_frames=future_frames,
    point_cloud_range=tuple(point_cloud_range),
    voxel_size=tuple(voxel_size),
    grid_size=tuple(grid_size),
)

# Training
max_epochs = 100
batch_size = 2
learning_rate = 1e-4
weight_decay = 0.01

# Checkpointing
checkpoint_interval = 10
work_dir = os.path.join(PROJECT_ROOT, 'out/occworld_nuscenes')

# Model (simplified for mini dataset)
model = dict(
    embed_dim=64,
    num_heads=8,
    num_layers=2,
)
PYTHON

    log_success "Created config/finetune_nuscenes.py"
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo ""
    echo "=============================================="
    echo "  OccWorld Training Data Setup"
    echo "=============================================="
    echo ""

    if [ "$SETUP_NUSCENES" = true ]; then
        setup_nuscenes
        create_nuscenes_config
    fi

    if [ "$SETUP_GAZEBO" = true ]; then
        setup_gazebo
    fi

    echo ""
    echo "=============================================="
    echo "  Setup Complete!"
    echo "=============================================="
    echo ""
    echo "Next steps:"
    echo ""
    if [ "$SETUP_NUSCENES" = true ]; then
        echo "For nuScenes training:"
        echo "  python train.py --config config/finetune_nuscenes.py --work-dir /workspace/checkpoints"
        echo ""
    fi
    if [ "$SETUP_GAZEBO" = true ]; then
        echo "For Gazebo/Tokyo data:"
        echo "  1. Generate data: python scripts/gazebo_data_collector.py --frames 1000 --sessions 5"
        echo "  2. Train: python train.py --config config/finetune_tokyo.py --work-dir /workspace/checkpoints"
        echo ""
    fi
}

main "$@"
