#!/usr/bin/env python3
"""
PLATEAU Mesh to VeryLargeWeebModel Training Data Converter

Converts Tokyo PLATEAU 3D city models directly into VeryLargeWeebModel training format
by voxelizing meshes and generating synthetic trajectories.

This bypasses the need for Gazebo simulation by:
1. Loading PLATEAU OBJ/DAE mesh files
2. Voxelizing meshes into occupancy grids
3. Generating drone/rover trajectories through the city
4. Creating training samples with proper temporal sequences

Usage:
    python scripts/plateau_to_occworld.py --input data/plateau/meshes/obj --output data/tokyo_gazebo
    python scripts/plateau_to_occworld.py --input data/plateau/gazebo_models --output data/tokyo_gazebo --frames 500
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("Warning: trimesh not installed. Run: pip install trimesh")

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# Numba-optimized voxelization kernel
if HAS_NUMBA:
    @jit(nopython=True, parallel=True, cache=True)
    def _voxelize_points_numba(
        vertices: np.ndarray,
        occupancy: np.ndarray,
        voxel_origin: np.ndarray,
        voxel_size: np.ndarray,
        grid_size: Tuple[int, int, int]
    ) -> np.ndarray:
        """Numba-optimized voxelization kernel (5-10x faster)."""
        for i in prange(vertices.shape[0]):
            # Calculate voxel indices
            vx = int((vertices[i, 0] - voxel_origin[0]) / voxel_size[0])
            vy = int((vertices[i, 1] - voxel_origin[1]) / voxel_size[1])
            vz = int((vertices[i, 2] - voxel_origin[2]) / voxel_size[2])

            # Bounds check and set
            if 0 <= vx < grid_size[0] and 0 <= vy < grid_size[1] and 0 <= vz < grid_size[2]:
                occupancy[vx, vy, vz] = 1

        return occupancy

    @jit(nopython=True, cache=True)
    def _apply_yaw_rotation_numba(
        vertices: np.ndarray,
        cos_yaw: float,
        sin_yaw: float
    ) -> np.ndarray:
        """Numba-optimized yaw rotation."""
        result = np.empty_like(vertices)
        for i in range(vertices.shape[0]):
            result[i, 0] = cos_yaw * vertices[i, 0] - sin_yaw * vertices[i, 1]
            result[i, 1] = sin_yaw * vertices[i, 0] + cos_yaw * vertices[i, 1]
            result[i, 2] = vertices[i, 2]
        return result
else:
    _voxelize_points_numba = None
    _apply_yaw_rotation_numba = None


@dataclass
class VoxelConfig:
    """Voxelization configuration matching VeryLargeWeebModel expectations."""
    # Point cloud range [xmin, ymin, zmin, xmax, ymax, zmax]
    point_cloud_range: Tuple[float, ...] = (-40.0, -40.0, -2.0, 40.0, 40.0, 150.0)
    voxel_size: Tuple[float, float, float] = (0.4, 0.4, 1.25)

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        return (
            int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0]),
            int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1]),
            int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.voxel_size[2]),
        )


class PLATEAUVoxelizer:
    """Voxelize PLATEAU mesh models into occupancy grids."""

    def __init__(self, config: VoxelConfig):
        self.config = config
        self.meshes = []
        self.combined_mesh = None

    def load_meshes(self, input_path: str, max_meshes: int = 50, num_workers: int = None) -> int:
        """Load mesh files from directory using parallel I/O."""
        input_path = Path(input_path)

        # Find mesh files
        mesh_files = []
        for ext in ['*.obj', '*.dae', '*.stl', '*.ply']:
            mesh_files.extend(input_path.rglob(ext))

        print(f"Found {len(mesh_files)} mesh files")

        if max_meshes > 0:
            mesh_files = mesh_files[:max_meshes]

        if num_workers is None:
            num_workers = min(8, multiprocessing.cpu_count())

        def load_single_mesh(mesh_file):
            """Load a single mesh file."""
            try:
                mesh = trimesh.load(str(mesh_file), force='mesh')
                if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                    return mesh, None
            except Exception as e:
                return None, f"  Skipped {mesh_file.name}: {e}"
            return None, None

        # Load meshes in parallel using ThreadPoolExecutor (I/O bound)
        print(f"Loading meshes with {num_workers} workers...")
        loaded = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(load_single_mesh, f): f for f in mesh_files}
            for future in as_completed(futures):
                mesh, error = future.result()
                if mesh is not None:
                    self.meshes.append(mesh)
                    loaded += 1
                elif error:
                    print(error)

        print(f"Loaded {loaded} meshes")
        return loaded

    def combine_meshes(self, spacing: float = 50.0) -> trimesh.Trimesh:
        """Combine meshes into a single scene with grid layout."""
        if not self.meshes:
            raise ValueError("No meshes loaded")

        # Arrange meshes in a grid
        grid_size = int(np.ceil(np.sqrt(len(self.meshes))))

        combined_vertices = []
        combined_faces = []
        vertex_offset = 0

        for i, mesh in enumerate(self.meshes):
            # Grid position
            x_offset = (i % grid_size) * spacing
            y_offset = (i // grid_size) * spacing

            # Offset vertices
            vertices = mesh.vertices.copy()
            vertices[:, 0] += x_offset
            vertices[:, 1] += y_offset

            # Offset face indices
            faces = mesh.faces.copy() + vertex_offset

            combined_vertices.append(vertices)
            combined_faces.append(faces)
            vertex_offset += len(mesh.vertices)

        combined_vertices = np.vstack(combined_vertices)
        combined_faces = np.vstack(combined_faces)

        self.combined_mesh = trimesh.Trimesh(
            vertices=combined_vertices,
            faces=combined_faces
        )

        print(f"Combined mesh: {len(combined_vertices)} vertices, {len(combined_faces)} faces")
        return self.combined_mesh

    def voxelize_at_position(
        self,
        position: np.ndarray,
        yaw: float = 0.0
    ) -> np.ndarray:
        """
        Create occupancy grid centered at given position.

        Args:
            position: [x, y, z] center position in world frame
            yaw: rotation angle in radians

        Returns:
            occupancy: [X, Y, Z] uint8 grid (0=empty, 1=occupied)
        """
        if self.combined_mesh is None:
            self.combine_meshes()

        cfg = self.config
        grid_size = cfg.grid_size

        # Calculate world bounds for this position
        range_min = np.array([
            position[0] + cfg.point_cloud_range[0],
            position[1] + cfg.point_cloud_range[1],
            position[2] + cfg.point_cloud_range[2],
        ])
        range_max = np.array([
            position[0] + cfg.point_cloud_range[3],
            position[1] + cfg.point_cloud_range[4],
            position[2] + cfg.point_cloud_range[5],
        ])

        # Get mesh vertices in range
        vertices = self.combined_mesh.vertices
        in_range = np.all((vertices >= range_min) & (vertices < range_max), axis=1)

        if not np.any(in_range):
            # No geometry in view, return ground plane only
            occupancy = np.zeros(grid_size, dtype=np.uint8)
            # Add ground plane at z=0
            ground_z = int((0 - cfg.point_cloud_range[2]) / cfg.voxel_size[2])
            if 0 <= ground_z < grid_size[2]:
                occupancy[:, :, ground_z] = 1
            return occupancy

        vertices_local = vertices[in_range] - np.array([position[0], position[1], 0])

        # Apply yaw rotation (use Numba if available)
        if yaw != 0:
            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            if HAS_NUMBA and _apply_yaw_rotation_numba is not None:
                vertices_local = _apply_yaw_rotation_numba(
                    vertices_local.astype(np.float64), cos_yaw, sin_yaw
                )
            else:
                rot_matrix = np.array([
                    [cos_yaw, -sin_yaw, 0],
                    [sin_yaw, cos_yaw, 0],
                    [0, 0, 1]
                ])
                vertices_local = (rot_matrix @ vertices_local.T).T

        # Convert to voxel indices and fill grid
        voxel_size = np.array(cfg.voxel_size, dtype=np.float64)
        voxel_origin = np.array(cfg.point_cloud_range[:3], dtype=np.float64)

        # Create occupancy grid
        occupancy = np.zeros(grid_size, dtype=np.uint8)

        # Use Numba-optimized voxelization if available (5-10x faster)
        if HAS_NUMBA and _voxelize_points_numba is not None:
            occupancy = _voxelize_points_numba(
                vertices_local.astype(np.float64),
                occupancy,
                voxel_origin,
                voxel_size,
                grid_size
            )
        else:
            # Fallback to numpy implementation
            voxel_indices = np.floor((vertices_local - voxel_origin) / voxel_size).astype(np.int32)
            valid = np.all((voxel_indices >= 0) & (voxel_indices < grid_size), axis=1)
            voxel_indices = voxel_indices[valid]
            if len(voxel_indices) > 0:
                occupancy[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1

        # Add ground plane
        ground_z = int((0 - cfg.point_cloud_range[2]) / cfg.voxel_size[2])
        if 0 <= ground_z < grid_size[2]:
            occupancy[:, :, ground_z] = np.maximum(occupancy[:, :, ground_z], 1)

        return occupancy


class TrajectoryGenerator:
    """Generate realistic drone/rover trajectories."""

    def __init__(self, bounds: Tuple[float, float, float, float]):
        """
        Args:
            bounds: (x_min, y_min, x_max, y_max) trajectory bounds
        """
        self.bounds = bounds

    def generate_survey_pattern(
        self,
        num_frames: int,
        altitude: float = 50.0,
        speed: float = 5.0
    ) -> List[Dict]:
        """Generate grid survey pattern for drone."""
        waypoints = []

        x_min, y_min, x_max, y_max = self.bounds

        # Calculate grid dimensions
        rows = int(np.sqrt(num_frames))
        cols = num_frames // rows

        x_spacing = (x_max - x_min) / cols
        y_spacing = (y_max - y_min) / rows

        frame = 0
        for row in range(rows):
            y = y_min + (row + 0.5) * y_spacing

            # Alternate direction each row (lawn mower pattern)
            col_range = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)

            for col in col_range:
                if frame >= num_frames:
                    break

                x = x_min + (col + 0.5) * x_spacing

                # Calculate yaw (heading direction)
                yaw = 0 if row % 2 == 0 else np.pi

                waypoints.append(self._create_waypoint(
                    x, y, altitude, yaw, speed, 'drone'
                ))
                frame += 1

        return waypoints

    def generate_orbit_pattern(
        self,
        num_frames: int,
        center: Tuple[float, float] = (0, 0),
        radius: float = 100.0,
        altitude: float = 80.0,
        speed: float = 5.0
    ) -> List[Dict]:
        """Generate circular orbit pattern."""
        waypoints = []

        for i in range(num_frames):
            angle = 2 * np.pi * i / num_frames
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)

            # Face toward center
            yaw = angle + np.pi

            waypoints.append(self._create_waypoint(
                x, y, altitude, yaw, speed, 'drone'
            ))

        return waypoints

    def generate_random_walk(
        self,
        num_frames: int,
        altitude_range: Tuple[float, float] = (30.0, 100.0),
        speed: float = 3.0,
        agent_type: str = 'drone'
    ) -> List[Dict]:
        """Generate random exploration trajectory."""
        waypoints = []

        x_min, y_min, x_max, y_max = self.bounds

        # Start position
        x = (x_min + x_max) / 2
        y = (y_min + y_max) / 2
        z = np.mean(altitude_range)
        yaw = 0

        for i in range(num_frames):
            # Random walk with momentum
            dx = np.random.uniform(-5, 5)
            dy = np.random.uniform(-5, 5)
            dz = np.random.uniform(-2, 2) if agent_type == 'drone' else 0
            dyaw = np.random.uniform(-0.2, 0.2)

            x = np.clip(x + dx, x_min, x_max)
            y = np.clip(y + dy, y_min, y_max)
            z = np.clip(z + dz, altitude_range[0], altitude_range[1])
            yaw = yaw + dyaw

            waypoints.append(self._create_waypoint(
                x, y, z, yaw, speed, agent_type
            ))

        return waypoints

    def generate_rover_patrol(
        self,
        num_frames: int,
        altitude: float = 1.5,  # Ground level camera height
        speed: float = 2.0
    ) -> List[Dict]:
        """Generate ground-level rover patrol trajectory."""
        waypoints = []

        x_min, y_min, x_max, y_max = self.bounds

        # Start at random position
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        yaw = np.random.uniform(0, 2 * np.pi)

        for i in range(num_frames):
            # Ground vehicle dynamics - smoother turns, no altitude change
            dyaw = np.random.uniform(-0.15, 0.15)  # Gentler turns than drone
            yaw = yaw + dyaw

            # Move forward with occasional speed variation
            current_speed = speed * np.random.uniform(0.8, 1.2)
            dx = current_speed * np.cos(yaw)
            dy = current_speed * np.sin(yaw)

            # Bounce off boundaries
            new_x = x + dx
            new_y = y + dy

            if new_x < x_min or new_x > x_max:
                yaw = np.pi - yaw  # Reflect
                new_x = np.clip(new_x, x_min, x_max)
            if new_y < y_min or new_y > y_max:
                yaw = -yaw  # Reflect
                new_y = np.clip(new_y, y_min, y_max)

            x, y = new_x, new_y

            waypoints.append(self._create_waypoint(
                x, y, altitude, yaw, current_speed, 'rover'
            ))

        return waypoints

    def _create_waypoint(
        self,
        x: float, y: float, z: float,
        yaw: float, speed: float,
        agent_type: str
    ) -> Dict:
        """Create waypoint in dataset format."""
        # Quaternion from yaw
        qw = np.cos(yaw / 2)
        qz = np.sin(yaw / 2)

        return {
            'position': {'x': float(x), 'y': float(y), 'z': float(z)},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': float(qz), 'w': float(qw)},
            'velocity': {
                'linear': {'x': float(speed * np.cos(yaw)), 'y': float(speed * np.sin(yaw)), 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
            },
            'agent_type': agent_type
        }


def generate_lidar_from_occupancy(
    occupancy: np.ndarray,
    position: np.ndarray,
    config: VoxelConfig,
    num_points: int = 10000
) -> np.ndarray:
    """Generate synthetic LiDAR points from occupancy grid."""
    # Find occupied voxels
    occupied_indices = np.argwhere(occupancy > 0)

    if len(occupied_indices) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Convert to world coordinates
    voxel_size = np.array(config.voxel_size)
    range_min = np.array(config.point_cloud_range[:3])

    # Sample points from occupied voxels
    if len(occupied_indices) > num_points:
        indices = np.random.choice(len(occupied_indices), num_points, replace=False)
        occupied_indices = occupied_indices[indices]

    # Add random offset within voxel
    points = (occupied_indices + np.random.uniform(0, 1, occupied_indices.shape)) * voxel_size + range_min

    # Transform to sensor frame (relative to position)
    points_local = points - position

    # Add intensity based on distance
    distances = np.linalg.norm(points_local, axis=1)
    intensities = np.clip(1.0 - distances / 100.0, 0.1, 1.0)

    # Combine into [x, y, z, intensity]
    lidar_points = np.column_stack([points_local, intensities]).astype(np.float32)

    return lidar_points


def create_dummy_image(frame_id: int, position: Dict, grid_size: Tuple[int, int] = (900, 1600)) -> np.ndarray:
    """Create a placeholder camera image."""
    img = np.zeros((grid_size[0], grid_size[1], 3), dtype=np.uint8)

    # Add gradient background (sky simulation)
    for y in range(grid_size[0]):
        blue_val = int(200 * (1 - y / grid_size[0]))
        img[y, :] = [blue_val + 55, blue_val + 30, blue_val]

    # Add position text
    try:
        import cv2
        pos_text = f"Frame {frame_id} | Pos: ({position['x']:.1f}, {position['y']:.1f}, {position['z']:.1f})"
        cv2.putText(img, pos_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    except ImportError:
        pass

    return img


def save_image(img: np.ndarray, path: str):
    """Save image using available library."""
    try:
        import cv2
        cv2.imwrite(path, img)
    except ImportError:
        try:
            from PIL import Image
            Image.fromarray(img).save(path)
        except ImportError:
            np.save(path.replace('.jpg', '.npy'), img)


def process_single_frame(args):
    """
    Process a single frame - designed for parallel execution.

    Args:
        args: tuple of (frame_idx, waypoint, session_dir, voxelizer, config)

    Returns:
        tuple of (frame_idx, occupied_voxel_count)
    """
    frame_idx, waypoint, session_dir, combined_mesh_data, config = args

    # Reconstruct minimal voxelizer state for this frame
    frame_id = f'{frame_idx:06d}'
    pos = waypoint['position']
    ori = waypoint['orientation']

    position = np.array([pos['x'], pos['y'], pos['z']])
    yaw = 2 * np.arctan2(ori['z'], ori['w'])

    # Create occupancy grid inline (avoiding shared state issues)
    cfg = config
    grid_size = cfg.grid_size

    range_min = np.array([
        position[0] + cfg.point_cloud_range[0],
        position[1] + cfg.point_cloud_range[1],
        position[2] + cfg.point_cloud_range[2],
    ])
    range_max = np.array([
        position[0] + cfg.point_cloud_range[3],
        position[1] + cfg.point_cloud_range[4],
        position[2] + cfg.point_cloud_range[5],
    ])

    # Use pre-computed mesh vertices from combined_mesh_data
    vertices = combined_mesh_data
    in_range = np.all((vertices >= range_min) & (vertices < range_max), axis=1)

    if not np.any(in_range):
        occupancy = np.zeros(grid_size, dtype=np.uint8)
        ground_z = int((0 - cfg.point_cloud_range[2]) / cfg.voxel_size[2])
        if 0 <= ground_z < grid_size[2]:
            occupancy[:, :, ground_z] = 1
    else:
        vertices_local = vertices[in_range] - np.array([position[0], position[1], 0])

        if yaw != 0:
            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            rot_matrix = np.array([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1]
            ])
            vertices_local = (rot_matrix @ vertices_local.T).T

        voxel_size = np.array(cfg.voxel_size)
        voxel_origin = np.array(cfg.point_cloud_range[:3])
        voxel_indices = np.floor((vertices_local - voxel_origin) / voxel_size).astype(np.int32)

        valid = np.all((voxel_indices >= 0) & (voxel_indices < grid_size), axis=1)
        voxel_indices = voxel_indices[valid]

        occupancy = np.zeros(grid_size, dtype=np.uint8)
        if len(voxel_indices) > 0:
            occupancy[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1

        ground_z = int((0 - cfg.point_cloud_range[2]) / cfg.voxel_size[2])
        if 0 <= ground_z < grid_size[2]:
            occupancy[:, :, ground_z] = np.maximum(occupancy[:, :, ground_z], 1)

    # Save occupancy
    np.savez_compressed(
        os.path.join(session_dir, 'occupancy', f'{frame_id}_occupancy.npz'),
        occupancy=occupancy
    )

    # Save pose
    with open(os.path.join(session_dir, 'poses', f'{frame_id}.json'), 'w') as f:
        json.dump(waypoint, f, indent=2)

    # Generate LiDAR
    lidar_points = generate_lidar_from_occupancy(occupancy, position, config)
    np.save(os.path.join(session_dir, 'lidar', f'{frame_id}_LIDAR.npy'), lidar_points)

    # Generate placeholder image
    img = create_dummy_image(frame_idx, pos)
    save_image(img, os.path.join(session_dir, 'images', f'{frame_id}_CAM_FRONT.jpg'))

    return frame_idx, np.sum(occupancy > 0)


def main():
    parser = argparse.ArgumentParser(description='Convert PLATEAU meshes to VeryLargeWeebModel training data')
    parser.add_argument('--input', '-i', required=True, help='Input mesh directory')
    parser.add_argument('--output', '-o', default='data/tokyo_gazebo', help='Output data directory')
    parser.add_argument('--frames', '-f', type=int, default=100, help='Frames per session')
    parser.add_argument('--sessions', '-s', type=int, default=3, help='Number of sessions')
    parser.add_argument('--max-meshes', '-m', type=int, default=50, help='Max meshes to load')
    parser.add_argument('--pattern', '-p', choices=['survey', 'orbit', 'random'],
                        default='survey', help='Trajectory pattern')
    parser.add_argument('--agent', '-a', choices=['drone', 'rover'], default='drone', help='Agent type')
    parser.add_argument('--allow-synthetic', action='store_true',
                        help='Allow synthetic building generation if no real meshes found (for testing only)')
    parser.add_argument('--synthetic-buildings', type=int, default=100,
                        help='Number of synthetic buildings to generate (only with --allow-synthetic)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of parallel workers (default: auto-detect CPU cores)')
    args = parser.parse_args()

    # Set number of workers
    if args.workers is None:
        args.workers = min(8, multiprocessing.cpu_count())
    print(f"Using {args.workers} parallel workers")

    if not HAS_TRIMESH:
        print("Error: trimesh is required. Install with: pip install trimesh")
        sys.exit(1)

    print("=" * 60)
    print("PLATEAU to VeryLargeWeebModel Converter")
    print("=" * 60)

    # Initialize voxelizer
    config = VoxelConfig()
    voxelizer = PLATEAUVoxelizer(config)

    # Load meshes
    print(f"\nLoading meshes from {args.input}...")
    loaded = voxelizer.load_meshes(args.input, args.max_meshes)

    if loaded == 0:
        if not args.allow_synthetic:
            print("=" * 60)
            print("ERROR: No real PLATEAU meshes found!")
            print("=" * 60)
            print(f"\nInput directory: {args.input}")
            print("\nTo download real Tokyo 3D city data, run:")
            print("  ./scripts/download_and_prepare_data.sh --plateau")
            print("\nOr download manually:")
            print("  mkdir -p data/plateau/raw data/plateau/meshes/obj")
            print("  wget -O data/plateau/raw/tokyo23ku_obj.zip \\")
            print("    'https://gic-plateau.s3.ap-northeast-1.amazonaws.com/2020/13100_tokyo23-ku_2020_obj_3_op.zip'")
            print("  unzip data/plateau/raw/tokyo23ku_obj.zip -d data/plateau/meshes/obj/")
            print("\nTo use synthetic data for TESTING ONLY, add: --allow-synthetic")
            print("WARNING: Synthetic data will cause model to overfit quickly!")
            sys.exit(1)

        print("=" * 60)
        print("WARNING: Using SYNTHETIC buildings (testing mode)")
        print("=" * 60)
        print("This data is for TESTING ONLY - model will overfit!")
        print("For real training, download PLATEAU data first.\n")

        # Create more diverse synthetic buildings
        buildings = []
        num_buildings = args.synthetic_buildings

        # Create varied building types for more diversity
        for i in range(num_buildings):
            x = np.random.uniform(-500, 500)
            y = np.random.uniform(-500, 500)

            # Vary building types
            building_type = np.random.choice(['small', 'medium', 'tall', 'wide'])
            if building_type == 'small':
                w, h, z = np.random.uniform(5, 15), np.random.uniform(5, 15), np.random.uniform(10, 30)
            elif building_type == 'medium':
                w, h, z = np.random.uniform(15, 30), np.random.uniform(15, 30), np.random.uniform(30, 60)
            elif building_type == 'tall':
                w, h, z = np.random.uniform(10, 20), np.random.uniform(10, 20), np.random.uniform(60, 150)
            else:  # wide
                w, h, z = np.random.uniform(30, 60), np.random.uniform(30, 60), np.random.uniform(15, 40)

            building = trimesh.creation.box(extents=[w, h, z])
            building.apply_translation([x, y, z/2])
            buildings.append(building)

        voxelizer.meshes = buildings
        print(f"Created {len(buildings)} synthetic buildings (testing mode)")

    # Combine meshes
    print("\nCombining meshes...")
    combined = voxelizer.combine_meshes(spacing=100.0)

    # Get bounds for trajectory
    mesh_bounds = combined.bounds
    bounds = (
        float(mesh_bounds[0, 0]), float(mesh_bounds[0, 1]),
        float(mesh_bounds[1, 0]), float(mesh_bounds[1, 1])
    )
    print(f"Scene bounds: X[{bounds[0]:.1f}, {bounds[2]:.1f}] Y[{bounds[1]:.1f}, {bounds[3]:.1f}]")

    # Initialize trajectory generator
    traj_gen = TrajectoryGenerator(bounds)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Generate sessions
    for session_idx in range(args.sessions):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_name = f'{args.agent}_{timestamp}_{session_idx:02d}'
        session_dir = os.path.join(args.output, session_name)

        print(f"\n{'=' * 40}")
        print(f"Session {session_idx + 1}/{args.sessions}: {session_name}")
        print(f"{'=' * 40}")

        # Create directories
        for subdir in ['occupancy', 'poses', 'lidar', 'images']:
            os.makedirs(os.path.join(session_dir, subdir), exist_ok=True)

        # Generate trajectory based on agent type
        if args.agent == 'rover':
            # Ground-level trajectories for rover
            if args.pattern == 'random':
                waypoints = traj_gen.generate_rover_patrol(args.frames)
            else:
                # Use random walk at ground level for other patterns
                waypoints = traj_gen.generate_random_walk(
                    args.frames,
                    altitude_range=(1.5, 2.0),  # Ground level
                    speed=2.0,
                    agent_type='rover'
                )
        else:
            # Aerial trajectories for drone
            if args.pattern == 'survey':
                waypoints = traj_gen.generate_survey_pattern(args.frames)
            elif args.pattern == 'orbit':
                center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
                waypoints = traj_gen.generate_orbit_pattern(args.frames, center=center)
            else:
                waypoints = traj_gen.generate_random_walk(args.frames, agent_type='drone')

        # Get mesh vertices once for parallel processing
        combined_vertices = voxelizer.combined_mesh.vertices.copy()

        # Prepare frame processing arguments
        frame_args = [
            (frame_idx, waypoint, session_dir, combined_vertices, config)
            for frame_idx, waypoint in enumerate(waypoints)
        ]

        # Process frames in parallel
        print(f"  Processing {len(waypoints)} frames with {args.workers} workers...")
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_frame, arg): arg[0] for arg in frame_args}
            for future in as_completed(futures):
                frame_idx, occ_count = future.result()
                completed += 1
                if completed % 50 == 0 or completed == len(waypoints):
                    print(f"  Processed {completed}/{len(waypoints)} frames | Last: {occ_count} occupied voxels")

        print(f"  Completed: {args.frames} frames")

    print(f"\n{'=' * 60}")
    print("Conversion complete!")
    print(f"{'=' * 60}")
    print(f"Output: {args.output}")
    print(f"Sessions: {args.sessions}")
    print(f"Frames per session: {args.frames}")
    print(f"Total frames: {args.sessions * args.frames}")
    print(f"\nTo train:")
    print(f"  python train.py --py-config config/finetune_tokyo.py --work-dir /workspace/checkpoints")


if __name__ == '__main__':
    main()
