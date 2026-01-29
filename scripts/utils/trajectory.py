"""Trajectory generation utilities for data collection.

Provides various flight/motion patterns for drone and rover simulations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation."""
    # Bounds for trajectory (x_min, y_min, x_max, y_max)
    bounds: Tuple[float, float, float, float] = (-200, -200, 200, 200)
    # Altitude range for drones
    altitude_range: Tuple[float, float] = (30.0, 100.0)
    # Ground height for rovers
    ground_height: float = 1.5
    # Default speed in m/s
    speed: float = 5.0
    # Minimum motion between frames in meters
    min_motion: float = 2.0


class TrajectoryGenerator:
    """Generate realistic drone/rover trajectories for data collection."""

    def __init__(self, config: TrajectoryConfig = None):
        self.config = config or TrajectoryConfig()

    def generate(
        self,
        num_frames: int,
        pattern: str = 'random',
        agent_type: str = 'drone'
    ) -> List[Dict]:
        """Generate trajectory with specified pattern.

        Args:
            num_frames: Number of waypoints to generate
            pattern: One of 'survey', 'orbit', 'random', 'figure8'
            agent_type: 'drone' or 'rover'

        Returns:
            List of waypoint dicts with position, orientation, velocity
        """
        if pattern == 'survey':
            return self.generate_survey_pattern(num_frames, agent_type)
        elif pattern == 'orbit':
            return self.generate_orbit_pattern(num_frames, agent_type)
        elif pattern == 'figure8':
            return self.generate_figure8_pattern(num_frames, agent_type)
        else:  # random
            return self.generate_random_walk(num_frames, agent_type)

    def generate_survey_pattern(
        self,
        num_frames: int,
        agent_type: str = 'drone'
    ) -> List[Dict]:
        """Generate lawn-mower survey pattern."""
        waypoints = []
        cfg = self.config
        x_min, y_min, x_max, y_max = cfg.bounds

        # Use cubic spline for smooth interpolation if scipy available
        try:
            from scipy.interpolate import CubicSpline
            has_scipy = True
        except ImportError:
            has_scipy = False

        # Create keypoints for lawn-mower pattern
        rows = max(3, int(np.sqrt(num_frames / 10)))
        keypoints = []

        y_vals = np.linspace(y_min + 20, y_max - 20, rows)
        for row, y in enumerate(y_vals):
            if row % 2 == 0:
                keypoints.append((x_min + 20, y))
                keypoints.append((x_max - 20, y))
            else:
                keypoints.append((x_max - 20, y))
                keypoints.append((x_min + 20, y))

        keypoints = np.array(keypoints)

        # Calculate cumulative distance
        distances = [0]
        for i in range(1, len(keypoints)):
            d = np.linalg.norm(keypoints[i] - keypoints[i-1])
            distances.append(distances[-1] + d)
        distances = np.array(distances)
        total_distance = distances[-1]

        # Interpolate positions
        t_keypoints = distances / total_distance
        t_frames = np.linspace(0, 1, num_frames)

        if has_scipy:
            try:
                cs_x = CubicSpline(t_keypoints, keypoints[:, 0])
                cs_y = CubicSpline(t_keypoints, keypoints[:, 1])
                x_interp = cs_x(t_frames)
                y_interp = cs_y(t_frames)
            except Exception:
                x_interp = np.interp(t_frames, t_keypoints, keypoints[:, 0])
                y_interp = np.interp(t_frames, t_keypoints, keypoints[:, 1])
        else:
            x_interp = np.interp(t_frames, t_keypoints, keypoints[:, 0])
            y_interp = np.interp(t_frames, t_keypoints, keypoints[:, 1])

        # Altitude variation
        if agent_type == 'drone':
            base_alt = np.mean(cfg.altitude_range)
            alt_variation = (cfg.altitude_range[1] - cfg.altitude_range[0]) / 4
            z_vals = base_alt + alt_variation * np.sin(np.linspace(0, 4*np.pi, num_frames))
        else:
            z_vals = np.full(num_frames, cfg.ground_height)

        for i in range(num_frames):
            x, y, z = float(x_interp[i]), float(y_interp[i]), float(z_vals[i])

            # Yaw from motion direction
            if i < num_frames - 1:
                dx = x_interp[i+1] - x_interp[i]
                dy = y_interp[i+1] - y_interp[i]
                yaw = np.arctan2(dy, dx)
            else:
                yaw = waypoints[-1]['orientation']['z'] * 2 if waypoints else 0

            waypoints.append(self._create_waypoint(x, y, z, yaw, cfg.speed, agent_type))

        return waypoints

    def generate_orbit_pattern(
        self,
        num_frames: int,
        agent_type: str = 'drone',
        center: Tuple[float, float] = None,
        radius: float = None
    ) -> List[Dict]:
        """Generate circular orbit pattern around a point."""
        waypoints = []
        cfg = self.config

        if center is None:
            center = (
                (cfg.bounds[0] + cfg.bounds[2]) / 2,
                (cfg.bounds[1] + cfg.bounds[3]) / 2
            )
        if radius is None:
            radius = min(cfg.bounds[2] - cfg.bounds[0], cfg.bounds[3] - cfg.bounds[1]) / 4

        altitude = np.mean(cfg.altitude_range) if agent_type == 'drone' else cfg.ground_height
        radius_var = radius * 0.2
        alt_var = 10.0 if agent_type == 'drone' else 0

        for i in range(num_frames):
            angle = 4 * np.pi * i / num_frames  # Two full orbits
            progress = i / num_frames

            r = radius + radius_var * np.sin(2 * np.pi * progress)
            z = altitude + alt_var * np.sin(3 * np.pi * progress)

            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)

            # Face toward center
            yaw = angle + np.pi + 0.2 * np.sin(angle * 3)

            waypoints.append(self._create_waypoint(x, y, z, yaw, cfg.speed, agent_type))

        return waypoints

    def generate_figure8_pattern(
        self,
        num_frames: int,
        agent_type: str = 'drone',
        center: Tuple[float, float] = None,
        size: float = None
    ) -> List[Dict]:
        """Generate figure-8 (lemniscate) pattern."""
        waypoints = []
        cfg = self.config

        if center is None:
            center = (
                (cfg.bounds[0] + cfg.bounds[2]) / 2,
                (cfg.bounds[1] + cfg.bounds[3]) / 2
            )
        if size is None:
            size = min(cfg.bounds[2] - cfg.bounds[0], cfg.bounds[3] - cfg.bounds[1]) / 4

        altitude = np.mean(cfg.altitude_range) if agent_type == 'drone' else cfg.ground_height

        for i in range(num_frames):
            t = 2 * np.pi * i / num_frames

            # Lemniscate of Bernoulli
            x = center[0] + size * np.sin(t)
            y = center[1] + size * np.sin(t) * np.cos(t)
            z = altitude + 10 * np.sin(2 * t) if agent_type == 'drone' else cfg.ground_height

            # Velocity from derivative
            dx = size * np.cos(t)
            dy = size * (np.cos(t)**2 - np.sin(t)**2)

            yaw = np.arctan2(dy, dx)

            waypoints.append(self._create_waypoint(x, y, z, yaw, cfg.speed, agent_type))

        return waypoints

    def generate_random_walk(
        self,
        num_frames: int,
        agent_type: str = 'drone'
    ) -> List[Dict]:
        """Generate smooth random exploration trajectory."""
        waypoints = []
        cfg = self.config
        x_min, y_min, x_max, y_max = cfg.bounds

        # Start position
        x = (x_min + x_max) / 2
        y = (y_min + y_max) / 2
        z = np.mean(cfg.altitude_range) if agent_type == 'drone' else cfg.ground_height
        yaw = np.random.uniform(0, 2 * np.pi)

        for i in range(num_frames):
            # Smooth steering (Ornstein-Uhlenbeck-like)
            theta_target = yaw + np.random.uniform(-0.5, 0.5)
            yaw = yaw + 0.3 * (theta_target - yaw) + np.random.uniform(-0.1, 0.1)

            # Speed with minimum motion guarantee
            current_speed = max(cfg.min_motion, cfg.speed * np.random.uniform(0.8, 1.2))

            # Motion
            dx = current_speed * np.cos(yaw)
            dy = current_speed * np.sin(yaw)
            dz = np.random.uniform(-1, 1) if agent_type == 'drone' else 0

            new_x = x + dx
            new_y = y + dy
            new_z = z + dz

            # Reflect off boundaries
            if new_x < x_min or new_x > x_max:
                yaw = np.pi - yaw
                new_x = np.clip(new_x, x_min + 5, x_max - 5)
            if new_y < y_min or new_y > y_max:
                yaw = -yaw
                new_y = np.clip(new_y, y_min + 5, y_max - 5)

            x, y = new_x, new_y
            if agent_type == 'drone':
                z = np.clip(new_z, cfg.altitude_range[0], cfg.altitude_range[1])

            waypoints.append(self._create_waypoint(x, y, z, yaw, current_speed, agent_type))

        return waypoints

    def _create_waypoint(
        self,
        x: float, y: float, z: float,
        yaw: float, speed: float,
        agent_type: str
    ) -> Dict:
        """Create waypoint in standard format."""
        # Quaternion from yaw (rotation around Z axis)
        qw = np.cos(yaw / 2)
        qz = np.sin(yaw / 2)

        return {
            'position': {'x': float(x), 'y': float(y), 'z': float(z)},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': float(qz), 'w': float(qw)},
            'velocity': {
                'linear': {
                    'x': float(speed * np.cos(yaw)),
                    'y': float(speed * np.sin(yaw)),
                    'z': 0.0
                },
                'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
            },
            'agent_type': agent_type
        }

    def validate_trajectory(self, waypoints: List[Dict]) -> Dict:
        """Validate trajectory has sufficient motion between frames."""
        positions = np.array([
            [w['position']['x'], w['position']['y'], w['position']['z']]
            for w in waypoints
        ])

        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)

        return {
            'num_frames': len(waypoints),
            'total_distance': float(np.sum(distances)),
            'mean_distance': float(np.mean(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances)),
            'static_frames': int(np.sum(distances < self.config.min_motion)),
            'valid': bool(np.all(distances >= self.config.min_motion * 0.5))
        }
