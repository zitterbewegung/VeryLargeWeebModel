#!/usr/bin/env python3
"""
Automated Data Collection Mission for OccWorld Training

This script automates drone/rover missions using DroneKit/pymavlink to collect
synchronized sensor data for OccWorld training. It executes predefined waypoint
patterns while ensuring stable data capture.

=== ARCHITECTURE OVERVIEW ===

    +------------------+
    | Mission Script   |
    | (this file)      |
    +--------+---------+
             |
             | pymavlink / MAVLink protocol
             |
    +--------v---------+
    | ArduPilot SITL   |
    | (ArduCopter/     |
    |  ArduRover)      |
    +--------+---------+
             |
             | JSON/UDP (physics state)
             |
    +--------v---------+
    | Gazebo Simulation|
    | + Sensors        |
    +------------------+

=== MISSION TYPES ===

1. Lawn-mower survey: Systematic grid coverage for comprehensive data
2. Orbit pattern: Circular path around a point of interest
3. Random waypoints: Varied trajectories for diverse training data

=== CONNECTION FLOW ===

1. connect_vehicle() establishes MAVLink connection to SITL
2. wait_heartbeat() confirms bidirectional communication
3. arm_and_takeoff() transitions vehicle to flight-ready state
4. execute_*_pattern() sends position commands and monitors arrival
5. land() safely returns vehicle to ground

=== DATA CAPTURE INTEGRATION ===

The mission script coordinates with the ROS 2 data recorder:
- Hover pauses (2-3 seconds) at each waypoint ensure sensor stability
- Controlled velocities (2-5 m/s) prevent motion blur
- Altitude variations provide diverse viewpoints
- Pattern overlap ensures complete scene coverage

Usage:
    python3 data_collection_mission.py --vehicle drone --pattern survey
    python3 data_collection_mission.py --vehicle rover --pattern random
"""

from pymavlink import mavutil
import time
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class VehicleType(Enum):
    """Supported vehicle types."""
    DRONE = "ArduCopter"
    ROVER = "ArduRover"


class MissionPattern(Enum):
    """Available mission patterns."""
    SURVEY = "survey"      # Lawn-mower grid pattern
    ORBIT = "orbit"        # Circular orbit
    RANDOM = "random"      # Random waypoints


@dataclass
class MissionConfig:
    """Mission configuration parameters."""
    vehicle_type: VehicleType = VehicleType.DRONE
    pattern: MissionPattern = MissionPattern.SURVEY

    # Connection
    connection_string: str = "udp:127.0.0.1:14550"

    # Survey pattern parameters
    survey_size: float = 80.0      # Grid size in meters
    survey_spacing: float = 20.0   # Line spacing in meters
    survey_altitude: float = 30.0  # Flight altitude in meters

    # Orbit pattern parameters
    orbit_radius: float = 30.0     # Orbit radius in meters
    orbit_altitude: float = 25.0   # Flight altitude
    orbit_points: int = 12         # Number of waypoints in orbit

    # Random pattern parameters
    random_count: int = 20         # Number of random waypoints
    random_bounds: float = 50.0    # Max distance from origin

    # Flight parameters
    cruise_speed: float = 3.0      # m/s
    hover_time: float = 2.0        # Seconds to hover at each waypoint
    arrival_threshold: float = 2.0 # Meters from waypoint to consider arrived

    # Rover-specific
    rover_speed: float = 2.0       # m/s for ground vehicles


class MAVLinkVehicle:
    """
    MAVLink vehicle interface using pymavlink.

    This class provides a high-level interface for vehicle control,
    abstracting the low-level MAVLink message handling.
    """

    def __init__(self, connection_string: str):
        """
        Initialize MAVLink connection.

        Args:
            connection_string: MAVLink connection URI
                Examples:
                - "udp:127.0.0.1:14550" - UDP to localhost
                - "tcp:192.168.1.1:5760" - TCP connection
                - "/dev/ttyACM0" - Serial connection
        """
        self.connection_string = connection_string
        self.master = None
        self.is_armed = False

    def connect(self) -> bool:
        """
        Establish MAVLink connection and wait for heartbeat.

        The heartbeat message confirms:
        1. The vehicle is powered and transmitting
        2. The autopilot type (ArduCopter, ArduRover, etc.)
        3. System and component IDs

        Returns:
            True if connection successful, False otherwise
        """
        print(f"Connecting to {self.connection_string}...")

        try:
            # mavlink_connection() creates the appropriate connection type
            # based on the URI scheme (udp:, tcp:, serial)
            self.master = mavutil.mavlink_connection(self.connection_string)

            # wait_heartbeat() blocks until a HEARTBEAT message is received
            # This confirms bidirectional communication
            self.master.wait_heartbeat(timeout=30)

            print(f"Connected to system {self.master.target_system}, "
                  f"component {self.master.target_component}")
            print(f"Autopilot: {mavutil.mavlink.enums['MAV_AUTOPILOT'][self.master.heartbeat().autopilot].name}")

            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def set_mode(self, mode: str) -> bool:
        """
        Set the vehicle flight mode.

        Common modes:
        - GUIDED: Accept position/velocity commands
        - AUTO: Follow mission waypoints
        - LOITER: Hold position
        - LAND: Autonomous landing
        - RTL: Return to launch

        Args:
            mode: Flight mode name (e.g., "GUIDED")

        Returns:
            True if mode change successful
        """
        # Get mode ID from name
        mode_mapping = self.master.mode_mapping()
        if mode not in mode_mapping:
            print(f"Unknown mode: {mode}")
            return False

        mode_id = mode_mapping[mode]

        # Send mode change command
        # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED tells the autopilot to use
        # the custom_mode field (mode_id) rather than base modes
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )

        # Wait for acknowledgment
        ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
        if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print(f"Mode changed to {mode}")
            return True

        return False

    def arm(self) -> bool:
        """
        Arm the vehicle motors.

        Arming enables motor output. Safety checks must pass:
        - GPS lock (for GPS modes)
        - Battery voltage sufficient
        - No failsafes active
        - Pre-arm checks passed

        Returns:
            True if arming successful
        """
        print("Arming motors...")

        # MAV_CMD_COMPONENT_ARM_DISARM command
        # param1: 1 = arm, 0 = disarm
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,      # confirmation
            1,      # param1: arm
            0, 0, 0, 0, 0, 0  # unused params
        )

        # Wait for motors to arm
        # motors_armed_wait() monitors the HEARTBEAT message's
        # base_mode field for the ARMED flag
        self.master.motors_armed_wait()
        self.is_armed = True
        print("Armed!")
        return True

    def disarm(self) -> bool:
        """Disarm the vehicle motors."""
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0  # param1=0 for disarm
        )
        self.is_armed = False
        print("Disarmed")
        return True

    def takeoff(self, altitude: float) -> bool:
        """
        Command the vehicle to take off to a specified altitude.

        The takeoff command:
        1. Increases throttle to lift off
        2. Climbs to target altitude
        3. Holds position once altitude reached

        Args:
            altitude: Target altitude in meters (AGL)

        Returns:
            True if takeoff successful
        """
        print(f"Taking off to {altitude}m...")

        # MAV_CMD_NAV_TAKEOFF command
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,           # confirmation
            0,           # param1: pitch (ignored)
            0,           # param2: empty
            0,           # param3: empty
            0,           # param4: yaw angle
            0,           # param5: latitude (0 = current)
            0,           # param6: longitude (0 = current)
            altitude     # param7: altitude
        )

        # Monitor altitude until target reached
        while True:
            msg = self.master.recv_match(
                type='GLOBAL_POSITION_INT',
                blocking=True,
                timeout=5
            )
            if msg:
                # relative_alt is in mm
                current_alt = msg.relative_alt / 1000.0
                print(f"  Altitude: {current_alt:.1f}m / {altitude}m")

                if current_alt >= altitude * 0.95:
                    print("Target altitude reached!")
                    return True

            time.sleep(0.5)

    def goto_local_ned(
        self,
        north: float,
        east: float,
        down: float,
        yaw: Optional[float] = None
    ):
        """
        Send position setpoint in local NED (North-East-Down) frame.

        The NED frame:
        - Origin: Vehicle's home position
        - North: Positive X
        - East: Positive Y
        - Down: Positive Z (negative = up)

        Args:
            north: North position in meters
            east: East position in meters
            down: Down position in meters (negative for altitude)
            yaw: Optional yaw angle in radians
        """
        # Build type mask to indicate which fields are valid
        # 0b0000_1111_1111_1000 = position only (ignore velocity, accel, yaw)
        type_mask = 0b0000111111111000

        if yaw is not None:
            # Enable yaw control
            type_mask = 0b0000011111111000

        self.master.mav.set_position_target_local_ned_send(
            0,                                      # time_boot_ms (0 = now)
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,    # Coordinate frame
            type_mask,                              # Fields to use
            north, east, down,                      # Position
            0, 0, 0,                                # Velocity (ignored)
            0, 0, 0,                                # Acceleration (ignored)
            yaw if yaw else 0,                      # Yaw
            0                                       # Yaw rate
        )

    def get_local_position(self) -> Tuple[float, float, float]:
        """
        Get current position in local NED frame.

        Returns:
            Tuple of (north, east, down) in meters
        """
        msg = self.master.recv_match(
            type='LOCAL_POSITION_NED',
            blocking=True,
            timeout=5
        )
        if msg:
            return (msg.x, msg.y, msg.z)
        return (0, 0, 0)

    def land(self):
        """Command the vehicle to land at current position."""
        print("Landing...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0, 0, 0, 0, 0, 0, 0, 0
        )


class DataCollectionMission:
    """
    Executes automated data collection missions for OccWorld training.

    The mission controller:
    1. Connects to vehicle via MAVLink
    2. Arms and takes off (drones) or sets mode (rovers)
    3. Executes waypoint pattern with hover pauses
    4. Lands/stops and disarms
    """

    def __init__(self, config: MissionConfig):
        self.config = config
        self.vehicle = MAVLinkVehicle(config.connection_string)
        self.waypoints: List[Tuple[float, float, float]] = []

    def generate_survey_pattern(self) -> List[Tuple[float, float, float]]:
        """
        Generate lawn-mower survey pattern.

        The pattern covers a square area with parallel lines:

            +------------------->
            |
            <-------------------+
                                |
            +------------------->
            |
            Start

        This ensures complete sensor coverage with minimal gaps.

        Returns:
            List of (north, east, down) waypoints
        """
        waypoints = []
        half_size = self.config.survey_size / 2
        spacing = self.config.survey_spacing
        altitude = -self.config.survey_altitude  # NED: down is positive

        east = -half_size
        direction = 1  # 1 = north, -1 = south

        while east <= half_size:
            if direction == 1:
                # Go north
                waypoints.append((-half_size, east, altitude))
                waypoints.append((half_size, east, altitude))
            else:
                # Go south
                waypoints.append((half_size, east, altitude))
                waypoints.append((-half_size, east, altitude))

            east += spacing
            direction *= -1

        print(f"Generated survey pattern: {len(waypoints)} waypoints")
        print(f"  Area: {self.config.survey_size}m x {self.config.survey_size}m")
        print(f"  Altitude: {self.config.survey_altitude}m")
        print(f"  Line spacing: {self.config.survey_spacing}m")

        return waypoints

    def generate_orbit_pattern(self) -> List[Tuple[float, float, float]]:
        """
        Generate circular orbit pattern around origin.

        Useful for:
        - Multi-angle views of a central object
        - Consistent distance from point of interest
        - Smooth, predictable motion for sensor calibration

        Returns:
            List of (north, east, down) waypoints
        """
        waypoints = []
        radius = self.config.orbit_radius
        altitude = -self.config.orbit_altitude
        num_points = self.config.orbit_points

        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            north = radius * math.cos(angle)
            east = radius * math.sin(angle)
            waypoints.append((north, east, altitude))

        # Return to start
        waypoints.append(waypoints[0])

        print(f"Generated orbit pattern: {len(waypoints)} waypoints")
        print(f"  Radius: {radius}m, Altitude: {self.config.orbit_altitude}m")

        return waypoints

    def generate_random_pattern(self) -> List[Tuple[float, float, float]]:
        """
        Generate random waypoints within bounds.

        Random patterns provide:
        - Diverse viewpoints for training
        - Varied motion dynamics
        - Unpredictable sensor coverage (tests generalization)

        Returns:
            List of (north, east, down) waypoints
        """
        import random
        waypoints = []
        bounds = self.config.random_bounds
        altitude = -self.config.survey_altitude

        for _ in range(self.config.random_count):
            north = random.uniform(-bounds, bounds)
            east = random.uniform(-bounds, bounds)
            # Vary altitude for drones
            if self.config.vehicle_type == VehicleType.DRONE:
                alt = random.uniform(-20, -50)  # 20-50m altitude
            else:
                alt = 0
            waypoints.append((north, east, alt))

        print(f"Generated random pattern: {len(waypoints)} waypoints")

        return waypoints

    def execute_mission(self):
        """
        Execute the complete data collection mission.

        Mission flow:
        1. Connect to vehicle
        2. Generate waypoint pattern
        3. Arm and prepare vehicle
        4. Takeoff (drones) or set mode (rovers)
        5. Execute waypoints with hover pauses
        6. Land and disarm
        """
        # Step 1: Connect
        if not self.vehicle.connect():
            print("Failed to connect!")
            return False

        # Step 2: Generate pattern
        if self.config.pattern == MissionPattern.SURVEY:
            self.waypoints = self.generate_survey_pattern()
        elif self.config.pattern == MissionPattern.ORBIT:
            self.waypoints = self.generate_orbit_pattern()
        else:
            self.waypoints = self.generate_random_pattern()

        # Step 3: Set mode to GUIDED
        if not self.vehicle.set_mode("GUIDED"):
            print("Failed to set GUIDED mode!")
            return False

        # Step 4: Arm
        if not self.vehicle.arm():
            print("Failed to arm!")
            return False

        # Step 5: Takeoff (drones only)
        if self.config.vehicle_type == VehicleType.DRONE:
            if not self.vehicle.takeoff(self.config.survey_altitude):
                print("Takeoff failed!")
                return False

        # Step 6: Execute waypoints
        print(f"\nExecuting mission with {len(self.waypoints)} waypoints...")
        print("=" * 50)

        for i, (north, east, down) in enumerate(self.waypoints):
            print(f"\nWaypoint {i+1}/{len(self.waypoints)}: "
                  f"N={north:.1f}, E={east:.1f}, D={down:.1f}")

            # Send position command
            self.vehicle.goto_local_ned(north, east, down)

            # Wait for arrival
            while True:
                current = self.vehicle.get_local_position()
                distance = math.sqrt(
                    (current[0] - north) ** 2 +
                    (current[1] - east) ** 2 +
                    (current[2] - down) ** 2
                )

                if distance < self.config.arrival_threshold:
                    print(f"  Arrived! Distance: {distance:.1f}m")
                    break

                time.sleep(0.1)

            # Hover for stable data capture
            # This pause ensures:
            # - Sensor readings are stable (no motion blur)
            # - Multiple frames captured at each position
            # - Time for data recorder to sync all sensors
            print(f"  Hovering for {self.config.hover_time}s...")
            time.sleep(self.config.hover_time)

        print("\n" + "=" * 50)
        print("Mission complete!")

        # Step 7: Land
        self.vehicle.land()

        # Wait for landing
        time.sleep(10)

        # Step 8: Disarm
        self.vehicle.disarm()

        return True


def main():
    parser = argparse.ArgumentParser(
        description='OccWorld Data Collection Mission',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--vehicle', '-v',
        choices=['drone', 'rover'],
        default='drone',
        help='Vehicle type (default: drone)'
    )
    parser.add_argument(
        '--pattern', '-p',
        choices=['survey', 'orbit', 'random'],
        default='survey',
        help='Mission pattern (default: survey)'
    )
    parser.add_argument(
        '--connection', '-c',
        default='udp:127.0.0.1:14550',
        help='MAVLink connection string (default: udp:127.0.0.1:14550)'
    )
    parser.add_argument(
        '--size',
        type=float,
        default=80.0,
        help='Survey grid size in meters (default: 80)'
    )
    parser.add_argument(
        '--spacing',
        type=float,
        default=20.0,
        help='Survey line spacing in meters (default: 20)'
    )
    parser.add_argument(
        '--altitude',
        type=float,
        default=30.0,
        help='Flight altitude in meters (default: 30)'
    )
    parser.add_argument(
        '--hover',
        type=float,
        default=2.0,
        help='Hover time at waypoints in seconds (default: 2)'
    )

    args = parser.parse_args()

    # Build configuration
    config = MissionConfig(
        vehicle_type=VehicleType.DRONE if args.vehicle == 'drone' else VehicleType.ROVER,
        pattern=MissionPattern(args.pattern),
        connection_string=args.connection,
        survey_size=args.size,
        survey_spacing=args.spacing,
        survey_altitude=args.altitude,
        hover_time=args.hover,
    )

    # Execute mission
    mission = DataCollectionMission(config)
    success = mission.execute_mission()

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
