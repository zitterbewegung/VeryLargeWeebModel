# OccWorld + ArduPilot Gazebo Integration Architecture Blueprint

## Component Interconnection Overview

This document explains how the OccWorld stack, Gazebo transport, ros_gz_bridge, MAVROS2, and BEVFusion components interconnect for multi-agent SITL/HITL simulation.

---

## 1. System Architecture

```
                          +--------------------------------------------------+
                          |              Gazebo Harmonic/Ionic                |
                          |                                                    |
    +---------------------|  +-------------+  +--------------+  +----------+  |
    | ArduPilotPlugin     |  | Physics     |  | Sensor Mgr   |  | Scene    |  |
    | (per vehicle)       |  | (ODE/Bullet)|  | (GPU LiDAR,  |  | Graph    |  |
    |   - FDM Interface   |  |             |  |  Cameras,    |  |          |  |
    |   - State I/O       |  |             |  |  IMU, GPS)   |  |          |  |
    +---------+-----------+  +------+------+  +------+-------+  +----+-----+
              |                     |                |               |
     JSON/UDP |                     |                |               |
   (bidirectional)           Physics        Sensor Data      Model State
              |                Update            |               |
              v                     |                |               |
    +------------------+            v                v               v
    | ArduPilot SITL   |     +-------------------------------------------+
    | Instance(s)      |     |           Gazebo Transport Layer           |
    |  - ArduCopter    |     |     (gz::transport::Node pub/sub)          |
    |  - ArduRover     |     +---------------------+---------------------+
    |  - ArduPlane     |                           |
    +--------+---------+                           |
             |                                     |
     MAVLink |                     +---------------+-----------------+
     (UDP)   |                     |         ros_gz_bridge           |
             |                     |   (Gazebo msgs <-> ROS 2 msgs)  |
             v                     +---------------+-----------------+
    +------------------+                           |
    | MAVROS2 Node(s)  |                           |
    |  - FCU URL cfg   |<--------------------------+
    |  - Topic remap   |           ROS 2 DDS
    +--------+---------+               |
             |                         |
             v                         v
    +--------------------------------------------------+
    |                ROS 2 Ecosystem (Humble)           |
    |                                                   |
    |  +-------------+  +-------------+  +------------+ |
    |  | BEVFusion   |  | OccWorld    |  | Trajectory | |
    |  | Node        |  | Inference   |  | Planner    | |
    |  +------+------+  +------+------+  +-----+------+ |
    |         |                |               |        |
    |         +-------+--------+-------+-------+        |
    |                 |                                 |
    |         +-------v-------+                         |
    |         | Navigation    |                         |
    |         | Controller    |                         |
    |         +---------------+                         |
    +--------------------------------------------------+
```

---

## 2. Component Details and Data Flows

### 2.1 ArduPilotPlugin <-> ArduPilot SITL

**Protocol:** JSON over UDP (bidirectional)

**Data Flow - Gazebo to SITL (State):**
```json
{
  "timestamp": 1234567890.123,
  "imu": {
    "gyro": [gx, gy, gz],           // rad/s (body frame)
    "accel_body": [ax, ay, az]      // m/s^2 (body frame)
  },
  "position": [lat, lon, alt],       // WGS84 degrees, meters
  "velocity": [vn, ve, vd],          // m/s NED frame
  "attitude": [roll, pitch, yaw]     // radians
}
```

**Data Flow - SITL to Gazebo (Commands):**
```json
{
  "servos": [pwm0, pwm1, ..., pwm15],  // PWM values 1000-2000
  "armed": true/false
}
```

**Port Conventions (Multi-Agent):**
| Instance | FDM Port In | FDM Port Out | MAVLink Port |
|----------|-------------|--------------|--------------|
| -I 0     | 9002        | 9003         | 14550        |
| -I 1     | 9012        | 9013         | 14560        |
| -I 2     | 9022        | 9023         | 14570        |

---

### 2.2 Gazebo Transport -> ros_gz_bridge -> ROS 2

**Sensor Topic Mappings:**

| Sensor | Gazebo Transport Topic | ROS 2 Topic | Message Type |
|--------|------------------------|-------------|--------------|
| Camera | `/world/{world}/model/{model}/link/{link}/sensor/{sensor}/image` | `/{vehicle}/camera/{position}/image` | `sensor_msgs/msg/Image` |
| LiDAR | `/world/{world}/model/{model}/link/{link}/sensor/{sensor}/scan/points` | `/{vehicle}/lidar/points` | `sensor_msgs/msg/PointCloud2` |
| IMU | `/world/{world}/model/{model}/link/{link}/sensor/{sensor}/imu` | `/{vehicle}/imu/data` | `sensor_msgs/msg/Imu` |
| GPS | `/world/{world}/model/{model}/link/{link}/sensor/{sensor}/navsat` | `/{vehicle}/gps/fix` | `sensor_msgs/msg/NavSatFix` |
| Odometry | `/model/{model}/odometry` | `/{vehicle}/odom` | `nav_msgs/msg/Odometry` |
| Clock | `/clock` | `/clock` | `rosgraph_msgs/msg/Clock` |

**Bridge Configuration (YAML Format):**
```yaml
- ros_topic_name: "/drone_1/lidar/points"
  gz_topic_name: "/world/tokyo/model/drone_1/link/lidar_link/sensor/gpu_lidar/scan/points"
  ros_type_name: "sensor_msgs/msg/PointCloud2"
  gz_type_name: "gz.msgs.PointCloudPacked"
  direction: GZ_TO_ROS
```

---

### 2.3 MAVROS2 <-> ArduPilot SITL

**Protocol:** MAVLink 2.0 over UDP

**Key ROS 2 Topics Published by MAVROS:**
| Topic | Type | Description |
|-------|------|-------------|
| `/{ns}/mavros/state` | `mavros_msgs/State` | Armed/disarmed, flight mode |
| `/{ns}/mavros/local_position/pose` | `geometry_msgs/PoseStamped` | Local position estimate |
| `/{ns}/mavros/global_position/global` | `sensor_msgs/NavSatFix` | GPS position |
| `/{ns}/mavros/imu/data` | `sensor_msgs/Imu` | IMU data from FCU |
| `/{ns}/mavros/battery` | `sensor_msgs/BatteryState` | Battery status |

**Key ROS 2 Topics Subscribed by MAVROS:**
| Topic | Type | Description |
|-------|------|-------------|
| `/{ns}/mavros/setpoint_position/local` | `geometry_msgs/PoseStamped` | Position setpoint |
| `/{ns}/mavros/setpoint_velocity/cmd_vel` | `geometry_msgs/TwistStamped` | Velocity setpoint |
| `/{ns}/mavros/setpoint_raw/attitude` | `mavros_msgs/AttitudeTarget` | Attitude setpoint |

**FCU URL Configuration:**
```python
# Drone instance 0
'fcu_url': 'udp://:14550@127.0.0.1:14555'

# Rover instance 1
'fcu_url': 'udp://:14560@127.0.0.1:14565'
```

---

### 2.4 BEVFusion Node Integration

**Input Topics (from ros_gz_bridge):**
```
/drone_1/camera/front/image          -> CAM_FRONT
/drone_1/camera/front_left/image     -> CAM_FRONT_LEFT
/drone_1/camera/front_right/image    -> CAM_FRONT_RIGHT
/drone_1/camera/back/image           -> CAM_BACK
/drone_1/camera/back_left/image      -> CAM_BACK_LEFT
/drone_1/camera/back_right/image     -> CAM_BACK_RIGHT
/drone_1/lidar/points                -> LIDAR_TOP
```

**Output Topics (to OccWorld):**
```
/drone_1/bevfusion/bev_features      -> [B, C, H, W] BEV feature tensor
/drone_1/bevfusion/detections        -> 3D object detections
```

---

### 2.5 OccWorld Inference Node

**Input:**
- BEV features from BEVFusion
- Historical occupancy states (4 frames)
- Ego poses from MAVROS

**Output:**
```
/drone_1/occworld/predicted_occupancy   -> Future occupancy (6 frames)
/drone_1/occworld/flow_field            -> 3D flow predictions
/drone_1/occworld/planning_cost         -> Cost volume for planning
```

---

## 3. Multi-Agent SITL/HITL Configuration

### 3.1 SITL Multi-Instance Setup

```
                    Gazebo World (tokyo_occworld.sdf)
                              |
        +---------------------+---------------------+
        |                     |                     |
   +----v----+           +----v----+           +----v----+
   | drone_1 |           | drone_2 |           | rover_1 |
   | Plugin  |           | Plugin  |           | Plugin  |
   | 9002/03 |           | 9022/23 |           | 9012/13 |
   +---------+           +---------+           +---------+
        |                     |                     |
   JSON/UDP              JSON/UDP              JSON/UDP
        |                     |                     |
   +----v----+           +----v----+           +----v----+
   |  SITL   |           |  SITL   |           |  SITL   |
   |   -I 0  |           |   -I 2  |           |   -I 1  |
   |Copter   |           |Copter   |           | Rover   |
   +---------+           +---------+           +---------+
        |                     |                     |
   MAVLink:14550         MAVLink:14570         MAVLink:14560
        |                     |                     |
   +----v----+           +----v----+           +----v----+
   | MAVROS  |           | MAVROS  |           | MAVROS  |
   | drone_1 |           | drone_2 |           | rover_1 |
   +---------+           +---------+           +---------+
```

### 3.2 HITL Configuration

For Hardware-in-the-Loop, replace JSON/UDP with serial connection:

```xml
<plugin filename="ArduPilotPlugin" name="ArduPilotPlugin">
  <!-- HITL uses serial instead of UDP -->
  <connectionType>serial</connectionType>
  <serialDevice>/dev/ttyACM0</serialDevice>
  <baudRate>921600</baudRate>
</plugin>
```

---

## 4. Data Flow Summary

```
[Physical World Simulation]
         |
    Gazebo Physics Engine
         |
         v
    +--------------------------------------------+
    |           Sensor Simulation                |
    |  Camera -> RGB Images (1600x900 @ 10Hz)   |
    |  LiDAR -> Point Clouds (115K pts @ 10Hz)  |
    |  IMU -> 6DoF @ 200Hz                       |
    |  GPS -> Lat/Lon/Alt @ 10Hz                |
    +--------------------------------------------+
                      |
         Gazebo Transport (protobuf msgs)
                      |
         +------------+------------+
         |                         |
    ros_gz_bridge             ArduPilotPlugin
         |                         |
    ROS 2 Topics              JSON/UDP
         |                         |
    +----v----+               +----v----+
    |BEVFusion|               |ArduPilot|
    |   Node  |               |  SITL   |
    +----+----+               +----+----+
         |                         |
    BEV Features              MAVLink
         |                         |
    +----v----+               +----v----+
    |OccWorld |<--------------| MAVROS2 |
    |Inference|   Ego Pose    |  Node   |
    +----+----+               +----+----+
         |                         ^
    Future                    Setpoints
    Occupancy                      |
         |                         |
    +----v-------------------------+----+
    |        Trajectory Planner         |
    |      (6-DoF Path Planning)        |
    +-----------------------------------+
```

---

## 5. Required Environment Variables

```bash
# Gazebo configuration
export GZ_VERSION=harmonic
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}
export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH}

# ROS 2 configuration
source /opt/ros/humble/setup.bash
source ~/occworld_ws/install/setup.bash

# ArduPilot paths
export PATH=$HOME/ardupilot/Tools/autotest:$PATH
```

---

## 6. Network Port Reference

| Service | Protocol | Port(s) | Direction |
|---------|----------|---------|-----------|
| Gazebo GUI | TCP | 11345 | Internal |
| FDM JSON (Instance 0) | UDP | 9002/9003 | Bidirectional |
| FDM JSON (Instance 1) | UDP | 9012/9013 | Bidirectional |
| MAVLink (Instance 0) | UDP | 14550-14555 | Bidirectional |
| MAVLink (Instance 1) | UDP | 14560-14565 | Bidirectional |
| GCS/MavProxy | TCP | 5760 | Bidirectional |
| ROS 2 DDS | UDP | 7400-7500 | Multicast |
