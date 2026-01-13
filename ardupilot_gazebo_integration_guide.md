# ArduPilot Gazebo Integration for OccWorld Simulation Framework

## Executive Summary

This guide provides a deep technical dive into integrating the **ArduPilot Gazebo plugin** (`ardupilot_gazebo`) as the simulation backbone for your OccWorld-based drone and ground robot navigation system. The integration enables:

- **Hardware-in-the-Loop (HITL)** and **Software-in-the-Loop (SITL)** testing
- **Realistic sensor simulation** (LiDAR, cameras, IMU) for BEVFusion input
- **Multi-vehicle scenarios** for swarm testing
- **ROS 2 integration** for perception pipeline connectivity
- **True simulation lockstepping** for deterministic training data generation

---

## 1. Architecture Overview

### 1.1 System Integration Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        OccWorld + ArduPilot Gazebo Integration                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                           Gazebo Harmonic/Ionic                               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │  │
│  │  │ Tokyo World │  │ Physics     │  │ Sensors     │  │ ArduPilotPlugin     │  │  │
│  │  │ (PLATEAU    │  │ Engine      │  │ (LiDAR,     │  │ (JSON UDP Bridge)   │  │  │
│  │  │  Models)    │  │ (ODE/Bullet)│  │  Camera,    │  │                     │  │  │
│  │  │             │  │             │  │  IMU, GPS)  │  │  ┌───────────────┐  │  │  │
│  │  └─────────────┘  └─────────────┘  └──────┬──────┘  │  │ FDM Interface │  │  │  │
│  │                                           │         │  └───────┬───────┘  │  │  │
│  └───────────────────────────────────────────┼─────────┴──────────┼──────────┘  │
│                                              │                    │              │
│           Gazebo Transport Topics            │         JSON/UDP   │              │
│                    │                         │                    │              │
│  ┌─────────────────┼─────────────────────────┼────────────────────┼──────────┐  │
│  │                 │      ROS 2 Bridge       │                    │          │  │
│  │  ┌──────────────▼──────────────┐          │       ┌────────────▼────────┐ │  │
│  │  │   ros_gz_bridge             │          │       │   ArduPilot SITL    │ │  │
│  │  │   - /scan (LaserScan)       │          │       │   - ArduCopter      │ │  │
│  │  │   - /camera/image (Image)   │          │       │   - ArduRover       │ │  │
│  │  │   - /imu (Imu)              │          │       │   - ArduPlane       │ │  │
│  │  │   - /gps (NavSatFix)        │          │       └─────────┬──────────┘ │  │
│  │  │   - /odom (Odometry)        │          │                 │            │  │
│  │  └──────────────┬──────────────┘          │        MAVLink  │            │  │
│  └─────────────────┼─────────────────────────┼─────────────────┼────────────┘  │
│                    │                         │                 │                │
│  ┌─────────────────▼─────────────────────────┼─────────────────▼────────────┐  │
│  │                     ROS 2 Ecosystem                                       │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐   │  │
│  │  │  BEVFusion Node │  │   MAVROS2       │  │   OccWorld Inference    │   │  │
│  │  │  (Perception)   │◄─┤   (MAVLink↔ROS) │  │   (World Model)         │   │  │
│  │  └────────┬────────┘  └─────────────────┘  └───────────┬─────────────┘   │  │
│  │           │                                            │                  │  │
│  │           └────────────────────┬───────────────────────┘                  │  │
│  │                                │                                          │  │
│  │                    ┌───────────▼───────────┐                              │  │
│  │                    │  Trajectory Planner   │                              │  │
│  │                    │  (6-DoF Commands)     │                              │  │
│  │                    └───────────────────────┘                              │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Integration Points

| Component | Purpose | Data Flow |
|-----------|---------|-----------|
| **ArduPilotPlugin** | Bridges Gazebo physics to ArduPilot SITL | Gazebo → JSON/UDP → SITL |
| **ros_gz_bridge** | Converts Gazebo topics to ROS 2 | Gazebo Transport → ROS 2 DDS |
| **MAVROS2** | MAVLink to ROS 2 translation | ArduPilot SITL → ROS 2 topics |
| **BEVFusion Node** | Sensor fusion for OccWorld | Camera + LiDAR → BEV features |
| **GstCameraPlugin** | Video streaming from cameras | H.264 UDP stream → OpenCV/ROS |

---

## 2. Installation and Setup

### 2.1 Prerequisites

```bash
# Ubuntu 22.04 (Jammy) recommended
# Gazebo Harmonic (LTS) or Ionic

# 1. Install Gazebo Harmonic
sudo apt-get update
sudo apt-get install curl lsb-release gnupg

sudo curl https://packages.osrfoundation.org/gazebo.gpg \
    --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
    http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

sudo apt-get update
sudo apt-get install gz-harmonic

# 2. Verify Gazebo installation
gz sim --version
```

### 2.2 ArduPilot SITL Setup

```bash
# Clone ArduPilot
cd ~
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git
cd ardupilot

# Install dependencies
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile

# Build ArduCopter (for drones)
./waf configure --board sitl
./waf copter

# Build ArduRover (for ground robots)
./waf rover
```

### 2.3 ArduPilot Gazebo Plugin Installation

```bash
# Install dependencies for Gazebo Harmonic
sudo apt update
sudo apt install libgz-sim8-dev rapidjson-dev
sudo apt install libopencv-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl

# Clone and build the plugin
cd ~
git clone https://github.com/ArduPilot/ardupilot_gazebo.git
cd ardupilot_gazebo

# Set Gazebo version
export GZ_VERSION=harmonic

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc)

# Configure environment (add to ~/.bashrc)
echo 'export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}' >> ~/.bashrc
echo 'export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc
source ~/.bashrc
```

### 2.4 ROS 2 Integration Setup

```bash
# Install ROS 2 Humble (if not already installed)
sudo apt install ros-humble-desktop

# Install ROS-Gazebo bridge packages
sudo apt install ros-humble-ros-gz-sim ros-humble-ros-gz-bridge ros-humble-ros-gz-interfaces

# Install MAVROS2
sudo apt install ros-humble-mavros ros-humble-mavros-extras

# Install geographic lib datasets for MAVROS
sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh
```

---

## 3. Sensor Configuration for OccWorld

### 3.1 Multi-Camera Setup (6-View for BEVFusion)

The OccWorld BEVFusion pipeline requires 6 surround-view cameras. Here's the SDF configuration:

```xml
<!-- models/occworld_sensor_rig/model.sdf -->
<?xml version="1.0"?>
<sdf version="1.9">
  <model name="occworld_sensor_rig">
    <static>false</static>
    
    <!-- Base link for sensor attachment -->
    <link name="sensor_base">
      <inertial>
        <mass>0.1</mass>
        <inertia>
          <ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz>
        </inertia>
      </inertial>
    </link>
    
    <!-- FRONT CAMERA -->
    <link name="camera_front_link">
      <pose relative_to="sensor_base">2.0 0 1.5 0 0 0</pose>
      <sensor name="camera_front" type="camera">
        <always_on>1</always_on>
        <update_rate>10</update_rate>
        <visualize>true</visualize>
        <topic>camera/front/image</topic>
        <camera>
          <horizontal_fov>1.2217</horizontal_fov>  <!-- 70 degrees -->
          <image>
            <width>1600</width>
            <height>900</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>300</far>
          </clip>
          <lens>
            <intrinsics>
              <fx>1142.5</fx>
              <fy>1142.5</fy>
              <cx>800</cx>
              <cy>450</cy>
              <s>0</s>
            </intrinsics>
          </lens>
        </camera>
      </sensor>
    </link>
    
    <!-- FRONT-LEFT CAMERA -->
    <link name="camera_front_left_link">
      <pose relative_to="sensor_base">1.5 0.8 1.5 0 0 0.7854</pose>  <!-- 45° yaw -->
      <sensor name="camera_front_left" type="camera">
        <always_on>1</always_on>
        <update_rate>10</update_rate>
        <topic>camera/front_left/image</topic>
        <camera>
          <horizontal_fov>1.2217</horizontal_fov>
          <image>
            <width>1600</width>
            <height>900</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>300</far>
          </clip>
        </camera>
      </sensor>
    </link>
    
    <!-- FRONT-RIGHT CAMERA -->
    <link name="camera_front_right_link">
      <pose relative_to="sensor_base">1.5 -0.8 1.5 0 0 -0.7854</pose>  <!-- -45° yaw -->
      <sensor name="camera_front_right" type="camera">
        <always_on>1</always_on>
        <update_rate>10</update_rate>
        <topic>camera/front_right/image</topic>
        <camera>
          <horizontal_fov>1.2217</horizontal_fov>
          <image>
            <width>1600</width>
            <height>900</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>300</far>
          </clip>
        </camera>
      </sensor>
    </link>
    
    <!-- BACK CAMERA -->
    <link name="camera_back_link">
      <pose relative_to="sensor_base">-2.0 0 1.5 0 0 3.1416</pose>  <!-- 180° yaw -->
      <sensor name="camera_back" type="camera">
        <always_on>1</always_on>
        <update_rate>10</update_rate>
        <topic>camera/back/image</topic>
        <camera>
          <horizontal_fov>1.2217</horizontal_fov>
          <image>
            <width>1600</width>
            <height>900</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>300</far>
          </clip>
        </camera>
      </sensor>
    </link>
    
    <!-- BACK-LEFT CAMERA -->
    <link name="camera_back_left_link">
      <pose relative_to="sensor_base">-1.5 0.8 1.5 0 0 2.3562</pose>  <!-- 135° yaw -->
      <sensor name="camera_back_left" type="camera">
        <always_on>1</always_on>
        <update_rate>10</update_rate>
        <topic>camera/back_left/image</topic>
        <camera>
          <horizontal_fov>1.2217</horizontal_fov>
          <image>
            <width>1600</width>
            <height>900</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>300</far>
          </clip>
        </camera>
      </sensor>
    </link>
    
    <!-- BACK-RIGHT CAMERA -->
    <link name="camera_back_right_link">
      <pose relative_to="sensor_base">-1.5 -0.8 1.5 0 0 -2.3562</pose>  <!-- -135° yaw -->
      <sensor name="camera_back_right" type="camera">
        <always_on>1</always_on>
        <update_rate>10</update_rate>
        <topic>camera/back_right/image</topic>
        <camera>
          <horizontal_fov>1.2217</horizontal_fov>
          <image>
            <width>1600</width>
            <height>900</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>300</far>
          </clip>
        </camera>
      </sensor>
    </link>
  </model>
</sdf>
```

### 3.2 LiDAR Configuration (Extended Range for Drones)

```xml
<!-- Ground Robot LiDAR: High resolution, moderate range -->
<link name="lidar_ground_link">
  <pose relative_to="sensor_base">0 0 1.8 0 0 0</pose>
  <sensor name="gpu_lidar_ground" type="gpu_lidar">
    <always_on>true</always_on>
    <update_rate>10</update_rate>
    <visualize>true</visualize>
    <topic>lidar/ground/points</topic>
    <gz_frame_id>lidar_ground_link</gz_frame_id>
    <lidar>
      <scan>
        <horizontal>
          <samples>1800</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
        <vertical>
          <samples>64</samples>
          <resolution>1</resolution>
          <min_angle>-0.4363</min_angle>  <!-- -25 degrees -->
          <max_angle>0.2618</max_angle>   <!-- +15 degrees -->
        </vertical>
      </scan>
      <range>
        <min>0.5</min>
        <max>120</max>
        <resolution>0.01</resolution>
      </range>
      <noise>
        <type>gaussian</type>
        <mean>0</mean>
        <stddev>0.02</stddev>
      </noise>
    </lidar>
  </sensor>
</link>

<!-- Drone LiDAR: Wider vertical FOV for aerial navigation -->
<link name="lidar_drone_link">
  <pose relative_to="sensor_base">0 0 0 0 0 0</pose>
  <sensor name="gpu_lidar_drone" type="gpu_lidar">
    <always_on>true</always_on>
    <update_rate>10</update_rate>
    <visualize>true</visualize>
    <topic>lidar/drone/points</topic>
    <gz_frame_id>lidar_drone_link</gz_frame_id>
    <lidar>
      <scan>
        <horizontal>
          <samples>1200</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
        <vertical>
          <samples>32</samples>
          <resolution>1</resolution>
          <min_angle>-0.7854</min_angle>  <!-- -45 degrees (look down) -->
          <max_angle>0.7854</max_angle>   <!-- +45 degrees (look up) -->
        </vertical>
      </scan>
      <range>
        <min>0.3</min>
        <max>100</max>
        <resolution>0.01</resolution>
      </range>
      <noise>
        <type>gaussian</type>
        <mean>0</mean>
        <stddev>0.03</stddev>
      </noise>
    </lidar>
  </sensor>
</link>
```

### 3.3 IMU and GPS for State Estimation

```xml
<!-- IMU Sensor -->
<link name="imu_link">
  <pose relative_to="sensor_base">0 0 0.5 0 0 0</pose>
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>200</update_rate>
    <visualize>false</visualize>
    <topic>imu/data</topic>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>0.0003</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>0.0003</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>0.0003</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>0.01</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>0.01</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>0.01</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</link>

<!-- NavSat (GPS) Sensor -->
<link name="gps_link">
  <pose relative_to="sensor_base">0 0 0.6 0 0 0</pose>
  <sensor name="navsat_sensor" type="navsat">
    <always_on>true</always_on>
    <update_rate>10</update_rate>
    <topic>gps/fix</topic>
    <navsat>
      <position_sensing>
        <horizontal>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>0.5</stddev>  <!-- ~0.5m horizontal error -->
          </noise>
        </horizontal>
        <vertical>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>1.0</stddev>  <!-- ~1m vertical error -->
          </noise>
        </vertical>
      </position_sensing>
      <velocity_sensing>
        <horizontal>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>0.05</stddev>
          </noise>
        </horizontal>
        <vertical>
          <noise type="gaussian">
            <mean>0</mean>
            <stddev>0.1</stddev>
          </noise>
        </vertical>
      </velocity_sensing>
    </navsat>
  </sensor>
</link>
```

### 3.4 RGBD Camera for Depth-Based Occupancy

```xml
<!-- RGBD Depth Camera (Intel RealSense D435 style) -->
<link name="rgbd_camera_link">
  <pose relative_to="sensor_base">1.0 0 1.2 0 0.1 0</pose>
  <sensor name="rgbd_camera" type="rgbd_camera">
    <always_on>1</always_on>
    <update_rate>30</update_rate>
    <visualize>true</visualize>
    <topic>rgbd/image</topic>
    <camera name="rgbd_front">
      <horizontal_fov>1.5009</horizontal_fov>  <!-- 86 degrees -->
      <lens>
        <intrinsics>
          <fx>343.159</fx>
          <fy>343.159</fy>
          <cx>319.5</cx>
          <cy>179.5</cy>
          <s>0</s>
        </intrinsics>
      </lens>
      <distortion>
        <k1>0.0</k1>
        <k2>0.0</k2>
        <k3>0.0</k3>
        <p1>0.0</p1>
        <p2>0.0</p2>
        <center>0.5 0.5</center>
      </distortion>
      <image>
        <width>640</width>
        <height>360</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.01</near>
        <far>300</far>
      </clip>
      <depth_camera>
        <clip>
          <near>0.1</near>
          <far>10</far>
        </clip>
      </depth_camera>
      <noise>
        <type>gaussian</type>
        <mean>0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
  </sensor>
</link>
```

---

## 4. ArduPilot Plugin Configuration

### 4.1 Plugin Integration in Vehicle Model

The ArduPilotPlugin connects Gazebo's physics simulation to ArduPilot SITL via JSON over UDP:

```xml
<!-- Iris Quadcopter with ArduPilot Plugin -->
<!-- models/iris_with_ardupilot/model.sdf -->
<?xml version="1.0"?>
<sdf version="1.9">
  <model name="iris_with_ardupilot">
    
    <!-- Include base Iris model structure -->
    <include>
      <uri>model://iris_with_standoffs</uri>
    </include>
    
    <!-- ArduPilot Plugin Configuration -->
    <plugin filename="ArduPilotPlugin" name="ArduPilotPlugin">
      <!-- SITL JSON interface -->
      <fdm_addr>127.0.0.1</fdm_addr>
      <fdm_port_in>9002</fdm_port_in>
      <fdm_port_out>9003</fdm_port_out>
      <modelXYZToAirplaneXForwardZDown>0 0 0 3.141593 0 0</modelXYZToAirplaneXForwardZDown>
      <gazeboXYZToNED>0 0 0 3.141593 0 0</gazeboXYZToNED>
      
      <!-- IMU Configuration -->
      <imuName>iris_with_standoffs::iris::imu_link::imu_sensor</imuName>
      
      <!-- Rotor Configuration (4 motors for quadcopter) -->
      <control channel="0">
        <jointName>iris_with_standoffs::iris::rotor_0_joint</jointName>
        <type>VELOCITY</type>
        <offset>0</offset>
        <p_gain>0.20</p_gain>
        <i_gain>0</i_gain>
        <d_gain>0</d_gain>
        <i_max>0</i_max>
        <i_min>0</i_min>
        <cmd_max>2.5</cmd_max>
        <cmd_min>-2.5</cmd_min>
        <multiplier>838</multiplier>
        <controlVelocitySlowdownSim>1</controlVelocitySlowdownSim>
      </control>
      
      <control channel="1">
        <jointName>iris_with_standoffs::iris::rotor_1_joint</jointName>
        <type>VELOCITY</type>
        <offset>0</offset>
        <p_gain>0.20</p_gain>
        <i_gain>0</i_gain>
        <d_gain>0</d_gain>
        <i_max>0</i_max>
        <i_min>0</i_min>
        <cmd_max>2.5</cmd_max>
        <cmd_min>-2.5</cmd_min>
        <multiplier>838</multiplier>
        <controlVelocitySlowdownSim>1</controlVelocitySlowdownSim>
      </control>
      
      <control channel="2">
        <jointName>iris_with_standoffs::iris::rotor_2_joint</jointName>
        <type>VELOCITY</type>
        <offset>0</offset>
        <p_gain>0.20</p_gain>
        <i_gain>0</i_gain>
        <d_gain>0</d_gain>
        <i_max>0</i_max>
        <i_min>0</i_min>
        <cmd_max>2.5</cmd_max>
        <cmd_min>-2.5</cmd_min>
        <multiplier>-838</multiplier>
        <controlVelocitySlowdownSim>1</controlVelocitySlowdownSim>
      </control>
      
      <control channel="3">
        <jointName>iris_with_standoffs::iris::rotor_3_joint</jointName>
        <type>VELOCITY</type>
        <offset>0</offset>
        <p_gain>0.20</p_gain>
        <i_gain>0</i_gain>
        <d_gain>0</d_gain>
        <i_max>0</i_max>
        <i_min>0</i_min>
        <cmd_max>2.5</cmd_max>
        <cmd_min>-2.5</cmd_min>
        <multiplier>-838</multiplier>
        <controlVelocitySlowdownSim>1</controlVelocitySlowdownSim>
      </control>
    </plugin>
    
    <!-- Lift/Drag plugins for realistic aerodynamics -->
    <plugin filename="gz-sim-lift-drag-system"
        name="gz::sim::systems::LiftDrag">
      <a0>0.3</a0>
      <alpha_stall>1.4</alpha_stall>
      <cla>4.2500</cla>
      <cda>0.10</cda>
      <cma>0.00</cma>
      <cla_stall>-0.025</cla_stall>
      <cda_stall>0.0</cda_stall>
      <cma_stall>0.0</cma_stall>
      <area>0.002</area>
      <air_density>1.2041</air_density>
      <cp>0.084 0 0</cp>
      <forward>0 1 0</forward>
      <upward>0 0 1</upward>
      <link_name>iris_with_standoffs::iris::rotor_0</link_name>
    </plugin>
    <!-- Repeat for other rotors... -->
    
  </model>
</sdf>
```

### 4.2 Ground Robot (Rover) Configuration

```xml
<!-- models/rover_with_ardupilot/model.sdf -->
<?xml version="1.0"?>
<sdf version="1.9">
  <model name="rover_with_ardupilot">
    
    <!-- Base rover structure with differential drive -->
    <link name="base_link">
      <inertial>
        <mass>20</mass>
        <inertia>
          <ixx>0.5</ixx><iyy>0.5</iyy><izz>0.5</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>0.8 0.5 0.3</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>0.8 0.5 0.3</size></box>
        </geometry>
      </visual>
    </link>
    
    <!-- Left wheel -->
    <link name="left_wheel">
      <pose relative_to="base_link">0 0.3 -0.1 -1.5708 0 0</pose>
      <inertial>
        <mass>1</mass>
        <inertia>
          <ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder><radius>0.15</radius><length>0.05</length></cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder><radius>0.15</radius><length>0.05</length></cylinder>
        </geometry>
      </visual>
    </link>
    
    <!-- Right wheel -->
    <link name="right_wheel">
      <pose relative_to="base_link">0 -0.3 -0.1 -1.5708 0 0</pose>
      <inertial>
        <mass>1</mass>
        <inertia>
          <ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder><radius>0.15</radius><length>0.05</length></cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder><radius>0.15</radius><length>0.05</length></cylinder>
        </geometry>
      </visual>
    </link>
    
    <!-- Joints -->
    <joint name="left_wheel_joint" type="revolute">
      <parent>base_link</parent>
      <child>left_wheel</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1e16</lower>
          <upper>1e16</upper>
        </limit>
      </axis>
    </joint>
    
    <joint name="right_wheel_joint" type="revolute">
      <parent>base_link</parent>
      <child>right_wheel</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1e16</lower>
          <upper>1e16</upper>
        </limit>
      </axis>
    </joint>
    
    <!-- ArduPilot Rover Plugin -->
    <plugin filename="ArduPilotPlugin" name="ArduPilotPlugin">
      <fdm_addr>127.0.0.1</fdm_addr>
      <fdm_port_in>9012</fdm_port_in>  <!-- Different port for second vehicle -->
      <fdm_port_out>9013</fdm_port_out>
      <modelXYZToAirplaneXForwardZDown>0 0 0 3.141593 0 0</modelXYZToAirplaneXForwardZDown>
      <gazeboXYZToNED>0 0 0 3.141593 0 0</gazeboXYZToNED>
      
      <imuName>rover::imu_link::imu_sensor</imuName>
      
      <!-- Skid-steer control: Channel 0 = Steering, Channel 2 = Throttle -->
      <control channel="0">
        <jointName>left_wheel_joint</jointName>
        <type>VELOCITY</type>
        <offset>0</offset>
        <p_gain>10</p_gain>
        <i_gain>0</i_gain>
        <d_gain>0</d_gain>
        <i_max>0</i_max>
        <i_min>0</i_min>
        <cmd_max>10</cmd_max>
        <cmd_min>-10</cmd_min>
        <multiplier>1</multiplier>
      </control>
      
      <control channel="2">
        <jointName>right_wheel_joint</jointName>
        <type>VELOCITY</type>
        <offset>0</offset>
        <p_gain>10</p_gain>
        <i_gain>0</i_gain>
        <d_gain>0</d_gain>
        <i_max>0</i_max>
        <i_min>0</i_min>
        <cmd_max>10</cmd_max>
        <cmd_min>-10</cmd_min>
        <multiplier>1</multiplier>
      </control>
    </plugin>
    
  </model>
</sdf>
```

---

## 5. World Configuration for OccWorld Training

### 5.1 Tokyo Urban World with Ground Truth Generation

```xml
<!-- worlds/tokyo_occworld.sdf -->
<?xml version="1.0"?>
<sdf version="1.9">
  <world name="tokyo_occworld">
    
    <!-- Physics Configuration for Training -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>0</real_time_factor>  <!-- Run as fast as possible -->
      <real_time_update_rate>0</real_time_update_rate>
    </physics>
    
    <!-- Required System Plugins -->
    <plugin filename="gz-sim-physics-system"
        name="gz::sim::systems::Physics">
    </plugin>
    
    <plugin filename="gz-sim-user-commands-system"
        name="gz::sim::systems::UserCommands">
    </plugin>
    
    <plugin filename="gz-sim-scene-broadcaster-system"
        name="gz::sim::systems::SceneBroadcaster">
    </plugin>
    
    <plugin filename="gz-sim-sensors-system"
        name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    
    <plugin filename="gz-sim-imu-system"
        name="gz::sim::systems::Imu">
    </plugin>
    
    <plugin filename="gz-sim-navsat-system"
        name="gz::sim::systems::NavSat">
    </plugin>
    
    <!-- Spherical Coordinates (Tokyo, Japan) -->
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>35.6762</latitude_deg>
      <longitude_deg>139.6503</longitude_deg>
      <elevation>40</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>
    
    <!-- Lighting -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 100 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    
    <!-- Ground Plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>1000 1000</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>1000 1000</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Tokyo Buildings (Load from PLATEAU meshes) -->
    <!-- Example building - replace with actual PLATEAU data -->
    <model name="building_01">
      <static>true</static>
      <pose>50 30 0 0 0 0.3</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>20 15 60</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>20 15 60</size></box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.6 1</ambient>
          </material>
        </visual>
      </link>
    </model>
    
    <model name="building_02">
      <static>true</static>
      <pose>-40 50 0 0 0 -0.2</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>25 20 80</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>25 20 80</size></box>
          </geometry>
          <material>
            <ambient>0.6 0.5 0.5 1</ambient>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Roads (semantic class for ground truth) -->
    <model name="road_main">
      <static>true</static>
      <pose>0 0 0.01 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box><size>200 8 0.02</size></box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.2 1</ambient>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Include Drone -->
    <include>
      <uri>model://iris_with_ardupilot</uri>
      <name>drone_1</name>
      <pose>0 0 0.2 0 0 0</pose>
    </include>
    
    <!-- Include Ground Robot -->
    <include>
      <uri>model://rover_with_ardupilot</uri>
      <name>rover_1</name>
      <pose>5 5 0.2 0 0 0</pose>
    </include>
    
  </world>
</sdf>
```

---

## 6. ROS 2 Bridge Configuration

### 6.1 Bridge Launch File

```python
# launch/occworld_gazebo_bridge.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    
    # Bridge configuration for all sensors
    bridge_config = os.path.join(
        os.getenv('HOME'),
        'occworld_ws/src/occworld_gazebo/config/bridge_config.yaml'
    )
    
    # Gazebo-ROS bridge node
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': bridge_config}],
        output='screen'
    )
    
    # Image bridge for cameras (separate for performance)
    camera_bridges = []
    camera_topics = [
        'camera/front/image',
        'camera/front_left/image',
        'camera/front_right/image',
        'camera/back/image',
        'camera/back_left/image',
        'camera/back_right/image',
    ]
    
    for topic in camera_topics:
        camera_bridges.append(
            Node(
                package='ros_gz_image',
                executable='image_bridge',
                arguments=[f'/world/tokyo_occworld/model/drone_1/link/sensor_base/{topic}'],
                output='screen'
            )
        )
    
    # MAVROS for ArduPilot communication
    mavros_drone = Node(
        package='mavros',
        executable='mavros_node',
        name='mavros_drone',
        parameters=[{
            'fcu_url': 'udp://:14550@127.0.0.1:14555',
            'gcs_url': '',
            'target_system_id': 1,
            'target_component_id': 1,
        }],
        remappings=[
            ('/mavros', '/drone_1/mavros'),
        ],
        output='screen'
    )
    
    mavros_rover = Node(
        package='mavros',
        executable='mavros_node',
        name='mavros_rover',
        parameters=[{
            'fcu_url': 'udp://:14560@127.0.0.1:14565',
            'gcs_url': '',
            'target_system_id': 2,
            'target_component_id': 1,
        }],
        remappings=[
            ('/mavros', '/rover_1/mavros'),
        ],
        output='screen'
    )
    
    return LaunchDescription([
        ros_gz_bridge,
        *camera_bridges,
        mavros_drone,
        mavros_rover,
    ])
```

### 6.2 Bridge Configuration YAML

```yaml
# config/bridge_config.yaml
---
# LiDAR Topics
- ros_topic_name: "/drone_1/lidar/points"
  gz_topic_name: "/world/tokyo_occworld/model/drone_1/link/lidar_drone_link/sensor/gpu_lidar_drone/scan/points"
  ros_type_name: "sensor_msgs/msg/PointCloud2"
  gz_type_name: "gz.msgs.PointCloudPacked"
  direction: GZ_TO_ROS

- ros_topic_name: "/rover_1/lidar/points"
  gz_topic_name: "/world/tokyo_occworld/model/rover_1/link/lidar_ground_link/sensor/gpu_lidar_ground/scan/points"
  ros_type_name: "sensor_msgs/msg/PointCloud2"
  gz_type_name: "gz.msgs.PointCloudPacked"
  direction: GZ_TO_ROS

# IMU Topics
- ros_topic_name: "/drone_1/imu/data"
  gz_topic_name: "/world/tokyo_occworld/model/drone_1/link/imu_link/sensor/imu_sensor/imu"
  ros_type_name: "sensor_msgs/msg/Imu"
  gz_type_name: "gz.msgs.IMU"
  direction: GZ_TO_ROS

- ros_topic_name: "/rover_1/imu/data"
  gz_topic_name: "/world/tokyo_occworld/model/rover_1/link/imu_link/sensor/imu_sensor/imu"
  ros_type_name: "sensor_msgs/msg/Imu"
  gz_type_name: "gz.msgs.IMU"
  direction: GZ_TO_ROS

# GPS Topics
- ros_topic_name: "/drone_1/gps/fix"
  gz_topic_name: "/world/tokyo_occworld/model/drone_1/link/gps_link/sensor/navsat_sensor/navsat"
  ros_type_name: "sensor_msgs/msg/NavSatFix"
  gz_type_name: "gz.msgs.NavSat"
  direction: GZ_TO_ROS

# Odometry
- ros_topic_name: "/drone_1/odom"
  gz_topic_name: "/model/drone_1/odometry"
  ros_type_name: "nav_msgs/msg/Odometry"
  gz_type_name: "gz.msgs.Odometry"
  direction: GZ_TO_ROS

# Clock (for synchronized playback)
- ros_topic_name: "/clock"
  gz_topic_name: "/clock"
  ros_type_name: "rosgraph_msgs/msg/Clock"
  gz_type_name: "gz.msgs.Clock"
  direction: GZ_TO_ROS
```

---

## 7. Data Recording for OccWorld Training

### 7.1 Custom ROS 2 Node for Synchronized Recording

```python
#!/usr/bin/env python3
# nodes/occworld_data_recorder.py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer

from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

import numpy as np
import cv2
from cv_bridge import CvBridge
import json
import os
from datetime import datetime


class OccWorldDataRecorder(Node):
    """
    Synchronized multi-sensor data recorder for OccWorld training.
    Records camera images, LiDAR scans, and pose ground truth.
    """
    
    def __init__(self):
        super().__init__('occworld_data_recorder')
        
        # Parameters
        self.declare_parameter('output_dir', '/data/occworld_training')
        self.declare_parameter('agent_type', 'drone')  # 'drone' or 'rover'
        self.declare_parameter('record_rate', 2.0)  # Hz
        
        self.output_dir = self.get_parameter('output_dir').value
        self.agent_type = self.get_parameter('agent_type').value
        self.record_rate = self.get_parameter('record_rate').value
        
        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_record_time = self.get_clock().now()
        
        # Create output directories
        self.session_dir = os.path.join(
            self.output_dir,
            f"{self.agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(os.path.join(self.session_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, 'lidar'), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, 'poses'), exist_ok=True)
        
        # QoS for sensor data
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        
        # Subscribers with message filters for synchronization
        prefix = f"/{self.agent_type}_1"
        
        self.cam_front_sub = Subscriber(
            self, Image, f'{prefix}/camera/front/image', qos_profile=sensor_qos
        )
        self.cam_fl_sub = Subscriber(
            self, Image, f'{prefix}/camera/front_left/image', qos_profile=sensor_qos
        )
        self.cam_fr_sub = Subscriber(
            self, Image, f'{prefix}/camera/front_right/image', qos_profile=sensor_qos
        )
        self.cam_back_sub = Subscriber(
            self, Image, f'{prefix}/camera/back/image', qos_profile=sensor_qos
        )
        self.cam_bl_sub = Subscriber(
            self, Image, f'{prefix}/camera/back_left/image', qos_profile=sensor_qos
        )
        self.cam_br_sub = Subscriber(
            self, Image, f'{prefix}/camera/back_right/image', qos_profile=sensor_qos
        )
        self.lidar_sub = Subscriber(
            self, PointCloud2, f'{prefix}/lidar/points', qos_profile=sensor_qos
        )
        self.odom_sub = Subscriber(
            self, Odometry, f'{prefix}/odom', qos_profile=sensor_qos
        )
        
        # Approximate time synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [
                self.cam_front_sub, self.cam_fl_sub, self.cam_fr_sub,
                self.cam_back_sub, self.cam_bl_sub, self.cam_br_sub,
                self.lidar_sub, self.odom_sub
            ],
            queue_size=10,
            slop=0.1  # 100ms tolerance
        )
        self.sync.registerCallback(self.sync_callback)
        
        self.get_logger().info(f'OccWorld Data Recorder initialized for {self.agent_type}')
        self.get_logger().info(f'Saving to: {self.session_dir}')
    
    def sync_callback(self, cam_f, cam_fl, cam_fr, cam_b, cam_bl, cam_br, lidar, odom):
        """Process synchronized sensor data."""
        
        # Rate limiting
        current_time = self.get_clock().now()
        dt = (current_time - self.last_record_time).nanoseconds / 1e9
        if dt < (1.0 / self.record_rate):
            return
        self.last_record_time = current_time
        
        timestamp = current_time.nanoseconds
        frame_id = f"{self.frame_count:06d}"
        
        # Save camera images
        cameras = {
            'CAM_FRONT': cam_f,
            'CAM_FRONT_LEFT': cam_fl,
            'CAM_FRONT_RIGHT': cam_fr,
            'CAM_BACK': cam_b,
            'CAM_BACK_LEFT': cam_bl,
            'CAM_BACK_RIGHT': cam_br,
        }
        
        for cam_name, cam_msg in cameras.items():
            try:
                cv_image = self.bridge.imgmsg_to_cv2(cam_msg, "bgr8")
                filename = os.path.join(
                    self.session_dir, 'images',
                    f'{frame_id}_{cam_name}.jpg'
                )
                cv2.imwrite(filename, cv_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            except Exception as e:
                self.get_logger().error(f'Failed to save {cam_name}: {e}')
        
        # Save LiDAR point cloud as numpy
        try:
            points = self.pointcloud2_to_array(lidar)
            np.save(
                os.path.join(self.session_dir, 'lidar', f'{frame_id}_LIDAR.npy'),
                points
            )
        except Exception as e:
            self.get_logger().error(f'Failed to save LiDAR: {e}')
        
        # Save pose/odometry
        pose_data = {
            'timestamp': timestamp,
            'frame_id': frame_id,
            'agent_type': self.agent_type,
            'position': {
                'x': odom.pose.pose.position.x,
                'y': odom.pose.pose.position.y,
                'z': odom.pose.pose.position.z,
            },
            'orientation': {
                'x': odom.pose.pose.orientation.x,
                'y': odom.pose.pose.orientation.y,
                'z': odom.pose.pose.orientation.z,
                'w': odom.pose.pose.orientation.w,
            },
            'velocity': {
                'linear': {
                    'x': odom.twist.twist.linear.x,
                    'y': odom.twist.twist.linear.y,
                    'z': odom.twist.twist.linear.z,
                },
                'angular': {
                    'x': odom.twist.twist.angular.x,
                    'y': odom.twist.twist.angular.y,
                    'z': odom.twist.twist.angular.z,
                }
            }
        }
        
        with open(os.path.join(self.session_dir, 'poses', f'{frame_id}.json'), 'w') as f:
            json.dump(pose_data, f, indent=2)
        
        self.frame_count += 1
        
        if self.frame_count % 100 == 0:
            self.get_logger().info(f'Recorded {self.frame_count} frames')
    
    def pointcloud2_to_array(self, cloud_msg):
        """Convert PointCloud2 to numpy array."""
        import sensor_msgs_py.point_cloud2 as pc2
        points = []
        for p in pc2.read_points(cloud_msg, skip_nans=True):
            points.append([p[0], p[1], p[2], p[3] if len(p) > 3 else 1.0])  # x, y, z, intensity
        return np.array(points, dtype=np.float32)


def main(args=None):
    rclpy.init(args=args)
    node = OccWorldDataRecorder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 7.2 Ground Truth Occupancy Generation

```python
#!/usr/bin/env python3
# scripts/generate_occupancy_gt.py
"""
Generate ground truth occupancy grids from Gazebo simulation.
Uses raycasting from vehicle pose to create voxelized occupancy.
"""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import json
import os
from glob import glob
from tqdm import tqdm


class OccupancyGroundTruthGenerator:
    """Generate OccWorld-format occupancy ground truth from LiDAR scans."""
    
    def __init__(self, config):
        self.config = config
        
        # Voxel grid configuration (matching OccWorld extended config)
        self.voxel_size = config.get('voxel_size', [0.4, 0.4, 0.4])
        self.point_cloud_range = config.get(
            'point_cloud_range',
            [-40, -40, -2, 40, 40, 150]  # Extended for drone
        )
        
        # Calculate grid dimensions
        self.grid_size = [
            int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0]),
            int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1]),
            int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.voxel_size[2]),
        ]
        
        # Semantic classes (nuScenes + aerial extensions)
        self.semantic_classes = {
            0: 'empty',
            1: 'barrier',
            2: 'bicycle',
            3: 'bus',
            4: 'car',
            5: 'construction_vehicle',
            6: 'motorcycle',
            7: 'pedestrian',
            8: 'traffic_cone',
            9: 'trailer',
            10: 'truck',
            11: 'driveable_surface',
            12: 'other_flat',
            13: 'sidewalk',
            14: 'terrain',
            15: 'manmade',  # buildings
            16: 'vegetation',
            17: 'sky',  # for aerial: unobstructed space
        }
    
    def process_frame(self, lidar_path, pose_path, output_path):
        """Process single frame to generate occupancy."""
        
        # Load LiDAR points
        points = np.load(lidar_path)  # [N, 4] - x, y, z, intensity
        
        # Load pose
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
        
        # Create transformation matrix
        position = pose_data['position']
        orientation = pose_data['orientation']
        
        T = np.eye(4)
        T[:3, 3] = [position['x'], position['y'], position['z']]
        R = Rotation.from_quat([
            orientation['x'], orientation['y'],
            orientation['z'], orientation['w']
        ]).as_matrix()
        T[:3, :3] = R
        
        # Transform points to world frame
        points_world = self.transform_points(points[:, :3], T)
        
        # Voxelize
        occupancy = self.voxelize_points(points_world)
        
        # Add semantic labels based on height heuristics
        # (In production, use semantic segmentation network)
        occupancy_semantic = self.add_semantic_labels(occupancy, pose_data)
        
        # Save
        np.savez_compressed(
            output_path,
            occupancy=occupancy_semantic,
            pose=T,
            grid_size=self.grid_size,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        
        return occupancy_semantic
    
    def transform_points(self, points, T):
        """Apply transformation to points."""
        ones = np.ones((points.shape[0], 1))
        points_homo = np.hstack([points, ones])
        points_transformed = (T @ points_homo.T).T
        return points_transformed[:, :3]
    
    def voxelize_points(self, points):
        """Convert point cloud to voxel grid."""
        
        # Filter points within range
        mask = (
            (points[:, 0] >= self.point_cloud_range[0]) &
            (points[:, 0] < self.point_cloud_range[3]) &
            (points[:, 1] >= self.point_cloud_range[1]) &
            (points[:, 1] < self.point_cloud_range[4]) &
            (points[:, 2] >= self.point_cloud_range[2]) &
            (points[:, 2] < self.point_cloud_range[5])
        )
        points_filtered = points[mask]
        
        # Calculate voxel indices
        voxel_indices = np.floor(
            (points_filtered - np.array(self.point_cloud_range[:3])) /
            np.array(self.voxel_size)
        ).astype(np.int32)
        
        # Create occupancy grid
        occupancy = np.zeros(self.grid_size, dtype=np.uint8)
        
        # Mark occupied voxels
        valid_mask = (
            (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < self.grid_size[0]) &
            (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < self.grid_size[1]) &
            (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < self.grid_size[2])
        )
        valid_indices = voxel_indices[valid_mask]
        
        occupancy[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = 1
        
        return occupancy
    
    def add_semantic_labels(self, occupancy, pose_data):
        """Add semantic labels based on height and position heuristics."""
        
        semantic_occ = np.zeros_like(occupancy, dtype=np.uint8)
        agent_height = pose_data['position']['z']
        
        # Height-based heuristics for Tokyo urban environment
        for z in range(self.grid_size[2]):
            actual_z = self.point_cloud_range[2] + z * self.voxel_size[2]
            
            occupied_mask = occupancy[:, :, z] > 0
            
            if actual_z < 0.5:
                # Ground level: driveable surface
                semantic_occ[:, :, z][occupied_mask] = 11  # driveable_surface
            elif actual_z < 2.0:
                # Low obstacles: could be vehicles, pedestrians
                semantic_occ[:, :, z][occupied_mask] = 15  # manmade (default)
            elif actual_z < 50:
                # Building level
                semantic_occ[:, :, z][occupied_mask] = 15  # manmade
            else:
                # High altitude: likely building tops or clear sky
                semantic_occ[:, :, z][occupied_mask] = 15  # manmade
        
        return semantic_occ
    
    def process_session(self, session_dir, output_dir):
        """Process entire recording session."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all frames
        lidar_files = sorted(glob(os.path.join(session_dir, 'lidar', '*.npy')))
        
        for lidar_path in tqdm(lidar_files, desc='Generating occupancy GT'):
            frame_id = os.path.basename(lidar_path).replace('_LIDAR.npy', '')
            pose_path = os.path.join(session_dir, 'poses', f'{frame_id}.json')
            output_path = os.path.join(output_dir, f'{frame_id}_occupancy.npz')
            
            if os.path.exists(pose_path):
                self.process_frame(lidar_path, pose_path, output_path)


if __name__ == '__main__':
    config = {
        'voxel_size': [0.4, 0.4, 1.25],  # Coarser Z for extended range
        'point_cloud_range': [-40, -40, -2, 40, 40, 150],
    }
    
    generator = OccupancyGroundTruthGenerator(config)
    generator.process_session(
        '/data/occworld_training/drone_20240101_120000',
        '/data/occworld_training/drone_20240101_120000/occupancy'
    )
```

---

## 8. Running the Complete Pipeline

### 8.1 Launch Sequence

```bash
# Terminal 1: Launch Gazebo with Tokyo world
gz sim -v4 -r tokyo_occworld.sdf

# Terminal 2: Launch ArduPilot SITL for drone
cd ~/ardupilot
sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console -I 0

# Terminal 3: Launch ArduPilot SITL for rover
cd ~/ardupilot
sim_vehicle.py -v Rover -f gazebo-rover --model JSON --map --console -I 1

# Terminal 4: Launch ROS 2 bridge
source /opt/ros/humble/setup.bash
source ~/occworld_ws/install/setup.bash
ros2 launch occworld_gazebo occworld_gazebo_bridge.launch.py

# Terminal 5: Launch data recorder
ros2 run occworld_gazebo occworld_data_recorder \
    --ros-args -p agent_type:=drone -p record_rate:=2.0
```

### 8.2 Automated Mission for Data Collection

```python
#!/usr/bin/env python3
# scripts/data_collection_mission.py
"""
Automated data collection mission using DroneKit/pymavlink.
Executes predefined waypoints while recording sensor data.
"""

from pymavlink import mavutil
import time
import math


def connect_vehicle(connection_string):
    """Connect to vehicle via MAVLink."""
    master = mavutil.mavlink_connection(connection_string)
    master.wait_heartbeat()
    print(f"Connected to vehicle (system {master.target_system})")
    return master


def arm_and_takeoff(master, altitude):
    """Arm and take off to specified altitude."""
    
    # Set mode to GUIDED
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        4  # GUIDED mode
    )
    time.sleep(1)
    
    # Arm
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    
    # Wait for arming
    master.motors_armed_wait()
    print("Armed!")
    
    # Takeoff
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude
    )
    
    # Wait for altitude
    while True:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        current_alt = msg.relative_alt / 1000.0
        print(f"Altitude: {current_alt:.1f}m")
        if current_alt >= altitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(0.5)


def goto_position_target_local_ned(master, north, east, down):
    """Send position command in local NED frame."""
    master.mav.set_position_target_local_ned_send(
        0,  # time_boot_ms
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,  # Position only
        north, east, down,
        0, 0, 0,  # velocity
        0, 0, 0,  # acceleration
        0, 0      # yaw, yaw_rate
    )


def execute_survey_pattern(master, grid_size=80, spacing=20, altitude=30):
    """Execute lawn-mower survey pattern for data collection."""
    
    waypoints = []
    
    # Generate lawn-mower pattern
    y = -grid_size / 2
    direction = 1
    while y <= grid_size / 2:
        if direction == 1:
            waypoints.append((-grid_size / 2, y, -altitude))
            waypoints.append((grid_size / 2, y, -altitude))
        else:
            waypoints.append((grid_size / 2, y, -altitude))
            waypoints.append((-grid_size / 2, y, -altitude))
        
        y += spacing
        direction *= -1
    
    print(f"Executing survey with {len(waypoints)} waypoints")
    
    for i, (north, east, down) in enumerate(waypoints):
        print(f"Waypoint {i+1}/{len(waypoints)}: N={north:.1f}, E={east:.1f}, D={down:.1f}")
        goto_position_target_local_ned(master, north, east, down)
        
        # Wait for arrival
        while True:
            msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=True)
            dist = math.sqrt(
                (msg.x - north)**2 +
                (msg.y - east)**2 +
                (msg.z - down)**2
            )
            if dist < 2.0:  # Within 2m
                break
            time.sleep(0.1)
        
        # Hover briefly for stable data capture
        time.sleep(2.0)
    
    print("Survey complete!")


def main():
    # Connect to drone SITL
    master = connect_vehicle('udp:127.0.0.1:14550')
    
    # Arm and takeoff
    arm_and_takeoff(master, altitude=30)
    
    # Execute survey pattern
    execute_survey_pattern(master, grid_size=80, spacing=20, altitude=30)
    
    # Land
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    
    print("Landing...")


if __name__ == '__main__':
    main()
```

---

## 9. Integration with OccWorld Training Pipeline

### 9.1 Dataset Loader for Gazebo-Generated Data

```python
# dataset/gazebo_occworld_dataset.py
"""
PyTorch dataset for OccWorld training using Gazebo-generated data.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
import os
from glob import glob


class GazeboOccWorldDataset(Dataset):
    """Dataset loader for ArduPilot Gazebo simulation data."""
    
    def __init__(
        self,
        data_root,
        split='train',
        history_frames=4,
        future_frames=6,
        agent_type='both',  # 'drone', 'rover', or 'both'
        transform=None
    ):
        self.data_root = data_root
        self.split = split
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.agent_type = agent_type
        self.transform = transform
        
        # Load session list
        self.sessions = self._load_sessions()
        
        # Build frame index
        self.frame_index = self._build_frame_index()
        
        # Camera names matching OccWorld/BEVFusion
        self.camera_names = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
    
    def _load_sessions(self):
        """Find all valid recording sessions."""
        sessions = []
        
        for session_dir in glob(os.path.join(self.data_root, '*')):
            if not os.path.isdir(session_dir):
                continue
            
            session_name = os.path.basename(session_dir)
            
            # Filter by agent type
            if self.agent_type == 'drone' and not session_name.startswith('drone'):
                continue
            if self.agent_type == 'rover' and not session_name.startswith('rover'):
                continue
            
            # Check required directories exist
            if (os.path.exists(os.path.join(session_dir, 'images')) and
                os.path.exists(os.path.join(session_dir, 'lidar')) and
                os.path.exists(os.path.join(session_dir, 'occupancy'))):
                sessions.append(session_dir)
        
        return sessions
    
    def _build_frame_index(self):
        """Build index of valid frame sequences."""
        index = []
        
        for session_dir in self.sessions:
            # Get sorted frame IDs
            occ_files = sorted(glob(os.path.join(session_dir, 'occupancy', '*.npz')))
            frame_ids = [
                os.path.basename(f).replace('_occupancy.npz', '')
                for f in occ_files
            ]
            
            # Find valid sequences (history + future frames available)
            total_frames_needed = self.history_frames + self.future_frames
            
            for i in range(len(frame_ids) - total_frames_needed + 1):
                sequence_frames = frame_ids[i:i + total_frames_needed]
                
                # Verify all frames exist
                valid = True
                for fid in sequence_frames:
                    for cam in self.camera_names:
                        if not os.path.exists(
                            os.path.join(session_dir, 'images', f'{fid}_{cam}.jpg')
                        ):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    index.append({
                        'session': session_dir,
                        'frames': sequence_frames,
                        'agent_type': 'drone' if 'drone' in session_dir else 'rover'
                    })
        
        return index
    
    def __len__(self):
        return len(self.frame_index)
    
    def __getitem__(self, idx):
        item = self.frame_index[idx]
        session_dir = item['session']
        frames = item['frames']
        agent_type = item['agent_type']
        
        # Split into history and future
        history_frames = frames[:self.history_frames]
        future_frames = frames[self.history_frames:]
        
        # Load history data
        history_images = []
        history_lidar = []
        history_poses = []
        history_occupancy = []
        
        for fid in history_frames:
            # Load images
            frame_images = {}
            for cam in self.camera_names:
                img_path = os.path.join(session_dir, 'images', f'{fid}_{cam}.jpg')
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame_images[cam] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            history_images.append(frame_images)
            
            # Load LiDAR
            lidar_path = os.path.join(session_dir, 'lidar', f'{fid}_LIDAR.npy')
            lidar_points = np.load(lidar_path)
            history_lidar.append(torch.from_numpy(lidar_points))
            
            # Load pose
            pose_path = os.path.join(session_dir, 'poses', f'{fid}.json')
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)
            history_poses.append(self._pose_to_tensor(pose_data))
            
            # Load occupancy
            occ_path = os.path.join(session_dir, 'occupancy', f'{fid}_occupancy.npz')
            occ_data = np.load(occ_path)
            history_occupancy.append(torch.from_numpy(occ_data['occupancy']))
        
        # Load future occupancy (ground truth for prediction)
        future_occupancy = []
        future_poses = []
        
        for fid in future_frames:
            occ_path = os.path.join(session_dir, 'occupancy', f'{fid}_occupancy.npz')
            occ_data = np.load(occ_path)
            future_occupancy.append(torch.from_numpy(occ_data['occupancy']))
            
            pose_path = os.path.join(session_dir, 'poses', f'{fid}.json')
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)
            future_poses.append(self._pose_to_tensor(pose_data))
        
        # Stack tensors
        sample = {
            'history_images': history_images,  # List of dicts
            'history_lidar': history_lidar,    # List of [N, 4] tensors
            'history_poses': torch.stack(history_poses),
            'history_occupancy': torch.stack(history_occupancy),
            'future_occupancy': torch.stack(future_occupancy),
            'future_poses': torch.stack(future_poses),
            'agent_type': 0 if agent_type == 'rover' else 1,  # 0=ground, 1=aerial
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _pose_to_tensor(self, pose_data):
        """Convert pose dict to tensor."""
        pos = pose_data['position']
        ori = pose_data['orientation']
        vel = pose_data['velocity']
        
        return torch.tensor([
            pos['x'], pos['y'], pos['z'],
            ori['x'], ori['y'], ori['z'], ori['w'],
            vel['linear']['x'], vel['linear']['y'], vel['linear']['z'],
            vel['angular']['x'], vel['angular']['y'], vel['angular']['z'],
        ], dtype=torch.float32)
```

---

## 10. Performance Optimization

### 10.1 Gazebo Physics Speedup

```xml
<!-- Fast training physics (in world SDF) -->
<physics name="fast" type="ode">
  <max_step_size>0.004</max_step_size>  <!-- 4ms steps -->
  <real_time_factor>0</real_time_factor>  <!-- Run as fast as possible -->
  <real_time_update_rate>0</real_time_update_rate>
  
  <ode>
    <solver>
      <type>quick</type>
      <iters>50</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### 10.2 Sensor Optimization

```xml
<!-- Reduced resolution sensors for faster simulation -->
<sensor name="camera_fast" type="camera">
  <update_rate>5</update_rate>  <!-- Reduced from 10Hz -->
  <camera>
    <image>
      <width>800</width>   <!-- Reduced from 1600 -->
      <height>450</height> <!-- Reduced from 900 -->
    </image>
  </camera>
</sensor>

<sensor name="lidar_fast" type="gpu_lidar">
  <update_rate>5</update_rate>  <!-- Reduced from 10Hz -->
  <lidar>
    <scan>
      <horizontal>
        <samples>900</samples>  <!-- Reduced from 1800 -->
      </horizontal>
      <vertical>
        <samples>32</samples>   <!-- Reduced from 64 -->
      </vertical>
    </scan>
  </lidar>
</sensor>
```

### 10.3 Headless Rendering

```bash
# Run Gazebo headless for data generation
gz sim -v1 -s -r tokyo_occworld.sdf  # -s for server only, no GUI

# Or with specific rendering for sensors
GZ_SIM_RENDER_ENGINE_PATH=/usr/lib/x86_64-linux-gnu/gz-rendering-8/engine-plugins \
GZ_SIM_RENDER_ENGINE=ogre2 \
gz sim -v1 --headless-rendering -r tokyo_occworld.sdf
```

---

## 11. Troubleshooting

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| "ArduPilot not connecting" | Port mismatch | Verify `fdm_port_in/out` match SITL JSON ports |
| "No sensor data on ROS topics" | Bridge not configured | Check `GZ_SIM_SYSTEM_PLUGIN_PATH` includes Sensors system |
| "Simulation too slow" | Physics too accurate | Increase `max_step_size`, reduce sensor rates |
| "Vehicle flying erratically" | PID tuning | Adjust `p_gain` in ArduPilotPlugin control elements |
| "LiDAR not working" | GPU rendering issue | Ensure `ogre2` renderer, check OpenGL support |
| "Memory exhausted" | Too many sensors | Reduce resolution, update rates, or use headless |

### Debug Commands

```bash
# Check Gazebo topics
gz topic -l

# Echo sensor data
gz topic -e -t /world/tokyo_occworld/model/drone_1/link/lidar_drone_link/sensor/gpu_lidar_drone/scan/points

# Check ROS 2 topics
ros2 topic list
ros2 topic hz /drone_1/lidar/points

# MAVLink inspection
mavproxy.py --master=udp:127.0.0.1:14550 --console
```

---

## 12. References

- **ArduPilot Gazebo Plugin**: https://github.com/ArduPilot/ardupilot_gazebo
- **ArduPilot SITL Documentation**: https://ardupilot.org/dev/docs/sitl-with-gazebo.html
- **Gazebo Harmonic Sensors**: https://gazebosim.org/docs/harmonic/sensors/
- **ROS 2 Gazebo Bridge**: https://github.com/gazebosim/ros_gz
- **MAVROS**: https://github.com/mavlink/mavros
- **OccWorld Paper**: https://arxiv.org/abs/2311.16038
- **BEVFusion**: https://github.com/mit-han-lab/bevfusion

---

## Appendix A: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ArduPilot Gazebo Quick Reference                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LAUNCH SEQUENCE:                                                    │
│  1. gz sim -v4 -r world.sdf                                         │
│  2. sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON        │
│  3. ros2 launch occworld_gazebo bridge.launch.py                    │
│                                                                      │
│  DEFAULT PORTS:                                                      │
│  ┌──────────┬────────────────┬────────────────┐                     │
│  │ Instance │ FDM In/Out     │ MAVLink        │                     │
│  ├──────────┼────────────────┼────────────────┤                     │
│  │ -I 0     │ 9002/9003      │ UDP:14550      │                     │
│  │ -I 1     │ 9012/9013      │ UDP:14560      │                     │
│  │ -I 2     │ 9022/9023      │ UDP:14570      │                     │
│  └──────────┴────────────────┴────────────────┘                     │
│                                                                      │
│  ENVIRONMENT VARIABLES:                                              │
│  GZ_VERSION=harmonic                                                │
│  GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build             │
│  GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:worlds          │
│                                                                      │
│  MAVPROXY COMMANDS:                                                  │
│  mode guided → arm throttle → takeoff 10                            │
│  mode land → disarm                                                  │
│  param set SIM_SPEEDUP 5                                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```
