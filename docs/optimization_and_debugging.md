# Optimization and Debugging Guide

This document summarizes physics/sensor tweaks for faster simulation and provides troubleshooting commands and fixes for the ArduPilot Gazebo integration.

---

## 1. Physics Optimization

### 1.1 Standard vs Fast Physics Settings

```xml
<!-- STANDARD: Accurate physics (1ms step) -->
<physics name="standard" type="ode">
  <max_step_size>0.001</max_step_size>     <!-- 1ms steps -->
  <real_time_factor>1.0</real_time_factor>  <!-- Real-time -->
  <real_time_update_rate>1000</real_time_update_rate>
</physics>

<!-- FAST: Training mode (4ms step, unlimited speed) -->
<physics name="fast" type="ode">
  <max_step_size>0.004</max_step_size>     <!-- 4ms steps (4x faster) -->
  <real_time_factor>0</real_time_factor>   <!-- Run as fast as possible -->
  <real_time_update_rate>0</real_time_update_rate>

  <ode>
    <solver>
      <type>quick</type>      <!-- Faster solver -->
      <iters>50</iters>       <!-- Reduced iterations -->
      <sor>1.3</sor>          <!-- Successive over-relaxation -->
    </solver>
    <constraints>
      <cfm>0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>

<!-- ULTRA FAST: Data generation only (may be unstable) -->
<physics name="ultra_fast" type="ode">
  <max_step_size>0.01</max_step_size>      <!-- 10ms steps -->
  <real_time_factor>0</real_time_factor>
</physics>
```

### 1.2 Physics Trade-offs

| Setting | `max_step_size` | Speed | Stability | Use Case |
|---------|-----------------|-------|-----------|----------|
| Accurate | 0.001 | 1x | Excellent | HITL, demos |
| Fast | 0.004 | 3-4x | Good | SITL training |
| Ultra | 0.01 | 8-10x | Fair | Bulk data gen |

**Warning:** Step sizes >4ms may cause:
- Vehicle oscillation
- Collision detection failures
- IMU data inaccuracy

---

## 2. Sensor Rate Optimization

### 2.1 Standard vs Reduced Sensor Rates

```xml
<!-- STANDARD: Full quality -->
<sensor name="camera_full" type="camera">
  <update_rate>10</update_rate>
  <camera>
    <image>
      <width>1600</width>
      <height>900</height>
    </image>
  </camera>
</sensor>

<sensor name="lidar_full" type="gpu_lidar">
  <update_rate>10</update_rate>
  <lidar>
    <scan>
      <horizontal><samples>1800</samples></horizontal>
      <vertical><samples>64</samples></vertical>
    </scan>
  </lidar>
</sensor>

<!-- FAST: Reduced resolution/rate -->
<sensor name="camera_fast" type="camera">
  <update_rate>5</update_rate>           <!-- 5Hz instead of 10Hz -->
  <camera>
    <image>
      <width>800</width>                 <!-- 50% resolution -->
      <height>450</height>
    </image>
  </camera>
</sensor>

<sensor name="lidar_fast" type="gpu_lidar">
  <update_rate>5</update_rate>
  <lidar>
    <scan>
      <horizontal><samples>900</samples></horizontal>   <!-- 50% points -->
      <vertical><samples>32</samples></vertical>
    </scan>
  </lidar>
</sensor>
```

### 2.2 Sensor Impact Analysis

| Sensor | Full Config | Fast Config | GPU Load Reduction |
|--------|-------------|-------------|-------------------|
| Camera (1600x900) | 10 Hz | 5 Hz, 800x450 | ~75% |
| GPU LiDAR (115K pts) | 10 Hz | 5 Hz, 29K pts | ~87% |
| IMU | 200 Hz | 100 Hz | ~50% |
| GPS | 10 Hz | 5 Hz | ~50% |

---

## 3. Headless Rendering

### 3.1 Launch Commands

```bash
# GUI mode (for visualization)
gz sim -v4 -r world.sdf

# Server only (no GUI, no rendering)
gz sim -v1 -s -r world.sdf

# Headless with rendering (for sensors)
gz sim -v1 --headless-rendering -r world.sdf

# Specify rendering engine
GZ_SIM_RENDER_ENGINE=ogre2 \
gz sim -v1 --headless-rendering -r world.sdf
```

### 3.2 Environment Variables for Rendering

```bash
# Force software rendering (useful on servers without GPU)
export LIBGL_ALWAYS_SOFTWARE=1

# Specify Ogre2 render engine
export GZ_SIM_RENDER_ENGINE_PATH=/usr/lib/x86_64-linux-gnu/gz-rendering-8/engine-plugins

# Disable GUI components
export QT_QPA_PLATFORM=offscreen
```

---

## 4. Memory Optimization

### 4.1 Reducing Memory Footprint

```bash
# Limit Gazebo rendering buffer
export OGRE_RENDER_SYSTEM=OpenGL 3+

# Reduce simulation verbosity
gz sim -v0 -r world.sdf   # Minimal logging

# Use compressed textures in models
# Edit material files to use DXT compression
```

### 4.2 Multi-Vehicle Memory Tips

- Share meshes across vehicle instances using `<uri>model://...</uri>`
- Use lower-poly models for non-visible vehicles
- Disable visualization for sensors: `<visualize>false</visualize>`
- Limit number of concurrent LiDAR sensors (GPU memory intensive)

---

## 5. Troubleshooting Guide

### 5.1 Common Issues Quick Reference

| Issue | Cause | Solution |
|-------|-------|----------|
| ArduPilot not connecting | Port mismatch | Verify `fdm_port_in/out` match SITL ports |
| No sensor data on ROS topics | Bridge not configured | Check bridge YAML paths, rebuild bridge |
| Simulation too slow | Physics too accurate | Increase `max_step_size` |
| Vehicle flying erratically | PID tuning | Adjust `p_gain` in ArduPilotPlugin |
| LiDAR not working | GPU rendering issue | Ensure `ogre2` renderer, check OpenGL |
| Memory exhausted | Too many sensors | Reduce resolution, rates, or use headless |
| MAVROS not receiving data | Wrong FCU URL | Check UDP port matches SITL output |
| Gazebo crashes on start | Plugin path wrong | Verify `GZ_SIM_SYSTEM_PLUGIN_PATH` |

### 5.2 Debug Commands

#### Gazebo Inspection

```bash
# List all Gazebo topics
gz topic -l

# Echo specific sensor topic
gz topic -e -t /world/tokyo_occworld/model/drone_1/link/lidar_drone_link/sensor/gpu_lidar_drone/scan/points

# Get topic info (message type, publishers)
gz topic -i -t /world/tokyo_occworld/clock

# Check model states
gz model -m drone_1 --list

# View running systems/plugins
gz service -l
```

#### ROS 2 Inspection

```bash
# List ROS 2 topics
ros2 topic list

# Check topic frequency
ros2 topic hz /drone_1/lidar/points

# Echo topic data
ros2 topic echo /drone_1/imu/data --once

# Check node status
ros2 node list
ros2 node info /mavros_drone

# View TF tree
ros2 run tf2_tools view_frames
```

#### MAVLink Inspection

```bash
# Connect MavProxy for debugging
mavproxy.py --master=udp:127.0.0.1:14550 --console

# Inside MavProxy:
# > status            # Show vehicle status
# > mode guided       # Change mode
# > arm throttle      # Arm vehicle
# > param show *      # List parameters
# > module load graph # Visual mode

# Direct pymavlink inspection
python3 -c "
from pymavlink import mavutil
m = mavutil.mavlink_connection('udp:127.0.0.1:14550')
m.wait_heartbeat()
print(f'Connected: sys={m.target_system}, comp={m.target_component}')
"
```

#### Network Port Debugging

```bash
# Check if ports are in use
netstat -tulpn | grep -E '(9002|9003|14550)'

# Test UDP connectivity
nc -u -l 9002   # Listen on port 9002
nc -u localhost 9002  # Send to port 9002

# Check for port conflicts
lsof -i :14550
```

### 5.3 Specific Issue Fixes

#### Issue: ArduPilot Plugin Not Connecting

```bash
# 1. Verify SITL is running and listening
ps aux | grep ardupilot

# 2. Check SITL output for JSON interface
# Should see: "Waiting for JSON input on port 9002"

# 3. Verify plugin ports match
# In model.sdf:
#   <fdm_port_in>9002</fdm_port_in>
#   <fdm_port_out>9003</fdm_port_out>

# 4. Check network interface
# SITL uses 127.0.0.1 by default, plugin must use same
```

#### Issue: No Bridge Topic Data

```bash
# 1. Check Gazebo is publishing
gz topic -l | grep sensor

# 2. Verify bridge config paths are correct
# Full path format:
# /world/{world_name}/model/{model_name}/link/{link_name}/sensor/{sensor_name}/...

# 3. Rebuild bridge node
colcon build --packages-select ros_gz_bridge

# 4. Check message type compatibility
# gz.msgs.PointCloudPacked -> sensor_msgs/msg/PointCloud2
```

#### Issue: Vehicle Unstable Flight

```xml
<!-- In ArduPilotPlugin, adjust PID gains -->
<control channel="0">
  <p_gain>0.20</p_gain>   <!-- Increase for more responsive -->
  <i_gain>0.01</i_gain>   <!-- Add if steady-state error -->
  <d_gain>0.05</d_gain>   <!-- Add if oscillating -->
</control>

<!-- Or use velocity slowdown -->
<controlVelocitySlowdownSim>1</controlVelocitySlowdownSim>
```

#### Issue: GPU LiDAR Not Rendering

```bash
# 1. Check GPU availability
nvidia-smi

# 2. Verify Ogre2 renderer
export GZ_SIM_RENDER_ENGINE=ogre2

# 3. Check OpenGL version
glxinfo | grep "OpenGL version"
# Needs OpenGL 3.3+

# 4. Test with basic GPU sensor
# Create minimal world with just LiDAR to isolate issue

# 5. Check Gazebo sensor system is loaded
# In world.sdf, verify:
<plugin filename="gz-sim-sensors-system"
    name="gz::sim::systems::Sensors">
  <render_engine>ogre2</render_engine>
</plugin>
```

### 5.4 Performance Profiling

```bash
# Gazebo built-in stats
gz stats -w tokyo_occworld

# Monitor CPU/GPU
htop
nvidia-smi -l 1

# ROS 2 performance
ros2 run ros2_tracing trace_launch

# Check message latencies
ros2 topic delay /drone_1/lidar/points
```

---

## 6. Quick Reference Card

```
╔═══════════════════════════════════════════════════════════════════╗
║              ArduPilot Gazebo Optimization Cheat Sheet            ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  SPEED OPTIMIZATION:                                               ║
║  • max_step_size: 0.001→0.004 = 3-4x faster                       ║
║  • real_time_factor: 0 = unlimited speed                          ║
║  • sensor rates: 10Hz→5Hz = 50% GPU reduction                     ║
║  • headless: gz sim -s = no GUI overhead                          ║
║                                                                    ║
║  STABILITY OPTIMIZATION:                                           ║
║  • max_step_size ≤0.004 for stable flight                         ║
║  • quick solver with 50+ iterations                               ║
║  • PID tuning in ArduPilotPlugin                                  ║
║                                                                    ║
║  DEBUG COMMANDS:                                                   ║
║  • gz topic -l              List Gazebo topics                    ║
║  • ros2 topic hz <topic>    Check publish rate                    ║
║  • mavproxy.py --console    MAVLink debug                         ║
║  • netstat -tulpn           Check port usage                      ║
║                                                                    ║
║  PORT CONVENTION:                                                  ║
║  ┌──────────┬────────────────┬────────────────┐                   ║
║  │ Instance │ FDM In/Out     │ MAVLink        │                   ║
║  ├──────────┼────────────────┼────────────────┤                   ║
║  │ -I 0     │ 9002/9003      │ UDP:14550      │                   ║
║  │ -I 1     │ 9012/9013      │ UDP:14560      │                   ║
║  │ -I 2     │ 9022/9023      │ UDP:14570      │                   ║
║  └──────────┴────────────────┴────────────────┘                   ║
║                                                                    ║
║  ENVIRONMENT SETUP:                                                ║
║  source ~/.occworld_env.sh                                        ║
║  GZ_VERSION=harmonic                                               ║
║  GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build           ║
║                                                                    ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## 7. Logging and Diagnostics

### 7.1 Enable Verbose Logging

```bash
# Gazebo verbose levels: 0-4
gz sim -v4 -r world.sdf 2>&1 | tee gazebo_debug.log

# ArduPilot SITL debug
sim_vehicle.py -v ArduCopter -f gazebo-iris --debug

# ROS 2 debug logging
ros2 run ros_gz_bridge parameter_bridge \
    --ros-args --log-level debug

# MAVROS verbose
ros2 run mavros mavros_node \
    --ros-args --log-level debug
```

### 7.2 Log Analysis Patterns

```bash
# Find ArduPilot connection issues
grep -i "json\|fdm\|connection" gazebo_debug.log

# Find sensor issues
grep -i "sensor\|lidar\|camera" gazebo_debug.log

# Find ROS bridge issues
grep -i "bridge\|topic\|subscriber" ros_gz_bridge.log

# Timestamp correlation
# All logs should use simulation time for matching
```

---

## 8. Performance Benchmarks

### 8.1 Expected Performance (RTX 3080, i9-10900K)

| Configuration | Sim Speed | GPU Load | RAM |
|---------------|-----------|----------|-----|
| 1 drone, full sensors | 1x real-time | 60% | 4GB |
| 1 drone, reduced sensors | 3x real-time | 30% | 2GB |
| 4 drones, full sensors | 0.5x real-time | 95% | 12GB |
| 4 drones, reduced sensors | 1.5x real-time | 70% | 6GB |
| Headless, 1 drone | 5x real-time | 40% | 2GB |

### 8.2 Bottleneck Identification

```bash
# If GPU-bound:
nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1
# → Reduce sensor resolution/rates

# If CPU-bound:
top -H -p $(pgrep -f "gz sim")
# → Increase physics step size, use quick solver

# If memory-bound:
free -h
# → Reduce number of vehicles, close unused sensors
```
