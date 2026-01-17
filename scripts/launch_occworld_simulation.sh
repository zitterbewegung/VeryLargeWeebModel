#!/bin/bash
# =============================================================================
# VeryLargeWeebModel Simulation End-to-End Launch Script
# =============================================================================
#
# This script launches the complete simulation stack:
# 1. Gazebo world with sensors
# 2. Multiple ArduPilot SITL instances (drone + rover)
# 3. ROS 2 bridge for sensor data
# 4. MAVROS2 for vehicle control
# 5. Data recorder for VeryLargeWeebModel training
#
# Usage:
#   ./launch_occworld_simulation.sh [OPTIONS]
#
# Options:
#   --world <name>      World file to load (default: tokyo_occworld)
#   --drones <n>        Number of drones (default: 1)
#   --rovers <n>        Number of rovers (default: 1)
#   --headless          Run Gazebo in headless mode
#   --record            Enable data recording
#   --fast              Use fast physics settings
#   --help              Show this help
#
# Environment Variables (set in ~/.occworld_env.sh):
#   GZ_SIM_SYSTEM_PLUGIN_PATH - Path to ArduPilot Gazebo plugins
#   GZ_SIM_RESOURCE_PATH - Path to models and worlds
#
# Port Convention:
#   Instance 0 (Drone 1):  FDM 9002/9003,  MAVLink 14550
#   Instance 1 (Rover 1):  FDM 9012/9013,  MAVLink 14560
#   Instance 2 (Drone 2):  FDM 9022/9023,  MAVLink 14570
#   Instance 3 (Rover 2):  FDM 9032/9033,  MAVLink 14580
# =============================================================================

set -e

# Default configuration
WORLD_NAME="tokyo_occworld"
NUM_DRONES=1
NUM_ROVERS=1
HEADLESS=false
RECORD=false
FAST_PHYSICS=false
VERBOSE=4

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs/$(date +%Y%m%d_%H%M%S)"
ARDUPILOT_DIR="${ARDUPILOT_DIR:-$HOME/ardupilot}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Process tracking
declare -a PIDS=()
declare -a PROCESS_NAMES=()

# =============================================================================
# Helper Functions
# =============================================================================

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

cleanup() {
    log_info "Shutting down simulation stack..."

    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        name=${PROCESS_NAMES[$i]}
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Stopping $name (PID: $pid)"
            kill -SIGTERM "$pid" 2>/dev/null || true
        fi
    done

    # Wait for graceful shutdown
    sleep 2

    # Force kill remaining
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -SIGKILL "$pid" 2>/dev/null || true
        fi
    done

    log_success "Cleanup complete"
}

trap cleanup EXIT INT TERM

add_process() {
    PIDS+=("$1")
    PROCESS_NAMES+=("$2")
}

wait_for_port() {
    local port=$1
    local timeout=${2:-30}
    local start_time=$(date +%s)

    while ! nc -z localhost "$port" 2>/dev/null; do
        local current_time=$(date +%s)
        if (( current_time - start_time > timeout )); then
            return 1
        fi
        sleep 0.5
    done
    return 0
}

show_help() {
    head -50 "$0" | grep "^#" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --world)
            WORLD_NAME="$2"
            shift 2
            ;;
        --drones)
            NUM_DRONES="$2"
            shift 2
            ;;
        --rovers)
            NUM_ROVERS="$2"
            shift 2
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        --record)
            RECORD=true
            shift
            ;;
        --fast)
            FAST_PHYSICS=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Environment Setup
# =============================================================================

log_info "Setting up environment..."

# Source environment
if [ -f ~/.occworld_env.sh ]; then
    source ~/.occworld_env.sh
else
    log_warn "~/.occworld_env.sh not found, using defaults"
    export GZ_VERSION=harmonic
    export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}
    export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH}
fi

# Add project paths
export GZ_SIM_RESOURCE_PATH="$PROJECT_DIR/models:$PROJECT_DIR/worlds:${GZ_SIM_RESOURCE_PATH}"

# ROS 2 setup
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

if [ -f ~/occworld_ws/install/setup.bash ]; then
    source ~/occworld_ws/install/setup.bash
fi

# Create log directory
mkdir -p "$LOG_DIR"

log_info "Configuration:"
log_info "  World: $WORLD_NAME"
log_info "  Drones: $NUM_DRONES"
log_info "  Rovers: $NUM_ROVERS"
log_info "  Headless: $HEADLESS"
log_info "  Recording: $RECORD"
log_info "  Log directory: $LOG_DIR"

# =============================================================================
# Step 1: Launch Gazebo
# =============================================================================

log_info "Step 1: Launching Gazebo..."

GZ_ARGS="-v${VERBOSE} -r"

if [ "$HEADLESS" = true ]; then
    GZ_ARGS="$GZ_ARGS --headless-rendering -s"
fi

WORLD_FILE="$PROJECT_DIR/worlds/${WORLD_NAME}.sdf"
if [ ! -f "$WORLD_FILE" ]; then
    WORLD_FILE="${WORLD_NAME}.sdf"  # Try built-in world
fi

gz sim $GZ_ARGS "$WORLD_FILE" > "$LOG_DIR/gazebo.log" 2>&1 &
add_process $! "Gazebo"

# Wait for Gazebo to initialize
log_info "Waiting for Gazebo to initialize..."
sleep 5

if ! kill -0 "${PIDS[-1]}" 2>/dev/null; then
    log_error "Gazebo failed to start. Check $LOG_DIR/gazebo.log"
    exit 1
fi

log_success "Gazebo started"

# =============================================================================
# Step 2: Launch ArduPilot SITL Instances
# =============================================================================

log_info "Step 2: Launching ArduPilot SITL instances..."

INSTANCE_ID=0

# Launch drones
for (( i=1; i<=NUM_DRONES; i++ )); do
    DRONE_NAME="drone_$i"
    FDM_PORT_IN=$((9002 + INSTANCE_ID * 10))
    FDM_PORT_OUT=$((9003 + INSTANCE_ID * 10))
    MAVLINK_PORT=$((14550 + INSTANCE_ID * 10))

    log_info "Starting $DRONE_NAME (Instance $INSTANCE_ID, MAVLink: $MAVLINK_PORT)"

    cd "$ARDUPILOT_DIR"

    # Note: --model JSON tells SITL to use JSON interface with Gazebo
    # -I sets instance ID for port calculation
    python3 Tools/autotest/sim_vehicle.py \
        -v ArduCopter \
        -f gazebo-iris \
        --model JSON \
        -I "$INSTANCE_ID" \
        --no-mavproxy \
        --out "udp:127.0.0.1:$MAVLINK_PORT" \
        > "$LOG_DIR/${DRONE_NAME}_sitl.log" 2>&1 &

    add_process $! "SITL-$DRONE_NAME"

    INSTANCE_ID=$((INSTANCE_ID + 1))
    sleep 3
done

# Launch rovers
for (( i=1; i<=NUM_ROVERS; i++ )); do
    ROVER_NAME="rover_$i"
    FDM_PORT_IN=$((9002 + INSTANCE_ID * 10))
    FDM_PORT_OUT=$((9003 + INSTANCE_ID * 10))
    MAVLINK_PORT=$((14550 + INSTANCE_ID * 10))

    log_info "Starting $ROVER_NAME (Instance $INSTANCE_ID, MAVLink: $MAVLINK_PORT)"

    cd "$ARDUPILOT_DIR"

    python3 Tools/autotest/sim_vehicle.py \
        -v Rover \
        -f gazebo-rover \
        --model JSON \
        -I "$INSTANCE_ID" \
        --no-mavproxy \
        --out "udp:127.0.0.1:$MAVLINK_PORT" \
        > "$LOG_DIR/${ROVER_NAME}_sitl.log" 2>&1 &

    add_process $! "SITL-$ROVER_NAME"

    INSTANCE_ID=$((INSTANCE_ID + 1))
    sleep 3
done

log_success "SITL instances started"

# =============================================================================
# Step 3: Launch ROS 2 Bridge
# =============================================================================

log_info "Step 3: Launching ROS 2 bridge..."

# Create bridge configuration dynamically
BRIDGE_CONFIG="$LOG_DIR/bridge_config.yaml"
cat > "$BRIDGE_CONFIG" << 'EOF'
# Auto-generated bridge configuration
---
# Clock (essential for time sync)
- ros_topic_name: "/clock"
  gz_topic_name: "/clock"
  ros_type_name: "rosgraph_msgs/msg/Clock"
  gz_type_name: "gz.msgs.Clock"
  direction: GZ_TO_ROS
EOF

# Add drone topics
for (( i=1; i<=NUM_DRONES; i++ )); do
    DRONE_NAME="drone_$i"
    cat >> "$BRIDGE_CONFIG" << EOF

# $DRONE_NAME sensors
- ros_topic_name: "/${DRONE_NAME}/lidar/points"
  gz_topic_name: "/world/${WORLD_NAME}/model/${DRONE_NAME}/link/lidar_drone_link/sensor/gpu_lidar_drone/scan/points"
  ros_type_name: "sensor_msgs/msg/PointCloud2"
  gz_type_name: "gz.msgs.PointCloudPacked"
  direction: GZ_TO_ROS

- ros_topic_name: "/${DRONE_NAME}/imu/data"
  gz_topic_name: "/world/${WORLD_NAME}/model/${DRONE_NAME}/link/imu_link/sensor/imu_sensor/imu"
  ros_type_name: "sensor_msgs/msg/Imu"
  gz_type_name: "gz.msgs.IMU"
  direction: GZ_TO_ROS

- ros_topic_name: "/${DRONE_NAME}/gps/fix"
  gz_topic_name: "/world/${WORLD_NAME}/model/${DRONE_NAME}/link/gps_link/sensor/navsat_sensor/navsat"
  ros_type_name: "sensor_msgs/msg/NavSatFix"
  gz_type_name: "gz.msgs.NavSat"
  direction: GZ_TO_ROS

- ros_topic_name: "/${DRONE_NAME}/odom"
  gz_topic_name: "/model/${DRONE_NAME}/odometry"
  ros_type_name: "nav_msgs/msg/Odometry"
  gz_type_name: "gz.msgs.Odometry"
  direction: GZ_TO_ROS
EOF
done

# Add rover topics
for (( i=1; i<=NUM_ROVERS; i++ )); do
    ROVER_NAME="rover_$i"
    cat >> "$BRIDGE_CONFIG" << EOF

# $ROVER_NAME sensors
- ros_topic_name: "/${ROVER_NAME}/lidar/points"
  gz_topic_name: "/world/${WORLD_NAME}/model/${ROVER_NAME}/link/lidar_ground_link/sensor/gpu_lidar_ground/scan/points"
  ros_type_name: "sensor_msgs/msg/PointCloud2"
  gz_type_name: "gz.msgs.PointCloudPacked"
  direction: GZ_TO_ROS

- ros_topic_name: "/${ROVER_NAME}/imu/data"
  gz_topic_name: "/world/${WORLD_NAME}/model/${ROVER_NAME}/link/imu_link/sensor/imu_sensor/imu"
  ros_type_name: "sensor_msgs/msg/Imu"
  gz_type_name: "gz.msgs.IMU"
  direction: GZ_TO_ROS

- ros_topic_name: "/${ROVER_NAME}/odom"
  gz_topic_name: "/model/${ROVER_NAME}/odometry"
  ros_type_name: "nav_msgs/msg/Odometry"
  gz_type_name: "gz.msgs.Odometry"
  direction: GZ_TO_ROS
EOF
done

# Launch parameter bridge
ros2 run ros_gz_bridge parameter_bridge \
    --ros-args -p config_file:="$BRIDGE_CONFIG" \
    > "$LOG_DIR/ros_gz_bridge.log" 2>&1 &
add_process $! "ROS-GZ-Bridge"

# Launch image bridges for cameras (separate processes for performance)
CAMERAS=("front" "front_left" "front_right" "back" "back_left" "back_right")
for (( i=1; i<=NUM_DRONES; i++ )); do
    DRONE_NAME="drone_$i"
    for cam in "${CAMERAS[@]}"; do
        ros2 run ros_gz_image image_bridge \
            "/world/${WORLD_NAME}/model/${DRONE_NAME}/link/camera_${cam}_link/sensor/camera_${cam}/image" \
            > "$LOG_DIR/${DRONE_NAME}_camera_${cam}.log" 2>&1 &
        add_process $! "Image-Bridge-${DRONE_NAME}-${cam}"
    done
done

log_success "ROS 2 bridge started"

# =============================================================================
# Step 4: Launch MAVROS2 Nodes
# =============================================================================

log_info "Step 4: Launching MAVROS2 nodes..."

INSTANCE_ID=0

# MAVROS for drones
for (( i=1; i<=NUM_DRONES; i++ )); do
    DRONE_NAME="drone_$i"
    MAVLINK_PORT=$((14550 + INSTANCE_ID * 10))

    ros2 run mavros mavros_node \
        --ros-args \
        -r __ns:=/${DRONE_NAME}/mavros \
        -p fcu_url:="udp://:$MAVLINK_PORT@127.0.0.1:$((MAVLINK_PORT + 5))" \
        -p gcs_url:="" \
        -p target_system_id:=$((INSTANCE_ID + 1)) \
        -p target_component_id:=1 \
        > "$LOG_DIR/${DRONE_NAME}_mavros.log" 2>&1 &
    add_process $! "MAVROS-$DRONE_NAME"

    INSTANCE_ID=$((INSTANCE_ID + 1))
    sleep 2
done

# MAVROS for rovers
for (( i=1; i<=NUM_ROVERS; i++ )); do
    ROVER_NAME="rover_$i"
    MAVLINK_PORT=$((14550 + INSTANCE_ID * 10))

    ros2 run mavros mavros_node \
        --ros-args \
        -r __ns:=/${ROVER_NAME}/mavros \
        -p fcu_url:="udp://:$MAVLINK_PORT@127.0.0.1:$((MAVLINK_PORT + 5))" \
        -p gcs_url:="" \
        -p target_system_id:=$((INSTANCE_ID + 1)) \
        -p target_component_id:=1 \
        > "$LOG_DIR/${ROVER_NAME}_mavros.log" 2>&1 &
    add_process $! "MAVROS-$ROVER_NAME"

    INSTANCE_ID=$((INSTANCE_ID + 1))
    sleep 2
done

log_success "MAVROS2 nodes started"

# =============================================================================
# Step 5: Launch Data Recorder (if enabled)
# =============================================================================

if [ "$RECORD" = true ]; then
    log_info "Step 5: Launching data recorder..."

    DATA_DIR="$PROJECT_DIR/data/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$DATA_DIR"

    # Launch recorder for each drone
    for (( i=1; i<=NUM_DRONES; i++ )); do
        DRONE_NAME="drone_$i"

        ros2 run occworld_gazebo occworld_data_recorder \
            --ros-args \
            -p output_dir:="$DATA_DIR" \
            -p agent_type:=drone \
            -p agent_id:=$i \
            -p record_rate:=2.0 \
            > "$LOG_DIR/${DRONE_NAME}_recorder.log" 2>&1 &
        add_process $! "Recorder-$DRONE_NAME"
    done

    # Launch recorder for each rover
    for (( i=1; i<=NUM_ROVERS; i++ )); do
        ROVER_NAME="rover_$i"

        ros2 run occworld_gazebo occworld_data_recorder \
            --ros-args \
            -p output_dir:="$DATA_DIR" \
            -p agent_type:=rover \
            -p agent_id:=$i \
            -p record_rate:=2.0 \
            > "$LOG_DIR/${ROVER_NAME}_recorder.log" 2>&1 &
        add_process $! "Recorder-$ROVER_NAME"
    done

    log_success "Data recorder started, saving to: $DATA_DIR"
fi

# =============================================================================
# Monitoring Loop
# =============================================================================

log_success "Simulation stack launched successfully!"
echo ""
echo "=============================================="
echo "           Active Processes                  "
echo "=============================================="
for i in "${!PIDS[@]}"; do
    printf "  %-25s PID: %s\n" "${PROCESS_NAMES[$i]}" "${PIDS[$i]}"
done
echo "=============================================="
echo ""
log_info "Press Ctrl+C to stop the simulation"
echo ""

# Keep script running and monitor processes
while true; do
    sleep 10

    # Check if critical processes are still running
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        name=${PROCESS_NAMES[$i]}

        if ! kill -0 "$pid" 2>/dev/null; then
            log_warn "$name (PID: $pid) has stopped"
            # Remove from tracking
            unset PIDS[$i]
            unset PROCESS_NAMES[$i]
        fi
    done

    # Exit if Gazebo died
    if ! kill -0 "${PIDS[0]}" 2>/dev/null; then
        log_error "Gazebo has stopped. Exiting..."
        exit 1
    fi
done
