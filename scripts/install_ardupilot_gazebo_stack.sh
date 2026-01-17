#!/bin/bash
# =============================================================================
# ArduPilot + Gazebo Harmonic + ROS 2 + MAVROS2 Installation Script
# For VeryLargeWeebModel Simulation Framework
# =============================================================================
# Usage: ./install_ardupilot_gazebo_stack.sh [--skip-ros] [--skip-ardupilot]
# Prerequisites: Ubuntu 22.04 (Jammy), sudo access

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_ROS=false
SKIP_ARDUPILOT=false
for arg in "$@"; do
    case $arg in
        --skip-ros) SKIP_ROS=true ;;
        --skip-ardupilot) SKIP_ARDUPILOT=true ;;
    esac
done

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Step 1: System Prerequisites
# =============================================================================
log_info "Step 1: Installing system prerequisites..."

sudo apt-get update
sudo apt-get install -y \
    curl \
    lsb-release \
    gnupg \
    wget \
    git \
    build-essential \
    cmake \
    python3-pip \
    python3-venv \
    software-properties-common

log_success "System prerequisites installed"

# =============================================================================
# Step 2: Install Gazebo Harmonic
# =============================================================================
log_info "Step 2: Installing Gazebo Harmonic..."

# Add OSRF repository
sudo curl https://packages.osrfoundation.org/gazebo.gpg \
    --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
    http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

sudo apt-get update
sudo apt-get install -y gz-harmonic

# Verify installation
GZ_VERSION=$(gz sim --version 2>/dev/null || echo "unknown")
log_success "Gazebo installed: $GZ_VERSION"

# Install Gazebo development libraries for plugin compilation
sudo apt-get install -y \
    libgz-sim8-dev \
    libgz-transport13-dev \
    libgz-sensors8-dev \
    libgz-rendering8-dev \
    libgz-msgs10-dev \
    rapidjson-dev

# GStreamer for camera streaming (optional but recommended)
sudo apt-get install -y \
    libopencv-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-gl

log_success "Gazebo Harmonic and dependencies installed"

# =============================================================================
# Step 3: Install ROS 2 Humble (if not skipped)
# =============================================================================
if [ "$SKIP_ROS" = false ]; then
    log_info "Step 3: Installing ROS 2 Humble..."

    # Check if ROS 2 is already installed
    if [ -f /opt/ros/humble/setup.bash ]; then
        log_warn "ROS 2 Humble already installed, skipping base installation"
    else
        # Add ROS 2 repository
        sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
            -o /usr/share/keyrings/ros-archive-keyring.gpg

        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
            http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | \
            sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

        sudo apt-get update
        sudo apt-get install -y ros-humble-desktop
    fi

    # Install ROS-Gazebo bridge packages
    log_info "Installing ROS-Gazebo bridge packages..."
    sudo apt-get install -y \
        ros-humble-ros-gz-sim \
        ros-humble-ros-gz-bridge \
        ros-humble-ros-gz-image \
        ros-humble-ros-gz-interfaces

    # Install MAVROS2
    log_info "Installing MAVROS2..."
    sudo apt-get install -y \
        ros-humble-mavros \
        ros-humble-mavros-extras \
        ros-humble-mavros-msgs

    # Install GeographicLib datasets for MAVROS
    log_info "Installing GeographicLib datasets..."
    sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh

    # Install additional ROS 2 packages for perception
    sudo apt-get install -y \
        ros-humble-cv-bridge \
        ros-humble-image-transport \
        ros-humble-message-filters \
        ros-humble-tf2-ros \
        ros-humble-tf2-geometry-msgs \
        ros-humble-pcl-ros \
        ros-humble-pcl-conversions

    log_success "ROS 2 Humble and MAVROS2 installed"
else
    log_warn "Skipping ROS 2 installation (--skip-ros flag set)"
fi

# =============================================================================
# Step 4: Clone and Build ArduPilot SITL (if not skipped)
# =============================================================================
if [ "$SKIP_ARDUPILOT" = false ]; then
    log_info "Step 4: Setting up ArduPilot SITL..."

    ARDUPILOT_DIR="$HOME/ardupilot"

    if [ -d "$ARDUPILOT_DIR" ]; then
        log_warn "ArduPilot directory exists. Pulling latest..."
        cd "$ARDUPILOT_DIR"
        git pull
        git submodule update --init --recursive
    else
        log_info "Cloning ArduPilot repository..."
        cd ~
        git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git
        cd ardupilot
    fi

    # Install ArduPilot prerequisites
    log_info "Installing ArduPilot prerequisites..."
    Tools/environment_install/install-prereqs-ubuntu.sh -y

    # Reload profile for new environment variables
    source ~/.profile 2>/dev/null || true

    # Build ArduCopter for drones
    log_info "Building ArduCopter SITL..."
    ./waf configure --board sitl
    ./waf copter

    # Build ArduRover for ground robots
    log_info "Building ArduRover SITL..."
    ./waf rover

    # Optionally build ArduPlane
    log_info "Building ArduPlane SITL..."
    ./waf plane

    log_success "ArduPilot SITL built (copter, rover, plane)"
else
    log_warn "Skipping ArduPilot installation (--skip-ardupilot flag set)"
fi

# =============================================================================
# Step 5: Clone and Build ardupilot_gazebo Plugin
# =============================================================================
log_info "Step 5: Building ardupilot_gazebo plugin..."

ARDUPILOT_GAZEBO_DIR="$HOME/ardupilot_gazebo"

if [ -d "$ARDUPILOT_GAZEBO_DIR" ]; then
    log_warn "ardupilot_gazebo directory exists. Pulling latest..."
    cd "$ARDUPILOT_GAZEBO_DIR"
    git pull
else
    log_info "Cloning ardupilot_gazebo repository..."
    cd ~
    git clone https://github.com/ArduPilot/ardupilot_gazebo.git
    cd ardupilot_gazebo
fi

# Set Gazebo version for build
export GZ_VERSION=harmonic

# Build the plugin
log_info "Building ardupilot_gazebo plugin..."
rm -rf build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc)

log_success "ardupilot_gazebo plugin built"

# =============================================================================
# Step 6: Create VeryLargeWeebModel ROS 2 Workspace
# =============================================================================
log_info "Step 6: Setting up VeryLargeWeebModel ROS 2 workspace..."

OCCWORLD_WS="$HOME/occworld_ws"
mkdir -p "$OCCWORLD_WS/src"
cd "$OCCWORLD_WS/src"

# Create package structure for occworld_gazebo
if [ ! -d "occworld_gazebo" ]; then
    log_info "Creating occworld_gazebo package..."
    source /opt/ros/humble/setup.bash
    ros2 pkg create --build-type ament_python occworld_gazebo \
        --dependencies rclpy sensor_msgs geometry_msgs nav_msgs cv_bridge
fi

# Create necessary directories
mkdir -p occworld_gazebo/config
mkdir -p occworld_gazebo/launch
mkdir -p occworld_gazebo/models
mkdir -p occworld_gazebo/worlds
mkdir -p occworld_gazebo/scripts

log_success "VeryLargeWeebModel workspace structure created at $OCCWORLD_WS"

# =============================================================================
# Step 7: Configure Environment Variables
# =============================================================================
log_info "Step 7: Configuring environment variables..."

# Create environment setup script
ENV_SETUP_FILE="$HOME/.occworld_env.sh"
cat > "$ENV_SETUP_FILE" << 'EOF'
# VeryLargeWeebModel + ArduPilot Gazebo Environment Setup
# Source this file: source ~/.occworld_env.sh

# Gazebo Harmonic configuration
export GZ_VERSION=harmonic
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}
export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH}

# Add VeryLargeWeebModel workspace resources
export GZ_SIM_RESOURCE_PATH=$HOME/occworld_ws/src/occworld_gazebo/models:$HOME/occworld_ws/src/occworld_gazebo/worlds:${GZ_SIM_RESOURCE_PATH}

# ArduPilot paths
export PATH=$HOME/ardupilot/Tools/autotest:$PATH

# ROS 2 setup
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

# VeryLargeWeebModel workspace setup
if [ -f $HOME/occworld_ws/install/setup.bash ]; then
    source $HOME/occworld_ws/install/setup.bash
fi

# Python path for custom modules
export PYTHONPATH=$HOME/occworld_ws/src:${PYTHONPATH}
EOF

# Add to .bashrc if not already present
if ! grep -q "occworld_env.sh" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# VeryLargeWeebModel Environment" >> ~/.bashrc
    echo "source ~/.occworld_env.sh" >> ~/.bashrc
    log_info "Added environment setup to ~/.bashrc"
fi

# Source the environment now
source "$ENV_SETUP_FILE"

log_success "Environment variables configured"

# =============================================================================
# Step 8: Install Python Dependencies
# =============================================================================
log_info "Step 8: Installing Python dependencies..."

pip3 install --user \
    pymavlink \
    dronekit \
    numpy \
    opencv-python \
    scipy \
    tqdm \
    open3d \
    pyyaml

log_success "Python dependencies installed"

# =============================================================================
# Step 9: Verification
# =============================================================================
log_info "Step 9: Verifying installation..."

echo ""
echo "=============================================="
echo "         Installation Verification           "
echo "=============================================="

# Check Gazebo
echo -n "Gazebo Harmonic: "
if command -v gz &> /dev/null; then
    gz sim --version 2>/dev/null || echo "installed (version check failed)"
else
    echo "NOT FOUND"
fi

# Check ROS 2
echo -n "ROS 2 Humble: "
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
    ros2 --version 2>/dev/null || echo "installed"
else
    echo "NOT FOUND"
fi

# Check ArduPilot
echo -n "ArduPilot SITL: "
if [ -f "$HOME/ardupilot/build/sitl/bin/arducopter" ]; then
    echo "FOUND (arducopter)"
else
    echo "NOT FOUND"
fi

# Check ardupilot_gazebo
echo -n "ardupilot_gazebo plugin: "
if [ -f "$HOME/ardupilot_gazebo/build/libArduPilotPlugin.so" ]; then
    echo "FOUND"
else
    echo "NOT FOUND"
fi

# Check MAVROS
echo -n "MAVROS2: "
if ros2 pkg list 2>/dev/null | grep -q mavros; then
    echo "FOUND"
else
    echo "NOT FOUND"
fi

echo "=============================================="
echo ""

# =============================================================================
# Final Instructions
# =============================================================================
log_success "Installation complete!"

echo ""
echo "=============================================="
echo "          Next Steps                         "
echo "=============================================="
echo ""
echo "1. Reload your shell environment:"
echo "   source ~/.bashrc"
echo ""
echo "2. Build the VeryLargeWeebModel ROS 2 workspace:"
echo "   cd ~/occworld_ws && colcon build"
echo ""
echo "3. Test Gazebo with ArduPilot plugin:"
echo "   # Terminal 1: Launch Gazebo"
echo "   gz sim -v4 -r iris_runway.sdf"
echo ""
echo "   # Terminal 2: Launch SITL"
echo "   cd ~/ardupilot"
echo "   sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console"
echo ""
echo "4. Test ROS 2 bridge:"
echo "   # Terminal 3:"
echo "   ros2 run ros_gz_bridge parameter_bridge /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"
echo ""
echo "=============================================="
