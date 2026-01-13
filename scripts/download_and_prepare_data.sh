#!/bin/bash
# =============================================================================
# OccWorld Data Download and Preparation Script
# =============================================================================
# Downloads:
#   - Tokyo PLATEAU 3D city models (for Gazebo simulation)
#   - OccWorld pretrained models (VQVAE + World Model)
#   - BEVFusion pretrained models
#   - nuScenes metadata pickle files
#
# Usage:
#   ./scripts/download_and_prepare_data.sh [OPTIONS]
#
# Options:
#   --all           Download everything (default)
#   --plateau       Download Tokyo PLATEAU 3D models only
#   --models        Download pretrained models only
#   --nuscenes      Download nuScenes pickle files only
#   --skip-plateau  Skip PLATEAU download (large files)
#   --skip-convert  Skip mesh conversion step
#   --output DIR    Output directory (default: ./data)
#   --help          Show this help message
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging functions
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()    { echo -e "${CYAN}[STEP]${NC} $1"; }

# Default settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/data"

# Download flags
DOWNLOAD_ALL=true
DOWNLOAD_PLATEAU=false
DOWNLOAD_MODELS=false
DOWNLOAD_NUSCENES=false
SKIP_PLATEAU=false
SKIP_CONVERT=false

# =============================================================================
# Parse Arguments
# =============================================================================
show_help() {
    head -25 "$0" | tail -20
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)          DOWNLOAD_ALL=true; shift ;;
        --plateau)      DOWNLOAD_ALL=false; DOWNLOAD_PLATEAU=true; shift ;;
        --models)       DOWNLOAD_ALL=false; DOWNLOAD_MODELS=true; shift ;;
        --nuscenes)     DOWNLOAD_ALL=false; DOWNLOAD_NUSCENES=true; shift ;;
        --skip-plateau) SKIP_PLATEAU=true; shift ;;
        --skip-convert) SKIP_CONVERT=true; shift ;;
        --output)       OUTPUT_DIR="$2"; shift 2 ;;
        --help|-h)      show_help ;;
        *)              log_error "Unknown option: $1"; show_help ;;
    esac
done

# Set download flags based on --all
if [ "$DOWNLOAD_ALL" = true ]; then
    DOWNLOAD_PLATEAU=true
    DOWNLOAD_MODELS=true
    DOWNLOAD_NUSCENES=true
fi

# =============================================================================
# Setup Directories
# =============================================================================
log_step "Setting up directories..."

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/plateau"
mkdir -p "${OUTPUT_DIR}/plateau/raw"
mkdir -p "${OUTPUT_DIR}/plateau/meshes"
mkdir -p "${OUTPUT_DIR}/plateau/gazebo_models"
mkdir -p "${OUTPUT_DIR}/tokyo_gazebo"
mkdir -p "${PROJECT_ROOT}/pretrained/vqvae"
mkdir -p "${PROJECT_ROOT}/pretrained/occworld"
mkdir -p "${PROJECT_ROOT}/pretrained/bevfusion"

log_success "Directories created"

# =============================================================================
# Download Functions
# =============================================================================

download_file() {
    local url="$1"
    local output="$2"
    local description="$3"

    if [ -f "$output" ]; then
        log_warn "$description already exists, skipping..."
        return 0
    fi

    log_info "Downloading $description..."
    log_info "  URL: $url"
    log_info "  Output: $output"

    # Try wget first, fall back to curl
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$output" "$url" || {
            log_error "Failed to download $description"
            rm -f "$output"
            return 1
        }
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$output" "$url" || {
            log_error "Failed to download $description"
            rm -f "$output"
            return 1
        }
    else
        log_error "Neither wget nor curl found. Please install one."
        return 1
    fi

    log_success "Downloaded $description"
}

# =============================================================================
# Download Tokyo PLATEAU 3D Models
# =============================================================================
download_plateau() {
    log_step "Downloading Tokyo PLATEAU 3D city models..."

    # ==========================================================================
    # ATTRIBUTION NOTICE
    # ==========================================================================
    # Tokyo PLATEAU 3D city model data is provided by:
    #   Ministry of Land, Infrastructure, Transport and Tourism (MLIT), Japan
    #   Project PLATEAU - https://www.mlit.go.jp/plateau/
    #
    # License: CC BY 4.0 (Commercial use allowed with attribution)
    # Source: https://www.geospatial.jp/ckan/dataset/plateau-tokyo23ku
    #
    # Required Attribution:
    #   "3D city model data provided by Ministry of Land, Infrastructure,
    #    Transport and Tourism (MLIT), Japan - Project PLATEAU.
    #    Licensed under CC BY 4.0."
    # ==========================================================================

    log_info "Data provided by MLIT Japan - Project PLATEAU (CC BY 4.0)"

    local PLATEAU_BASE="https://gic-plateau.s3.ap-northeast-1.amazonaws.com/2020"

    # OBJ format - best for Gazebo/Blender conversion (~2.1 GB)
    download_file \
        "${PLATEAU_BASE}/13100_tokyo23-ku_2020_obj_3_op.zip" \
        "${OUTPUT_DIR}/plateau/raw/tokyo23ku_obj.zip" \
        "Tokyo PLATEAU OBJ models"

    # FBX format - alternative format (~2.8 GB)
    # Uncomment if needed:
    # download_file \
    #     "${PLATEAU_BASE}/13100_tokyo23-ku_2020_fbx_3_op.zip" \
    #     "${OUTPUT_DIR}/plateau/raw/tokyo23ku_fbx.zip" \
    #     "Tokyo PLATEAU FBX models"

    # 3D Tiles + GeoJSON (for visualization/reference)
    download_file \
        "${PLATEAU_BASE}/13100_tokyo23ku_2020_3Dtiles_etc_1_op.zip" \
        "${OUTPUT_DIR}/plateau/raw/tokyo23ku_3dtiles.zip" \
        "Tokyo PLATEAU 3D Tiles"

    # CityGML (semantic data - useful for ground truth labels)
    download_file \
        "https://assets.cms.plateau.reearth.io/assets/ec/d51c64-a47f-4a56-aa64-340d1d3c720b/13100_tokyo23-ku_2020_citygml_4_2_op.zip" \
        "${OUTPUT_DIR}/plateau/raw/tokyo23ku_citygml.zip" \
        "Tokyo PLATEAU CityGML"

    log_success "PLATEAU downloads complete"
}

# =============================================================================
# Download Pretrained Models
# =============================================================================
download_pretrained_models() {
    log_step "Downloading pretrained models..."

    # -------------------------------------------------------------------------
    # OccWorld Models (Tsinghua Cloud)
    # -------------------------------------------------------------------------
    log_info "Downloading OccWorld models from Tsinghua Cloud..."

    # Note: Tsinghua Cloud links may require manual download
    # These are placeholder URLs - check the actual links from:
    # https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/

    local TSINGHUA_BASE="https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5"

    cat << 'EOF' > "${PROJECT_ROOT}/pretrained/DOWNLOAD_INSTRUCTIONS.md"
# Manual Download Required

Tsinghua Cloud requires manual download. Please:

1. Visit: https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/
2. Download the following files:
   - vqvae_epoch_125.pth -> pretrained/vqvae/epoch_125.pth
   - occworld_latest.pth -> pretrained/occworld/latest.pth

3. Visit: https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/
4. Download pickle files:
   - nuscenes_infos_train_temporal_v3_scene.pkl -> data/
   - nuscenes_infos_val_temporal_v3_scene.pkl -> data/

Alternative: Use gdown for Google Drive mirrors if available.
EOF

    log_warn "OccWorld models require manual download from Tsinghua Cloud"
    log_warn "See: ${PROJECT_ROOT}/pretrained/DOWNLOAD_INSTRUCTIONS.md"

    # -------------------------------------------------------------------------
    # BEVFusion Models
    # -------------------------------------------------------------------------
    log_info "Setting up BEVFusion model download..."

    if [ -d "${PROJECT_ROOT}/bevfusion" ]; then
        log_warn "BEVFusion directory exists, skipping clone..."
    else
        log_info "Cloning BEVFusion repository..."
        git clone https://github.com/mit-han-lab/bevfusion.git "${PROJECT_ROOT}/bevfusion" || {
            log_warn "BEVFusion clone failed (repo may be archived)"
        }
    fi

    # Create download script for BEVFusion
    cat << 'EOF' > "${PROJECT_ROOT}/pretrained/bevfusion/download.sh"
#!/bin/bash
# BEVFusion model download script
# Run this manually if automatic download fails

PRETRAINED_DIR="$(dirname "$0")"

# Detection model (Camera + LiDAR)
wget -O "${PRETRAINED_DIR}/bevfusion-det.pth" \
    "https://www.dropbox.com/s/xxx/bevfusion-det.pth?dl=1"

# Segmentation model (Camera + LiDAR)
wget -O "${PRETRAINED_DIR}/bevfusion-seg.pth" \
    "https://www.dropbox.com/s/xxx/bevfusion-seg.pth?dl=1"

echo "Download complete. Check files in ${PRETRAINED_DIR}"
EOF
    chmod +x "${PROJECT_ROOT}/pretrained/bevfusion/download.sh"

    log_success "Pretrained model setup complete (some manual downloads required)"
}

# =============================================================================
# Download nuScenes Metadata
# =============================================================================
download_nuscenes_metadata() {
    log_step "Downloading nuScenes metadata pickle files..."

    # These files are required for OccWorld training with nuScenes
    # They contain preprocessed scene information

    log_warn "nuScenes pickle files require manual download from Tsinghua Cloud"
    log_info "Visit: https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/"
    log_info "Download to: ${OUTPUT_DIR}/"

    # Create placeholder info file
    cat << EOF > "${OUTPUT_DIR}/NUSCENES_DOWNLOAD.md"
# nuScenes Data Download

## Required Files

Download from: https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/

1. nuscenes_infos_train_temporal_v3_scene.pkl
2. nuscenes_infos_val_temporal_v3_scene.pkl

Place in: ${OUTPUT_DIR}/

## Full nuScenes Dataset (Optional)

For training from scratch on nuScenes:
1. Register at https://www.nuscenes.org/
2. Download Full dataset (v1.0)
3. Download lidarseg add-on
4. Extract to ${OUTPUT_DIR}/nuscenes/

## Occ3D Ground Truth (Optional)

For occupancy supervision:
1. Visit: https://github.com/Tsinghua-MARS-Lab/Occ3D
2. Follow their download instructions
3. Extract to ${OUTPUT_DIR}/gts/
EOF

    log_success "nuScenes download instructions created"
}

# =============================================================================
# Extract and Convert PLATEAU Data
# =============================================================================
extract_plateau() {
    log_step "Extracting PLATEAU data..."

    local RAW_DIR="${OUTPUT_DIR}/plateau/raw"
    local MESH_DIR="${OUTPUT_DIR}/plateau/meshes"

    # Extract OBJ files
    if [ -f "${RAW_DIR}/tokyo23ku_obj.zip" ]; then
        log_info "Extracting OBJ models..."
        unzip -q -o "${RAW_DIR}/tokyo23ku_obj.zip" -d "${MESH_DIR}/obj/" || {
            log_warn "Extraction failed or already extracted"
        }
        log_success "OBJ models extracted"
    fi

    # Extract 3D Tiles
    if [ -f "${RAW_DIR}/tokyo23ku_3dtiles.zip" ]; then
        log_info "Extracting 3D Tiles..."
        unzip -q -o "${RAW_DIR}/tokyo23ku_3dtiles.zip" -d "${MESH_DIR}/3dtiles/" || {
            log_warn "Extraction failed or already extracted"
        }
        log_success "3D Tiles extracted"
    fi

    log_success "PLATEAU extraction complete"
}

# =============================================================================
# Convert PLATEAU to Gazebo SDF Format
# =============================================================================
convert_to_gazebo() {
    log_step "Converting PLATEAU models to Gazebo SDF format..."

    # Create Python conversion script
    cat << 'PYTHON_SCRIPT' > "${OUTPUT_DIR}/plateau/convert_to_sdf.py"
#!/usr/bin/env python3
"""
Convert PLATEAU OBJ models to Gazebo SDF format.

This script:
1. Reads OBJ mesh files from PLATEAU dataset
2. Simplifies meshes for real-time simulation
3. Generates Gazebo-compatible SDF model files
4. Creates collision meshes

Usage:
    python convert_to_sdf.py --input meshes/obj/ --output gazebo_models/
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

try:
    import trimesh
    import numpy as np
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("Warning: trimesh not installed. Install with: pip install trimesh")


def create_sdf_model(name: str, mesh_path: str, output_dir: str,
                     scale: float = 1.0, simplify_ratio: float = 0.1):
    """Create a Gazebo SDF model from an OBJ mesh."""

    model_dir = os.path.join(output_dir, name)
    meshes_dir = os.path.join(model_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)

    # Copy/convert mesh
    if HAS_TRIMESH:
        try:
            mesh = trimesh.load(mesh_path)

            # Simplify mesh for performance
            if hasattr(mesh, 'simplify_quadric_decimation'):
                target_faces = int(len(mesh.faces) * simplify_ratio)
                if target_faces > 100:
                    mesh = mesh.simplify_quadric_decimation(target_faces)

            # Export as DAE (Collada) for Gazebo
            output_mesh = os.path.join(meshes_dir, f"{name}.dae")
            mesh.export(output_mesh)
            mesh_file = f"meshes/{name}.dae"

            # Get bounding box for collision
            bounds = mesh.bounds
            size = bounds[1] - bounds[0]
            center = (bounds[0] + bounds[1]) / 2

        except Exception as e:
            print(f"  Error processing {mesh_path}: {e}")
            return None
    else:
        # Just copy the OBJ file
        shutil.copy(mesh_path, os.path.join(meshes_dir, f"{name}.obj"))
        mesh_file = f"meshes/{name}.obj"
        size = [10, 10, 30]  # Default building size
        center = [0, 0, 15]

    # Create model.config
    config_content = f"""<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author>
    <name>PLATEAU/OccWorld</name>
    <email>auto-generated</email>
  </author>
  <description>
    Tokyo PLATEAU building model converted for Gazebo simulation.
    Original data from MLIT Project PLATEAU.
  </description>
</model>
"""

    with open(os.path.join(model_dir, "model.config"), "w") as f:
        f.write(config_content)

    # Create model.sdf
    sdf_content = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <mesh>
            <uri>model://{name}/{mesh_file}</uri>
            <scale>{scale} {scale} {scale}</scale>
          </mesh>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>model://{name}/{mesh_file}</uri>
            <scale>{scale} {scale} {scale}</scale>
          </mesh>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.7 0.7 0.7 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""

    with open(os.path.join(model_dir, "model.sdf"), "w") as f:
        f.write(sdf_content)

    return model_dir


def process_plateau_directory(input_dir: str, output_dir: str,
                              max_models: int = 100, simplify: float = 0.1):
    """Process all OBJ files in PLATEAU directory."""

    input_path = Path(input_dir)
    obj_files = list(input_path.rglob("*.obj"))

    print(f"Found {len(obj_files)} OBJ files")

    if max_models > 0:
        obj_files = obj_files[:max_models]
        print(f"Processing first {max_models} models")

    created = 0
    for i, obj_file in enumerate(obj_files):
        name = obj_file.stem.replace(" ", "_").replace("-", "_")
        name = f"plateau_{name}"

        print(f"[{i+1}/{len(obj_files)}] Converting: {name}")

        result = create_sdf_model(
            name=name,
            mesh_path=str(obj_file),
            output_dir=output_dir,
            simplify_ratio=simplify
        )

        if result:
            created += 1

    print(f"\nCreated {created} Gazebo models in {output_dir}")
    return created


def create_world_file(models_dir: str, output_file: str,
                      world_name: str = "tokyo_plateau"):
    """Create a Gazebo world file including all models."""

    model_dirs = [d for d in os.listdir(models_dir)
                  if os.path.isdir(os.path.join(models_dir, d))]

    # Generate model includes with grid positioning
    includes = []
    grid_size = int(np.ceil(np.sqrt(len(model_dirs))))
    spacing = 50  # meters between models

    for i, model_name in enumerate(model_dirs):
        x = (i % grid_size) * spacing
        y = (i // grid_size) * spacing

        includes.append(f"""
    <include>
      <uri>model://{model_name}</uri>
      <name>{model_name}</name>
      <pose>{x} {y} 0 0 0 0</pose>
    </include>""")

    world_content = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="{world_name}">

    <!-- Physics -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>

    <!-- Plugins -->
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <!-- Coordinates (Tokyo) -->
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <latitude_deg>35.6762</latitude_deg>
      <longitude_deg>139.6503</longitude_deg>
      <elevation>40</elevation>
    </spherical_coordinates>

    <!-- Lighting -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 100 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane><normal>0 0 1</normal><size>2000 2000</size></plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane><normal>0 0 1</normal><size>2000 2000</size></plane>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
          </material>
        </visual>
      </link>
    </model>

    <!-- PLATEAU Buildings -->
    {"".join(includes)}

  </world>
</sdf>
"""

    with open(output_file, "w") as f:
        f.write(world_content)

    print(f"Created world file: {output_file}")
    print(f"Included {len(model_dirs)} building models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PLATEAU to Gazebo SDF")
    parser.add_argument("--input", "-i", required=True, help="Input OBJ directory")
    parser.add_argument("--output", "-o", required=True, help="Output model directory")
    parser.add_argument("--max-models", "-m", type=int, default=50,
                        help="Max models to convert (0=all)")
    parser.add_argument("--simplify", "-s", type=float, default=0.1,
                        help="Mesh simplification ratio (0.1 = 10% of faces)")
    parser.add_argument("--world", "-w", help="Generate world file")
    args = parser.parse_args()

    # Convert models
    process_plateau_directory(
        args.input,
        args.output,
        max_models=args.max_models,
        simplify=args.simplify
    )

    # Generate world file
    if args.world:
        create_world_file(args.output, args.world)
PYTHON_SCRIPT

    chmod +x "${OUTPUT_DIR}/plateau/convert_to_sdf.py"

    # Check if we should run conversion
    if [ "$SKIP_CONVERT" = true ]; then
        log_warn "Skipping mesh conversion (--skip-convert)"
        return 0
    fi

    # Check for trimesh
    if ! python3 -c "import trimesh" 2>/dev/null; then
        log_warn "trimesh not installed. Installing..."
        pip3 install --user trimesh numpy || {
            log_warn "Could not install trimesh. Manual conversion required."
            log_info "Run: pip install trimesh && python ${OUTPUT_DIR}/plateau/convert_to_sdf.py --help"
            return 0
        }
    fi

    # Run conversion if OBJ files exist
    local OBJ_DIR="${OUTPUT_DIR}/plateau/meshes/obj"
    local GAZEBO_DIR="${OUTPUT_DIR}/plateau/gazebo_models"
    local WORLD_FILE="${PROJECT_ROOT}/worlds/tokyo_plateau.sdf"

    if [ -d "$OBJ_DIR" ]; then
        log_info "Converting PLATEAU models to Gazebo format..."
        python3 "${OUTPUT_DIR}/plateau/convert_to_sdf.py" \
            --input "$OBJ_DIR" \
            --output "$GAZEBO_DIR" \
            --max-models 50 \
            --simplify 0.1 \
            --world "$WORLD_FILE" || {
            log_warn "Conversion completed with warnings"
        }
        log_success "Gazebo models created in $GAZEBO_DIR"
    else
        log_warn "OBJ directory not found. Extract PLATEAU data first."
    fi
}

# =============================================================================
# Create Simulation Data Directory Structure
# =============================================================================
setup_simulation_data_dirs() {
    log_step "Setting up simulation data directories..."

    # Create standard OccWorld data structure
    local DATA_DIR="${OUTPUT_DIR}/tokyo_gazebo"

    mkdir -p "${DATA_DIR}/train"
    mkdir -p "${DATA_DIR}/val"
    mkdir -p "${DATA_DIR}/test"

    # Create info file
    cat << EOF > "${DATA_DIR}/README.md"
# Tokyo Gazebo Dataset

This directory contains training data generated from Gazebo simulation.

## Directory Structure

\`\`\`
tokyo_gazebo/
├── drone_YYYYMMDD_HHMMSS/      # Drone recording sessions
│   ├── images/                  # 6-camera surround images
│   │   ├── 000001_CAM_FRONT.jpg
│   │   ├── 000001_CAM_FRONT_LEFT.jpg
│   │   └── ...
│   ├── lidar/                   # LiDAR point clouds
│   │   └── 000001_LIDAR.npy
│   ├── poses/                   # 6-DoF poses
│   │   └── 000001.json
│   └── occupancy/               # Ground truth occupancy
│       └── 000001_occupancy.npz
└── rover_YYYYMMDD_HHMMSS/       # Rover recording sessions
    └── ...
\`\`\`

## Generating Data

\`\`\`bash
# Launch simulation with recording
./scripts/launch_occworld_simulation.sh --record --headless

# Run data collection mission
python3 scripts/data_collection_mission.py --vehicle drone --pattern survey

# Generate occupancy ground truth
python3 scripts/depth_occupancy_processor.py --input . --output .
\`\`\`

## Using with OccWorld Training

\`\`\`python
from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset, DatasetConfig

config = DatasetConfig(
    history_frames=4,
    future_frames=6,
    agent_type='both',
)
dataset = GazeboOccWorldDataset('data/tokyo_gazebo', config)
\`\`\`
EOF

    log_success "Simulation data directories ready"
}

# =============================================================================
# Summary
# =============================================================================
print_summary() {
    echo ""
    echo "=============================================================================="
    echo "                        Download and Setup Complete                           "
    echo "=============================================================================="
    echo ""
    echo "Directory Structure:"
    echo "  ${OUTPUT_DIR}/"
    echo "  ├── plateau/                  # Tokyo 3D city models"
    echo "  │   ├── raw/                  # Downloaded archives"
    echo "  │   ├── meshes/               # Extracted mesh files"
    echo "  │   └── gazebo_models/        # Converted Gazebo SDF models"
    echo "  ├── tokyo_gazebo/             # Simulation training data (to be generated)"
    echo "  └── nuscenes/                 # nuScenes data (if downloaded)"
    echo ""
    echo "  ${PROJECT_ROOT}/pretrained/"
    echo "  ├── vqvae/                    # VQVAE checkpoint"
    echo "  ├── occworld/                 # OccWorld checkpoint"
    echo "  └── bevfusion/                # BEVFusion checkpoints"
    echo ""
    echo "=============================================================================="
    echo "                              Next Steps                                      "
    echo "=============================================================================="
    echo ""
    echo "1. MANUAL DOWNLOADS REQUIRED:"
    echo "   - OccWorld models: https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/"
    echo "   - Pickle files:    https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/"
    echo "   See: ${PROJECT_ROOT}/pretrained/DOWNLOAD_INSTRUCTIONS.md"
    echo ""
    echo "2. Configure Gazebo environment:"
    echo "   export GZ_SIM_RESOURCE_PATH=${OUTPUT_DIR}/plateau/gazebo_models:\${GZ_SIM_RESOURCE_PATH}"
    echo ""
    echo "3. Generate training data:"
    echo "   ./scripts/launch_occworld_simulation.sh --record --headless"
    echo "   python3 scripts/data_collection_mission.py --vehicle drone --pattern survey"
    echo ""
    echo "4. Start training:"
    echo "   python train.py --py-config config/finetune_tokyo.py --work-dir out/occworld_tokyo"
    echo ""
    echo "=============================================================================="
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    echo ""
    echo "=============================================================================="
    echo "           OccWorld Data Download and Preparation Script                      "
    echo "=============================================================================="
    echo ""
    log_info "Output directory: ${OUTPUT_DIR}"
    log_info "Project root: ${PROJECT_ROOT}"
    echo ""

    # Check dependencies
    log_step "Checking dependencies..."
    for cmd in wget unzip python3; do
        if ! command -v $cmd &> /dev/null; then
            log_warn "$cmd not found, some features may not work"
        fi
    done

    # Run selected downloads
    if [ "$DOWNLOAD_PLATEAU" = true ] && [ "$SKIP_PLATEAU" = false ]; then
        download_plateau
        extract_plateau
        convert_to_gazebo
    elif [ "$SKIP_PLATEAU" = true ]; then
        log_warn "Skipping PLATEAU download (--skip-plateau)"
    fi

    if [ "$DOWNLOAD_MODELS" = true ]; then
        download_pretrained_models
    fi

    if [ "$DOWNLOAD_NUSCENES" = true ]; then
        download_nuscenes_metadata
    fi

    # Setup directories
    setup_simulation_data_dirs

    # Print summary
    print_summary
}

# Run main
main "$@"
