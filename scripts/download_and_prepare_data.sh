#!/bin/bash
# =============================================================================
# VeryLargeWeebModel Data Download and Preparation Script
# =============================================================================
# Downloads and prepares:
#   - Tokyo PLATEAU 3D city models (for Gazebo simulation)
#   - Pretrained models (VQVAE + World Model)
#   - BEVFusion pretrained models
#   - nuScenes metadata pickle files
#   - Generates training data from PLATEAU meshes
#
# Usage:
#   ./scripts/download_and_prepare_data.sh [OPTIONS]
#
# Options:
#   --all            Download everything + generate training data (default)
#   --plateau        Download Tokyo PLATEAU 3D models only
#   --models         Download pretrained models only
#   --nuscenes       Download nuScenes pickle files only
#   --generate-data  Generate training data from PLATEAU meshes
#   --skip-plateau   Skip PLATEAU download (large files)
#   --skip-convert   Skip mesh conversion step
#   --output DIR     Output directory (default: ./data)
#   --help           Show this help message
#
# Examples:
#   ./scripts/download_and_prepare_data.sh --all
#   ./scripts/download_and_prepare_data.sh --models --generate-data
#   ./scripts/download_and_prepare_data.sh --plateau --generate-data
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
GENERATE_DATA=false

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
        --generate-data) GENERATE_DATA=true; shift ;;
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
    GENERATE_DATA=true
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
# Download Functions (Fast parallel downloads with aria2c/axel fallback)
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

    # Create output directory if needed
    mkdir -p "$(dirname "$output")"

    # Try aria2c first (fastest - 16 parallel connections)
    if command -v aria2c &> /dev/null; then
        log_info "  Using aria2c (16 connections)..."
        if aria2c -x 16 -s 16 --file-allocation=none -o "$output" "$url"; then
            log_success "Downloaded $description (aria2c)"
            return 0
        else
            log_warn "aria2c failed, trying fallback..."
            rm -f "$output"
        fi
    fi

    # Try axel (also fast - 16 parallel connections)
    if command -v axel &> /dev/null; then
        log_info "  Using axel (16 connections)..."
        if axel -n 16 -o "$output" "$url"; then
            log_success "Downloaded $description (axel)"
            return 0
        else
            log_warn "axel failed, trying fallback..."
            rm -f "$output"
        fi
    fi

    # Fallback to curl (single connection, but supports resume)
    if command -v curl &> /dev/null; then
        log_info "  Using curl..."
        if curl -L -C - -o "$output" "$url"; then
            log_success "Downloaded $description (curl)"
            return 0
        else
            rm -f "$output"
        fi
    fi

    # Final fallback to wget
    if command -v wget &> /dev/null; then
        log_info "  Using wget..."
        if wget -c -O "$output" "$url"; then
            log_success "Downloaded $description (wget)"
            return 0
        else
            rm -f "$output"
        fi
    fi

    log_error "Failed to download $description"
    return 1
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
    # VeryLargeWeebModel Models (Tsinghua Cloud)
    # -------------------------------------------------------------------------
    log_info "Downloading VeryLargeWeebModel models from Tsinghua Cloud..."

    # VeryLargeWeebModel pretrained models from Tsinghua Cloud
    # Direct download URL discovered via Seafile API
    # Source: https://github.com/wzzheng/VeryLargeWeebModel

    local OCCWORLD_FILE="${PROJECT_ROOT}/pretrained/occworld/latest.pth"
    local TSINGHUA_URL="https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/"
    local DOWNLOAD_URL="https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/files/?p=/latest.pth&dl=1"

    # Note: The share only contains latest.pth (721MB) which includes both VeryLargeWeebModel and VQVAE weights
    # There is no separate epoch_125.pth file in the official share

    # Check if model already exists
    if [ -f "$OCCWORLD_FILE" ]; then
        local occworld_size=$(stat -f%z "$OCCWORLD_FILE" 2>/dev/null || stat -c%s "$OCCWORLD_FILE" 2>/dev/null || echo 0)

        if [ "$occworld_size" -gt 100000000 ]; then  # > 100MB
            log_success "VeryLargeWeebModel checkpoint already downloaded! ($(numfmt --to=iec $occworld_size 2>/dev/null || echo "${occworld_size} bytes"))"
            return 0
        else
            log_warn "Existing file seems incomplete, re-downloading..."
            rm -f "$OCCWORLD_FILE"
        fi
    fi

    # Download VeryLargeWeebModel checkpoint (includes VQVAE weights)
    log_info "Downloading VeryLargeWeebModel checkpoint (~721MB)..."
    log_info "  URL: ${DOWNLOAD_URL}"

    mkdir -p "$(dirname "$OCCWORLD_FILE")"

    if curl -L --progress-bar -o "$OCCWORLD_FILE" "$DOWNLOAD_URL" 2>&1; then
        local downloaded_size=$(stat -f%z "$OCCWORLD_FILE" 2>/dev/null || stat -c%s "$OCCWORLD_FILE" 2>/dev/null || echo 0)

        if [ "$downloaded_size" -gt 100000000 ]; then  # > 100MB
            log_success "VeryLargeWeebModel checkpoint downloaded! ($(numfmt --to=iec $downloaded_size 2>/dev/null || echo "${downloaded_size} bytes"))"
        else
            log_error "Download failed or file is too small (${downloaded_size} bytes)"
            log_info "Try manual download: curl -L -o ${OCCWORLD_FILE} '${DOWNLOAD_URL}'"
            rm -f "$OCCWORLD_FILE"
        fi
    elif wget -q --show-progress -O "$OCCWORLD_FILE" "$DOWNLOAD_URL" 2>&1; then
        log_success "VeryLargeWeebModel checkpoint downloaded!"
    else
        log_error "Download failed. Try manually:"
        log_info "  curl -L -o ${OCCWORLD_FILE} '${DOWNLOAD_URL}'"
        rm -f "$OCCWORLD_FILE"
    fi

    # Create a helper script to verify downloads
    cat << 'SCRIPT' > "${PROJECT_ROOT}/pretrained/verify_downloads.sh"
#!/bin/bash
# Verify VeryLargeWeebModel pretrained model download

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCCWORLD="${SCRIPT_DIR}/occworld/latest.pth"

echo "Checking pretrained models..."
echo ""

if [ -f "$OCCWORLD" ]; then
    size=$(stat -f%z "$OCCWORLD" 2>/dev/null || stat -c%s "$OCCWORLD" 2>/dev/null || echo 0)
    size_mb=$((size / 1024 / 1024))

    if [ "$size" -gt 700000000 ]; then  # > 700MB
        echo "✓ VeryLargeWeebModel checkpoint: OK (${size_mb}MB)"
        echo ""
        echo "Pretrained model ready! You can now run training."
    elif [ "$size" -gt 100000000 ]; then  # > 100MB
        echo "⚠ VeryLargeWeebModel checkpoint: ${size_mb}MB (expected ~721MB, may be incomplete)"
    else
        echo "✗ VeryLargeWeebModel checkpoint: File too small (${size_mb}MB), download failed"
    fi
else
    echo "✗ VeryLargeWeebModel checkpoint: NOT FOUND"
    echo ""
    echo "Download with:"
    echo "  curl -L -o pretrained/occworld/latest.pth \\"
    echo "    'https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/files/?p=/latest.pth&dl=1'"
fi
SCRIPT
    chmod +x "${PROJECT_ROOT}/pretrained/verify_downloads.sh"

    # Create instructions file
    cat << 'EOF' > "${PROJECT_ROOT}/pretrained/DOWNLOAD_INSTRUCTIONS.md"
# VeryLargeWeebModel Pretrained Model Download

## Direct Download (Recommended)

```bash
mkdir -p pretrained/occworld

curl -L -o pretrained/occworld/latest.pth \
  "https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/files/?p=/latest.pth&dl=1"
```

This downloads `latest.pth` (~721MB) which contains both VeryLargeWeebModel and VQVAE weights.

## Verify Download

```bash
./pretrained/verify_downloads.sh

# Or manually:
ls -lh pretrained/occworld/latest.pth   # Should be ~721MB
```

## Alternative: Browser Download

1. Visit: https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/
2. Download `latest.pth`
3. Move to `pretrained/occworld/latest.pth`

## Training Without Pretrained Models

If download fails, you can train from scratch (slower):

```bash
# First train VQVAE
python train.py --py-config config/train_vqvae.py --work-dir out/vqvae

# Then train VeryLargeWeebModel (update vqvae path in config first)
python train.py --py-config config/finetune_tokyo.py --work-dir out/occworld
```
EOF

    log_info "Verify download with: ./pretrained/verify_downloads.sh"

    # -------------------------------------------------------------------------
    # BEVFusion Models (Optional)
    # -------------------------------------------------------------------------
    log_info "Setting up BEVFusion model download..."

    mkdir -p "${PROJECT_ROOT}/pretrained/bevfusion"

    # BEVFusion direct download links from MIT HAN Lab
    local BEVFUSION_DET_URL="https://bevfusion.mit.edu/files/pretrained/bevfusion-det.pth"
    local BEVFUSION_SEG_URL="https://bevfusion.mit.edu/files/pretrained/bevfusion-seg.pth"

    cat << EOF > "${PROJECT_ROOT}/pretrained/bevfusion/download.sh"
#!/bin/bash
# BEVFusion model download script

PRETRAINED_DIR="\$(dirname "\$0")"

echo "Downloading BEVFusion detection model..."
wget -c -O "\${PRETRAINED_DIR}/bevfusion-det.pth" \\
    "${BEVFUSION_DET_URL}" || \\
    curl -L -o "\${PRETRAINED_DIR}/bevfusion-det.pth" "${BEVFUSION_DET_URL}"

echo "Downloading BEVFusion segmentation model..."
wget -c -O "\${PRETRAINED_DIR}/bevfusion-seg.pth" \\
    "${BEVFUSION_SEG_URL}" || \\
    curl -L -o "\${PRETRAINED_DIR}/bevfusion-seg.pth" "${BEVFUSION_SEG_URL}"

echo "Download complete!"
ls -lh "\${PRETRAINED_DIR}"/*.pth
EOF
    chmod +x "${PROJECT_ROOT}/pretrained/bevfusion/download.sh"

    log_success "Pretrained model setup complete"
}

# =============================================================================
# Download nuScenes Metadata
# =============================================================================
download_nuscenes_metadata() {
    log_step "Downloading nuScenes metadata pickle files..."

    # Direct download links from Tsinghua Cloud
    # These are preprocessed scene information files for VeryLargeWeebModel training
    local TRAIN_PKL_URL="https://cloud.tsinghua.edu.cn/f/a05c25067a864e0eb7d0/?dl=1"
    local VAL_PKL_URL="https://cloud.tsinghua.edu.cn/f/8c8f1e9b5f4a47a3b7c2/?dl=1"

    local TRAIN_PKL="${OUTPUT_DIR}/nuscenes_infos_train_temporal_v3_scene.pkl"
    local VAL_PKL="${OUTPUT_DIR}/nuscenes_infos_val_temporal_v3_scene.pkl"

    # Download training pickle
    if [ -f "$TRAIN_PKL" ]; then
        log_warn "Training pickle already exists, skipping..."
    else
        log_info "Downloading nuScenes training pickle (~100MB)..."
        if wget -q --show-progress -O "$TRAIN_PKL" "$TRAIN_PKL_URL" 2>/dev/null; then
            log_success "Training pickle downloaded"
        elif curl -L --progress-bar -o "$TRAIN_PKL" "$TRAIN_PKL_URL" 2>/dev/null; then
            log_success "Training pickle downloaded"
        else
            log_warn "Auto-download failed for training pickle"
            rm -f "$TRAIN_PKL"
        fi
    fi

    # Download validation pickle
    if [ -f "$VAL_PKL" ]; then
        log_warn "Validation pickle already exists, skipping..."
    else
        log_info "Downloading nuScenes validation pickle (~30MB)..."
        if wget -q --show-progress -O "$VAL_PKL" "$VAL_PKL_URL" 2>/dev/null; then
            log_success "Validation pickle downloaded"
        elif curl -L --progress-bar -o "$VAL_PKL" "$VAL_PKL_URL" 2>/dev/null; then
            log_success "Validation pickle downloaded"
        else
            log_warn "Auto-download failed for validation pickle"
            rm -f "$VAL_PKL"
        fi
    fi

    # Create info file with manual download instructions
    cat << EOF > "${OUTPUT_DIR}/NUSCENES_DOWNLOAD.md"
# nuScenes Data Download

## Pickle Files (Auto-downloaded)

Direct download commands if auto-download failed:
\`\`\`bash
# Training pickle
wget -O data/nuscenes_infos_train_temporal_v3_scene.pkl \\
    "https://cloud.tsinghua.edu.cn/f/a05c25067a864e0eb7d0/?dl=1"

# Validation pickle
wget -O data/nuscenes_infos_val_temporal_v3_scene.pkl \\
    "https://cloud.tsinghua.edu.cn/f/8c8f1e9b5f4a47a3b7c2/?dl=1"
\`\`\`

Or visit: https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/

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

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


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

            # Simplify mesh for performance using pyvista if available
            if HAS_PYVISTA and hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                try:
                    # Convert to pyvista for simplification
                    faces_pv = np.hstack([[3] + list(f) for f in mesh.faces])
                    pv_mesh = pv.PolyData(mesh.vertices, faces_pv)
                    # Decimate to target reduction (0.9 = remove 90% of faces)
                    target_reduction = 1.0 - simplify_ratio  # simplify_ratio=0.1 means keep 10%
                    if target_reduction > 0 and target_reduction < 1:
                        pv_mesh = pv_mesh.decimate(target_reduction)
                        # Convert back to trimesh
                        faces_out = pv_mesh.faces.reshape(-1, 4)[:, 1:4]
                        mesh = trimesh.Trimesh(vertices=pv_mesh.points, faces=faces_out)
                except Exception as e:
                    print(f"  Warning: simplification failed, using original mesh: {e}")
            elif hasattr(mesh, 'simplify_quadric_decimation'):
                # Fallback to trimesh simplification
                try:
                    target_faces = max(100, int(len(mesh.faces) * simplify_ratio))
                    mesh = mesh.simplify_quadric_decimation(target_faces)
                except Exception as e:
                    print(f"  Warning: simplification failed, using original mesh: {e}")

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
    <name>PLATEAU/VeryLargeWeebModel</name>
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

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    model_dirs = [d for d in os.listdir(models_dir)
                  if os.path.isdir(os.path.join(models_dir, d))]

    if not model_dirs:
        print(f"Warning: No models found in {models_dir}, skipping world file creation")
        return

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
# Generate Training Data from PLATEAU Meshes
# =============================================================================
generate_training_data() {
    log_step "Generating VeryLargeWeebModel training data from PLATEAU meshes..."

    local MESH_DIR="${OUTPUT_DIR}/plateau/meshes/obj"
    local TOKYO_DATA="${OUTPUT_DIR}/tokyo_gazebo"
    local CONVERTER="${SCRIPT_DIR}/plateau_to_occworld.py"

    # Check if converter script exists
    if [ ! -f "$CONVERTER" ]; then
        log_warn "Converter script not found: $CONVERTER"
        log_info "Falling back to dummy data generator..."

        if [ -f "${SCRIPT_DIR}/create_dummy_data.py" ]; then
            log_info "Generating dummy training data..."
            python3 "${SCRIPT_DIR}/create_dummy_data.py" \
                --output "$TOKYO_DATA" \
                --frames 100 \
                --sessions 3
            log_success "Dummy training data generated"
        fi
        return 0
    fi

    # Check if meshes exist
    if [ -d "$MESH_DIR" ] && [ "$(ls -A $MESH_DIR 2>/dev/null)" ]; then
        log_info "Found PLATEAU meshes, generating training data..."

        # Install trimesh if needed
        python3 -c "import trimesh" 2>/dev/null || {
            log_info "Installing trimesh..."
            pip3 install trimesh numpy
        }

        # Generate training data
        python3 "$CONVERTER" \
            --input "$MESH_DIR" \
            --output "$TOKYO_DATA" \
            --frames 200 \
            --sessions 5 \
            --pattern survey \
            --max-meshes 30 || {
            log_warn "PLATEAU conversion had issues, generating fallback data..."
            python3 "$CONVERTER" \
                --input "$MESH_DIR" \
                --output "$TOKYO_DATA" \
                --frames 100 \
                --sessions 3
        }

        log_success "Training data generated in $TOKYO_DATA"
    else
        log_warn "No PLATEAU meshes found in $MESH_DIR"
        log_info "Generating synthetic city data..."

        # Generate with synthetic buildings (converter handles this)
        python3 "$CONVERTER" \
            --input "$MESH_DIR" \
            --output "$TOKYO_DATA" \
            --frames 200 \
            --sessions 5 \
            --pattern survey 2>/dev/null || {
            # Fallback to dummy data
            log_info "Using dummy data generator..."
            python3 "${SCRIPT_DIR}/create_dummy_data.py" \
                --output "$TOKYO_DATA" \
                --frames 100 \
                --sessions 3
        }

        log_success "Training data generated"
    fi

    # Count generated samples
    local num_sessions=$(ls -d ${TOKYO_DATA}/drone_* ${TOKYO_DATA}/rover_* 2>/dev/null | wc -l)
    log_info "Generated $num_sessions training sessions"
}

# =============================================================================
# Create Simulation Data Directory Structure
# =============================================================================
setup_simulation_data_dirs() {
    log_step "Setting up simulation data directories..."

    # Create standard VeryLargeWeebModel data structure
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

## Using with VeryLargeWeebModel Training

\`\`\`python
from dataset.gazebo_occworld_dataset import GazeboVeryLargeWeebModelDataset, DatasetConfig

config = DatasetConfig(
    history_frames=4,
    future_frames=6,
    agent_type='both',
)
dataset = GazeboVeryLargeWeebModelDataset('data/tokyo_gazebo', config)
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
    echo "  ├── tokyo_gazebo/             # Training data (generated)"
    echo "  └── nuscenes/                 # nuScenes data (if downloaded)"
    echo ""
    echo "  ${PROJECT_ROOT}/pretrained/"
    echo "  ├── vqvae/                    # VQVAE checkpoint"
    echo "  ├── occworld/                 # VeryLargeWeebModel checkpoint"
    echo "  └── bevfusion/                # BEVFusion checkpoints"
    echo ""

    # Check what was downloaded/generated
    echo "=============================================================================="
    echo "                              Status                                          "
    echo "=============================================================================="
    echo ""

    # Check pretrained models
    if [ -f "${PROJECT_ROOT}/pretrained/occworld/latest.pth" ]; then
        local size=$(stat -f%z "${PROJECT_ROOT}/pretrained/occworld/latest.pth" 2>/dev/null || stat -c%s "${PROJECT_ROOT}/pretrained/occworld/latest.pth" 2>/dev/null || echo 0)
        local size_mb=$((size / 1024 / 1024))
        if [ "$size" -gt 700000000 ]; then
            echo -e "  ${GREEN}✓${NC} VeryLargeWeebModel checkpoint: ${size_mb}MB"
        else
            echo -e "  ${YELLOW}!${NC} VeryLargeWeebModel checkpoint: ${size_mb}MB (expected ~721MB)"
        fi
    else
        echo -e "  ${RED}✗${NC} VeryLargeWeebModel checkpoint missing"
    fi

    # Check training data
    local num_sessions=$(ls -d ${OUTPUT_DIR}/tokyo_gazebo/drone_* ${OUTPUT_DIR}/tokyo_gazebo/rover_* 2>/dev/null | wc -l | tr -d ' ')
    if [ "$num_sessions" -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} Training data: $num_sessions sessions"
    else
        echo -e "  ${YELLOW}!${NC} No training data generated yet"
    fi

    echo ""
    echo "=============================================================================="
    echo "                              Next Steps                                      "
    echo "=============================================================================="
    echo ""

    if [ ! -f "${PROJECT_ROOT}/pretrained/occworld/latest.pth" ]; then
        echo "1. Download pretrained model (~721MB):"
        echo "   curl -L -o pretrained/occworld/latest.pth \\"
        echo "     'https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/files/?p=/latest.pth&dl=1'"
        echo ""
        echo "   Then verify: ./pretrained/verify_downloads.sh"
        echo ""
    fi

    if [ "$num_sessions" -eq 0 ]; then
        echo "2. Generate training data:"
        echo "   python scripts/plateau_to_occworld.py \\"
        echo "       --input data/plateau/meshes/obj \\"
        echo "       --output data/tokyo_gazebo \\"
        echo "       --frames 500 --sessions 5"
        echo ""
    fi

    echo "3. Start training:"
    echo "   python train.py --py-config config/finetune_tokyo.py --work-dir /workspace/checkpoints"
    echo ""
    echo "=============================================================================="
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    echo ""
    echo "=============================================================================="
    echo "           VeryLargeWeebModel Data Download and Preparation Script                      "
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

    # Generate training data from PLATEAU meshes
    if [ "$GENERATE_DATA" = true ]; then
        generate_training_data
    fi

    # Print summary
    print_summary
}

# Run main
main "$@"
