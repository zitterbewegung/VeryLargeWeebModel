#!/bin/bash
# Download all research/non-commercial datasets for AerialWorld training
#
# Datasets included:
#   - nuScenes pickle files (CC BY-NC-SA 4.0)
#   - Tokyo PLATEAU 3D city data (CC BY 4.0 - commercial OK)
#   - VisDrone aerial dataset (research use)
#
# Usage:
#   ./scripts/download_research_data.sh [options]
#
# Options:
#   --all           Download everything (default)
#   --nuscenes      Download nuScenes only
#   --plateau       Download PLATEAU only
#   --visdrone      Download VisDrone only
#   --skip-large    Skip large downloads (nuScenes full, VisDrone)
#   --output DIR    Output directory (default: ./data)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()    { echo -e "\n${GREEN}==>${NC} $1"; }

# Defaults
OUTPUT_DIR="./data"
DOWNLOAD_ALL=true
DOWNLOAD_NUSCENES=false
DOWNLOAD_PLATEAU=false
DOWNLOAD_VISDRONE=false
SKIP_LARGE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)          DOWNLOAD_ALL=true; shift ;;
        --nuscenes)     DOWNLOAD_ALL=false; DOWNLOAD_NUSCENES=true; shift ;;
        --plateau)      DOWNLOAD_ALL=false; DOWNLOAD_PLATEAU=true; shift ;;
        --visdrone)     DOWNLOAD_ALL=false; DOWNLOAD_VISDRONE=true; shift ;;
        --skip-large)   SKIP_LARGE=true; shift ;;
        --output)       OUTPUT_DIR="$2"; shift 2 ;;
        --help|-h)      head -20 "$0" | tail -15; exit 0 ;;
        *)              log_warn "Unknown option: $1"; shift ;;
    esac
done

if [ "$DOWNLOAD_ALL" = true ]; then
    DOWNLOAD_NUSCENES=true
    DOWNLOAD_PLATEAU=true
    DOWNLOAD_VISDRONE=true
fi

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  AerialWorld Research Data Download"
echo "=============================================="
echo ""
echo "License Notice:"
echo "  - nuScenes: CC BY-NC-SA 4.0 (NON-COMMERCIAL)"
echo "  - VisDrone: Research use only"
echo "  - PLATEAU:  CC BY 4.0 (Commercial OK)"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# =============================================================================
# nuScenes Download
# =============================================================================
if [ "$DOWNLOAD_NUSCENES" = true ]; then
    log_step "Downloading nuScenes data..."

    NUSCENES_DIR="${OUTPUT_DIR}/nuscenes"
    mkdir -p "$NUSCENES_DIR"

    # Download pickle files (always - small and essential)
    log_info "Downloading nuScenes pickle files (~400MB total)..."

    TRAIN_PKL="${OUTPUT_DIR}/nuscenes_infos_train_temporal_v3_scene.pkl"
    VAL_PKL="${OUTPUT_DIR}/nuscenes_infos_val_temporal_v3_scene.pkl"

    if [ ! -f "$TRAIN_PKL" ]; then
        wget -q --show-progress -O "$TRAIN_PKL" \
            "https://cloud.tsinghua.edu.cn/f/a05c25067a864e0eb7d0/?dl=1" || \
        curl -L --progress-bar -o "$TRAIN_PKL" \
            "https://cloud.tsinghua.edu.cn/f/a05c25067a864e0eb7d0/?dl=1"
        log_success "Training pickle downloaded"
    else
        log_info "Training pickle exists, skipping"
    fi

    if [ ! -f "$VAL_PKL" ]; then
        wget -q --show-progress -O "$VAL_PKL" \
            "https://cloud.tsinghua.edu.cn/f/8c8f1e9b5f4a47a3b7c2/?dl=1" || \
        curl -L --progress-bar -o "$VAL_PKL" \
            "https://cloud.tsinghua.edu.cn/f/8c8f1e9b5f4a47a3b7c2/?dl=1"
        log_success "Validation pickle downloaded"
    else
        log_info "Validation pickle exists, skipping"
    fi

    # Download Occ3D ground truth
    log_info "Setting up Occ3D ground truth download instructions..."
    OCC3D_DIR="${OUTPUT_DIR}/gts"
    mkdir -p "$OCC3D_DIR"

    cat << 'EOF' > "${OCC3D_DIR}/DOWNLOAD_INSTRUCTIONS.md"
# Occ3D Ground Truth Download

For occupancy supervision, download from Occ3D:

1. Visit: https://github.com/Tsinghua-MARS-Lab/Occ3D
2. Download the nuScenes-Occupancy annotations
3. Extract to this directory (data/gts/)

Expected structure:
```
data/gts/
├── scene-0001/
│   ├── 0001.npz
│   ├── 0002.npz
│   └── ...
├── scene-0002/
└── ...
```

Direct download (if available):
```bash
# Check Occ3D repo for latest links
wget -O occ3d_nuscenes.tar.gz "URL_FROM_OCC3D_REPO"
tar -xzf occ3d_nuscenes.tar.gz -C data/gts/
```
EOF

    if [ "$SKIP_LARGE" = false ]; then
        log_info "For full nuScenes dataset, register at: https://www.nuscenes.org/"
        log_info "Then download v1.0-mini (~4GB) or v1.0-trainval (~300GB)"
    fi

    log_success "nuScenes setup complete"
fi

# =============================================================================
# PLATEAU Download
# =============================================================================
if [ "$DOWNLOAD_PLATEAU" = true ]; then
    log_step "Downloading Tokyo PLATEAU 3D city data..."

    PLATEAU_DIR="${OUTPUT_DIR}/plateau"
    PLATEAU_RAW="${PLATEAU_DIR}/raw"
    PLATEAU_MESHES="${PLATEAU_DIR}/meshes/obj"

    mkdir -p "$PLATEAU_RAW"
    mkdir -p "$PLATEAU_MESHES"

    PLATEAU_ARCHIVE="${PLATEAU_RAW}/tokyo23ku_obj.zip"
    PLATEAU_URL="https://gic-plateau.s3.ap-northeast-1.amazonaws.com/2020/13100_tokyo23-ku_2020_obj_3_op.zip"

    if [ -d "$PLATEAU_MESHES" ] && [ "$(ls -A $PLATEAU_MESHES 2>/dev/null | head -1)" ]; then
        MESH_COUNT=$(find "$PLATEAU_MESHES" -name "*.obj" 2>/dev/null | wc -l)
        log_info "PLATEAU meshes already extracted: $MESH_COUNT files"
    else
        if [ ! -f "$PLATEAU_ARCHIVE" ]; then
            log_info "Downloading PLATEAU OBJ models (~2.1GB)..."
            log_info "Source: Project PLATEAU (MLIT Japan) - CC BY 4.0"
            wget -q --show-progress -O "$PLATEAU_ARCHIVE" "$PLATEAU_URL" || \
            curl -L --progress-bar -o "$PLATEAU_ARCHIVE" "$PLATEAU_URL"
        fi

        if [ -f "$PLATEAU_ARCHIVE" ]; then
            log_info "Extracting PLATEAU meshes..."
            unzip -q -o "$PLATEAU_ARCHIVE" -d "$PLATEAU_MESHES/" || {
                cd "$PLATEAU_MESHES" && unzip -o "$PLATEAU_ARCHIVE"
            }
            MESH_COUNT=$(find "$PLATEAU_MESHES" -name "*.obj" 2>/dev/null | wc -l)
            log_success "Extracted $MESH_COUNT OBJ mesh files"
        fi
    fi

    log_success "PLATEAU setup complete"
fi

# =============================================================================
# VisDrone Download
# =============================================================================
if [ "$DOWNLOAD_VISDRONE" = true ]; then
    log_step "Setting up VisDrone aerial dataset..."

    VISDRONE_DIR="${OUTPUT_DIR}/visdrone"
    mkdir -p "$VISDRONE_DIR"

    cat << 'EOF' > "${VISDRONE_DIR}/DOWNLOAD_INSTRUCTIONS.md"
# VisDrone Dataset Download

VisDrone is a large-scale aerial dataset for drone-based detection and tracking.

## License
Research use only. See: https://github.com/VisDrone/VisDrone-Dataset

## Download Links

### Detection Dataset
- VisDrone2019-DET-train: https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view
- VisDrone2019-DET-val: https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view
- VisDrone2019-DET-test: https://drive.google.com/file/d/1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V/view

### Video Dataset
- VisDrone2019-VID-train: Check official repo
- VisDrone2019-VID-val: Check official repo

## Quick Download (using gdown)
```bash
pip install gdown

# Detection training set
gdown 1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn -O VisDrone2019-DET-train.zip

# Detection validation set
gdown 1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59 -O VisDrone2019-DET-val.zip

# Extract
unzip VisDrone2019-DET-train.zip -d data/visdrone/
unzip VisDrone2019-DET-val.zip -d data/visdrone/
```

## Citation
```bibtex
@article{zhu2021detection,
  title={Detection and tracking meet drones challenge},
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and others},
  journal={IEEE TPAMI},
  year={2021}
}
```
EOF

    if [ "$SKIP_LARGE" = false ]; then
        log_info "VisDrone requires manual download from Google Drive"
        log_info "See: ${VISDRONE_DIR}/DOWNLOAD_INSTRUCTIONS.md"

        # Try to download with gdown if available
        if command -v gdown &> /dev/null; then
            log_info "gdown found, attempting VisDrone download..."

            if [ ! -f "${VISDRONE_DIR}/VisDrone2019-DET-val.zip" ]; then
                gdown 1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59 -O "${VISDRONE_DIR}/VisDrone2019-DET-val.zip" || \
                    log_warn "VisDrone download failed - try manual download"
            fi
        else
            log_info "Install gdown for automatic download: pip install gdown"
        fi
    fi

    log_success "VisDrone setup complete"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "  Download Summary"
echo "=============================================="

if [ "$DOWNLOAD_NUSCENES" = true ]; then
    echo ""
    echo "nuScenes (CC BY-NC-SA 4.0 - NON-COMMERCIAL):"
    ls -lh "${OUTPUT_DIR}/"*.pkl 2>/dev/null || echo "  Pickle files: Not found"
fi

if [ "$DOWNLOAD_PLATEAU" = true ]; then
    echo ""
    echo "PLATEAU (CC BY 4.0 - Commercial OK):"
    MESH_COUNT=$(find "${OUTPUT_DIR}/plateau/meshes/obj" -name "*.obj" 2>/dev/null | wc -l)
    echo "  OBJ meshes: $MESH_COUNT files"
fi

if [ "$DOWNLOAD_VISDRONE" = true ]; then
    echo ""
    echo "VisDrone (Research only):"
    echo "  See: ${OUTPUT_DIR}/visdrone/DOWNLOAD_INSTRUCTIONS.md"
fi

echo ""
echo "=============================================="
echo "  Next Steps"
echo "=============================================="
echo ""
echo "1. Generate training data from PLATEAU:"
echo "   python scripts/plateau_to_occworld.py \\"
echo "     --input data/plateau/meshes/obj \\"
echo "     --output data/tokyo_gazebo \\"
echo "     --sessions 50 --frames 300 --pattern random"
echo ""
echo "2. Train with nuScenes + PLATEAU:"
echo "   python train.py --py-config config/finetune_tokyo.py"
echo ""
echo "3. Verify data:"
echo "   ./scripts/verify_real_data.sh"
echo ""
