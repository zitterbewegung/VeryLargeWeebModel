#!/bin/bash
# Verify that real PLATEAU data is being used (not synthetic)
# Usage: ./scripts/verify_real_data.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

DATA_DIR="${DATA_DIR:-data}"
PLATEAU_MESHES="${DATA_DIR}/plateau/meshes/obj"
PLATEAU_ARCHIVE="${DATA_DIR}/plateau/raw/tokyo23ku_obj.zip"
TRAINING_DIR="${DATA_DIR}/tokyo_gazebo"

echo "=============================================="
echo "  VeryLargeWeebModel Data Verification"
echo "=============================================="
echo ""

REAL_DATA=true
WARNINGS=0

# Check 1: PLATEAU archive
echo "1. Checking PLATEAU archive..."
if [ -f "$PLATEAU_ARCHIVE" ]; then
    ARCHIVE_SIZE=$(stat -f%z "$PLATEAU_ARCHIVE" 2>/dev/null || stat -c%s "$PLATEAU_ARCHIVE" 2>/dev/null || echo 0)
    ARCHIVE_SIZE_GB=$(echo "scale=2; $ARCHIVE_SIZE / 1024 / 1024 / 1024" | bc 2>/dev/null || echo "0")
    if [ "$ARCHIVE_SIZE" -gt 1000000000 ]; then  # > 1GB
        echo -e "   ${GREEN}✓${NC} Archive found: ${ARCHIVE_SIZE_GB}GB"
    else
        echo -e "   ${YELLOW}⚠${NC} Archive too small: ${ARCHIVE_SIZE_GB}GB (expected ~2.1GB)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "   ${RED}✗${NC} Archive not found: $PLATEAU_ARCHIVE"
    REAL_DATA=false
fi

# Check 2: Extracted meshes
echo ""
echo "2. Checking extracted meshes..."
if [ -d "$PLATEAU_MESHES" ]; then
    MESH_COUNT=$(find "$PLATEAU_MESHES" -name "*.obj" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$MESH_COUNT" -gt 100 ]; then
        echo -e "   ${GREEN}✓${NC} Found $MESH_COUNT OBJ mesh files"

        # Check a sample mesh size
        SAMPLE_MESH=$(find "$PLATEAU_MESHES" -name "*.obj" 2>/dev/null | head -1)
        if [ -n "$SAMPLE_MESH" ]; then
            SAMPLE_SIZE=$(stat -f%z "$SAMPLE_MESH" 2>/dev/null || stat -c%s "$SAMPLE_MESH" 2>/dev/null || echo 0)
            SAMPLE_SIZE_KB=$((SAMPLE_SIZE / 1024))
            if [ "$SAMPLE_SIZE" -gt 10000 ]; then  # > 10KB
                echo -e "   ${GREEN}✓${NC} Sample mesh size: ${SAMPLE_SIZE_KB}KB (real mesh)"
            else
                echo -e "   ${YELLOW}⚠${NC} Sample mesh very small: ${SAMPLE_SIZE_KB}KB"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    elif [ "$MESH_COUNT" -gt 0 ]; then
        echo -e "   ${YELLOW}⚠${NC} Only $MESH_COUNT meshes (expected 1000+)"
        WARNINGS=$((WARNINGS + 1))
    else
        echo -e "   ${RED}✗${NC} No OBJ meshes found in $PLATEAU_MESHES"
        REAL_DATA=false
    fi
else
    echo -e "   ${RED}✗${NC} Mesh directory not found: $PLATEAU_MESHES"
    REAL_DATA=false
fi

# Check 3: Training data sessions
echo ""
echo "3. Checking training data..."
if [ -d "$TRAINING_DIR" ]; then
    SESSION_COUNT=$(ls -d ${TRAINING_DIR}/drone_* ${TRAINING_DIR}/rover_* 2>/dev/null | wc -l | tr -d ' ')
    if [ "$SESSION_COUNT" -gt 30 ]; then
        echo -e "   ${GREEN}✓${NC} Found $SESSION_COUNT training sessions (good diversity)"
    elif [ "$SESSION_COUNT" -gt 0 ]; then
        echo -e "   ${YELLOW}⚠${NC} Only $SESSION_COUNT sessions (recommend 30+ to prevent overfitting)"
        WARNINGS=$((WARNINGS + 1))
    else
        echo -e "   ${RED}✗${NC} No training sessions found"
        REAL_DATA=false
    fi

    # Check total frames
    if [ "$SESSION_COUNT" -gt 0 ]; then
        TOTAL_FRAMES=$(find "$TRAINING_DIR" -name "*_occupancy.npz" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$TOTAL_FRAMES" -gt 5000 ]; then
            echo -e "   ${GREEN}✓${NC} Total frames: $TOTAL_FRAMES (sufficient for training)"
        elif [ "$TOTAL_FRAMES" -gt 0 ]; then
            echo -e "   ${YELLOW}⚠${NC} Only $TOTAL_FRAMES frames (recommend 10000+)"
            WARNINGS=$((WARNINGS + 1))
        fi
    fi
else
    echo -e "   ${YELLOW}⚠${NC} Training directory not found: $TRAINING_DIR"
    WARNINGS=$((WARNINGS + 1))
fi

# Summary
echo ""
echo "=============================================="
echo "  VERIFICATION RESULT"
echo "=============================================="

if [ "$REAL_DATA" = true ] && [ "$WARNINGS" -eq 0 ]; then
    echo -e "  ${GREEN}✓ USING REAL PLATEAU DATA${NC}"
    echo ""
    echo "  Your training data is from real Tokyo 3D city scans."
    echo "  Training should produce meaningful results."
    exit 0
elif [ "$REAL_DATA" = true ]; then
    echo -e "  ${YELLOW}⚠ REAL DATA WITH WARNINGS${NC}"
    echo ""
    echo "  Real PLATEAU meshes detected but with $WARNINGS warning(s)."
    echo "  Consider generating more training data for better results."
    exit 0
else
    echo -e "  ${RED}✗ NOT USING REAL DATA${NC}"
    echo ""
    echo "  You are likely using SYNTHETIC data which will cause overfitting!"
    echo ""
    echo "  To download real Tokyo PLATEAU data, run:"
    echo "    ./scripts/download_and_prepare_data.sh --plateau"
    echo ""
    echo "  Or manually:"
    echo "    wget -O data/plateau/raw/tokyo23ku_obj.zip \\"
    echo "      'https://gic-plateau.s3.ap-northeast-1.amazonaws.com/2020/13100_tokyo23-ku_2020_obj_3_op.zip'"
    echo "    mkdir -p data/plateau/meshes/obj"
    echo "    unzip data/plateau/raw/tokyo23ku_obj.zip -d data/plateau/meshes/obj/"
    exit 1
fi
