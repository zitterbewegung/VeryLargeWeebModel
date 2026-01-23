#!/bin/bash
# =============================================================================
# Sanity Check Script for VeryLargeWeebModel Tokyo Training Pipeline
# =============================================================================
# Validates the codebase for:
#   - Python syntax errors
#   - Shell script syntax errors
#   - Required files exist
#   - Directory structure
#   - Configuration validity
#   - Import dependencies
#
# Usage:
#   ./scripts/sanity_check.sh [OPTIONS]
#
# Options:
#   --full        Run all checks including dependency imports
#   --quick       Quick syntax check only
#   --fix         Attempt to fix common issues
#   --verbose     Show detailed output
#   --help        Show this help message
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# Options
FULL_CHECK=false
QUICK_CHECK=false
FIX_ISSUES=false
VERBOSE=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# =============================================================================
# Helper Functions
# =============================================================================

log_pass()    { echo -e "${GREEN}[PASS]${NC} $1"; ((PASSED++)); }
log_fail()    { echo -e "${RED}[FAIL]${NC} $1"; ((FAILED++)); }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; ((WARNINGS++)); }
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_check()   { echo -e "${CYAN}[CHECK]${NC} $1"; }

show_help() {
    head -22 "$0" | tail -17
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)     FULL_CHECK=true; shift ;;
        --quick)    QUICK_CHECK=true; shift ;;
        --fix)      FIX_ISSUES=true; shift ;;
        --verbose)  VERBOSE=true; shift ;;
        --help|-h)  show_help ;;
        *)          echo "Unknown option: $1"; show_help ;;
    esac
done

cd "$PROJECT_ROOT"

echo ""
echo "=============================================="
echo "    VeryLargeWeebModel Sanity Check                    "
echo "=============================================="
echo "Project: $PROJECT_ROOT"
echo "Date: $(date)"
echo "=============================================="
echo ""

# =============================================================================
# 1. Check Required Files Exist
# =============================================================================
log_check "Checking required files..."

REQUIRED_FILES=(
    "train.py"
    "config/finetune_tokyo.py"
    "dataset/gazebo_occworld_dataset.py"
    "scripts/download_and_prepare_data.sh"
    "scripts/vastai_setup.sh"
    "scripts/lambda_setup.sh"
    "ATTRIBUTION.md"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        log_pass "$file exists"
    else
        log_fail "$file is missing!"
    fi
done

echo ""

# =============================================================================
# 2. Check Directory Structure
# =============================================================================
log_check "Checking directory structure..."

REQUIRED_DIRS=(
    "config"
    "dataset"
    "scripts"
    "docs"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        log_pass "$dir/ directory exists"
    else
        log_fail "$dir/ directory is missing!"
        if [ "$FIX_ISSUES" = true ]; then
            mkdir -p "$dir"
            log_info "Created $dir/"
        fi
    fi
done

echo ""

# =============================================================================
# 3. Python Syntax Check
# =============================================================================
log_check "Checking Python syntax..."

PYTHON_FILES=$(find . -name "*.py" -not -path "./.venv/*" -not -path "./__pycache__/*" 2>/dev/null)

if [ -z "$PYTHON_FILES" ]; then
    log_warn "No Python files found"
else
    for pyfile in $PYTHON_FILES; do
        if python3 -m py_compile "$pyfile" 2>/dev/null; then
            log_pass "$pyfile"
        else
            log_fail "$pyfile has syntax errors"
            if [ "$VERBOSE" = true ]; then
                python3 -m py_compile "$pyfile" 2>&1 | head -5
            fi
        fi
    done
fi

echo ""

# =============================================================================
# 4. Shell Script Syntax Check
# =============================================================================
log_check "Checking shell script syntax..."

SHELL_FILES=$(find . -name "*.sh" -not -path "./.venv/*" 2>/dev/null)

if [ -z "$SHELL_FILES" ]; then
    log_warn "No shell scripts found"
else
    for shfile in $SHELL_FILES; do
        if bash -n "$shfile" 2>/dev/null; then
            log_pass "$shfile"
        else
            log_fail "$shfile has syntax errors"
            if [ "$VERBOSE" = true ]; then
                bash -n "$shfile" 2>&1 | head -5
            fi
        fi
    done
fi

echo ""

# =============================================================================
# 5. Check Shell Scripts Are Executable
# =============================================================================
log_check "Checking shell scripts are executable..."

for shfile in $SHELL_FILES; do
    if [ -x "$shfile" ]; then
        log_pass "$shfile is executable"
    else
        log_warn "$shfile is not executable"
        if [ "$FIX_ISSUES" = true ]; then
            chmod +x "$shfile"
            log_info "Made $shfile executable"
        fi
    fi
done

echo ""

# =============================================================================
# 6. Check Documentation Files
# =============================================================================
log_check "Checking documentation..."

DOC_FILES=(
    "docs/training_guide.md"
    "docs/lambda_cloud_deployment.md"
    "docs/vastai_deployment.md"
    "docs/blog_post.md"
    "docs/slides.md"
    "docs/youtube_video_script.md"
)

for doc in "${DOC_FILES[@]}"; do
    if [ -f "$doc" ]; then
        # Check file is not empty
        if [ -s "$doc" ]; then
            log_pass "$doc"
        else
            log_warn "$doc is empty"
        fi
    else
        log_warn "$doc is missing (optional)"
    fi
done

echo ""

# =============================================================================
# 7. Check Configuration File Validity
# =============================================================================
log_check "Validating configuration files..."

if [ -f "config/finetune_tokyo.py" ]; then
    # Check config can be imported
    python3 -c "
import sys
sys.path.insert(0, '.')
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location('config', 'config/finetune_tokyo.py')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # Check required attributes
    required = ['point_cloud_range', 'voxel_size', 'history_frames', 'future_frames']
    missing = [attr for attr in required if not hasattr(config, attr)]

    if missing:
        print(f'MISSING_ATTRS:{missing}')
        sys.exit(1)
    else:
        print('CONFIG_OK')
        sys.exit(0)
except Exception as e:
    print(f'ERROR:{e}')
    sys.exit(1)
" 2>/dev/null && log_pass "config/finetune_tokyo.py is valid" || log_fail "config/finetune_tokyo.py has issues"
fi

echo ""

# =============================================================================
# 8. Check for Common Issues
# =============================================================================
log_check "Checking for common issues..."

# Check for placeholder text that should be replaced
PLACEHOLDERS=("YOUR_USERNAME" "YOUR_IP" "YOUR_NAME" "[DATE]" "[LINK]")

for placeholder in "${PLACEHOLDERS[@]}"; do
    count=$(grep -r "$placeholder" --include="*.md" --include="*.sh" --include="*.py" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -gt 0 ]; then
        log_warn "Found $count occurrences of '$placeholder' - may need to be replaced"
        if [ "$VERBOSE" = true ]; then
            grep -r "$placeholder" --include="*.md" --include="*.sh" --include="*.py" 2>/dev/null | head -3
        fi
    fi
done

# Check for debug prints
debug_count=$(grep -r "print.*DEBUG\|breakpoint()\|pdb.set_trace" --include="*.py" 2>/dev/null | wc -l | tr -d ' ')
if [ "$debug_count" -gt 0 ]; then
    log_warn "Found $debug_count debug statements in Python files"
fi

# Check for hardcoded paths
hardcoded=$(grep -r "/home/ubuntu\|/Users/" --include="*.py" 2>/dev/null | grep -v "# " | wc -l | tr -d ' ')
if [ "$hardcoded" -gt 0 ]; then
    log_warn "Found $hardcoded potential hardcoded paths"
fi

echo ""

# =============================================================================
# 9. Full Dependency Check (Optional)
# =============================================================================
if [ "$FULL_CHECK" = true ] && [ "$QUICK_CHECK" = false ]; then
    log_check "Checking Python dependencies..."

    CORE_DEPS=("torch" "numpy" "cv2" "scipy" "tqdm" "PIL")

    for dep in "${CORE_DEPS[@]}"; do
        if python3 -c "import $dep" 2>/dev/null; then
            log_pass "$dep is installed"
        else
            log_warn "$dep is not installed"
        fi
    done

    # Check PyTorch CUDA
    cuda_status=$(python3 -c "import torch; print('available' if torch.cuda.is_available() else 'not_available')" 2>/dev/null || echo "not_installed")
    if [ "$cuda_status" = "available" ]; then
        log_pass "PyTorch CUDA is available"
    elif [ "$cuda_status" = "not_available" ]; then
        log_warn "PyTorch installed but CUDA not available"
    else
        log_warn "PyTorch not installed"
    fi

    echo ""
fi

# =============================================================================
# 10. Check Dataset Loader
# =============================================================================
if [ "$FULL_CHECK" = true ] && [ "$QUICK_CHECK" = false ]; then
    log_check "Testing dataset loader import..."

    python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from dataset.gazebo_occworld_dataset import GazeboOccWorldDataset, DatasetConfig
    print('IMPORT_OK')
except ImportError as e:
    print(f'IMPORT_ERROR:{e}')
    sys.exit(1)
except Exception as e:
    print(f'ERROR:{e}')
    sys.exit(1)
" 2>/dev/null && log_pass "Dataset loader imports successfully" || log_fail "Dataset loader import failed"

    echo ""
fi

# =============================================================================
# 11. Check train.py
# =============================================================================
log_check "Validating train.py..."

if [ -f "train.py" ]; then
    # Check train.py has main components
    python3 -c "
import ast
import sys

with open('train.py', 'r') as f:
    tree = ast.parse(f.read())

# Check for required functions/classes
names = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))]

required = ['parse_args', 'main', 'train_epoch']
missing = [r for r in required if r not in names]

if missing:
    print(f'MISSING:{missing}')
    sys.exit(1)
else:
    print('STRUCTURE_OK')
    sys.exit(0)
" 2>/dev/null && log_pass "train.py has required components" || log_warn "train.py may be missing some components"
fi

echo ""

# =============================================================================
# 12. Check Git Status (if git repo)
# =============================================================================
if [ -d ".git" ]; then
    log_check "Checking git status..."

    # Check for uncommitted changes
    if git diff --quiet 2>/dev/null; then
        log_pass "No uncommitted changes"
    else
        uncommitted=$(git diff --stat 2>/dev/null | tail -1)
        log_warn "Uncommitted changes: $uncommitted"
    fi

    # Check for untracked files
    untracked=$(git ls-files --others --exclude-standard 2>/dev/null | wc -l | tr -d ' ')
    if [ "$untracked" -gt 0 ]; then
        log_warn "$untracked untracked files"
    else
        log_pass "No untracked files"
    fi
fi

echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "              SANITY CHECK SUMMARY           "
echo "=============================================="
echo ""
echo -e "  ${GREEN}Passed:${NC}   $PASSED"
echo -e "  ${RED}Failed:${NC}   $FAILED"
echo -e "  ${YELLOW}Warnings:${NC} $WARNINGS"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All critical checks passed!${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}Review warnings above.${NC}"
    fi
    EXIT_CODE=0
else
    echo -e "${RED}Some checks failed. Please fix the issues above.${NC}"
    EXIT_CODE=1
fi

echo ""
echo "=============================================="

# Quick tips based on results
if [ $FAILED -gt 0 ] || [ $WARNINGS -gt 0 ]; then
    echo ""
    echo "Tips:"
    if [ $FAILED -gt 0 ]; then
        echo "  - Run with --verbose for more details on failures"
        echo "  - Run with --fix to auto-fix some issues"
    fi
    if [ $WARNINGS -gt 0 ]; then
        echo "  - Warnings are non-critical but should be reviewed"
        echo "  - Replace placeholder text (YOUR_USERNAME, etc.)"
    fi
    echo ""
fi

exit $EXIT_CODE
