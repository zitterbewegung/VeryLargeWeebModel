#!/bin/bash
# =============================================================================
# Manual UAVScenes Download Helper
# =============================================================================
# Opens OneDrive links in browser and auto-extracts when downloads complete.
#
# Usage:
#   ./scripts/download_uavscenes_manual.sh              # Interactive mode (watch Downloads)
#   ./scripts/download_uavscenes_manual.sh --parallel   # Parallel download from direct URLs
#   ./scripts/download_uavscenes_manual.sh --urls file  # Read URLs from file, download parallel
#   ./scripts/download_uavscenes_manual.sh --threaded   # Download 3 files with 3 threads, then extract
#
# For parallel/threaded mode, set URLs by (in priority order):
#   1. Edit DOWNLOAD_URL_* variables directly in this script (recommended)
#   2. Set environment variables: UAVSCENES_URL_AMTOWN=https://...
#   3. Use --urls file with NAME=URL format
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data/uavscenes"
DOWNLOADS_DIR="$HOME/Downloads"
PARALLEL_MODE=false
THREADED_MODE=false
URL_FILE=""
MAX_PARALLEL=3

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel|-p)
            PARALLEL_MODE=true
            shift
            ;;
        --threaded|-t)
            THREADED_MODE=true
            shift
            ;;
        --urls|-u)
            URL_FILE="$2"
            PARALLEL_MODE=true
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --parallel, -p        Enable parallel download mode"
            echo "  --threaded, -t        Download with 3 threads, then extract multithreaded"
            echo "  --urls, -u FILE       Read download URLs from file"
            echo "  --data-dir DIR        Data directory (default: ./data/uavscenes)"
            echo "  --max-parallel N      Max parallel downloads (default: 3)"
            echo ""
            echo "URL file format (one per line):"
            echo "  AMtown=https://direct-download-url..."
            echo "  AMvalley=https://direct-download-url..."
            echo ""
            echo "Or set environment variables:"
            echo "  UAVSCENES_URL_AMTOWN=https://..."
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# OneDrive folder (for manual browser download mode)
ONEDRIVE_FOLDER="https://entuedu-my.sharepoint.com/:f:/g/personal/wang1679_e_ntu_edu_sg/EgY6DU5GBchIiAIa-eQZmEAB0vJx3khCPHbFW3LnR77RFw"

# =============================================================================
# Download URLs - Set these directly here for parallel/threaded mode
# =============================================================================
# Get direct download URLs from OneDrive by:
# 1. Right-click the file in OneDrive
# 2. Select "Share" > "Copy link" > "Anyone with the link"
# 3. Paste the URL here
#
# Leave empty ("") to skip a scene, or use environment variables as fallback
DOWNLOAD_URL_AMTOWN=""
DOWNLOAD_URL_AMVALLEY=""
DOWNLOAD_URL_HKAIRPORT=""
DOWNLOAD_URL_HKISLAND=""

mkdir -p "$DATA_DIR"

echo "=============================================="
echo "  UAVScenes Manual Download Helper"
echo "=============================================="
echo ""
echo -e "${BLUE}Data directory:${NC} $DATA_DIR"
echo -e "${BLUE}Downloads folder:${NC} $DOWNLOADS_DIR"
echo ""

# Check what's already downloaded
check_scene() {
    local scene=$1
    local path="$DATA_DIR/interval1_${scene}01"
    if [ -d "$path/interval1_LIDAR" ]; then
        return 0
    fi
    return 1
}

# =============================================================================
# Parallel Download Functions
# =============================================================================

# Load URLs from script variables, environment, or file
declare -A SCENE_URLS
load_urls() {
    # Priority 1: Load from script-defined variables (set at top of script)
    [ -n "$DOWNLOAD_URL_AMTOWN" ] && SCENE_URLS["AMtown"]="$DOWNLOAD_URL_AMTOWN"
    [ -n "$DOWNLOAD_URL_AMVALLEY" ] && SCENE_URLS["AMvalley"]="$DOWNLOAD_URL_AMVALLEY"
    [ -n "$DOWNLOAD_URL_HKAIRPORT" ] && SCENE_URLS["HKairport"]="$DOWNLOAD_URL_HKAIRPORT"
    [ -n "$DOWNLOAD_URL_HKISLAND" ] && SCENE_URLS["HKisland"]="$DOWNLOAD_URL_HKISLAND"

    # Priority 2: Load from environment variables (fallback if script vars empty)
    [ -z "${SCENE_URLS[AMtown]}" ] && [ -n "$UAVSCENES_URL_AMTOWN" ] && SCENE_URLS["AMtown"]="$UAVSCENES_URL_AMTOWN"
    [ -z "${SCENE_URLS[AMvalley]}" ] && [ -n "$UAVSCENES_URL_AMVALLEY" ] && SCENE_URLS["AMvalley"]="$UAVSCENES_URL_AMVALLEY"
    [ -z "${SCENE_URLS[HKairport]}" ] && [ -n "$UAVSCENES_URL_HKAIRPORT" ] && SCENE_URLS["HKairport"]="$UAVSCENES_URL_HKAIRPORT"
    [ -z "${SCENE_URLS[HKisland]}" ] && [ -n "$UAVSCENES_URL_HKISLAND" ] && SCENE_URLS["HKisland"]="$UAVSCENES_URL_HKISLAND"

    # Priority 3: Load from file if specified (overrides all above)
    if [ -n "$URL_FILE" ] && [ -f "$URL_FILE" ]; then
        while IFS='=' read -r scene url; do
            # Skip comments and empty lines
            [[ "$scene" =~ ^#.*$ ]] && continue
            [ -z "$scene" ] && continue
            # Trim whitespace
            scene=$(echo "$scene" | xargs)
            url=$(echo "$url" | xargs)
            SCENE_URLS["$scene"]="$url"
        done < "$URL_FILE"
    fi
}

# Download a single scene in foreground (used by parallel wrapper)
download_scene() {
    local scene=$1
    local url=$2
    local output="$DATA_DIR/interval1_${scene}01.zip"
    local log_file="$DATA_DIR/.download_${scene}.log"

    echo "[$(date '+%H:%M:%S')] Starting download: $scene" | tee "$log_file"

    # Use aria2c if available (best for large files), otherwise wget/curl
    if command -v aria2c &>/dev/null; then
        aria2c \
            --continue=true \
            --max-connection-per-server=4 \
            --split=4 \
            --min-split-size=10M \
            --file-allocation=none \
            --max-tries=5 \
            --retry-wait=10 \
            --timeout=60 \
            --dir="$DATA_DIR" \
            --out="$(basename "$output")" \
            "$url" 2>&1 | tee -a "$log_file"
    elif command -v wget &>/dev/null; then
        wget \
            --continue \
            --progress=bar:force \
            --timeout=60 \
            --tries=5 \
            -O "$output" \
            "$url" 2>&1 | tee -a "$log_file"
    elif command -v curl &>/dev/null; then
        curl \
            --continue-at - \
            --location \
            --retry 5 \
            --retry-delay 10 \
            --progress-bar \
            -o "$output" \
            "$url" 2>&1 | tee -a "$log_file"
    else
        echo "ERROR: No download tool found (aria2c, wget, or curl required)" | tee -a "$log_file"
        return 1
    fi

    local status=$?
    if [ $status -eq 0 ] && [ -f "$output" ]; then
        echo "[$(date '+%H:%M:%S')] Download complete: $scene" | tee -a "$log_file"
        return 0
    else
        echo "[$(date '+%H:%M:%S')] Download FAILED: $scene" | tee -a "$log_file"
        return 1
    fi
}

# Extract a single scene (used by parallel wrapper)
extract_scene() {
    local scene=$1
    local zip_file="$DATA_DIR/interval1_${scene}01.zip"
    local log_file="$DATA_DIR/.extract_${scene}.log"

    if [ ! -f "$zip_file" ]; then
        echo "ERROR: Zip file not found: $zip_file" | tee "$log_file"
        return 1
    fi

    echo "[$(date '+%H:%M:%S')] Extracting: $scene" | tee "$log_file"

    # Use unzip with overwrite
    if command -v unzip &>/dev/null; then
        unzip -o "$zip_file" -d "$DATA_DIR" 2>&1 | tee -a "$log_file"
    else
        # Python fallback
        python3 -c "import zipfile; zipfile.ZipFile('$zip_file').extractall('$DATA_DIR')" 2>&1 | tee -a "$log_file"
    fi

    local status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Extraction complete: $scene" | tee -a "$log_file"
        # Verify extraction
        if check_scene "$scene"; then
            echo "[$(date '+%H:%M:%S')] Verified: $scene" | tee -a "$log_file"
            return 0
        else
            echo "[$(date '+%H:%M:%S')] Verification FAILED: $scene" | tee -a "$log_file"
            return 1
        fi
    else
        echo "[$(date '+%H:%M:%S')] Extraction FAILED: $scene" | tee -a "$log_file"
        return 1
    fi
}

# Run parallel downloads for all scenes with URLs
run_parallel_downloads() {
    load_urls

    local scenes_to_download=()
    local pids=()
    local scene_for_pid=()

    # Find scenes that need downloading
    for scene in AMtown AMvalley HKairport HKisland; do
        if check_scene "$scene"; then
            echo -e "${GREEN}✓${NC} $scene - already downloaded"
            continue
        fi

        if [ -z "${SCENE_URLS[$scene]}" ]; then
            echo -e "${YELLOW}○${NC} $scene - no URL provided, skipping"
            continue
        fi

        scenes_to_download+=("$scene")
    done

    if [ ${#scenes_to_download[@]} -eq 0 ]; then
        echo -e "${GREEN}Nothing to download!${NC}"
        return 0
    fi

    echo ""
    echo "=============================================="
    echo "  Parallel Download: ${#scenes_to_download[@]} scenes"
    echo "=============================================="
    echo ""

    # Start downloads in parallel (up to MAX_PARALLEL)
    local running=0
    local idx=0

    while [ $idx -lt ${#scenes_to_download[@]} ] || [ $running -gt 0 ]; do
        # Start new downloads if under limit
        while [ $running -lt $MAX_PARALLEL ] && [ $idx -lt ${#scenes_to_download[@]} ]; do
            local scene="${scenes_to_download[$idx]}"
            local url="${SCENE_URLS[$scene]}"

            echo -e "${BLUE}[STARTING]${NC} $scene"
            download_scene "$scene" "$url" &
            pids+=($!)
            scene_for_pid+=("$scene")
            ((running++))
            ((idx++))
        done

        # Wait for any process to finish
        if [ $running -gt 0 ]; then
            sleep 2

            # Check which processes have finished
            for i in "${!pids[@]}"; do
                if [ -n "${pids[$i]}" ]; then
                    if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                        # Process finished, check exit status
                        wait "${pids[$i]}" 2>/dev/null
                        local status=$?
                        local finished_scene="${scene_for_pid[$i]}"

                        if [ $status -eq 0 ]; then
                            echo -e "${GREEN}[DONE]${NC} $finished_scene download complete"
                        else
                            echo -e "${RED}[FAILED]${NC} $finished_scene download failed"
                        fi

                        unset 'pids[i]'
                        unset 'scene_for_pid[i]'
                        ((running--))
                    fi
                fi
            done
        fi
    done

    echo ""
    echo "=============================================="
    echo "  Parallel Extraction"
    echo "=============================================="
    echo ""

    # Now extract in parallel
    pids=()
    scene_for_pid=()
    running=0
    idx=0

    # Find downloaded zips to extract
    local scenes_to_extract=()
    for scene in "${scenes_to_download[@]}"; do
        local zip_file="$DATA_DIR/interval1_${scene}01.zip"
        if [ -f "$zip_file" ]; then
            if ! check_scene "$scene"; then
                scenes_to_extract+=("$scene")
            fi
        fi
    done

    while [ $idx -lt ${#scenes_to_extract[@]} ] || [ $running -gt 0 ]; do
        # Start new extractions if under limit
        while [ $running -lt $MAX_PARALLEL ] && [ $idx -lt ${#scenes_to_extract[@]} ]; do
            local scene="${scenes_to_extract[$idx]}"

            echo -e "${BLUE}[EXTRACTING]${NC} $scene"
            extract_scene "$scene" &
            pids+=($!)
            scene_for_pid+=("$scene")
            ((running++))
            ((idx++))
        done

        # Wait for any process to finish
        if [ $running -gt 0 ]; then
            sleep 2

            for i in "${!pids[@]}"; do
                if [ -n "${pids[$i]}" ]; then
                    if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                        wait "${pids[$i]}" 2>/dev/null
                        local status=$?
                        local finished_scene="${scene_for_pid[$i]}"

                        if [ $status -eq 0 ]; then
                            echo -e "${GREEN}[DONE]${NC} $finished_scene extracted"
                        else
                            echo -e "${RED}[FAILED]${NC} $finished_scene extraction failed"
                        fi

                        unset 'pids[i]'
                        unset 'scene_for_pid[i]'
                        ((running--))
                    fi
                fi
            done
        fi
    done

    # Final verification
    echo ""
    echo "=============================================="
    echo "  Final Status"
    echo "=============================================="
    echo ""

    local total_frames=0
    for scene in AMtown AMvalley HKairport HKisland; do
        local path="$DATA_DIR/interval1_${scene}01/interval1_LIDAR"
        if [ -d "$path" ]; then
            local count=$(ls -1 "$path" 2>/dev/null | wc -l | tr -d ' ')
            echo -e "  ${GREEN}✓${NC} $scene: $count LiDAR frames"
            total_frames=$((total_frames + count))
        else
            echo -e "  ${RED}✗${NC} $scene: not found"
        fi
    done

    echo ""
    echo -e "${GREEN}Total: $total_frames LiDAR frames${NC}"
    echo ""
    echo "To start training:"
    echo "  python train.py --config config/finetune_uavscenes.py --model-type 6dof --amp"
}

# =============================================================================
# Threaded Download Functions (3 threads for 3 files)
# =============================================================================

# Worker function for threaded downloads - runs in subshell
download_worker() {
    local scene=$1
    local url=$2
    local worker_id=$3
    local output="$DATA_DIR/interval1_${scene}01.zip"
    local log_file="$DATA_DIR/.thread_${worker_id}_${scene}.log"

    echo "[Thread $worker_id] Starting download: $scene" > "$log_file"
    echo "[Thread $worker_id] URL: ${url:0:50}..." >> "$log_file"

    local start_time=$(date +%s)

    # Download with progress
    if command -v aria2c &>/dev/null; then
        aria2c \
            --continue=true \
            --max-connection-per-server=4 \
            --split=4 \
            --min-split-size=10M \
            --file-allocation=none \
            --max-tries=5 \
            --retry-wait=10 \
            --timeout=60 \
            --console-log-level=warn \
            --summary-interval=30 \
            --dir="$DATA_DIR" \
            --out="$(basename "$output")" \
            "$url" >> "$log_file" 2>&1
    elif command -v wget &>/dev/null; then
        wget \
            --continue \
            --progress=dot:mega \
            --timeout=60 \
            --tries=5 \
            -O "$output" \
            "$url" >> "$log_file" 2>&1
    elif command -v curl &>/dev/null; then
        curl \
            --continue-at - \
            --location \
            --retry 5 \
            --retry-delay 10 \
            --progress-bar \
            -o "$output" \
            "$url" >> "$log_file" 2>&1
    else
        echo "[Thread $worker_id] ERROR: No download tool found" >> "$log_file"
        return 1
    fi

    local status=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ $status -eq 0 ] && [ -f "$output" ]; then
        local size=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output" 2>/dev/null || echo "0")
        local size_gb=$(echo "scale=2; $size/1024/1024/1024" | bc 2>/dev/null || echo "?")
        echo "[Thread $worker_id] COMPLETE: $scene (${size_gb}GB in ${duration}s)" >> "$log_file"
        # Write success marker
        echo "SUCCESS" > "$DATA_DIR/.thread_${worker_id}_status"
        return 0
    else
        echo "[Thread $worker_id] FAILED: $scene (exit code $status)" >> "$log_file"
        echo "FAILED" > "$DATA_DIR/.thread_${worker_id}_status"
        return 1
    fi
}

# Worker function for threaded extraction
extract_worker() {
    local scene=$1
    local worker_id=$2
    local zip_file="$DATA_DIR/interval1_${scene}01.zip"
    local log_file="$DATA_DIR/.extract_thread_${worker_id}_${scene}.log"

    echo "[Thread $worker_id] Starting extraction: $scene" > "$log_file"

    if [ ! -f "$zip_file" ]; then
        echo "[Thread $worker_id] ERROR: Zip not found: $zip_file" >> "$log_file"
        echo "FAILED" > "$DATA_DIR/.extract_thread_${worker_id}_status"
        return 1
    fi

    local start_time=$(date +%s)

    # Extract
    if command -v unzip &>/dev/null; then
        unzip -o -q "$zip_file" -d "$DATA_DIR" >> "$log_file" 2>&1
    else
        python3 -c "import zipfile; zipfile.ZipFile('$zip_file').extractall('$DATA_DIR')" >> "$log_file" 2>&1
    fi

    local status=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ $status -eq 0 ] && check_scene "$scene"; then
        local lidar_path="$DATA_DIR/interval1_${scene}01/interval1_LIDAR"
        local frame_count=$(ls -1 "$lidar_path" 2>/dev/null | wc -l | tr -d ' ')
        echo "[Thread $worker_id] COMPLETE: $scene ($frame_count frames in ${duration}s)" >> "$log_file"
        echo "SUCCESS" > "$DATA_DIR/.extract_thread_${worker_id}_status"
        return 0
    else
        echo "[Thread $worker_id] FAILED: $scene extraction" >> "$log_file"
        echo "FAILED" > "$DATA_DIR/.extract_thread_${worker_id}_status"
        return 1
    fi
}

# Monitor progress of threaded operations
monitor_threads() {
    local phase=$1
    shift
    local pids=("$@")
    local count=${#pids[@]}

    while true; do
        local running=0
        local completed=0
        local failed=0

        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                ((running++))
            else
                wait "$pid" 2>/dev/null
                if [ $? -eq 0 ]; then
                    ((completed++))
                else
                    ((failed++))
                fi
            fi
        done

        # Show live progress from log files
        printf "\r${BLUE}[$phase]${NC} Running: $running | Completed: $completed | Failed: $failed    "

        if [ $running -eq 0 ]; then
            echo ""
            break
        fi

        sleep 2
    done
}

# Main threaded download function - 3 threads for 3 files
run_threaded_downloads() {
    load_urls

    echo ""
    echo -e "${CYAN}=============================================="
    echo "  Threaded Download Mode (3 threads)"
    echo "==============================================${NC}"
    echo ""

    # Find scenes to download (max 3 for threaded mode)
    local scenes_to_download=()
    local urls_to_download=()

    for scene in AMtown AMvalley HKairport HKisland; do
        if check_scene "$scene"; then
            echo -e "  ${GREEN}✓${NC} $scene - already downloaded"
            continue
        fi

        if [ -z "${SCENE_URLS[$scene]}" ]; then
            echo -e "  ${YELLOW}○${NC} $scene - no URL provided"
            continue
        fi

        scenes_to_download+=("$scene")
        urls_to_download+=("${SCENE_URLS[$scene]}")

        # Limit to 3 for threaded mode
        if [ ${#scenes_to_download[@]} -ge 3 ]; then
            break
        fi
    done

    if [ ${#scenes_to_download[@]} -eq 0 ]; then
        echo ""
        echo -e "${GREEN}Nothing to download! All scenes complete.${NC}"
        return 0
    fi

    echo ""
    echo "=============================================="
    echo "  Phase 1: Downloading ${#scenes_to_download[@]} files with ${#scenes_to_download[@]} threads"
    echo "=============================================="
    echo ""

    # Start download threads
    local download_pids=()
    for i in "${!scenes_to_download[@]}"; do
        local scene="${scenes_to_download[$i]}"
        local url="${urls_to_download[$i]}"
        local thread_id=$((i + 1))

        echo -e "  ${BLUE}[Thread $thread_id]${NC} Starting: $scene"
        download_worker "$scene" "$url" "$thread_id" &
        download_pids+=($!)
    done

    echo ""
    echo "All ${#download_pids[@]} download threads started."
    echo ""

    # Monitor downloads
    monitor_threads "DOWNLOADING" "${download_pids[@]}"

    # Wait for all downloads to complete
    local download_success=0
    local download_failed=0
    for i in "${!download_pids[@]}"; do
        wait "${download_pids[$i]}" 2>/dev/null
        local status=$?
        local scene="${scenes_to_download[$i]}"
        local thread_id=$((i + 1))

        if [ $status -eq 0 ]; then
            echo -e "  ${GREEN}✓${NC} Thread $thread_id: $scene downloaded"
            ((download_success++))
        else
            echo -e "  ${RED}✗${NC} Thread $thread_id: $scene failed"
            ((download_failed++))
        fi
    done

    echo ""
    echo "Download phase complete: $download_success success, $download_failed failed"

    if [ $download_success -eq 0 ]; then
        echo -e "${RED}No successful downloads. Aborting.${NC}"
        return 1
    fi

    # Find files to extract
    local scenes_to_extract=()
    for scene in "${scenes_to_download[@]}"; do
        local zip_file="$DATA_DIR/interval1_${scene}01.zip"
        if [ -f "$zip_file" ] && ! check_scene "$scene"; then
            scenes_to_extract+=("$scene")
        fi
    done

    if [ ${#scenes_to_extract[@]} -eq 0 ]; then
        echo ""
        echo "No files to extract."
        return 0
    fi

    echo ""
    echo "=============================================="
    echo "  Phase 2: Extracting ${#scenes_to_extract[@]} files with ${#scenes_to_extract[@]} threads"
    echo "=============================================="
    echo ""

    # Start extraction threads
    local extract_pids=()
    for i in "${!scenes_to_extract[@]}"; do
        local scene="${scenes_to_extract[$i]}"
        local thread_id=$((i + 1))

        echo -e "  ${BLUE}[Thread $thread_id]${NC} Starting: $scene"
        extract_worker "$scene" "$thread_id" &
        extract_pids+=($!)
    done

    echo ""
    echo "All ${#extract_pids[@]} extraction threads started."
    echo ""

    # Monitor extractions
    monitor_threads "EXTRACTING" "${extract_pids[@]}"

    # Wait for all extractions to complete
    local extract_success=0
    local extract_failed=0
    for i in "${!extract_pids[@]}"; do
        wait "${extract_pids[$i]}" 2>/dev/null
        local status=$?
        local scene="${scenes_to_extract[$i]}"
        local thread_id=$((i + 1))

        if [ $status -eq 0 ]; then
            echo -e "  ${GREEN}✓${NC} Thread $thread_id: $scene extracted"
            ((extract_success++))
        else
            echo -e "  ${RED}✗${NC} Thread $thread_id: $scene failed"
            ((extract_failed++))
        fi
    done

    echo ""
    echo "Extraction phase complete: $extract_success success, $extract_failed failed"

    # Final verification
    echo ""
    echo "=============================================="
    echo "  Final Status"
    echo "=============================================="
    echo ""

    local total_frames=0
    for scene in AMtown AMvalley HKairport HKisland; do
        local path="$DATA_DIR/interval1_${scene}01/interval1_LIDAR"
        if [ -d "$path" ]; then
            local count=$(ls -1 "$path" 2>/dev/null | wc -l | tr -d ' ')
            echo -e "  ${GREEN}✓${NC} $scene: $count LiDAR frames"
            total_frames=$((total_frames + count))
        else
            echo -e "  ${RED}✗${NC} $scene: not found"
        fi
    done

    echo ""
    echo -e "${GREEN}Total: $total_frames LiDAR frames${NC}"
    echo ""

    # Cleanup status files
    rm -f "$DATA_DIR"/.thread_*_status "$DATA_DIR"/.extract_thread_*_status 2>/dev/null

    echo "To start training:"
    echo "  python train.py --config config/finetune_uavscenes.py --model-type 6dof --amp"
}

# =============================================================================
# Main Script
# =============================================================================

# If threaded mode, run threaded downloads and exit
if [ "$THREADED_MODE" = true ]; then
    run_threaded_downloads
    exit $?
fi

# If parallel mode, run parallel downloads and exit
if [ "$PARALLEL_MODE" = true ]; then
    run_parallel_downloads
    exit $?
fi

# List scenes to download
NEED_DOWNLOAD=""
echo "Scene Status:"
for scene in AMtown AMvalley HKairport HKisland; do
    if check_scene "$scene"; then
        echo -e "  ${GREEN}✓${NC} $scene - already downloaded"
    else
        echo -e "  ${YELLOW}○${NC} $scene - needs download"
        NEED_DOWNLOAD="$NEED_DOWNLOAD $scene"
    fi
done
echo ""

NEED_DOWNLOAD=$(echo $NEED_DOWNLOAD | xargs)  # trim whitespace

if [ -z "$NEED_DOWNLOAD" ]; then
    echo -e "${GREEN}All scenes already downloaded!${NC}"

    # Show final count
    echo ""
    echo "LiDAR frames per scene:"
    total_frames=0
    for scene in AMtown AMvalley HKairport HKisland; do
        path="$DATA_DIR/interval1_${scene}01/interval1_LIDAR"
        if [ -d "$path" ]; then
            count=$(ls -1 "$path" 2>/dev/null | wc -l | tr -d ' ')
            echo "  $scene: $count frames"
            total_frames=$((total_frames + count))
        fi
    done
    echo ""
    echo -e "${GREEN}Total: $total_frames LiDAR frames${NC}"
    echo ""
    echo "To start training:"
    echo "  python train.py --config config/finetune_uavscenes.py --model-type 6dof --amp"
    exit 0
fi

echo "=============================================="
echo "  Step 1: Open OneDrive in Browser"
echo "=============================================="
echo ""
echo "Opening OneDrive folder in your browser..."
sleep 1

# Open the main OneDrive folder
open "$ONEDRIVE_FOLDER" 2>/dev/null || xdg-open "$ONEDRIVE_FOLDER" 2>/dev/null || {
    echo "Could not open browser. Please open manually:"
    echo "$ONEDRIVE_FOLDER"
}

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  INSTRUCTIONS - Download these folders:${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
for scene in $NEED_DOWNLOAD; do
    echo -e "  ${YELLOW}→${NC} interval1_${scene}01"
done
echo ""
echo "For each folder:"
echo "  1. Click on the folder name"
echo "  2. Click the 'Download' button (top menu)"
echo "  3. Wait for the zip to download (~10-20 GB each)"
echo ""
echo "=============================================="
echo "  Step 2: Waiting for Downloads"
echo "=============================================="
echo ""
echo "I'll watch your Downloads folder and auto-extract."
echo ""
echo -e "${YELLOW}Press Ctrl+C when done or to stop${NC}"
echo ""

# Watch for downloads and extract
extract_if_found() {
    local scene=$1

    # Check various possible filenames
    for zip_file in \
        "$DOWNLOADS_DIR/interval1_${scene}01.zip" \
        "$DOWNLOADS_DIR/interval1_${scene}01 (1).zip" \
        "$DOWNLOADS_DIR/interval1_${scene}01-1.zip" \
        "$DOWNLOADS_DIR/${scene}.zip" \
        "$DOWNLOADS_DIR/${scene}01.zip"
    do
        if [ -f "$zip_file" ]; then
            # Check if download is complete (file not growing)
            local size1=$(stat -f%z "$zip_file" 2>/dev/null || stat -c%s "$zip_file" 2>/dev/null || echo "0")
            sleep 3
            local size2=$(stat -f%z "$zip_file" 2>/dev/null || stat -c%s "$zip_file" 2>/dev/null || echo "0")

            # File must be >100MB and not growing
            if [ "$size1" = "$size2" ] && [ "$size1" -gt 100000000 ]; then
                echo ""
                echo -e "${GREEN}Found completed download:${NC} $(basename "$zip_file")"
                echo "Size: $(echo "scale=1; $size1/1024/1024/1024" | bc 2>/dev/null || echo "$((size1/1024/1024)) MB") GB"
                echo ""
                echo "Extracting to $DATA_DIR..."
                echo "(This may take a few minutes)"
                echo ""

                unzip -o "$zip_file" -d "$DATA_DIR"

                echo ""
                echo -e "${GREEN}✓ Extracted ${scene}!${NC}"

                # Count frames
                local lidar_path="$DATA_DIR/interval1_${scene}01/interval1_LIDAR"
                if [ -d "$lidar_path" ]; then
                    local frame_count=$(ls -1 "$lidar_path" 2>/dev/null | wc -l | tr -d ' ')
                    echo "  LiDAR frames: $frame_count"
                fi
                echo ""

                # Move zip to data dir
                echo "Moving zip to $DATA_DIR for backup..."
                mv "$zip_file" "$DATA_DIR/"

                return 0
            fi
        fi
    done
    return 1
}

# Main watch loop
extracted_count=0
need_count=$(echo $NEED_DOWNLOAD | wc -w | tr -d ' ')

while [ $extracted_count -lt $need_count ]; do
    for scene in $NEED_DOWNLOAD; do
        # Skip if already exists
        if check_scene "$scene"; then
            continue
        fi

        # Try to extract
        if extract_if_found "$scene"; then
            extracted_count=$((extracted_count + 1))
        fi
    done

    # Show progress
    remaining=$((need_count - extracted_count))

    # Recount what still needs download
    still_need=""
    for scene in $NEED_DOWNLOAD; do
        if ! check_scene "$scene"; then
            still_need="$still_need $scene"
        fi
    done
    still_need=$(echo $still_need | xargs)

    if [ -n "$still_need" ]; then
        printf "\r${BLUE}Waiting for:${NC} $still_need ${BLUE}(checking every 5s)${NC}          "
        sleep 5
    else
        break
    fi
done

echo ""
echo ""
echo "=============================================="
echo "  Download Complete!"
echo "=============================================="
echo ""

# Final count
echo "Verifying downloads..."
echo ""
total_frames=0
for scene in AMtown AMvalley HKairport HKisland; do
    path="$DATA_DIR/interval1_${scene}01/interval1_LIDAR"
    if [ -d "$path" ]; then
        count=$(ls -1 "$path" 2>/dev/null | wc -l | tr -d ' ')
        echo -e "  ${GREEN}✓${NC} $scene: $count LiDAR frames"
        total_frames=$((total_frames + count))
    else
        echo -e "  ${YELLOW}○${NC} $scene: not found"
    fi
done

echo ""
echo "========================================"
echo -e "${GREEN}Total: ~$total_frames training frames${NC}"
echo "========================================"
echo ""
echo "With batch_size=3 on A100:"
echo "  - ~$((total_frames / 3)) iterations per epoch"
echo "  - 50 epochs @ \$1.00/hr on Vast.ai"
echo ""
echo "To start training:"
echo "  python train.py --config config/finetune_uavscenes.py --model-type 6dof --amp"
echo ""
