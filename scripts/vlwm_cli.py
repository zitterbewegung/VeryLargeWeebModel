#!/usr/bin/env python3
"""VeryLargeWeebModel unified CLI.

Consolidates setup, download, train, deploy, sanity check, and info
commands into a single entry point.

Usage:
    python scripts/vlwm_cli.py setup       # Environment setup
    python scripts/vlwm_cli.py download    # Download data
    python scripts/vlwm_cli.py train       # Run training
    python scripts/vlwm_cli.py deploy      # Deploy to remote instance
    python scripts/vlwm_cli.py sanity      # Pre-flight sanity checks
    python scripts/vlwm_cli.py info        # Show environment info
    python scripts/vlwm_cli.py pack        # Pack/unpack dataset archives
    python scripts/vlwm_cli.py sync        # Fast S3 sync (no compression)
"""

import argparse
import ast
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure scripts/ is on the path so utils can be imported
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils.logging import log_info, log_success, log_warn, log_error, log_step
from utils.gpu import GPUInfo, detect_gpu_info, gpu_tier, auto_batch_size, select_precision
from utils.environment import detect_cloud_environment, CloudEnvironment
from utils.download import fast_download, verify_download, available_download_tool
from utils.system_packages import install_system_packages


# =============================================================================
# Setup subcommand
# =============================================================================

def cmd_setup(args: argparse.Namespace) -> int:
    """Set up the training environment for a cloud GPU instance."""
    log_step("Environment Setup")

    env = detect_cloud_environment()
    log_info(f"Detected provider: {env.provider}")
    log_info(f"Working directory: {env.work_dir}")
    log_info(f"Persistent storage: {env.has_persistent_storage}")

    if args.dry_run:
        log_info("[DRY RUN] Would install system packages and Python deps")
        return 0

    # System packages
    base_packages = ["git", "curl", "wget", "unzip", "htop", "tmux", "screen"]
    if args.provider in ("vastai", "lambda", "runpod", None):
        base_packages.append("aria2")

    log_step("Installing system packages")
    install_system_packages(base_packages)

    # Python dependencies — use requirements files
    log_step("Installing Python dependencies")
    repo_root = Path(__file__).resolve().parent.parent
    req_file = "requirements-full.txt" if args.full else "requirements.txt"
    req_path = repo_root / req_file

    if not req_path.exists():
        log_error(f"{req_path} not found")
        return 1

    log_info(f"Installing from {req_file}")
    pip_cmd = [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_path)]

    try:
        result = subprocess.run(pip_cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            log_error(f"pip install failed (exit code {result.returncode})")
            if result.stderr:
                # Show last few lines of stderr for actionable context
                err_lines = result.stderr.strip().split("\n")
                for line in err_lines[-3:]:
                    log_error(line)
            return 1
        log_success("Python dependencies installed")
    except OSError as e:
        log_error(f"pip install failed: {e}")
        return 1

    log_success("Setup complete")
    return 0


# =============================================================================
# Download subcommand
# =============================================================================

# Data URLs (from existing shell scripts)
PLATEAU_URL = "https://assets.cms.plateau.reearth.io/assets/d6/70821e-7f58-4f69-bc74-43f240713f1e/13100_tokyo23-ku_2022_3dtiles_1_1_op_bldg_13101_chiyoda-ku_lod2.zip"
NUSCENES_MINI_URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"

# Drone/aerial dataset info
UAVSCENES_HF_REPO = "sijieaaa/UAVScenes"
UAVSCENES_GDRIVE_FOLDER = "1HSJWc5qmIKLdpaS8w8pqrWch4F9MHIeN"
MIDAIR_URL = "https://midair.ulg.ac.be/"
TARTANAIR_URL = "https://theairlab.org/tartanair-dataset/"


def _download_uavscenes(data_dir: Path, scene: str = None) -> bool:
    """Download UAVScenes from HuggingFace (preferred) or Google Drive.

    Args:
        data_dir: Base data directory.
        scene: Specific scene to download (AMtown, AMvalley, HKairport, HKisland)
               or None for all scenes.

    Returns:
        True if download succeeded.
    """
    import zipfile

    uav_dir = data_dir / "uavscenes"
    uav_dir.mkdir(parents=True, exist_ok=True)

    # Try HuggingFace Hub first (preferred — no quota limits)
    # HF repo has: interval5_CAM_LIDAR.zip, interval5_LIDAR_label.zip, etc.
    try:
        from huggingface_hub import snapshot_download
        log_info(f"Downloading UAVScenes from HuggingFace ({UAVSCENES_HF_REPO})...")

        # Download main data zip (LiDAR + Camera)
        patterns = ["interval5_CAM_LIDAR.zip", "interval5_LIDAR_label.zip"]
        snapshot_download(
            repo_id=UAVSCENES_HF_REPO,
            repo_type="dataset",
            local_dir=str(uav_dir),
            allow_patterns=patterns,
        )

        # Extract zips
        for pattern in patterns:
            zip_path = uav_dir / pattern
            if zip_path.exists():
                log_info(f"Extracting {pattern}...")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(str(uav_dir))

        log_success(f"UAVScenes downloaded and extracted to {uav_dir}")
        return True
    except ImportError:
        log_warn("huggingface_hub not installed, trying gdown...")
    except Exception as e:
        log_warn(f"HuggingFace download failed: {e}, trying gdown...")

    # Try gdown (Google Drive)
    try:
        import subprocess
        gdown_cmd = [sys.executable, "-m", "gdown", "--folder", "--remaining-ok",
                     f"https://drive.google.com/drive/folders/{UAVSCENES_GDRIVE_FOLDER}",
                     "-O", str(uav_dir)]
        log_info("Downloading UAVScenes from Google Drive via gdown...")
        log_warn("Google Drive may impose quota limits on large files")
        result = subprocess.run(gdown_cmd, timeout=7200)
        if result.returncode == 0:
            log_success(f"UAVScenes downloaded to {uav_dir}")
            return True
        log_warn("gdown failed or incomplete")
    except ImportError:
        log_warn("gdown not installed")
    except Exception as e:
        log_warn(f"gdown failed: {e}")

    # Manual instructions
    log_error("Automatic UAVScenes download failed. Install a download tool:")
    log_info("  pip install huggingface_hub   # Recommended")
    log_info("  pip install gdown             # Alternative (may hit quota)")
    log_info("")
    log_info("Or use the shell script:")
    log_info("  ./scripts/setup_uavscenes.sh --all")
    return False


def _download_midair(data_dir: Path) -> bool:
    """Download Mid-Air dataset."""
    midair_dir = data_dir / "midair"
    midair_dir.mkdir(parents=True, exist_ok=True)

    # Mid-Air requires registration — provide instructions
    log_info("Mid-Air dataset requires manual download (academic registration).")
    log_info(f"  1. Visit: {MIDAIR_URL}")
    log_info("  2. Register and download the desired environments")
    log_info(f"  3. Extract to: {midair_dir}")
    log_info("")
    log_info("Expected structure:")
    log_info(f"  {midair_dir}/Kite_training/")
    log_info(f"  {midair_dir}/PLE_training/")
    return True


def _download_tartanair(data_dir: Path) -> bool:
    """Download TartanAir dataset."""
    tartanair_dir = data_dir / "tartanair"
    tartanair_dir.mkdir(parents=True, exist_ok=True)

    # Try tartanair Python package
    try:
        import tartanair as ta
        log_info("Downloading TartanAir via tartanair package...")
        ta.init(str(tartanair_dir))
        ta.download(env=["abandonedfactory", "neighborhood"], difficulty=["Easy"])
        log_success(f"TartanAir downloaded to {tartanair_dir}")
        return True
    except ImportError:
        pass
    except Exception as e:
        log_warn(f"tartanair package download failed: {e}")

    log_info("TartanAir dataset download options:")
    log_info(f"  1. Visit: {TARTANAIR_URL}")
    log_info("  2. Or install: pip install tartanair")
    log_info(f"  3. Extract to: {tartanair_dir}")
    return True


def cmd_download(args: argparse.Namespace) -> int:
    """Download training data and pretrained models."""
    log_step("Data Download")

    env = detect_cloud_environment()
    data_dir = Path(args.data_dir or os.path.join(env.work_dir, "data"))

    if args.dry_run:
        log_info(f"[DRY RUN] Would download to {data_dir}")
        if args.plateau or args.all:
            log_info(f"  - PLATEAU data from {PLATEAU_URL}")
        if args.nuscenes or args.all:
            log_info(f"  - nuScenes mini from {NUSCENES_MINI_URL}")
        if args.uavscenes or args.all:
            log_info(f"  - UAVScenes from HuggingFace ({UAVSCENES_HF_REPO})")
        if args.midair or args.all:
            log_info(f"  - Mid-Air (manual download from {MIDAIR_URL})")
        if args.tartanair or args.all:
            log_info(f"  - TartanAir from {TARTANAIR_URL}")
        return 0

    data_dir.mkdir(parents=True, exist_ok=True)
    failed = []

    if args.plateau or args.all:
        log_step("Downloading PLATEAU data")
        tool = available_download_tool()
        if not tool:
            log_error("No download tool found (need curl, wget, aria2c, or axel)")
            failed.append("PLATEAU")
        else:
            output = str(data_dir / "plateau" / "tokyo_chiyoda.zip")
            if verify_download(output, min_size=1_000_000):
                log_info("PLATEAU data already downloaded, skipping")
            elif not fast_download(PLATEAU_URL, output, "PLATEAU Tokyo Chiyoda"):
                log_error("PLATEAU download failed")
                failed.append("PLATEAU")

    if args.nuscenes or args.all:
        log_step("Downloading nuScenes mini")
        tool = available_download_tool()
        if not tool:
            log_error("No download tool found (need curl, wget, aria2c, or axel)")
            failed.append("nuScenes")
        else:
            output = str(data_dir / "nuscenes" / "v1.0-mini.tgz")
            if verify_download(output, min_size=1_000_000):
                log_info("nuScenes mini already downloaded, skipping")
            elif not fast_download(NUSCENES_MINI_URL, output, "nuScenes mini"):
                log_error("nuScenes download failed")
                failed.append("nuScenes")

    if args.uavscenes or args.all:
        log_step("Downloading UAVScenes")
        scene = getattr(args, 'scene', None)
        if not _download_uavscenes(data_dir, scene=scene):
            failed.append("UAVScenes")

    if args.midair or args.all:
        log_step("Mid-Air Dataset")
        _download_midair(data_dir)

    if args.tartanair or args.all:
        log_step("TartanAir Dataset")
        _download_tartanair(data_dir)

    if args.models or args.all:
        log_step("Downloading pretrained models")
        log_info("Pretrained model download: use scripts/download_pretrained.py")

    if failed:
        log_error(f"Some downloads failed: {', '.join(failed)}")
        return 1

    log_success("Download complete")
    return 0


# =============================================================================
# Train subcommand
# =============================================================================

def cmd_train(args: argparse.Namespace) -> int:
    """Run training with auto-detected GPU settings."""
    log_step("Training")

    gpu = detect_gpu_info()
    if gpu:
        log_info(f"GPU: {gpu.name} ({gpu.memory_mb} MB) x{gpu.count}")
        log_info(f"Tier: {gpu_tier(gpu)}")
    else:
        log_warn("No GPU detected, training will be slow")

    batch_size = args.batch_size or auto_batch_size(gpu)
    precision = args.precision or select_precision(gpu)
    num_gpus = args.gpus or (gpu.count if gpu else 1)
    config = args.config
    epochs = args.epochs
    work_dir = args.work_dir

    log_info(f"Config: {config}")
    log_info(f"Batch size: {batch_size}")
    log_info(f"Precision: {precision}")
    log_info(f"GPUs: {num_gpus}")
    log_info(f"Epochs: {epochs}")
    log_info(f"Work dir: {work_dir}")

    if args.dry_run:
        log_info("[DRY RUN] Would start training with above settings")
        return 0

    os.makedirs(work_dir, exist_ok=True)

    # Build training command matching train.py's argparse interface
    train_script = str(PROJECT_ROOT / "train.py")
    cmd = [
        sys.executable, train_script,
        "--config", config,
        "--work-dir", work_dir,
        "--batch-size", str(batch_size),
    ]

    if num_gpus > 1:
        gpu_ids = ",".join(str(i) for i in range(num_gpus))
        cmd.extend(["--gpu-ids", gpu_ids])

    if precision in ("fp16", "bf16"):
        cmd.append("--amp")
        if precision == "bf16":
            cmd.extend(["--amp-dtype", "bfloat16"])
        else:
            cmd.extend(["--amp-dtype", "float16"])

    if epochs:
        cmd.extend(["--epochs", str(epochs)])

    if args.resume:
        cmd.extend(["--resume-from", args.resume])

    if getattr(args, 'interval', None):
        cmd.extend(["--interval", str(args.interval)])

    log_step("Starting training")
    log_info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            log_success("Training complete")
        else:
            log_error(f"Training failed with exit code {result.returncode}")
        return result.returncode
    except OSError as e:
        log_error(f"Failed to start training: {e}")
        return 1


# =============================================================================
# Deploy subcommand
# =============================================================================

def cmd_deploy(args: argparse.Namespace) -> int:
    """Deploy code to a remote GPU instance."""
    log_step("Deploy")

    host = args.host
    key = args.key
    remote_dir = args.remote_dir

    if not host:
        log_error("--host is required for deploy")
        return 1

    if args.dry_run:
        log_info(f"[DRY RUN] Would deploy to {host}:{remote_dir}")
        return 0

    # Test SSH connection
    log_info(f"Testing SSH connection to {host}...")
    ssh_base = ["ssh"]
    if key:
        ssh_base.extend(["-i", key])
    ssh_base.extend(["-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no"])

    try:
        result = subprocess.run(
            ssh_base + [host, "echo", "ok"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            log_error(f"SSH connection failed: {result.stderr.strip()}")
            return 1
        log_success("SSH connection OK")
    except (subprocess.TimeoutExpired, OSError) as e:
        log_error(f"SSH connection failed: {e}")
        return 1

    # Rsync project files
    log_step("Syncing files")
    rsync_cmd = [
        "rsync", "-avz", "--progress",
        "--exclude", ".git",
        "--exclude", "__pycache__",
        "--exclude", "*.pyc",
        "--exclude", "data/",
        "--exclude", "checkpoints/",
    ]
    if key:
        rsync_cmd.extend(["-e", f"ssh -i {shlex.quote(key)}"])
    import re
    if not re.match(r'^[\w.\-]+@[\w.\-]+$', host):
        log_error(f"Invalid host format: {host}. Expected: user@hostname")
        return 1
    rsync_cmd.extend([str(PROJECT_ROOT) + "/", f"{host}:{shlex.quote(remote_dir)}/"])

    try:
        result = subprocess.run(rsync_cmd)
        if result.returncode == 0:
            log_success("Files synced")
        else:
            log_error("Rsync failed")
            return 1
    except OSError as e:
        log_error(f"Rsync failed: {e}")
        return 1

    # Optionally start training
    if args.start_train:
        log_step("Starting remote training")
        train_cmd = f"cd {shlex.quote(remote_dir)} && python scripts/vlwm_cli.py train"
        try:
            subprocess.run(ssh_base + [host, train_cmd])
        except OSError as e:
            log_error(f"Remote training start failed: {e}")
            return 1

    log_success("Deploy complete")
    return 0


# =============================================================================
# Sanity subcommand
# =============================================================================

def cmd_sanity(args: argparse.Namespace) -> int:
    """Run pre-flight sanity checks on the codebase."""
    log_step("Sanity Check")

    passed = 0
    failed = 0
    warnings = 0

    def check_pass(msg: str) -> None:
        nonlocal passed
        print(f"\033[0;32m[PASS]\033[0m {msg}", flush=True)
        passed += 1

    def check_fail(msg: str) -> None:
        nonlocal failed
        print(f"\033[0;31m[FAIL]\033[0m {msg}", flush=True)
        failed += 1

    def check_warn(msg: str) -> None:
        nonlocal warnings
        print(f"\033[1;33m[WARN]\033[0m {msg}", flush=True)
        warnings += 1

    # Check project structure
    log_step("Project Structure")
    required_files = [
        "train.py",
        "config/finetune_tokyo.py",
        "dataset/gazebo_occworld_dataset.py",
        "scripts/plateau_to_occworld.py",
    ]
    for f in required_files:
        path = PROJECT_ROOT / f
        if path.exists():
            check_pass(f"Required file exists: {f}")
        else:
            check_fail(f"Missing required file: {f}")

    required_dirs = ["config", "dataset", "scripts", "scripts/utils"]
    for d in required_dirs:
        path = PROJECT_ROOT / d
        if path.is_dir():
            check_pass(f"Required directory exists: {d}")
        else:
            check_fail(f"Missing required directory: {d}")

    if args.quick:
        _print_summary(passed, failed, warnings)
        return 1 if failed > 0 else 0

    # Python syntax validation
    log_step("Python Syntax")
    py_files = list(PROJECT_ROOT.glob("**/*.py"))
    py_files = [f for f in py_files if ".git" not in str(f)]
    syntax_errors = 0
    for py_file in py_files:
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            ast.parse(source, filename=str(py_file))
        except SyntaxError as e:
            check_fail(f"Syntax error in {py_file.relative_to(PROJECT_ROOT)}: {e.msg} (line {e.lineno})")
            syntax_errors += 1
    if syntax_errors == 0:
        check_pass(f"All {len(py_files)} Python files have valid syntax")

    # Shell syntax validation
    log_step("Shell Syntax")
    sh_files = list(PROJECT_ROOT.glob("scripts/*.sh"))
    if shutil.which("bash"):
        sh_errors = 0
        for sh_file in sh_files:
            result = subprocess.run(
                ["bash", "-n", str(sh_file)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                check_fail(f"Shell syntax error in {sh_file.name}: {result.stderr.strip()}")
                sh_errors += 1
        if sh_errors == 0:
            check_pass(f"All {len(sh_files)} shell scripts have valid syntax")
    else:
        check_warn("bash not found, skipping shell syntax checks")

    # Placeholder detection
    log_step("Placeholder Detection")
    placeholders = ["YOUR_USERNAME", "YOUR_API_KEY", "CHANGEME", "TODO_REPLACE"]
    for sh_file in sh_files:
        try:
            content = sh_file.read_text(encoding="utf-8", errors="ignore")
            for placeholder in placeholders:
                if placeholder in content:
                    check_warn(f"Placeholder '{placeholder}' found in {sh_file.name}")
        except OSError:
            pass

    # Config validation
    log_step("Config Validation")
    config_dir = PROJECT_ROOT / "config"
    if config_dir.is_dir():
        config_files = list(config_dir.glob("*.py"))
        for cf in config_files:
            try:
                source = cf.read_text(encoding="utf-8", errors="ignore")
                ast.parse(source, filename=str(cf))
                check_pass(f"Config valid: {cf.name}")
            except SyntaxError as e:
                check_fail(f"Config error in {cf.name}: {e.msg}")

    _print_summary(passed, failed, warnings)
    return 1 if failed > 0 else 0


def _print_summary(passed: int, failed: int, warnings: int) -> None:
    """Print sanity check summary."""
    print()
    print("=" * 50)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed, {failed} failed, {warnings} warnings")
    print("=" * 50)
    if failed == 0:
        log_success("All checks passed!")
    else:
        log_error(f"{failed} check(s) failed")


# =============================================================================
# Info subcommand
# =============================================================================

def cmd_info(args: argparse.Namespace) -> int:
    """Show environment info: GPU, data, checkpoints."""
    log_step("Environment Info")

    # Cloud environment
    env = detect_cloud_environment()
    print(f"  Provider:    {env.provider}")
    print(f"  Work dir:    {env.work_dir}")
    print(f"  Persistent:  {env.has_persistent_storage}")
    print()

    # GPU info
    log_step("GPU Info")
    gpu = detect_gpu_info()
    if gpu:
        print(f"  Name:       {gpu.name}")
        print(f"  Memory:     {gpu.memory_mb} MB")
        print(f"  Count:      {gpu.count}")
        print(f"  Tier:       {gpu_tier(gpu)}")
        print(f"  Batch size: {auto_batch_size(gpu)} (auto)")
        print(f"  Precision:  {select_precision(gpu)}")
    else:
        print("  No GPU detected")
    print()

    # Data directories
    log_step("Data")
    data_dirs = [
        PROJECT_ROOT / "data",
        Path(env.work_dir) / "data",
        PROJECT_ROOT / "data" / "nuscenes",
        PROJECT_ROOT / "data" / "plateau",
    ]
    for d in data_dirs:
        if d.is_dir():
            n_files = sum(1 for _ in d.rglob("*") if _.is_file())
            print(f"  {d}: {n_files} files")

    # Checkpoints
    log_step("Checkpoints")
    ckpt_dirs = [
        PROJECT_ROOT / "checkpoints",
        Path(env.work_dir) / "checkpoints",
    ]
    for d in ckpt_dirs:
        if d.is_dir():
            ckpts = list(d.glob("*.pth")) + list(d.glob("**/*.pth"))
            if ckpts:
                for c in sorted(ckpts):
                    size_mb = c.stat().st_size / (1024 * 1024)
                    print(f"  {c.name}: {size_mb:.1f} MB")
            else:
                print(f"  {d}: no .pth files found")

    # Download tools
    log_step("Download Tools")
    for tool in ["aria2c", "axel", "curl", "wget", "rsync"]:
        status = "available" if shutil.which(tool) else "not found"
        print(f"  {tool}: {status}")

    return 0


# =============================================================================
# Pack subcommand
# =============================================================================

# Known dataset names -> directory names
KNOWN_DATASETS = {
    "tokyo_gazebo": "tokyo_gazebo",
    "uavscenes": "uavscenes",
    "midair": "midair",
    "tartanair": "tartanair",
    "nuscenes": "nuscenes",
}


def _resolve_dataset_path(dataset, data_dir):
    # type: (str, str) -> tuple
    """Resolve a dataset name or path to (name, directory_path).

    Args:
        dataset: Known dataset name or arbitrary directory path.
        data_dir: Base data directory for known dataset names.

    Returns:
        Tuple of (dataset_name, dataset_path).
    """
    if dataset in KNOWN_DATASETS:
        name = dataset
        path = os.path.join(data_dir, KNOWN_DATASETS[dataset])
    else:
        # Treat as a directory path
        path = os.path.abspath(dataset)
        name = os.path.basename(path)
    return name, path


def _compress_dataset(source_dir, archive_path, compression_level, dry_run):
    # type: (str, str, int, bool) -> bool
    """Compress a dataset directory to a .tar.xz archive.

    Args:
        source_dir: Path to the dataset directory.
        archive_path: Output archive path.
        compression_level: xz compression level (0-9).
        dry_run: If True, show what would be done.

    Returns:
        True if successful.
    """
    import tarfile

    if not os.path.isdir(source_dir):
        log_error(f"Dataset directory not found: {source_dir}")
        return False

    if dry_run:
        log_info(f"[DRY RUN] Would compress: {source_dir} -> {archive_path}")
        # Count files for informational output
        file_count = sum(1 for _ in Path(source_dir).rglob("*") if _.is_file())
        log_info(f"[DRY RUN] Source contains {file_count} files")
        return True

    log_info(f"Compressing: {source_dir} -> {archive_path}")
    log_info(f"Compression level: {compression_level}")

    try:
        # Count files first for progress reporting
        file_count = sum(1 for _ in Path(source_dir).rglob("*") if _.is_file())
        log_info(f"Total files: {file_count}")

        added = 0
        with tarfile.open(archive_path, "w:xz") as tar:
            base_name = os.path.basename(source_dir)
            for root, dirs, files in os.walk(source_dir):
                for fname in files:
                    filepath = os.path.join(root, fname)
                    arcname = os.path.join(
                        base_name,
                        os.path.relpath(filepath, source_dir)
                    )
                    tar.add(filepath, arcname=arcname)
                    added += 1
                    if added % 500 == 0:
                        log_info(f"  Added {added}/{file_count} files...")

        archive_size = os.path.getsize(archive_path)
        size_mb = archive_size / (1024 * 1024)
        log_success(f"Archive created: {archive_path} ({size_mb:.1f} MB, {added} files)")
        return True

    except Exception as e:
        log_error(f"Compression failed: {e}")
        # Clean up partial archive
        if os.path.exists(archive_path):
            try:
                os.remove(archive_path)
                log_info("Cleaned up partial archive")
            except OSError:
                pass
        return False


def _decompress_archive(archive_path, target_dir, dry_run):
    # type: (str, str, bool) -> bool
    """Decompress a .tar.xz archive to a target directory.

    Args:
        archive_path: Path to the archive.
        target_dir: Directory to extract into.
        dry_run: If True, show what would be done.

    Returns:
        True if successful.
    """
    import tarfile

    if not os.path.isfile(archive_path):
        log_error(f"Archive not found: {archive_path}")
        return False

    if dry_run:
        log_info(f"[DRY RUN] Would decompress: {archive_path} -> {target_dir}")
        return True

    log_info(f"Decompressing: {archive_path} -> {target_dir}")

    try:
        with tarfile.open(archive_path, "r:xz") as tar:
            # Path traversal protection: skip members with .. or absolute paths
            safe_members = []
            skipped = 0
            for member in tar.getmembers():
                # Skip absolute paths
                if member.name.startswith("/") or member.name.startswith("\\"):
                    skipped += 1
                    continue
                # Skip path traversal
                normalized = os.path.normpath(member.name)
                if normalized.startswith("..") or "/.." in normalized or "\\.." in normalized:
                    skipped += 1
                    continue
                safe_members.append(member)

            if skipped > 0:
                log_warn(f"Skipped {skipped} archive members with unsafe paths")

            # Use filter='data' on Python 3.12+ for additional safety
            if sys.version_info >= (3, 12):
                tar.extractall(path=target_dir, members=safe_members,
                               filter='data')
            else:
                tar.extractall(path=target_dir, members=safe_members)

        log_success(f"Decompressed {len(safe_members)} members to {target_dir}")
        return True

    except Exception as e:
        log_error(f"Decompression failed: {e}")
        return False


def _make_s3_progress_callback(total_size, description):
    # type: (int, str) -> Any
    """Create a progress callback for S3 transfers.

    Uses tqdm if available, otherwise prints periodic text updates.

    Args:
        total_size: Total file size in bytes.
        description: Description for the progress display.

    Returns:
        Callback function that accepts bytes_transferred.
    """
    try:
        from tqdm import tqdm as tqdm_cls
        pbar = tqdm_cls(total=total_size, unit='B', unit_scale=True, desc=description)

        def callback(bytes_transferred):
            pbar.update(bytes_transferred)

        # Attach close method so caller can finalize
        callback.close = pbar.close  # type: ignore[attr-defined]
        return callback
    except ImportError:
        pass

    # Text fallback
    state = {"transferred": 0, "last_pct": -1}

    def callback(bytes_transferred):
        state["transferred"] += bytes_transferred
        if total_size > 0:
            pct = int(state["transferred"] * 100 / total_size)
            # Print every 10%
            if pct >= state["last_pct"] + 10:
                state["last_pct"] = pct
                mb = state["transferred"] / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                log_info(f"  {description}: {mb:.1f}/{total_mb:.1f} MB ({pct}%)")

    callback.close = lambda: None  # type: ignore[attr-defined]
    return callback


def _upload_archive(archive_path, bucket, s3_key, dry_run):
    # type: (str, str, str, bool) -> bool
    """Upload an archive to S3 with progress tracking.

    Args:
        archive_path: Local archive file path.
        bucket: S3 bucket name.
        s3_key: S3 object key.
        dry_run: If True, show what would be done.

    Returns:
        True if successful.
    """
    from utils.s3 import (
        check_aws_credentials, get_s3_client, ensure_bucket_exists,
        upload_file, get_transfer_config, s3_file_exists,
    )

    if dry_run:
        if os.path.exists(archive_path):
            file_size = os.path.getsize(archive_path)
            size_mb = file_size / (1024 * 1024)
            log_info(f"[DRY RUN] Would upload: {archive_path} ({size_mb:.1f} MB)")
        else:
            log_info(f"[DRY RUN] Would upload: {archive_path}")
        log_info(f"[DRY RUN] Destination: s3://{bucket}/{s3_key}")
        return True

    log_info("Checking AWS credentials...")
    if not check_aws_credentials():
        log_error("AWS credentials not configured. Run 'aws configure' first.")
        return False

    s3_client = get_s3_client()

    if not ensure_bucket_exists(s3_client, bucket):
        return False

    file_size = os.path.getsize(archive_path)
    size_mb = file_size / (1024 * 1024)
    log_info(f"Uploading: {archive_path} ({size_mb:.1f} MB) -> s3://{bucket}/{s3_key}")

    transfer_config = get_transfer_config()
    progress = _make_s3_progress_callback(file_size, "Upload")

    try:
        success = upload_file(
            s3_client, archive_path, bucket, s3_key,
            transfer_config=transfer_config, callback=progress,
        )
    finally:
        if hasattr(progress, 'close'):
            progress.close()

    if not success:
        return False

    # Verify upload size
    if s3_file_exists(s3_client, bucket, s3_key, local_size=file_size):
        log_success(f"Upload verified: s3://{bucket}/{s3_key}")
        return True
    else:
        log_error("Upload verification failed: size mismatch")
        return False


def _download_archive(bucket, s3_key, archive_path, dry_run):
    # type: (str, str, str, bool) -> bool
    """Download an archive from S3 with progress tracking.

    Args:
        bucket: S3 bucket name.
        s3_key: S3 object key.
        archive_path: Local path to save archive.
        dry_run: If True, show what would be done.

    Returns:
        True if successful.
    """
    from utils.s3 import (
        check_aws_credentials, get_s3_client,
        download_file, get_transfer_config,
    )

    if dry_run:
        log_info(f"[DRY RUN] Would download: s3://{bucket}/{s3_key}")
        log_info(f"[DRY RUN] Destination: {archive_path}")
        return True

    log_info("Checking AWS credentials...")
    if not check_aws_credentials():
        log_error("AWS credentials not configured. Run 'aws configure' first.")
        return False

    s3_client = get_s3_client()

    # Get file size for progress tracking
    try:
        response = s3_client.head_object(Bucket=bucket, Key=s3_key)
        remote_size = response['ContentLength']
    except Exception as e:
        log_error(f"Archive not found in S3: s3://{bucket}/{s3_key} ({e})")
        return False

    size_mb = remote_size / (1024 * 1024)
    log_info(f"Downloading: s3://{bucket}/{s3_key} ({size_mb:.1f} MB) -> {archive_path}")

    transfer_config = get_transfer_config()
    progress = _make_s3_progress_callback(remote_size, "Download")

    try:
        success = download_file(
            s3_client, bucket, s3_key, archive_path,
            transfer_config=transfer_config, callback=progress,
        )
    finally:
        if hasattr(progress, 'close'):
            progress.close()

    if not success:
        return False

    # Verify download size
    local_size = os.path.getsize(archive_path)
    if local_size == remote_size:
        log_success(f"Download verified: {archive_path} ({size_mb:.1f} MB)")
        return True
    else:
        log_error(f"Download verification failed: expected {remote_size} bytes, got {local_size}")
        return False


def cmd_pack(args):
    # type: (argparse.Namespace) -> int
    """Pack/unpack dataset archives for transfer to/from S3.

    Modes:
        compress (default): Create a .tar.xz archive from a dataset directory.
        upload (--upload):  Compress and upload to S3.
        download (--download): Download from S3 and decompress.
    """
    log_step("Pack")

    env = detect_cloud_environment()
    data_dir = args.data_dir or os.path.join(env.work_dir, "data")
    dataset_name, dataset_path = _resolve_dataset_path(args.dataset, data_dir)
    bucket = args.bucket
    prefix = args.prefix
    s3_key = "{}/{}.tar.xz".format(prefix, dataset_name)
    compression_level = args.compression_level
    dry_run = args.dry_run
    keep_archive = args.keep_archive
    force = args.force

    # Determine archive path
    if args.output:
        archive_path = os.path.abspath(args.output)
    else:
        archive_path = os.path.join(data_dir, "{}.tar.xz".format(dataset_name))

    log_info(f"Dataset: {dataset_name}")
    log_info(f"Dataset path: {dataset_path}")
    log_info(f"Archive: {archive_path}")

    if args.download:
        # === Download mode ===
        log_info(f"Mode: download from s3://{bucket}/{s3_key}")

        if os.path.isdir(dataset_path) and not force:
            log_error(f"Dataset directory already exists: {dataset_path}")
            log_info("Use --force to overwrite")
            return 1

        if not _download_archive(bucket, s3_key, archive_path, dry_run):
            return 1

        if not dry_run:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(archive_path) or ".", exist_ok=True)

            # Extract to data_dir (archive contains dataset_name/ as top dir)
            extract_dir = os.path.dirname(dataset_path) or data_dir
            if not _decompress_archive(archive_path, extract_dir, dry_run):
                return 1

            if not keep_archive:
                log_info(f"Removing archive: {archive_path}")
                try:
                    os.remove(archive_path)
                except OSError as e:
                    log_warn(f"Could not remove archive (operation still succeeded): {e}")

        log_success("Download and extract complete")
        return 0

    else:
        # === Compress / Upload mode ===
        upload = args.upload

        if os.path.exists(archive_path) and not force:
            log_error(f"Archive already exists: {archive_path}")
            log_info("Use --force to overwrite")
            return 1

        # Ensure archive parent directory exists
        if not dry_run:
            os.makedirs(os.path.dirname(archive_path) or ".", exist_ok=True)

        if not _compress_dataset(dataset_path, archive_path, compression_level, dry_run):
            return 1

        if upload:
            log_info(f"Mode: upload to s3://{bucket}/{s3_key}")

            if not _upload_archive(archive_path, bucket, s3_key, dry_run):
                return 1

            if not dry_run and not keep_archive:
                log_info(f"Removing local archive: {archive_path}")
                try:
                    os.remove(archive_path)
                except OSError as e:
                    log_warn(f"Could not remove archive (operation still succeeded): {e}")

            log_success("Compress and upload complete")
        else:
            log_success("Compression complete")

        return 0


# =============================================================================
# Sync subcommand
# =============================================================================

# Files/patterns to always exclude from sync (not training data)
SYNC_EXCLUDES = [
    "*.DS_Store",
    "*.zip",
    "*.download_state",
    "*.sh",
    "*.claude/*",
    "*__pycache__/*",
    "*.pyc",
    "*Thumbs.db",
]


def cmd_sync(args):
    # type: (argparse.Namespace) -> int
    """Fast S3 sync for large datasets using aws CLI.

    Uses 'aws s3 sync' directly — no compression, parallel multipart
    transfers, resumable, incremental. Much faster than tar+upload for
    repeated transfers of large datasets.
    """
    log_step("Sync")

    if not shutil.which("aws"):
        log_error("AWS CLI not found. Install with: pip install awscli")
        return 1

    env = detect_cloud_environment()
    data_dir = args.data_dir or os.path.join(env.work_dir, "data")
    dataset_name, dataset_path = _resolve_dataset_path(args.dataset, data_dir)
    bucket = args.bucket
    prefix = args.prefix
    s3_uri = "s3://{}/{}/{}".format(bucket, prefix, dataset_name)
    dry_run = args.dry_run
    delete = args.delete
    max_concurrent = args.jobs

    log_info("Dataset: {}".format(dataset_name))
    log_info("Local:   {}".format(dataset_path))
    log_info("Remote:  {}".format(s3_uri))
    log_info("Jobs:    {}".format(max_concurrent))

    # Build aws s3 sync command
    cmd = ["aws", "s3", "sync"]

    if args.download:
        # S3 -> local
        log_info("Direction: download (S3 -> local)")
        cmd.extend([s3_uri + "/", dataset_path + "/"])
    else:
        # local -> S3 (default = upload)
        log_info("Direction: upload (local -> S3)")
        if not os.path.isdir(dataset_path):
            log_error("Dataset directory not found: {}".format(dataset_path))
            return 1
        cmd.extend([dataset_path + "/", s3_uri + "/"])

    # Exclude junk files
    excludes = list(SYNC_EXCLUDES)
    if args.exclude:
        excludes.extend(args.exclude)
    for pattern in excludes:
        cmd.extend(["--exclude", pattern])

    # Include filter (e.g. only specific subdirs)
    if args.include:
        for pattern in args.include:
            cmd.extend(["--include", pattern])

    if delete:
        cmd.append("--delete")

    if dry_run:
        cmd.append("--dryrun")

    # Set concurrency via AWS CLI config env var
    cli_env = os.environ.copy()
    cli_env["AWS_MAX_ATTEMPTS"] = "5"

    log_step("Running sync")
    log_info("Command: {}".format(" ".join(cmd)))

    # Configure max concurrency via aws configure set (in a subprocess-safe way)
    config_cmd = [
        "aws", "configure", "set",
        "default.s3.max_concurrent_requests", str(max_concurrent),
    ]
    subprocess.run(config_cmd, env=cli_env, check=False)

    try:
        result = subprocess.run(cmd, env=cli_env)
        if result.returncode == 0:
            log_success("Sync complete")
            if not args.download:
                log_info("To download on a GPU instance:")
                log_info("  aws s3 sync {} ./data/{}/".format(
                    s3_uri, dataset_name))
        else:
            log_error("Sync failed with exit code {}".format(result.returncode))
        return result.returncode
    except KeyboardInterrupt:
        log_warn("Sync interrupted — re-run to resume (incremental)")
        return 1
    except OSError as e:
        log_error("Sync failed: {}".format(e))
        return 1


# =============================================================================
# Argument parsing
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="vlwm_cli",
        description="VeryLargeWeebModel unified CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- setup ---
    p_setup = subparsers.add_parser("setup", help="Set up training environment")
    p_setup.add_argument("--provider", choices=["vastai", "lambda", "runpod", "generic"],
                         help="Override cloud provider detection")
    p_setup.add_argument("--full", action="store_true",
                         help="Install full environment (requirements-full.txt)")
    p_setup.add_argument("--dry-run", action="store_true", help="Show what would be done")

    # --- download ---
    p_dl = subparsers.add_parser("download", help="Download data and models")
    p_dl.add_argument("--plateau", action="store_true", help="Download PLATEAU 3D city data")
    p_dl.add_argument("--nuscenes", action="store_true", help="Download nuScenes mini")
    p_dl.add_argument("--uavscenes", action="store_true", help="Download UAVScenes drone dataset")
    p_dl.add_argument("--midair", action="store_true", help="Download Mid-Air drone dataset")
    p_dl.add_argument("--tartanair", action="store_true", help="Download TartanAir drone dataset")
    p_dl.add_argument("--scene", help="UAVScenes scene filter (AMtown, AMvalley, HKairport, HKisland)")
    p_dl.add_argument("--models", action="store_true", help="Download pretrained models")
    p_dl.add_argument("--all", action="store_true", help="Download everything")
    p_dl.add_argument("--data-dir", help="Override data directory")
    p_dl.add_argument("--dry-run", action="store_true", help="Show what would be done")

    # --- train ---
    p_train = subparsers.add_parser("train", help="Run training")
    p_train.add_argument("--config", default="config/finetune_tokyo.py",
                         help="Config file (default: config/finetune_tokyo.py)")
    p_train.add_argument("--work-dir", default="checkpoints",
                         help="Output directory (default: checkpoints)")
    p_train.add_argument("--epochs", type=int, default=None,
                         help="Number of epochs (default: use config value)")
    p_train.add_argument("--batch-size", type=int, help="Override batch size")
    p_train.add_argument("--precision", choices=["fp16", "bf16"], help="Override precision")
    p_train.add_argument("--gpus", type=int, help="Number of GPUs")
    p_train.add_argument("--resume", help="Resume from checkpoint path")
    p_train.add_argument("--interval", type=int, help="UAVScenes interval (5=keyframes, 1=full)")
    p_train.add_argument("--dry-run", action="store_true", help="Show config without training")

    # --- deploy ---
    p_deploy = subparsers.add_parser("deploy", help="Deploy to remote instance")
    p_deploy.add_argument("--host", required=False, help="SSH host (user@ip)")
    p_deploy.add_argument("--key", help="SSH key path")
    p_deploy.add_argument("--remote-dir", default="/workspace/VeryLargeWeebModel",
                          help="Remote directory")
    p_deploy.add_argument("--start-train", action="store_true",
                          help="Start training after deploy")
    p_deploy.add_argument("--dry-run", action="store_true", help="Show what would be done")

    # --- sanity ---
    p_sanity = subparsers.add_parser("sanity", help="Pre-flight sanity checks")
    p_sanity.add_argument("--quick", action="store_true", help="Quick check only")

    # --- info ---
    subparsers.add_parser("info", help="Show environment info")

    # --- pack ---
    p_pack = subparsers.add_parser("pack", help="Pack/unpack dataset archives for S3 transfer")
    p_pack.add_argument("dataset",
                         help="Dataset name ({}) or directory path".format(
                             ", ".join(sorted(KNOWN_DATASETS.keys()))))
    pack_mode = p_pack.add_mutually_exclusive_group()
    pack_mode.add_argument("--upload", action="store_true",
                           help="Compress and upload archive to S3")
    pack_mode.add_argument("--download", action="store_true",
                           help="Download archive from S3 and decompress")
    p_pack.add_argument("--bucket", default="verylargeweebmodel",
                         help="S3 bucket (default: verylargeweebmodel)")
    p_pack.add_argument("--prefix", default="packed",
                         help="S3 key prefix (default: packed)")
    p_pack.add_argument("--data-dir",
                         help="Base data directory override")
    p_pack.add_argument("--output",
                         help="Override archive output path")
    p_pack.add_argument("--keep-archive", action="store_true",
                         help="Don't delete archive after upload/download")
    p_pack.add_argument("--force", action="store_true",
                         help="Overwrite existing archive or directory")
    p_pack.add_argument("--compression-level", type=int, default=6,
                         choices=range(0, 10), metavar="N",
                         help="xz compression level 0-9 (default: 6)")
    p_pack.add_argument("--dry-run", action="store_true",
                         help="Show what would be done")

    # --- sync ---
    p_sync = subparsers.add_parser("sync",
        help="Fast S3 sync for large datasets (no compression)")
    p_sync.add_argument("dataset",
                         help="Dataset name ({}) or directory path".format(
                             ", ".join(sorted(KNOWN_DATASETS.keys()))))
    p_sync.add_argument("--download", action="store_true",
                         help="Download from S3 (default is upload)")
    p_sync.add_argument("--bucket", default="verylargeweebmodel",
                         help="S3 bucket (default: verylargeweebmodel)")
    p_sync.add_argument("--prefix", default="datasets",
                         help="S3 key prefix (default: datasets)")
    p_sync.add_argument("--data-dir",
                         help="Base data directory override")
    p_sync.add_argument("--exclude", action="append",
                         help="Additional exclude patterns (can repeat)")
    p_sync.add_argument("--include", action="append",
                         help="Include patterns (can repeat)")
    p_sync.add_argument("--delete", action="store_true",
                         help="Delete files in destination not in source")
    p_sync.add_argument("--jobs", "-j", type=int, default=50,
                         help="Max concurrent requests (default: 50)")
    p_sync.add_argument("--dry-run", action="store_true",
                         help="Show what would be synced")

    return parser


def main(argv=None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    handlers = {
        "setup": cmd_setup,
        "download": cmd_download,
        "train": cmd_train,
        "deploy": cmd_deploy,
        "sanity": cmd_sanity,
        "info": cmd_info,
        "pack": cmd_pack,
        "sync": cmd_sync,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
