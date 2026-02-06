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

    # Python dependencies
    log_step("Installing Python dependencies")
    pip_cmd = [sys.executable, "-m", "pip", "install", "-q"]
    core_deps = ["torch", "torchvision", "mmcv", "mmdet", "mmdet3d",
                 "numpy", "pillow", "tqdm", "tensorboard"]

    try:
        result = subprocess.run(pip_cmd + core_deps, check=False, capture_output=True)
        if result.returncode != 0:
            log_error(f"pip install failed (exit code {result.returncode})")
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
    uav_dir = data_dir / "uavscenes"
    uav_dir.mkdir(parents=True, exist_ok=True)

    # Try HuggingFace Hub first (preferred — no quota limits)
    try:
        from huggingface_hub import snapshot_download
        log_info(f"Downloading UAVScenes from HuggingFace ({UAVSCENES_HF_REPO})...")
        if scene:
            log_info(f"Scene filter: {scene}")
            snapshot_download(
                repo_id=UAVSCENES_HF_REPO,
                repo_type="dataset",
                local_dir=str(uav_dir),
                allow_patterns=[f"*{scene}*"],
            )
        else:
            snapshot_download(
                repo_id=UAVSCENES_HF_REPO,
                repo_type="dataset",
                local_dir=str(uav_dir),
            )
        log_success(f"UAVScenes downloaded to {uav_dir}")
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

    if epochs:
        cmd.extend(["--epochs", str(epochs)])

    if args.resume:
        cmd.extend(["--resume-from", args.resume])

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
    rsync_cmd.extend([str(PROJECT_ROOT) + "/", f"{host}:{remote_dir}/"])

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
    p_train.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    p_train.add_argument("--batch-size", type=int, help="Override batch size")
    p_train.add_argument("--precision", choices=["fp16", "bf16"], help="Override precision")
    p_train.add_argument("--gpus", type=int, help="Number of GPUs")
    p_train.add_argument("--resume", help="Resume from checkpoint path")
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
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
