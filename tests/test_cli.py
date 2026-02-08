"""Tests for the unified CLI and utility modules."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Add scripts/ to path for imports
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.gpu import GPUInfo, detect_gpu_info, gpu_tier, auto_batch_size, select_precision
from utils.environment import (
    CloudEnvironment, detect_cloud_environment, work_dir_for_provider,
)
from utils.download import fast_download, verify_download, available_download_tool
from utils.system_packages import _detect_package_manager


# =============================================================================
# GPU utility tests
# =============================================================================

class TestGPUInfo:
    """Tests for GPU detection and configuration."""

    def test_gpu_info_dataclass(self):
        """GPUInfo should store name, memory, and count."""
        info = GPUInfo(name="NVIDIA A100-SXM4-80GB", memory_mb=81920, count=4)
        assert info.name == "NVIDIA A100-SXM4-80GB"
        assert info.memory_mb == 81920
        assert info.count == 4

    @patch("utils.gpu.shutil.which", return_value=None)
    def test_detect_gpu_no_nvidia_smi(self, mock_which):
        """Should return None when nvidia-smi is not found."""
        assert detect_gpu_info() is None

    @patch("utils.gpu.subprocess.run")
    @patch("utils.gpu.shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_detect_gpu_info_a100(self, mock_which, mock_run):
        """Should parse A100 GPU info from nvidia-smi output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA A100-SXM4-80GB, 81920\nNVIDIA A100-SXM4-80GB, 81920\nNVIDIA A100-SXM4-80GB, 81920\nNVIDIA A100-SXM4-80GB, 81920\n",
        )
        info = detect_gpu_info()
        assert info is not None
        assert info.name == "NVIDIA A100-SXM4-80GB"
        assert info.memory_mb == 81920
        assert info.count == 4

    @patch("utils.gpu.subprocess.run")
    @patch("utils.gpu.shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_detect_gpu_info_rtx3090(self, mock_which, mock_run):
        """Should parse RTX 3090 GPU info."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 3090, 24576\n",
        )
        info = detect_gpu_info()
        assert info is not None
        assert info.name == "NVIDIA GeForce RTX 3090"
        assert info.memory_mb == 24576
        assert info.count == 1

    @patch("utils.gpu.subprocess.run")
    @patch("utils.gpu.shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_detect_gpu_info_failure(self, mock_which, mock_run):
        """Should return None when nvidia-smi fails."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert detect_gpu_info() is None

    @patch("utils.gpu.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 10))
    @patch("utils.gpu.shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_detect_gpu_info_timeout(self, mock_which, mock_run):
        """Should return None on timeout."""
        assert detect_gpu_info() is None


class TestGPUTier:
    """Tests for GPU tier classification."""

    def test_tier_high(self):
        gpu = GPUInfo(name="A100-80GB", memory_mb=81920, count=1)
        assert gpu_tier(gpu) == "high"

    def test_tier_mid_high(self):
        gpu = GPUInfo(name="A100-40GB", memory_mb=40960, count=1)
        assert gpu_tier(gpu) == "mid-high"

    def test_tier_mid(self):
        gpu = GPUInfo(name="RTX 3090", memory_mb=24576, count=1)
        assert gpu_tier(gpu) == "mid"

    def test_tier_low(self):
        gpu = GPUInfo(name="RTX 3080", memory_mb=12288, count=1)
        assert gpu_tier(gpu) == "low"

    def test_tier_none_small_gpu(self):
        gpu = GPUInfo(name="GTX 1650", memory_mb=4096, count=1)
        assert gpu_tier(gpu) == "none"

    def test_tier_none_no_gpu(self):
        assert gpu_tier(None) == "none"


class TestAutoBatchSize:
    """Tests for automatic batch size selection."""

    def test_batch_size_80gb(self):
        gpu = GPUInfo(name="A100-80GB", memory_mb=81920, count=1)
        assert auto_batch_size(gpu) == 12

    def test_batch_size_40gb(self):
        gpu = GPUInfo(name="A100-40GB", memory_mb=40960, count=1)
        assert auto_batch_size(gpu) == 6

    def test_batch_size_24gb(self):
        gpu = GPUInfo(name="RTX 3090", memory_mb=24576, count=1)
        assert auto_batch_size(gpu) == 4

    def test_batch_size_10gb(self):
        gpu = GPUInfo(name="RTX 3080", memory_mb=12288, count=1)
        assert auto_batch_size(gpu) == 2

    def test_batch_size_no_gpu(self):
        assert auto_batch_size(None) == 1


class TestPrecisionSelection:
    """Tests for training precision selection."""

    def test_precision_a100(self):
        gpu = GPUInfo(name="NVIDIA A100-SXM4-80GB", memory_mb=81920, count=1)
        assert select_precision(gpu) == "bf16"

    def test_precision_h100(self):
        gpu = GPUInfo(name="NVIDIA H100", memory_mb=81920, count=1)
        assert select_precision(gpu) == "bf16"

    def test_precision_rtx3090(self):
        gpu = GPUInfo(name="NVIDIA GeForce RTX 3090", memory_mb=24576, count=1)
        assert select_precision(gpu) == "bf16"

    def test_precision_rtx4090(self):
        gpu = GPUInfo(name="NVIDIA GeForce RTX 4090", memory_mb=24576, count=1)
        assert select_precision(gpu) == "bf16"

    def test_precision_v100(self):
        gpu = GPUInfo(name="Tesla V100-SXM2-16GB", memory_mb=16384, count=1)
        assert select_precision(gpu) == "fp16"

    def test_precision_unknown(self):
        gpu = GPUInfo(name="Unknown GPU", memory_mb=8192, count=1)
        assert select_precision(gpu) == "fp16"

    def test_precision_no_gpu(self):
        assert select_precision(None) == "fp16"


# =============================================================================
# Environment utility tests
# =============================================================================

class TestCloudEnvironment:
    """Tests for cloud environment detection."""

    @patch.dict(os.environ, {}, clear=False)
    @patch("utils.environment.os.path.isdir")
    @patch("utils.environment.os.path.isfile")
    def test_detect_vastai(self, mock_isfile, mock_isdir):
        """Should detect Vast.ai when /workspace exists."""
        # Remove RUNPOD_POD_ID if present
        env_copy = os.environ.copy()
        env_copy.pop("RUNPOD_POD_ID", None)
        mock_isdir.side_effect = lambda p: p == "/workspace"
        mock_isfile.return_value = False
        with patch.dict(os.environ, env_copy, clear=True):
            env = detect_cloud_environment()
        assert env.provider == "vastai"
        assert env.work_dir == "/workspace"

    @patch.dict(os.environ, {}, clear=True)
    @patch("utils.environment.os.path.isdir")
    @patch("utils.environment.os.path.isfile")
    def test_detect_lambda(self, mock_isfile, mock_isdir):
        """Should detect Lambda when /etc/lambda-stack-version exists."""
        mock_isdir.side_effect = lambda p: p in ("/home/ubuntu/persistent", "/home/ubuntu")
        mock_isfile.side_effect = lambda p: p == "/etc/lambda-stack-version"
        env = detect_cloud_environment()
        assert env.provider == "lambda"
        assert env.work_dir == "/home/ubuntu/persistent"

    @patch.dict(os.environ, {"RUNPOD_POD_ID": "abc123"}, clear=False)
    @patch("utils.environment.os.path.isdir")
    @patch("utils.environment.os.path.isfile")
    def test_detect_runpod(self, mock_isfile, mock_isdir):
        """Should detect RunPod via RUNPOD_POD_ID env var."""
        mock_isdir.side_effect = lambda p: p in ("/workspace", "/root/workspace")
        mock_isfile.return_value = False
        env = detect_cloud_environment()
        assert env.provider == "runpod"

    @patch.dict(os.environ, {}, clear=True)
    @patch("utils.environment.os.path.isdir", return_value=False)
    @patch("utils.environment.os.path.isfile", return_value=False)
    def test_detect_generic(self, mock_isfile, mock_isdir):
        """Should fall back to generic."""
        env = detect_cloud_environment()
        assert env.provider == "generic"

    def test_work_dir_for_provider(self):
        """Should return correct work dirs for known providers."""
        assert work_dir_for_provider("vastai") == "/workspace"
        assert work_dir_for_provider("lambda") == "/home/ubuntu"
        assert work_dir_for_provider("runpod") == "/root/workspace"
        assert work_dir_for_provider("generic") == os.getcwd()
        assert work_dir_for_provider("unknown") == os.getcwd()


# =============================================================================
# Download utility tests
# =============================================================================

class TestDownloadUtils:
    """Tests for download utilities."""

    @patch("utils.download.shutil.which")
    def test_available_download_tool_aria2c(self, mock_which):
        """Should prefer aria2c when available."""
        mock_which.side_effect = lambda t: "/usr/bin/aria2c" if t == "aria2c" else None
        assert available_download_tool() == "aria2c"

    @patch("utils.download.shutil.which")
    def test_available_download_tool_curl_fallback(self, mock_which):
        """Should fall back to curl when aria2c/axel unavailable."""
        def which_side_effect(tool):
            return "/usr/bin/curl" if tool == "curl" else None
        mock_which.side_effect = which_side_effect
        assert available_download_tool() == "curl"

    @patch("utils.download.shutil.which", return_value=None)
    def test_available_download_tool_none(self, mock_which):
        """Should return None when no tools available."""
        assert available_download_tool() is None

    def test_verify_download_existing(self, tmp_path):
        """Should verify an existing file that meets minimum size."""
        f = tmp_path / "test.zip"
        f.write_bytes(b"x" * 2048)
        assert verify_download(str(f), min_size=1024) is True

    def test_verify_download_too_small(self, tmp_path):
        """Should reject files smaller than minimum size."""
        f = tmp_path / "test.zip"
        f.write_bytes(b"x" * 100)
        assert verify_download(str(f), min_size=1024) is False

    def test_verify_download_missing(self):
        """Should reject missing files."""
        assert verify_download("/nonexistent/file.zip") is False

    @patch("utils.download.subprocess.run")
    @patch("utils.download.shutil.which")
    def test_fast_download_aria2c(self, mock_which, mock_run, tmp_path):
        """Should use aria2c when available."""
        mock_which.return_value = "/usr/bin/aria2c"
        mock_run.return_value = MagicMock(returncode=0)
        output = str(tmp_path / "out.zip")
        result = fast_download("https://example.com/file.zip", output, "test file")
        assert result is True
        # Verify aria2c was called
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "aria2c"

    @patch("utils.download.subprocess.run")
    @patch("utils.download.shutil.which")
    def test_fast_download_fallback_to_curl(self, mock_which, mock_run, tmp_path):
        """Should fall back to curl when aria2c fails."""
        def which_side_effect(tool):
            if tool in ("aria2c", "curl"):
                return f"/usr/bin/{tool}"
            return None
        mock_which.side_effect = which_side_effect

        # aria2c fails, curl succeeds
        call_count = [0]

        def run_side_effect(cmd, **kwargs):
            call_count[0] += 1
            if cmd[0] == "aria2c":
                return MagicMock(returncode=1)
            return MagicMock(returncode=0)

        mock_run.side_effect = run_side_effect
        output = str(tmp_path / "out.zip")
        result = fast_download("https://example.com/file.zip", output, "test file")
        assert result is True
        assert call_count[0] >= 2  # aria2c tried, then curl


# =============================================================================
# System packages tests
# =============================================================================

class TestSystemPackages:
    """Tests for system package utilities."""

    @patch("utils.system_packages.shutil.which")
    def test_detect_apt(self, mock_which):
        mock_which.side_effect = lambda t: "/usr/bin/apt-get" if t == "apt-get" else None
        assert _detect_package_manager() == "apt"

    @patch("utils.system_packages.shutil.which")
    def test_detect_yum(self, mock_which):
        mock_which.side_effect = lambda t: "/usr/bin/yum" if t == "yum" else None
        assert _detect_package_manager() == "yum"

    @patch("utils.system_packages.shutil.which", return_value=None)
    def test_detect_none(self, mock_which):
        assert _detect_package_manager() is None


# =============================================================================
# CLI integration tests
# =============================================================================

class TestCLIHelp:
    """Test CLI help and argument parsing."""

    def test_main_help(self):
        """Main --help should exit 0."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"), "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "vlwm_cli" in result.stdout

    def test_no_args_shows_help(self):
        """No args should show help and exit 1."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py")],
            capture_output=True, text=True,
        )
        assert result.returncode == 1

    @pytest.mark.parametrize("subcmd", ["setup", "download", "train", "deploy", "sanity", "info", "pack"])
    def test_subcommand_help(self, subcmd):
        """Each subcommand should have --help."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"), subcmd, "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0


class TestCLICommands:
    """Test CLI subcommands in dry-run / safe mode."""

    def test_info_subcommand(self):
        """info should run without error."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"), "info"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Provider" in result.stdout

    def test_sanity_quick(self):
        """sanity --quick should run and report results."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"), "sanity", "--quick"],
            capture_output=True, text=True,
        )
        # May pass or fail depending on project state, but should not crash
        assert result.returncode in (0, 1)
        assert "Results" in result.stdout

    def test_setup_dry_run(self):
        """setup --dry-run should not actually install anything."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"), "setup", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout

    def test_train_dry_run(self):
        """train --dry-run should show config without training."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"), "train", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout
        assert "Batch size" in result.stdout

    def test_download_dry_run(self):
        """download --dry-run --all should show what would download."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"),
             "download", "--dry-run", "--all"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout

    def test_deploy_dry_run(self):
        """deploy --dry-run should show what would deploy."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"),
             "deploy", "--dry-run", "--host", "test@example.com"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout


class TestCLISanity:
    """Test the sanity subcommand in detail."""

    def test_python_syntax_validation(self):
        """Full sanity should validate Python syntax."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"), "sanity"],
            capture_output=True, text=True,
        )
        assert "Python Syntax" in result.stdout

    def test_shell_syntax_validation(self):
        """Full sanity should validate shell script syntax."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"), "sanity"],
            capture_output=True, text=True,
        )
        assert "Shell Syntax" in result.stdout

    def test_placeholder_detection(self):
        """Full sanity should check for placeholders."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"), "sanity"],
            capture_output=True, text=True,
        )
        assert "Placeholder" in result.stdout

    def test_config_validation(self):
        """Full sanity should validate config files."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"), "sanity"],
            capture_output=True, text=True,
        )
        assert "Config" in result.stdout


class TestCLIMainFunction:
    """Test the main() function directly."""

    def test_main_returns_int(self):
        """main() should return an integer exit code."""
        from vlwm_cli import main
        result = main(["info"])
        assert isinstance(result, int)
        assert result == 0

    def test_main_no_args_returns_1(self):
        """main() with no args should return 1."""
        from vlwm_cli import main
        result = main([])
        assert result == 1

    def test_main_help_exits(self):
        """main(['--help']) should raise SystemExit(0)."""
        from vlwm_cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_build_parser(self):
        """build_parser() should return a valid parser."""
        from vlwm_cli import build_parser
        parser = build_parser()
        # Parse known subcommands
        args = parser.parse_args(["train", "--dry-run", "--epochs", "10"])
        assert args.command == "train"
        assert args.dry_run is True
        assert args.epochs == 10

    def test_build_parser_train_uncertainty_weight(self):
        """train parser should accept --uncertainty-weight."""
        from vlwm_cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["train", "--uncertainty-weight", "0.05"])
        assert args.command == "train"
        assert args.uncertainty_weight == 0.05

    def test_build_parser_pack(self):
        """build_parser() should parse pack subcommand."""
        from vlwm_cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["pack", "uavscenes", "--dry-run"])
        assert args.command == "pack"
        assert args.dataset == "uavscenes"
        assert args.dry_run is True
        assert args.upload is False
        assert args.download is False


# =============================================================================
# Pack subcommand tests
# =============================================================================

class TestPackSubcommand:
    """Test CLI pack subcommand."""

    def test_pack_help(self):
        """pack --help should exit 0."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"), "pack", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "dataset" in result.stdout
        assert "--upload" in result.stdout
        assert "--download" in result.stdout

    def test_pack_dry_run_compress(self, tmp_path):
        """pack --dry-run should show compress plan for a temp dataset."""
        # Create a temp dataset directory with a file
        dataset_dir = tmp_path / "test_dataset"
        dataset_dir.mkdir()
        (dataset_dir / "sample.txt").write_text("hello world")

        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"),
             "pack", str(dataset_dir), "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout
        assert "1 files" in result.stdout

    def test_pack_dry_run_upload(self, tmp_path):
        """pack --upload --dry-run should show S3 path."""
        dataset_dir = tmp_path / "mydata"
        dataset_dir.mkdir()
        (dataset_dir / "file.bin").write_bytes(b"\x00" * 100)

        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"),
             "pack", str(dataset_dir), "--upload", "--dry-run"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout

    def test_pack_dry_run_download(self):
        """pack --download --dry-run should show S3 download plan."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"),
             "pack", "uavscenes", "--download", "--dry-run", "--force"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout
        assert "s3://" in result.stdout

    def test_pack_mutually_exclusive_flags(self):
        """--upload and --download together should be an argparse error."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"),
             "pack", "uavscenes", "--upload", "--download"],
            capture_output=True, text=True,
        )
        assert result.returncode == 2  # argparse error

    def test_pack_nonexistent_dataset(self, tmp_path):
        """Compressing a nonexistent directory should return 1."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"),
             "pack", str(tmp_path / "does_not_exist"), "--force"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1

    def test_pack_known_dataset_names(self):
        """Known dataset names should be listed in help."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "vlwm_cli.py"), "pack", "--help"],
            capture_output=True, text=True,
        )
        assert "uavscenes" in result.stdout
        assert "tokyo_gazebo" in result.stdout


class TestPackCompressDecompress:
    """Integration test: compress and decompress a small dataset."""

    def test_roundtrip(self, tmp_path):
        """Compress a temp dir, decompress to new location, verify contents match."""
        from vlwm_cli import _compress_dataset, _decompress_archive

        # Create source dataset
        source = tmp_path / "source_dataset"
        source.mkdir()
        (source / "file1.txt").write_text("hello")
        sub = source / "subdir"
        sub.mkdir()
        (sub / "file2.bin").write_bytes(b"\x00\x01\x02\x03")

        # Compress
        archive = str(tmp_path / "source_dataset.tar.xz")
        assert _compress_dataset(str(source), archive, compression_level=0, dry_run=False) is True
        assert os.path.isfile(archive)

        # Decompress to new location
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        assert _decompress_archive(archive, str(extract_dir), dry_run=False) is True

        # Verify contents
        extracted_root = extract_dir / "source_dataset"
        assert (extracted_root / "file1.txt").read_text() == "hello"
        assert (extracted_root / "subdir" / "file2.bin").read_bytes() == b"\x00\x01\x02\x03"

    def test_compress_dry_run_no_file(self, tmp_path):
        """Dry-run compress should not create an archive file."""
        from vlwm_cli import _compress_dataset

        source = tmp_path / "data"
        source.mkdir()
        (source / "f.txt").write_text("x")

        archive = str(tmp_path / "data.tar.xz")
        assert _compress_dataset(str(source), archive, compression_level=0, dry_run=True) is True
        assert not os.path.exists(archive)

    def test_decompress_dry_run_no_extract(self, tmp_path):
        """Dry-run decompress should not extract anything."""
        from vlwm_cli import _compress_dataset, _decompress_archive

        source = tmp_path / "data2"
        source.mkdir()
        (source / "f.txt").write_text("x")

        archive = str(tmp_path / "data2.tar.xz")
        _compress_dataset(str(source), archive, compression_level=0, dry_run=False)

        extract_dir = tmp_path / "output"
        extract_dir.mkdir()
        assert _decompress_archive(archive, str(extract_dir), dry_run=True) is True
        # Nothing should be extracted
        assert list(extract_dir.iterdir()) == []
