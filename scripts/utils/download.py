"""Download utilities with fast-download fallback chain."""

import os
import shutil
import subprocess
from typing import Optional

from .logging import log_info, log_success, log_warn, log_error


def _run_download(cmd: list[str], description: str) -> bool:
    """Run a download command, return True on success."""
    try:
        result = subprocess.run(cmd, timeout=3600)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def fast_download(url: str, output: str, description: str = "file") -> bool:
    """Download a file using the fastest available tool.

    Fallback chain: aria2c -> axel -> curl -> wget

    Args:
        url: URL to download.
        output: Output file path.
        description: Human-readable description for logging.

    Returns:
        True if download succeeded.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    log_info(f"Downloading {description}...")
    log_info(f"URL: {url}")
    log_info(f"Output: {output}")

    # Try aria2c (fastest - 16 parallel connections)
    if shutil.which("aria2c"):
        log_info("Using aria2c (16 connections)...")
        if _run_download(
            ["aria2c", "-x", "16", "-s", "16", "--file-allocation=none",
             "-d", os.path.dirname(os.path.abspath(output)),
             "-o", os.path.basename(output), url],
            description,
        ):
            log_success(f"Download complete (aria2c): {description}")
            return True
        log_warn("aria2c failed, trying fallback...")
        _remove_partial(output)

    # Try axel
    if shutil.which("axel"):
        log_info("Using axel (16 connections)...")
        if _run_download(["axel", "-n", "16", "-o", output, url], description):
            log_success(f"Download complete (axel): {description}")
            return True
        log_warn("axel failed, trying fallback...")
        _remove_partial(output)

    # Try curl
    if shutil.which("curl"):
        log_info("Using curl...")
        if _run_download(
            ["curl", "-L", "-C", "-", "-o", output, url], description
        ):
            log_success(f"Download complete (curl): {description}")
            return True
        log_warn("curl failed, trying fallback...")
        _remove_partial(output)

    # Try wget
    if shutil.which("wget"):
        log_info("Using wget...")
        if _run_download(["wget", "-c", "-O", output, url], description):
            log_success(f"Download complete (wget): {description}")
            return True
        _remove_partial(output)

    log_error(f"All download methods failed for {description}")
    return False


def _remove_partial(path: str) -> None:
    """Remove a partially downloaded file."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def verify_download(path: str, min_size: int = 1024) -> bool:
    """Verify a downloaded file exists and meets minimum size.

    Args:
        path: File path to check.
        min_size: Minimum file size in bytes.

    Returns:
        True if file exists and is large enough.
    """
    if not os.path.isfile(path):
        return False
    return os.path.getsize(path) >= min_size


def available_download_tool() -> Optional[str]:
    """Return the name of the best available download tool, or None."""
    for tool in ["aria2c", "axel", "curl", "wget"]:
        if shutil.which(tool):
            return tool
    return None
