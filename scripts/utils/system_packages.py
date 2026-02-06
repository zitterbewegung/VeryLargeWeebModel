"""System package installation utilities."""

import os
import re
import shutil
import subprocess

from .logging import log_info, log_warn, log_error

# Only allow safe package names (alphanumeric, hyphens, dots, plus, underscores)
_SAFE_PACKAGE_RE = re.compile(r"^[a-zA-Z0-9._+\-]+$")


def _detect_package_manager():
    """Detect the system package manager.

    Returns:
        'apt' or 'yum' or None.
    """
    if shutil.which("apt-get"):
        return "apt"
    if shutil.which("yum"):
        return "yum"
    return None


def install_system_packages(packages, update_first: bool = True) -> bool:
    """Install system packages using the detected package manager.

    Args:
        packages: List of package names to install.
        update_first: Whether to run update before install.

    Returns:
        True if installation succeeded or no package manager found.
    """
    pkg_mgr = _detect_package_manager()
    if pkg_mgr is None:
        log_warn("No supported package manager found (apt/yum)")
        return False

    # Validate package names to prevent injection
    for pkg in packages:
        if not _SAFE_PACKAGE_RE.match(pkg):
            log_error(f"Invalid package name rejected: {pkg!r}")
            return False

    env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}

    try:
        if update_first and pkg_mgr == "apt":
            log_info("Updating package lists...")
            subprocess.run(
                ["apt-get", "update", "-qq"],
                env=env, check=False, capture_output=True,
            )

        if pkg_mgr == "apt":
            cmd = ["apt-get", "install", "-y", "-qq"] + packages
        else:
            cmd = ["yum", "install", "-y", "-q"] + packages

        log_info(f"Installing: {' '.join(packages)}")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        return result.returncode == 0
    except OSError as e:
        log_error(f"Failed to install packages: {e}")
        return False
