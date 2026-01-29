"""Colored logging utilities for CLI scripts."""


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color / Reset


def log_info(msg: str):
    """Print an info message in blue."""
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}", flush=True)


def log_success(msg: str):
    """Print a success message in green."""
    print(f"{Colors.GREEN}[OK]{Colors.NC} {msg}", flush=True)


def log_warn(msg: str):
    """Print a warning message in yellow."""
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}", flush=True)


def log_error(msg: str):
    """Print an error message in red."""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}", flush=True)


def log_step(msg: str):
    """Print a step header in cyan with bold."""
    print(f"\n{Colors.CYAN}==>{Colors.NC} {Colors.BOLD}{msg}{Colors.NC}", flush=True)
