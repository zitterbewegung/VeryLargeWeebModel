"""Colored logging utilities for CLI scripts."""


class Colors:
    """ANSI color codes for terminal output."""
    RED: str = '\033[0;31m'
    GREEN: str = '\033[0;32m'
    YELLOW: str = '\033[1;33m'
    BLUE: str = '\033[0;34m'
    CYAN: str = '\033[0;36m'
    BOLD: str = '\033[1m'
    NC: str = '\033[0m'  # No Color / Reset


def log_info(msg: str) -> None:
    """Print an info message in blue.

    Args:
        msg: Message to print
    """
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}", flush=True)


def log_success(msg: str) -> None:
    """Print a success message in green.

    Args:
        msg: Message to print
    """
    print(f"{Colors.GREEN}[OK]{Colors.NC} {msg}", flush=True)


def log_warn(msg: str) -> None:
    """Print a warning message in yellow.

    Args:
        msg: Message to print
    """
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}", flush=True)


def log_error(msg: str) -> None:
    """Print an error message in red.

    Args:
        msg: Message to print
    """
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}", flush=True)


def log_step(msg: str) -> None:
    """Print a step header in cyan with bold.

    Args:
        msg: Message to print
    """
    print(f"\n{Colors.CYAN}==>{Colors.NC} {Colors.BOLD}{msg}{Colors.NC}", flush=True)
