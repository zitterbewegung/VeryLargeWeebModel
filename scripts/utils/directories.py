"""Directory utilities for data collection sessions."""

import os
from typing import Dict, Tuple


def create_session_dirs(
    base_dir: str,
    session_name: str,
    subdirs: Tuple[str, ...] = ('images', 'lidar', 'poses', 'occupancy')
) -> Dict[str, str]:
    """Create directory structure for a data collection session.

    Args:
        base_dir: Base directory for sessions
        session_name: Name of this recording session
        subdirs: Tuple of subdirectory names to create

    Returns:
        Dict mapping directory names to their full paths
    """
    session_dir: str = os.path.join(str(base_dir), session_name)
    dirs: Dict[str, str] = {'root': session_dir}

    for subdir in subdirs:
        dirs[subdir] = os.path.join(session_dir, subdir)

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    return dirs
