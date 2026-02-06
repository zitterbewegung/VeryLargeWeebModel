"""GPU detection, batch size auto-sizing, and precision selection."""

import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInfo:
    """Information about a detected GPU."""
    name: str
    memory_mb: int
    count: int
    index: int = 0


def detect_gpu_info() -> Optional[GPUInfo]:
    """Detect GPU info via nvidia-smi.

    Returns:
        GPUInfo if a GPU is detected, None otherwise.
    """
    if not shutil.which("nvidia-smi"):
        return None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None

        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if not lines:
            return None

        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) < 2:
            return None

        return GPUInfo(
            name=parts[0],
            memory_mb=int(parts[1]),
            count=len(lines),
        )
    except (subprocess.TimeoutExpired, ValueError, IndexError, OSError):
        return None


def gpu_tier(gpu: Optional[GPUInfo]) -> str:
    """Classify GPU into a performance tier.

    Returns one of: 'high', 'mid-high', 'mid', 'low', 'none'.
    """
    if gpu is None:
        return "none"
    mem = gpu.memory_mb
    if mem >= 70000:
        return "high"
    if mem >= 35000:
        return "mid-high"
    if mem >= 20000:
        return "mid"
    if mem >= 10000:
        return "low"
    return "none"


def auto_batch_size(gpu: Optional[GPUInfo]) -> int:
    """Select batch size based on GPU memory.

    Returns:
        Recommended batch size (1-12).
    """
    tier = gpu_tier(gpu)
    return {
        "high": 12,      # A100-80GB, H100-80GB
        "mid-high": 6,   # A100-40GB, A6000
        "mid": 4,        # RTX 3090/4090, A5000
        "low": 2,        # RTX 3080/4080
        "none": 1,
    }[tier]


def select_precision(gpu: Optional[GPUInfo]) -> str:
    """Select training precision based on GPU architecture.

    Returns 'bf16' for Ampere/Hopper GPUs, 'fp16' otherwise.
    """
    if gpu is None:
        return "fp16"

    name = gpu.name.lower()
    # Ampere (A100, A6000, RTX 30xx) and Hopper (H100) support bf16
    bf16_patterns = ["a100", "a6000", "a5000", "h100", "h200", "rtx 30", "rtx 40",
                     "l40", "l4"]
    for pattern in bf16_patterns:
        if pattern in name:
            return "bf16"
    return "fp16"
