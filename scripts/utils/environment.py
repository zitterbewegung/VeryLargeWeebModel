"""Cloud environment detection for GPU instances."""

import os
from dataclasses import dataclass


@dataclass
class CloudEnvironment:
    """Detected cloud environment information."""
    provider: str          # 'vastai', 'lambda', 'runpod', 'generic'
    work_dir: str          # Best working directory
    has_persistent_storage: bool


def detect_cloud_environment() -> CloudEnvironment:
    """Detect which cloud GPU provider we're running on.

    Detection logic (matches existing shell scripts):
    - Vast.ai: /workspace directory exists
    - Lambda: /etc/lambda-stack-version file exists
    - RunPod: /root/workspace or RUNPOD_POD_ID env var
    - Generic: fallback
    """
    if os.path.isdir("/workspace") and not os.environ.get("RUNPOD_POD_ID"):
        return CloudEnvironment(
            provider="vastai",
            work_dir="/workspace",
            has_persistent_storage=True,
        )

    if os.path.isfile("/etc/lambda-stack-version"):
        # Lambda persistent storage at /home/ubuntu/persistent or /home/ubuntu
        persistent = "/home/ubuntu/persistent"
        work_dir = persistent if os.path.isdir(persistent) else "/home/ubuntu"
        return CloudEnvironment(
            provider="lambda",
            work_dir=work_dir,
            has_persistent_storage=os.path.isdir(persistent),
        )

    if os.environ.get("RUNPOD_POD_ID") or os.path.isdir("/root/workspace"):
        return CloudEnvironment(
            provider="runpod",
            work_dir="/root/workspace" if os.path.isdir("/root/workspace") else "/workspace",
            has_persistent_storage=True,
        )

    # Generic Linux
    return CloudEnvironment(
        provider="generic",
        work_dir=os.getcwd(),
        has_persistent_storage=False,
    )


def work_dir_for_provider(provider: str) -> str:
    """Return the default working directory for a given provider name."""
    dirs = {
        "vastai": "/workspace",
        "lambda": "/home/ubuntu",
        "runpod": "/root/workspace",
        "generic": os.getcwd(),
    }
    return dirs.get(provider, os.getcwd())
