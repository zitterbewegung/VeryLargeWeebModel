#!/usr/bin/env python3
"""
Research Experiment Management for AerialWorld

Provides:
- Experiment tracking with Weights & Biases
- Reproducibility (seeds, config hashing, git tracking)
- Metrics logging (mIoU, VPQ, FID, etc.)
- Ablation study support
- Automatic paper-ready tables/figures

Usage:
    python scripts/experiment.py run --config config/finetune_tokyo.py --name "baseline"
    python scripts/experiment.py ablation --sweep config/ablations.yaml
    python scripts/experiment.py evaluate --checkpoint out/best.pth
    python scripts/experiment.py export --format latex
"""

import os
import sys
import json
import hashlib
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import random
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ExperimentConfig:
    """Experiment configuration with full reproducibility tracking."""

    # Identification
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Git tracking
    git_commit: str = ""
    git_branch: str = ""
    git_dirty: bool = False

    # Config tracking
    config_path: str = ""
    config_hash: str = ""

    # Hardware
    gpu_name: str = ""
    gpu_count: int = 1
    gpu_memory_gb: float = 0.0

    # Training
    batch_size: int = 1
    learning_rate: float = 1e-4
    epochs: int = 50
    precision: str = "bf16"

    # Data
    dataset: str = "plateau"
    train_samples: int = 0
    val_samples: int = 0

    # Timestamps
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self._capture_git_info()
        self._capture_gpu_info()

    def _capture_git_info(self):
        """Capture git state for reproducibility."""
        try:
            self.git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL
            ).decode().strip()[:8]

            self.git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            self.git_dirty = len(status) > 0
        except:
            pass

    def _capture_gpu_info(self):
        """Capture GPU information."""
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_name = torch.cuda.get_device_name(0)
                self.gpu_count = torch.cuda.device_count()
                self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        except:
            pass

    def hash_config(self, config_path: str) -> str:
        """Generate hash of config file for tracking changes."""
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                self.config_hash = hashlib.md5(f.read()).hexdigest()[:8]
            self.config_path = config_path
        return self.config_hash

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str):
        """Save experiment config to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load experiment config from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class ExperimentTracker:
    """
    Unified experiment tracking supporting:
    - Weights & Biases
    - TensorBoard
    - Local JSON logging
    """

    def __init__(
        self,
        config: ExperimentConfig,
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        project_name: str = "aerialworld",
        output_dir: Optional[str] = None,
    ):
        self.config = config
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.project_name = project_name

        self.output_dir = Path(output_dir or f"experiments/{config.name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history: List[Dict[str, Any]] = []
        self.wandb_run = None
        self.tb_writer = None

        self._init_tracking()

    def _init_tracking(self):
        """Initialize tracking backends."""
        # Save config
        self.config.save(self.output_dir / "config.json")

        # Weights & Biases
        if self.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.project_name,
                    name=self.config.name,
                    config=self.config.to_dict(),
                    tags=self.config.tags,
                    dir=str(self.output_dir),
                )
                print(f"[Experiment] W&B run: {wandb.run.url}")
            except ImportError:
                print("[Experiment] wandb not installed, skipping")
                self.use_wandb = False
            except Exception as e:
                print(f"[Experiment] W&B init failed: {e}")
                self.use_wandb = False

        # TensorBoard
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
                print(f"[Experiment] TensorBoard: {self.output_dir / 'tensorboard'}")
            except ImportError:
                print("[Experiment] tensorboard not installed, skipping")
                self.use_tensorboard = False

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to all backends."""
        metrics['step'] = step
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_history.append(metrics)

        # W&B
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.log(metrics, step=step)

        # TensorBoard
        if self.use_tensorboard and self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)

        # Local JSON
        with open(self.output_dir / "metrics.jsonl", 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def log_artifact(self, path: str, name: str, type: str = "model"):
        """Log artifact (model checkpoint, etc.)."""
        if self.use_wandb and self.wandb_run:
            import wandb
            artifact = wandb.Artifact(name, type=type)
            artifact.add_file(path)
            self.wandb_run.log_artifact(artifact)

    def finish(self, final_metrics: Optional[Dict[str, float]] = None):
        """Finish experiment and save summary."""
        self.config.finished_at = datetime.now().isoformat()
        self.config.save(self.output_dir / "config.json")

        # Save final summary
        summary = {
            'config': self.config.to_dict(),
            'final_metrics': final_metrics or {},
            'metrics_history': self.metrics_history,
        }
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        if self.use_wandb and self.wandb_run:
            import wandb
            if final_metrics:
                wandb.summary.update(final_metrics)
            wandb.finish()

        if self.tb_writer:
            self.tb_writer.close()


@dataclass
class EvaluationMetrics:
    """Standard evaluation metrics for occupancy prediction."""

    # Occupancy metrics
    mIoU: float = 0.0           # Mean Intersection over Union
    mAP: float = 0.0            # Mean Average Precision
    accuracy: float = 0.0       # Overall accuracy
    precision: float = 0.0      # Precision
    recall: float = 0.0         # Recall
    f1: float = 0.0             # F1 score

    # Temporal metrics
    VPQ: float = 0.0            # Video Panoptic Quality
    temporal_consistency: float = 0.0

    # Generation metrics (if applicable)
    FID: float = 0.0            # Fréchet Inception Distance

    # Per-class IoU
    class_iou: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_latex_row(self, name: str) -> str:
        """Generate LaTeX table row."""
        return f"{name} & {self.mIoU:.1f} & {self.VPQ:.1f} & {self.accuracy:.1f} \\\\"


def set_seed(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_ablation_study(
    base_config: str,
    ablation_params: Dict[str, List[Any]],
    output_dir: str = "experiments/ablations",
) -> List[Dict[str, Any]]:
    """
    Run ablation study over specified parameters.

    Args:
        base_config: Path to base configuration file
        ablation_params: Dict of parameter names to list of values to try
        output_dir: Output directory for results

    Returns:
        List of results for each ablation
    """
    results = []

    # Generate all combinations
    import itertools
    param_names = list(ablation_params.keys())
    param_values = list(ablation_params.values())

    for values in itertools.product(*param_values):
        params = dict(zip(param_names, values))

        # Create experiment name
        name_parts = [f"{k}={v}" for k, v in params.items()]
        exp_name = "ablation_" + "_".join(name_parts)

        print(f"\n{'='*60}")
        print(f"Running ablation: {exp_name}")
        print(f"{'='*60}")

        # TODO: Run training with modified params
        # This would modify the config and run training

        results.append({
            'name': exp_name,
            'params': params,
            'metrics': {},  # Would be filled after training
        })

    # Save ablation results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "ablation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def export_results_latex(
    experiments_dir: str,
    output_file: str = "paper/tables/results.tex",
):
    """Export experiment results to LaTeX tables."""

    experiments = []
    exp_dir = Path(experiments_dir)

    for config_file in exp_dir.glob("*/config.json"):
        try:
            config = ExperimentConfig.load(str(config_file))
            summary_file = config_file.parent / "summary.json"

            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                experiments.append({
                    'name': config.name,
                    'config': config,
                    'metrics': summary.get('final_metrics', {}),
                })
        except Exception as e:
            print(f"Error loading {config_file}: {e}")

    if not experiments:
        print("No experiments found")
        return

    # Generate LaTeX table
    latex = r"""
\begin{table}[t]
\centering
\caption{Experimental Results on Tokyo PLATEAU Dataset}
\label{tab:results}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{mIoU} $\uparrow$ & \textbf{VPQ} $\uparrow$ & \textbf{Acc.} $\uparrow$ & \textbf{Epochs} & \textbf{Time} \\
\midrule
"""

    for exp in experiments:
        m = exp['metrics']
        latex += f"{exp['name']} & {m.get('mIoU', 0):.1f} & {m.get('VPQ', 0):.1f} & "
        latex += f"{m.get('accuracy', 0):.1f} & {exp['config'].epochs} & -- \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex)

    print(f"Exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="AerialWorld Experiment Management")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run experiment
    run_parser = subparsers.add_parser('run', help='Run an experiment')
    run_parser.add_argument('--config', required=True, help='Config file path')
    run_parser.add_argument('--name', required=True, help='Experiment name')
    run_parser.add_argument('--description', default='', help='Experiment description')
    run_parser.add_argument('--tags', nargs='+', default=[], help='Tags')
    run_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    run_parser.add_argument('--no-wandb', action='store_true', help='Disable W&B')
    run_parser.add_argument('--output-dir', help='Output directory')

    # Ablation study
    ablation_parser = subparsers.add_parser('ablation', help='Run ablation study')
    ablation_parser.add_argument('--config', required=True, help='Base config')
    ablation_parser.add_argument('--sweep', required=True, help='Sweep config YAML')
    ablation_parser.add_argument('--output-dir', default='experiments/ablations')

    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate checkpoint')
    eval_parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    eval_parser.add_argument('--config', help='Config file')
    eval_parser.add_argument('--output', help='Output file')

    # Export
    export_parser = subparsers.add_parser('export', help='Export results')
    export_parser.add_argument('--format', choices=['latex', 'csv', 'json'], default='latex')
    export_parser.add_argument('--experiments-dir', default='experiments')
    export_parser.add_argument('--output', help='Output file')

    args = parser.parse_args()

    if args.command == 'run':
        # Create experiment config
        config = ExperimentConfig(
            name=args.name,
            description=args.description,
            tags=args.tags,
            seed=args.seed,
        )
        config.hash_config(args.config)

        # Set seed
        set_seed(config.seed, config.deterministic)

        # Initialize tracker
        tracker = ExperimentTracker(
            config=config,
            use_wandb=not args.no_wandb,
            output_dir=args.output_dir,
        )

        print(f"\nExperiment: {config.name}")
        print(f"Config hash: {config.config_hash}")
        print(f"Git: {config.git_branch}@{config.git_commit}" +
              (" (dirty)" if config.git_dirty else ""))
        print(f"Seed: {config.seed}")
        print(f"Output: {tracker.output_dir}")

        # TODO: Run actual training
        print("\nTo integrate with training, use:")
        print("  from scripts.experiment import ExperimentTracker")
        print("  tracker.log_metrics({'loss': 0.1, 'mIoU': 50.0}, step=100)")

    elif args.command == 'ablation':
        import yaml
        with open(args.sweep) as f:
            sweep_config = yaml.safe_load(f)

        run_ablation_study(
            args.config,
            sweep_config.get('parameters', {}),
            args.output_dir,
        )

    elif args.command == 'evaluate':
        print(f"Evaluating {args.checkpoint}...")
        # TODO: Run evaluation

    elif args.command == 'export':
        if args.format == 'latex':
            export_results_latex(
                args.experiments_dir,
                args.output or 'paper/tables/results.tex',
            )

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
