# AerialWorld (VeryLargeWeebModel)

Occupancy world models for urban aerial navigation. Fine-tune [OccWorld](https://github.com/wzzheng/OccWorld) on Tokyo PLATEAU 3D city data with Gazebo simulation for 4D occupancy forecasting.

**Paper:** [AerialWorld: Occupancy World Models for Urban Aerial Navigation](paper/main.tex)

## Key Contributions

1. First occupancy world model for aerial urban navigation
2. Data generation pipeline using Tokyo PLATEAU 3D city data (CC BY 4.0)
3. Multi-agent training (drone + rover perspectives)
4. Anti-collapse loss design for extreme sparsity (~1% occupied voxels)
5. Full open-source release

## Quick Start

```bash
# Clone
git clone https://github.com/zitterbewegung/VeryLargeWeebModel.git
cd VeryLargeWeebModel

# Option A: Cloud GPU (recommended)
python scripts/vlwm_cli.py setup        # Auto-detects Vast.ai/Lambda/RunPod
python scripts/vlwm_cli.py download     # Download data and pretrained models
python scripts/vlwm_cli.py train        # Train with auto GPU detection

# Option B: Manual
pip install torch torchvision tqdm scipy opencv-python wandb
python train.py --config config/finetune_tokyo.py --work-dir ./checkpoints
```

## Architecture

```
Input: 4 history frames                    Output: 6 future frames
[200×200×121 voxel grids]                  [200×200×121 voxel grids]
         │                                          ▲
         ▼                                          │
┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐
│  VQVAE Encoder  │─▶│   Transformer    │─▶│ VQVAE Decoder │
│  (frozen)       │  │   (trainable)    │  │ (frozen)      │
└─────────────────┘  └──────────────────┘  └───────────────┘
                            │
                     ┌──────┴──────┐
                     │ Pose Module │  ←── 13D pose (xyz + quat + vel)
                     │ (trainable) │
                     └─────────────┘
```

### Voxel Grid

| Parameter | Value |
|-----------|-------|
| Grid Size | 200 × 200 × 121 |
| Voxel Size | 0.4m × 0.4m × 1.25m |
| Range | 80m × 80m × 152m (X, Y, Z) |
| Z Range | -2m to +150m (aerial) |
| Occupancy Rate | ~0.83% |

### Loss Function

Focal Loss + Dice Loss + Mean-Matching to handle extreme class imbalance:
- **Focal Loss** (alpha=0.99, gamma=2): Focus on rare occupied voxels
- **Dice Loss**: Overlap-based, robust to imbalance
- **Mean-Matching** (weight=10): Prevents all-zero collapse

## Training Data

| Dataset | Type | Use Case |
|---------|------|----------|
| Tokyo PLATEAU + Gazebo | Simulated urban | Aerial drone navigation |
| nuScenes | Real driving | Ground vehicle baseline |
| UAVScenes | Real aerial | Drone validation |
| Mid-Air | Simulated flight | Multi-environment pretraining |

## Hardware

| GPU | VRAM | Batch Size | ~Training Time |
|-----|------|------------|----------------|
| A100 80GB | 80GB | 6-8 | ~8 hours |
| A100 40GB | 40GB | 3-4 | ~12 hours |
| RTX 4090 | 24GB | 2 | ~20 hours |
| RTX 3090 | 24GB | 1-2 | ~24 hours |

## Project Structure

```
VeryLargeWeebModel/
├── train.py                        # Main training script
├── train_6dof.py                   # 6DoF training variant
├── config/                         # Training configurations
│   ├── finetune_tokyo.py           # Tokyo PLATEAU (primary)
│   ├── finetune_nuscenes.py        # nuScenes baseline
│   ├── finetune_uavscenes.py       # UAVScenes aerial
│   └── finetune_6dof.py            # 6DoF pose model
├── dataset/                        # Dataset loaders
│   ├── gazebo_occworld_dataset.py  # Tokyo Gazebo loader
│   ├── uavscenes_dataset.py        # UAVScenes loader
│   ├── midair_dataset.py           # Mid-Air loader
│   └── nuscenes_*.py               # nuScenes loaders
├── models/                         # Model architectures
│   └── occworld_6dof.py            # 6DoF OccWorld model
├── scripts/                        # Utilities and CLI
│   ├── vlwm_cli.py                 # Unified CLI (setup/download/train/deploy)
│   └── utils/                      # Shared utilities
├── tests/                          # Test suite
├── paper/                          # LaTeX paper
└── docs/                           # Detailed documentation
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/TRAINING.md](docs/TRAINING.md) | Complete training guide (commands, configs, troubleshooting) |
| [docs/vastai_deployment.md](docs/vastai_deployment.md) | Vast.ai cloud deployment |
| [docs/lambda_cloud_deployment.md](docs/lambda_cloud_deployment.md) | Lambda Cloud deployment |
| [TRAINING_LOG.md](TRAINING_LOG.md) | Development notes (loss collapse fix, W&B setup) |
| [ATTRIBUTION.md](ATTRIBUTION.md) | Data licensing and citations |
| [paper/](paper/) | Academic paper source |

## CLI Tool

```bash
python scripts/vlwm_cli.py <command> [options]

Commands:
  setup       Environment setup (auto-detects cloud provider)
  download    Download data and pretrained models
  train       Run training with GPU auto-detection
  deploy      Deploy to remote GPU instance
  sanity      Pre-flight checks (syntax, configs, data)
  info        Show GPU, data, and environment info
```

## Tests

```bash
python -m pytest tests/ -v --tb=short
```

## Citation

```bibtex
@article{aerialworld2025,
  title={AerialWorld: Occupancy World Models for Urban Aerial Navigation},
  author={Anonymous},
  journal={arXiv preprint},
  year={2025},
  url={https://github.com/zitterbewegung/VeryLargeWeebModel}
}
```

## License

MIT License. See [ATTRIBUTION.md](ATTRIBUTION.md) for data source licenses.
