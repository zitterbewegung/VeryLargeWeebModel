# OccWorld Tokyo Training Pipeline

Fine-tune [OccWorld](https://github.com/wzzheng/OccWorld) on Tokyo PLATEAU 3D city data using Gazebo simulation.

## Training Data Options

This project supports two training data sources:

### 1. nuScenes Dataset (Recommended for Quick Start)

Real-world autonomous driving data from nuScenes. OccWorld was originally trained on this dataset, so it works out of the box.

| Pros | Cons |
|------|------|
| Real sensor data (camera, LiDAR) | Ground-level driving only |
| Proven to work with OccWorld | No aerial/drone perspectives |
| Well-documented format | Fixed environments (Boston, Singapore) |

```bash
# Setup nuScenes
./scripts/setup_training_data.sh --nuscenes

# Train on nuScenes
python train.py --config config/finetune_nuscenes.py --work-dir /workspace/checkpoints
```

### 2. Tokyo PLATEAU + Gazebo Simulation (Custom Urban Data)

Simulated data using Tokyo's official 3D city models in Gazebo. Supports drones and custom trajectories.

| Pros | Cons |
|------|------|
| Tokyo urban environment | Simulated (not real sensors) |
| Aerial drone perspectives | Requires Gazebo setup |
| Custom flight patterns | Data generation takes time |
| Scalable to all Japan | |

```bash
# Setup Gazebo + PLATEAU models
./scripts/setup_training_data.sh --gazebo

# Generate training data (1000 frames × 5 sessions = 5000 samples)
python scripts/gazebo_data_collector.py --frames 1000 --sessions 5

# Train on Tokyo data
python train.py --config config/finetune_tokyo.py --work-dir /workspace/checkpoints
```

### Combined Setup

Run both data sources:
```bash
./scripts/setup_training_data.sh --all
```

---

## Quick Start (Vast.ai - Recommended)

**Cost: ~$15 | Time: ~30 hours | GPU: A100 40GB**

```bash
# 1. Rent A100 40GB on vast.ai (~$0.50/hr)
#    - Image: vastai/pytorch
#    - Disk: 50GB+
#    - Reliability: >95%

# 2. SSH into your instance
ssh -p PORT root@ssh.vast.ai

# 3. Clone and setup
git clone https://github.com/YOUR_USERNAME/VeryLargeWeebModel.git
cd VeryLargeWeebModel
./scripts/vastai_setup.sh

# 4. Download data
./scripts/download_and_prepare_data.sh --all

# 5. Train (use screen to keep alive)
screen -S training
python train.py --config config/finetune_tokyo.py --work-dir /workspace/checkpoints
# Detach: Ctrl+A, then D

# 6. Download results (from local machine)
scp -P PORT root@ssh.vast.ai:/workspace/checkpoints/best.pth ./

# 7. DESTROY INSTANCE (stop billing!)
```

## Training Options

| Provider | GPU | $/hr | Time | Total Cost |
|----------|-----|------|------|------------|
| **Vast.ai** | A100 40GB | $0.50 | ~30 hrs | **~$15** |
| Vast.ai | A100 80GB | $0.79 | ~24 hrs | ~$19 |
| Vast.ai | RTX 3090 | $0.25 | ~48 hrs | ~$12 |
| Lambda Cloud | A100 | $1.79 | ~24 hrs | ~$43 |
| Local Mac | M1/M2/M3 | $0 | 5-7 days | $0 |

### Vast.ai (Cheapest)
```bash
./scripts/vastai_setup.sh
python train.py --config config/finetune_tokyo.py --work-dir /workspace/checkpoints
```

### Lambda Cloud (Easiest)
```bash
./scripts/lambda_setup.sh
python train.py --config config/finetune_tokyo.py --work-dir ~/checkpoints
```

### Local Mac (Free)
```bash
pip install torch torchvision tqdm scipy opencv-python
python train.py --config config/finetune_tokyo.py --work-dir ./out --batch-size 1
```

## GPU Comparison

| GPU | VRAM | Est. Time | Notes |
|-----|------|-----------|-------|
| A100 80GB | 80GB | ~24 hrs | Fastest, batch 2-4 |
| A100 40GB | 40GB | ~30 hrs | Best value, batch 1-2 |
| RTX 4090 | 24GB | ~36 hrs | Batch 1 |
| RTX 3090 | 24GB | ~48 hrs | Batch 1 |
| RTX 3080 | 10GB | ~60-80 hrs | Tight on VRAM |
| Mac M3 Max | 64GB unified | 5-7 days | CPU/MPS training |

## Project Structure

```
VeryLargeWeebModel/
├── train.py                     # Training script
├── config/
│   ├── finetune_tokyo.py        # Tokyo/Gazebo dataset config
│   └── finetune_nuscenes.py     # nuScenes dataset config
├── dataset/
│   ├── gazebo_occworld_dataset.py   # Tokyo/Gazebo dataset loader
│   └── nuscenes_occworld_dataset.py # nuScenes dataset loader
├── scripts/
│   ├── vastai_setup.sh          # Vast.ai environment setup
│   ├── lambda_setup.sh          # Lambda Cloud setup
│   ├── setup_training_data.sh   # Setup nuScenes + Gazebo data
│   ├── download_and_prepare_data.sh  # Download PLATEAU models
│   ├── gazebo_data_collector.py # Generate simulation data
│   ├── integration_test.py      # Validate pipeline
│   ├── create_dummy_data.py     # Create test data
│   └── sanity_check.sh          # Validate codebase
├── docs/
│   ├── training_guide.md        # Detailed training guide
│   ├── vastai_deployment.md     # Vast.ai deployment
│   ├── lambda_cloud_deployment.md  # Lambda deployment
│   └── ...
├── HOW_TO_TRAIN.md              # Quick training reference
├── ATTRIBUTION.md               # Data licensing info
└── README.md                    # This file
```

## Training Commands

### Basic
```bash
python train.py --config config/finetune_tokyo.py --work-dir ./checkpoints
```

### With Options
```bash
python train.py \
    --config config/finetune_tokyo.py \
    --work-dir ./checkpoints \
    --batch-size 1 \
    --lr 0.0001 \
    --epochs 50
```

### Resume Training
```bash
python train.py \
    --config config/finetune_tokyo.py \
    --work-dir ./checkpoints \
    --resume
```

## Monitoring

```bash
# GPU usage
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir ./checkpoints --port 6006

# Reattach to training
screen -r training
```

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Val Loss | 0.45 | 0.28 |
| mIoU | 0.32 | 0.47 |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Use `--batch-size 1` |
| Training too slow | Use A100 instead of consumer GPU |
| Connection dropped | Use `screen`, reattach with `screen -r` |
| NaN loss | Lower learning rate: `--lr 0.00001` |
| Import errors | Run setup script again |
| Safari certificate error | See [Jupyter SSL Certificate Setup](#jupyter-ssl-certificate-safari) below |

### Jupyter SSL Certificate (Safari)

When using Vast.ai's direct Jupyter access, Safari blocks the self-signed certificate. To fix:

1. Open the Jupyter URL in Safari → "This Connection Is Not Private" appears
2. Click **"Show Details"** → **"visit this website"** → enter password

**If that doesn't work**, add the certificate to Keychain:

1. In Safari, click the lock icon in the address bar
2. Click **"Show Certificate"**
3. Drag the certificate icon to your Desktop (creates a `.cer` file)
4. **Double-click the `.cer` file** - Keychain Access opens automatically
5. Enter your password to add the certificate
6. In Keychain Access, double-click the certificate → expand **"Trust"**
7. Set **"When using this certificate"** to **"Always Trust"**
8. Close and enter your password to confirm
9. Restart Safari and revisit the Jupyter URL

**Chrome/Firefox users**: Click "Advanced" → "Proceed anyway" when prompted.

## Sanity Check

Validate the codebase before deploying:

```bash
./scripts/sanity_check.sh
./scripts/sanity_check.sh --full    # Include dependency checks
./scripts/sanity_check.sh --fix     # Auto-fix common issues
```

## Data Attribution

This project uses Tokyo PLATEAU 3D city data, licensed under **CC BY 4.0** (commercial use allowed).

> 3D city model data provided by Ministry of Land, Infrastructure, Transport and Tourism (MLIT), Japan - Project PLATEAU.

See [ATTRIBUTION.md](ATTRIBUTION.md) for full licensing details.

## Documentation

- [HOW_TO_TRAIN.md](HOW_TO_TRAIN.md) - Quick training reference
- [docs/training_guide.md](docs/training_guide.md) - Detailed training guide
- [docs/vastai_deployment.md](docs/vastai_deployment.md) - Vast.ai deployment
- [docs/lambda_cloud_deployment.md](docs/lambda_cloud_deployment.md) - Lambda Cloud deployment
- [ATTRIBUTION.md](ATTRIBUTION.md) - Data licensing and attribution

## License

MIT License. See [ATTRIBUTION.md](ATTRIBUTION.md) for data source licenses.
