---
marp: true
theme: default
paginate: true
backgroundColor: #1a1a2e
color: #eaeaea
style: |
  section {
    font-family: 'Inter', 'Helvetica Neue', sans-serif;
  }
  h1 {
    color: #00d9ff;
  }
  h2 {
    color: #9d4edd;
  }
  code {
    background-color: #2d2d44;
  }
  a {
    color: #00d9ff;
  }
  img {
    background-color: transparent;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
---

<!-- _class: lead -->
<!-- _backgroundColor: #0d0d1a -->

# Training OccWorld on Tokyo 3D City Data

## A Complete Pipeline for Custom World Models

![bg right:40% 80%](https://raw.githubusercontent.com/wzzheng/OccWorld/main/assets/teaser.png)

**Lambda Cloud GPU Tutorial**

---

# Agenda

1. **What is OccWorld?** — 3D world models for autonomous systems
2. **Tokyo PLATEAU** — Japan's open 3D city data
3. **Our Pipeline** — Simulation to training
4. **Cloud Training** — Lambda Cloud walkthrough
5. **Results** — What we achieved
6. **Get Started** — Code and resources

---

<!-- _class: lead -->

# Part 1
## What is OccWorld?

---

# The Challenge: Predicting the Future

![bg right:45% 90%](https://www.nuscenes.org/public/images/data.png)

### Autonomous systems need to anticipate

- Where will that pedestrian go?
- Will the car ahead brake?
- Is this path safe in 3 seconds?

### Reactive systems are too slow

- By the time you see the problem...
- It may be too late to react

---

# World Models: A Paradigm Shift

<div class="columns">
<div>

### Traditional Approach
```
Sensors → Detect → React
         (frame-by-frame)
```

### World Model Approach
```
Sensors → Learn Dynamics → Predict → Plan
          (understand physics)
```

</div>
<div>

### Benefits
- **Anticipation** over reaction
- **Safety** through foresight
- **Planning** with predictions
- **Simulation** for testing

</div>
</div>

---

# OccWorld: 3D Occupancy World Model

![bg right:55% 95%](https://raw.githubusercontent.com/wzzheng/OccWorld/main/assets/framework.jpg)

### ECCV 2024 | Tsinghua University

**Key Idea:** Represent the world as voxels, predict future voxels

1. **VQVAE** encodes 3D grids to tokens
2. **Transformer** models temporal dynamics
3. **Decoder** generates future predictions

---

# How OccWorld Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   HISTORY (t-3 to t)              FUTURE (t+1 to t+6)               │
│                                                                      │
│   ┌───┐ ┌───┐ ┌───┐ ┌───┐       ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐│
│   │ ░ │ │ ░ │ │ ░ │ │ ░ │  ───▶ │ ? │ │ ? │ │ ? │ │ ? │ │ ? │ │ ? ││
│   └───┘ └───┘ └───┘ └───┘       └───┘ └───┘ └───┘ └───┘ └───┘ └───┘│
│     ▲                                   │                            │
│     │     VQVAE + Transformer           │                            │
│     │         (GPT-style)               ▼                            │
│   INPUT                              OUTPUT                          │
│   4 frames (~2 sec)               6 frames (~3 sec)                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

# OccWorld Performance

| Metric | OccWorld | Previous SOTA |
|--------|----------|---------------|
| **mIoU** (4D Occupancy) | 25.1 | 21.4 |
| **VPQ** (Video Quality) | 32.8 | 28.2 |
| **ADE** (Trajectory) | 1.23m | 1.56m |

### But there's a limitation...

> Trained on **nuScenes** (Boston & Singapore)
>
> What about **other cities**? Other environments?

---

<!-- _class: lead -->

# Part 2
## Tokyo PLATEAU

---

# Project PLATEAU: Japan's Digital Twin

![bg right:50% 90%](https://www.mlit.go.jp/plateau/assets/img/top/about_img.jpg)

### Ministry of Land, Infrastructure, Transport and Tourism

- **250+ cities** in 3D
- **Building-level** detail (LOD2)
- **Semantic** annotations
- **Completely free** (even commercial)

---

# Tokyo 23 Wards Dataset

<div class="columns">
<div>

### Coverage
- 621 km² of urban area
- All 23 special wards
- Dense downtown + residential

### Formats Available
- CityGML (semantic)
- OBJ / FBX (meshes)
- 3D Tiles (web)

</div>
<div>

### Data Specifications

| Attribute | Value |
|-----------|-------|
| Buildings | ~900,000 |
| LOD | 2 (with roofs) |
| Size | ~2.1 GB (OBJ) |
| License | CC BY 4.0 |

</div>
</div>

---

# Why PLATEAU for Training?

<div class="columns">
<div>

### ✅ Advantages

- **Real geometry** (not procedural)
- **Diverse areas** (urban, suburban)
- **Semantic labels** (building types)
- **Free & legal** to use
- **Active maintenance** by MLIT

</div>
<div>

### 🎯 Use Cases

- Autonomous vehicle training
- Drone navigation
- Urban robotics
- Digital twin research
- Game development

</div>
</div>

---

<!-- _class: lead -->

# Part 3
## Our Pipeline

---

# Pipeline Overview

```
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│                │      │                │      │                │
│    PLATEAU     │─────▶│    GAZEBO      │─────▶│   OCCWORLD     │
│   3D Models    │      │  Simulation    │      │   Training     │
│                │      │                │      │                │
└────────────────┘      └────────────────┘      └────────────────┘
       │                       │                       │
       ▼                       ▼                       ▼
   ┌────────┐            ┌──────────┐           ┌──────────┐
   │OBJ/FBX │            │ Sensors  │           │ Trained  │
   │Meshes  │            │ + Poses  │           │  Model   │
   └────────┘            └──────────┘           └──────────┘
```

---

# Step 1: PLATEAU → Gazebo

### Convert OBJ meshes to SDF models

```python
def convert_plateau_to_gazebo(obj_path, output_dir):
    # Load and simplify mesh
    mesh = trimesh.load(obj_path)
    mesh = mesh.simplify_quadric_decimation(target_faces=10000)

    # Generate Gazebo SDF
    sdf = create_sdf_model(mesh, name="tokyo_building")
    sdf.save(output_dir / "model.sdf")
```

### Result: Gazebo world with Tokyo buildings

```bash
./scripts/download_and_prepare_data.sh --plateau
# Downloads ~2GB, converts to Gazebo format
```

---

# Step 2: Simulate & Record

![bg right:40% 90%](https://gazebosim.org/assets/images/gazebo_horz_pos.svg)

### Spawn Vehicles
- Drones (aerial data)
- Rovers (ground data)

### Record Sensors
- 6 RGB cameras
- LiDAR (360°)
- 6-DoF poses
- Ground truth occupancy

```bash
./scripts/launch_occworld_simulation.sh \
    --drones 1 --rovers 1 --record
```

---

# Data Collection Missions

<div class="columns">
<div>

### Survey Pattern
```
    ──────────────
    │            │
    │  ┌──────┐  │
    │  │      │  │
    │  │      │  │
    │  └──────┘  │
    │            │
    ──────────────
```
Systematic coverage

</div>
<div>

### Orbit Pattern
```
        ╭───╮
      ╱       ╲
     │    ●    │
      ╲       ╱
        ╰───╯
```
Point-of-interest focus

</div>
</div>

```bash
python scripts/data_collection_mission.py \
    --vehicle drone --pattern survey --altitude 30
```

---

# Step 3: Dataset Format

```
data/tokyo_gazebo/
├── drone_20240115_120000/
│   ├── images/
│   │   ├── 000001_CAM_FRONT.jpg
│   │   ├── 000001_CAM_FRONT_LEFT.jpg
│   │   └── ... (6 cameras)
│   ├── lidar/
│   │   └── 000001_LIDAR.npy      # [N, 4] points
│   ├── poses/
│   │   └── 000001.json           # 6-DoF + velocity
│   └── occupancy/
│       └── 000001_occupancy.npz  # [X, Y, Z] voxels
```

---

# Custom Dataset Loader

```python
from dataset.gazebo_occworld_dataset import (
    GazeboOccWorldDataset,
    DatasetConfig
)

config = DatasetConfig(
    history_frames=4,
    future_frames=6,
    agent_type='both',  # drone + rover
    point_cloud_range=(-40, -40, -2, 40, 40, 150),
)

dataset = GazeboOccWorldDataset('data/tokyo_gazebo', config)
loader = DataLoader(dataset, batch_size=1, shuffle=True)
```

---

<!-- _class: lead -->

# Part 4
## Training on Lambda Cloud

---

# Why Lambda Cloud?

| Feature | Lambda | Alternatives |
|---------|--------|--------------|
| **A100 80GB** | $1.79/hr | $2-4/hr |
| **Billing** | Per-minute | Hourly |
| **Setup** | Pre-installed | Manual |
| **Egress** | Free | Charged |
| **Availability** | Good | Varies |

### 24-hour training run ≈ **$43**

---

# Setup: 5 Easy Steps

```bash
# 1. Launch A100 instance from Lambda dashboard

# 2. SSH in
ssh ubuntu@YOUR_INSTANCE_IP

# 3. Clone and setup
git clone https://github.com/USER/VeryLargeWeebModel.git
cd VeryLargeWeebModel
./scripts/lambda_setup.sh

# 4. Download data
./scripts/download_and_prepare_data.sh --all

# 5. Start training (in tmux!)
tmux new -s training
python train.py --config config/finetune_tokyo.py
```

---

# Training Configuration

```python
# config/finetune_tokyo.py

# Lower LR for fine-tuning (preserve pretrained knowledge)
optimizer = dict(lr=1e-4)  # vs 1e-3 from scratch

# Extended range for aerial vehicles
point_cloud_range = [-40, -40, -2, 40, 40, 150]

# Freeze encoder, train transformer
freeze_vae = True
freeze_transformer = False

# Fine-tuning epochs
max_epochs = 50
```

---

# Monitoring Training

<div class="columns">
<div>

### GPU Utilization
```bash
nvtop
```
Should be ~100%

### Training Logs
```bash
tmux attach -t training
```

</div>
<div>

### TensorBoard
```bash
tensorboard --logdir ~/checkpoints \
    --port 6006 --bind_all
```
Open `http://IP:6006`

</div>
</div>

---

# Critical Reminders

## ⚠️ Use tmux!
SSH disconnect = lost training (without tmux)

## ⚠️ Download checkpoints before terminating!
```bash
scp -r ubuntu@IP:~/checkpoints/best.pth ./
```

## ⚠️ TERMINATE when done!
Lambda charges per minute. Don't forget!

---

<!-- _class: lead -->

# Part 5
## Results

---

# Training Curves

![bg right:55% 90%](assets/training_curves.png)

### Observations

- Fast initial convergence
- Stable after epoch 20
- No overfitting (val tracks train)
- Fine-tuning >> from scratch

---

# Quantitative Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Val Loss | 0.45 | 0.28 | -38% |
| mIoU | 0.32 | 0.47 | +47% |
| Inference | 45ms | 45ms | Same |

### Key Improvements

- Better building edge prediction
- Correct aerial perspectives
- Accurate open space rendering

---

# Visual Comparison

<div class="columns">
<div>

### Ground Truth
![Ground Truth](assets/gt_occupancy.png)

</div>
<div>

### Prediction
![Prediction](assets/pred_occupancy.png)

</div>
</div>

Tokyo urban structure accurately captured

---

<!-- _class: lead -->

# Part 6
## Get Started

---

# Open Source Resources

<div class="columns">
<div>

### Our Repository
```
github.com/USER/VeryLargeWeebModel
```

- Training pipeline
- Dataset loaders
- Lambda scripts
- Documentation

### OccWorld
```
github.com/wzzheng/OccWorld
```

</div>
<div>

### Tokyo PLATEAU
```
geospatial.jp/ckan/dataset/plateau
```

### Lambda Cloud
```
cloud.lambda.ai
```

### Paper
```
arxiv.org/abs/2311.16038
```

</div>
</div>

---

# Quick Start Commands

```bash
# Clone
git clone https://github.com/USER/VeryLargeWeebModel.git
cd VeryLargeWeebModel

# Setup (local or Lambda)
./scripts/lambda_setup.sh

# Download everything
./scripts/download_and_prepare_data.sh --all

# Train
python train.py \
    --config config/finetune_tokyo.py \
    --work-dir out/occworld_tokyo
```

---

# Cost Summary

| Component | Cost |
|-----------|------|
| Lambda A100 80GB (24 hrs) | ~$43 |
| Storage (100 GB) | ~$0.60/day |
| Data transfer | Free |
| PLATEAU data | Free |
| OccWorld code | Free |
| **Total for one training run** | **~$44** |

---

# Applications

<div class="columns">
<div>

### Autonomous Vehicles
- Urban navigation
- Path planning
- Safety validation

### Drones / UAVs
- Delivery planning
- Inspection routes
- Obstacle avoidance

</div>
<div>

### Robotics
- Warehouse navigation
- Service robots
- Construction automation

### Research
- World model development
- Sim-to-real transfer
- Urban AI systems

</div>
</div>

---

---

# Data Attribution

### Tokyo PLATEAU

> 3D city model data provided by **Ministry of Land, Infrastructure, Transport and Tourism (MLIT), Japan** - Project PLATEAU. Licensed under **CC BY 4.0**.

- **License:** CC BY 4.0 (Commercial use allowed)
- **Source:** https://www.geospatial.jp/ckan/dataset/plateau-tokyo23ku

### OccWorld

> Zheng et al. "OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving." ECCV 2024.

---

<!-- _class: lead -->
<!-- _backgroundColor: #0d0d1a -->

# Thank You!

## Questions?

<br>

**GitHub:** github.com/USER/VeryLargeWeebModel

**OccWorld Paper:** arxiv.org/abs/2311.16038

**Tokyo PLATEAU:** mlit.go.jp/plateau (CC BY 4.0)

---

# Appendix: Architecture Details

```
┌─────────────────────────────────────────────────────────────────────┐
│                         OCCWORLD ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   3D Input   │───▶│    VQVAE     │───▶│   Tokens     │          │
│  │ [B,T,X,Y,Z]  │    │   Encoder    │    │  [B,T,N]     │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                 │                    │
│                                                 ▼                    │
│                                          ┌──────────────┐           │
│                                          │  Transformer │           │
│                                          │   (GPT-2)    │           │
│                                          └──────────────┘           │
│                                                 │                    │
│  ┌──────────────┐    ┌──────────────┐          │                    │
│  │   3D Output  │◀───│    VQVAE     │◀─────────┘                    │
│  │ [B,T',X,Y,Z] │    │   Decoder    │                               │
│  └──────────────┘    └──────────────┘                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

# Appendix: Sensor Configuration

| Sensor | Specs | Rate |
|--------|-------|------|
| Front Camera | 1600×900, 70° FOV | 10 Hz |
| Front-Left Camera | 1600×900, 70° FOV | 10 Hz |
| Front-Right Camera | 1600×900, 70° FOV | 10 Hz |
| Back Camera | 1600×900, 70° FOV | 10 Hz |
| Back-Left Camera | 1600×900, 70° FOV | 10 Hz |
| Back-Right Camera | 1600×900, 70° FOV | 10 Hz |
| LiDAR | 32 beams, 120m range | 10 Hz |
| IMU | 6-axis | 100 Hz |

---

# Appendix: File Reference

| File | Purpose |
|------|---------|
| `train.py` | Main training script |
| `config/finetune_tokyo.py` | Fine-tuning configuration |
| `dataset/gazebo_occworld_dataset.py` | PyTorch dataset |
| `scripts/lambda_setup.sh` | Cloud instance setup |
| `scripts/download_and_prepare_data.sh` | Data download |
| `scripts/deploy_to_lambda.sh` | One-click deploy |
| `docs/training_guide.md` | Full documentation |
