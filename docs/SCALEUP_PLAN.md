# Scale-Up Plan: All Japan PLATEAU Models

## Overview

Scale training from Tokyo-only to all available PLATEAU 3D city models across Japan.

---

## Phase 1: Data Inventory

### Available PLATEAU Cities
PLATEAU provides 3D models for 100+ cities across Japan:

| Region | Major Cities | Est. Size |
|--------|--------------|-----------|
| Kanto | Tokyo (23 wards), Yokohama, Kawasaki, Saitama, Chiba | ~15GB |
| Kansai | Osaka, Kyoto, Kobe, Nara | ~8GB |
| Chubu | Nagoya, Shizuoka, Niigata | ~5GB |
| Kyushu | Fukuoka, Kitakyushu, Kumamoto | ~4GB |
| Tohoku | Sendai, Morioka | ~3GB |
| Hokkaido | Sapporo | ~2GB |
| Other | 90+ smaller cities | ~20GB |
| **Total** | **100+ cities** | **~60GB** |

### Tasks
- [ ] Scrape full PLATEAU city list from geospatial.jp
- [ ] Create manifest of all available CityGML/OBJ downloads
- [ ] Estimate total storage requirements
- [ ] Prioritize cities by data quality and coverage

---

## Phase 2: Data Pipeline

### Storage Architecture
```
/data/
├── plateau/
│   ├── tokyo/           # Current
│   ├── osaka/
│   ├── kyoto/
│   ├── nagoya/
│   └── .../
├── processed/
│   ├── occupancy_grids/
│   └── metadata/
└── splits/
    ├── train.json
    ├── val.json
    └── test.json
```

### Tasks
- [ ] Extend `download_and_prepare_data.sh` to accept city list
- [ ] Create `scripts/download_all_japan.sh` for bulk download
- [ ] Implement parallel download with resume capability
- [ ] Add data validation and integrity checks
- [ ] Create unified dataset index across all cities

### Cloud Storage Options
| Provider | Cost (1TB/mo) | Egress | Recommendation |
|----------|---------------|--------|----------------|
| Vast.ai disk | Included | N/A | Small scale |
| AWS S3 | ~$23 | $0.09/GB | Medium scale |
| GCS | ~$20 | $0.12/GB | Medium scale |
| Backblaze B2 | ~$5 | Free (some) | Budget option |
| Cloudflare R2 | ~$15 | Free | Best value |

---

## Phase 3: Multi-GPU Training

### Single Node Multi-GPU
```python
# train.py modifications
torch.nn.DataParallel(model)  # Simple
# or
torch.nn.parallel.DistributedDataParallel(model)  # Better scaling
```

### Launch Command
```bash
# 4x A100 on single node
torchrun --nproc_per_node=4 train.py \
    --config config/finetune_japan_full.py \
    --work-dir /workspace/checkpoints
```

### Tasks
- [ ] Add DDP support to `train.py`
- [ ] Create `config/finetune_japan_full.py` for full dataset
- [ ] Implement gradient accumulation for effective larger batches
- [ ] Add learning rate scaling for multi-GPU

---

## Phase 4: Multi-Node Distributed Training

### Architecture
```
┌─────────────────┐     ┌─────────────────┐
│  Node 0 (Main)  │────▶│    Node 1       │
│  4x A100        │     │    4x A100      │
└─────────────────┘     └─────────────────┘
         │                       │
         ▼                       ▼
   ┌─────────────────────────────────┐
   │     Shared Storage (NFS/S3)     │
   └─────────────────────────────────┘
```

### Launch Commands
```bash
# Node 0 (master)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=NODE0_IP --master_port=29500 \
    train.py --config config/finetune_japan_full.py

# Node 1
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=NODE0_IP --master_port=29500 \
    train.py --config config/finetune_japan_full.py
```

### Tasks
- [ ] Create multi-node launch scripts
- [ ] Set up shared storage between nodes
- [ ] Implement checkpoint saving to shared storage
- [ ] Add fault tolerance (checkpoint recovery)

---

## Phase 5: Dataset Sharding

### Strategy
Split Japan data geographically for efficient loading:

```python
# dataset/sharded_dataset.py
class ShardedJapanDataset:
    def __init__(self, shards=['kanto', 'kansai', 'chubu']):
        self.shards = [load_shard(s) for s in shards]

    def __getitem__(self, idx):
        shard_idx = idx % len(self.shards)
        return self.shards[shard_idx][idx // len(self.shards)]
```

### Tasks
- [ ] Implement dataset sharding by region
- [ ] Create shard manifest files
- [ ] Add dynamic shard loading (memory efficient)
- [ ] Implement weighted sampling across regions

---

## Phase 6: Optimizations

### Memory Optimization
- [ ] Gradient checkpointing for larger batches
- [ ] Mixed precision training (AMP)
- [ ] Activation offloading to CPU

### Speed Optimization
- [ ] NVIDIA DALI for GPU data loading
- [ ] Compile model with `torch.compile()`
- [ ] Profile and optimize bottlenecks

### Config Changes
```python
# config/finetune_japan_full.py
data_root = '/data/plateau/all_japan'
cities = ['tokyo', 'osaka', 'kyoto', 'nagoya', ...]  # All cities
batch_size = 4  # Per GPU
gradient_accumulation = 4  # Effective batch = 16 per GPU
use_amp = True
gradient_checkpointing = True
```

---

## Phase 7: GPU Selection & Cost Estimates

### Vast.ai GPU Pricing (as of Jan 2026)

| GPU | $/hr | VRAM | Viable for OccWorld? |
|-----|------|------|----------------------|
| RTX 3090 | $0.13 | 24GB | Yes (batch 1) |
| RTX 3090 Ti | $0.13 | 24GB | Yes (batch 1) |
| RTX 4090 | $0.29 | 24GB | Yes (batch 1-2) |
| RTX 5090 | $0.35 | 32GB | Yes (batch 2) |
| L40S | $0.47 | 48GB | Yes (batch 2-3) |
| A100 PCIE | $0.52 | 40GB | Yes (batch 2) |
| A100 SXM4 | $0.64 | 40/80GB | Yes (batch 2-4) |
| RTX A6000 | $0.37 | 48GB | Yes (batch 2-3) |
| H100 SXM | $1.56 | 80GB | Yes (batch 4+) |
| H200 | $2.19 | 141GB | Overkill |

### Not Recommended (VRAM too low)

| GPU | VRAM | Issue |
|-----|------|-------|
| RTX 3060 | 12GB | Too tight, very slow |
| RTX 4060/4070 | 8-12GB | Will OOM |
| RTX 3080 | 10GB | High OOM risk |
| RTX 2080 Ti | 11GB | Old, too slow |
| Tesla V100 | 16-32GB | Old architecture |

### Tokyo Only (~2GB data)

| GPU | $/hr | Batch | Time | Total Cost |
|-----|------|-------|------|------------|
| **RTX 3090** | $0.13 | 1 | ~48 hrs | **~$6** |
| RTX 3090 Ti | $0.13 | 1 | ~45 hrs | ~$6 |
| RTX 4090 | $0.29 | 1-2 | ~36 hrs | ~$10 |
| A100 40GB | $0.52 | 2 | ~30 hrs | ~$16 |
| A100 80GB | $0.79 | 4 | ~24 hrs | ~$19 |

**Recommendation:** RTX 3090 @ $0.13/hr for Tokyo-only (~$6 total)

### Full Japan (~60GB data, ~30x more)

| GPU | $/hr | Batch | Est. Time | Est. Cost |
|-----|------|-------|-----------|-----------|
| RTX 3090 | $0.13 | 1 | ~200-300 hrs | ~$26-39 |
| A100 40GB | $0.52 | 2 | ~100-150 hrs | ~$52-78 |
| A100 80GB | $0.79 | 4 | ~60-80 hrs | ~$47-63 |
| **4x A100** | ~$2.50 | 8 | ~20-30 hrs | **~$50-75** |
| **8x A100** | ~$5.00 | 16 | ~10-15 hrs | **~$50-75** |

### Why Single GPU Doesn't Scale

For full Japan with a single RTX 3090:
- Batch size 1 = sees one city sample at a time
- Model learns slower (less gradient diversity)
- 200+ hours = 8+ days of rental
- High risk of instance interruption
- No fault tolerance

### Recommendation by Data Size

| Data Size | Best GPU | Why |
|-----------|----------|-----|
| Tokyo only | RTX 3090 ($0.13/hr) | Cheapest viable |
| 5-10 cities | A100 40GB ($0.52/hr) | Good balance |
| Full Japan | 4-8x A100 node ($2.50-5/hr) | Faster, same cost |

### Cost Summary

| Stage | Setup | Est. Cost |
|-------|-------|-----------|
| Prototype | RTX 3090, Tokyo only | **$6** |
| Validate | A100 40GB, 5 major cities | ~$30 |
| Full train | 8x A100, all Japan | ~$50-75 |

---

## Phase 8: Monitoring & Logging

### Weights & Biases Integration
```python
# train.py additions
import wandb

wandb.init(project="occworld-japan", config=config)
wandb.log({"loss": loss, "epoch": epoch, "city": current_city})
wandb.watch(model)
```

### Tasks
- [ ] Add W&B integration to train.py
- [ ] Create dashboards for multi-city training
- [ ] Set up alerts for training failures
- [ ] Log per-city metrics

---

## Phase 9: Validation Strategy

### Geographic Validation Split
```
Train: 80 cities (randomly selected)
Val: 10 cities (held out regions)
Test: 10 cities (completely unseen)
```

### Tasks
- [ ] Create geographic train/val/test splits
- [ ] Implement per-city evaluation metrics
- [ ] Add visualization for predictions per region
- [ ] Create evaluation scripts

---

## Implementation Order

1. **Data pipeline** - Download and organize all Japan data
2. **Dataset loader** - Extend to handle multiple cities
3. **Multi-GPU** - Add DDP support for single node
4. **W&B integration** - Add monitoring
5. **Validation** - Create proper geographic splits
6. **Multi-node** - Scale to multiple machines if needed
7. **Optimizations** - Profile and optimize

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `scripts/download_all_japan.sh` | Create | Bulk download all cities |
| `config/finetune_japan_full.py` | Create | Full Japan config |
| `dataset/sharded_dataset.py` | Create | Multi-city dataset loader |
| `train.py` | Modify | Add DDP, W&B, AMP support |
| `scripts/launch_distributed.sh` | Create | Multi-node launch script |
| `docs/CITIES.md` | Create | List of all available cities |

---

## Questions to Resolve

1. Which cities have the highest quality data?
2. Should we weight samples by city size/importance?
3. Do we need separate models per region or one unified model?
4. What's the minimum viable dataset for good generalization?
5. Should we include rural areas or focus on urban centers?

---

## Resources

- [PLATEAU Portal](https://www.mlit.go.jp/plateau/)
- [G-Portal Data Download](https://www.geospatial.jp/ckan/dataset/plateau)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Weights & Biases Docs](https://docs.wandb.ai/)
