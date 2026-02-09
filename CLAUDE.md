# Project Context for Claude

## Training Environment

**Training runs on remote GPU servers, not locally.**

- Code is developed locally, pushed to git, then pulled on remote servers
- Remote servers: Vast.ai, Lambda Cloud, or similar GPU providers
- Always commit and push changes before they can be tested
- When debugging training issues, provide code changes that can be pushed

## Project Overview

OccWorld training pipeline for Tokyo PLATEAU 3D city data with Gazebo simulation.

## Key Files

- `train.py` - Main training script
- `config/finetune_tokyo.py` - Tokyo dataset configuration
- `dataset/gazebo_occworld_dataset.py` - Custom dataset loader
- `scripts/plateau_to_occworld.py` - PLATEAU mesh to training data converter

## Common Issues

- Occupancy data is very sparse (~1% occupied voxels)
- Loss functions need to handle extreme class imbalance (Focal Loss, Dice Loss)

## Training Bug Log

**All training bugs and fixes are documented in `TRAINING_LOG.md`.**

When fixing training issues, always add an entry with:
- Symptom (what the training output looked like)
- Root cause (why it happened)
- Fix (what was changed)
- Lesson learned (what to watch for next time)

## Testing

**Always run tests after making changes:**

```bash
python -m pytest tests/ -v --tb=short
```
