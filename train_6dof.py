#!/usr/bin/env python3
"""
OccWorld 6DoF Training Script

Training script for the enhanced OccWorld model with 6DoF pose prediction,
uncertainty estimation, relocalization, and place recognition.

Usage:
    # Standard training
    python train_6dof.py --config config/finetune_6dof.py --work-dir out/6dof

    # Disable specific heads
    python train_6dof.py --config config/finetune_6dof.py --no-uncertainty --no-relocalization

    # Use transformer temporal encoder
    python train_6dof.py --config config/finetune_6dof.py --use-transformer
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import model and dataset
from models.occworld_6dof import OccWorld6DoF, OccWorld6DoFConfig, OccWorld6DoFLoss, count_parameters
from dataset.gazebo_occworld_dataset import (
    GazeboOccWorldDataset,
    DatasetConfig,
    collate_fn,
)


def parse_args():
    parser = argparse.ArgumentParser(description='OccWorld 6DoF Training')

    # Config
    parser.add_argument('--config', type=str, default='config/finetune_6dof.py',
                        help='Path to config file')
    parser.add_argument('--work-dir', type=str, default='out/6dof',
                        help='Directory to save outputs')

    # Model architecture
    parser.add_argument('--use-transformer', action='store_true',
                        help='Use transformer for temporal encoding')
    parser.add_argument('--no-uncertainty', action='store_true',
                        help='Disable uncertainty estimation')
    parser.add_argument('--no-relocalization', action='store_true',
                        help='Disable relocalization head')
    parser.add_argument('--no-place-recognition', action='store_true',
                        help='Disable place recognition head')

    # Training
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from specific checkpoint')

    # Hardware
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='GPU IDs to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Overrides
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)

    # Loss weights
    parser.add_argument('--occ-weight', type=float, default=1.0)
    parser.add_argument('--pose-weight', type=float, default=0.5)
    parser.add_argument('--uncertainty-weight', type=float, default=0.1)
    parser.add_argument('--reloc-weight', type=float, default=0.2)
    parser.add_argument('--place-weight', type=float, default=0.1)

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from Python file."""
    import importlib.util

    config_path = Path(config_path).absolute()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Using default configuration.")
        return None

    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    return config


def setup_environment(args):
    """Setup training environment."""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / 'checkpoints').mkdir(exist_ok=True)
    (work_dir / 'logs').mkdir(exist_ok=True)

    return device, work_dir


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, writer, scaler=None):
    """Train for one epoch."""
    model.train()
    total_losses = {}
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        history_occ = batch['history_occupancy'].to(device)
        future_occ = batch['future_occupancy'].to(device)
        history_poses = batch['history_poses'].to(device)
        future_poses = batch['future_poses'].to(device)

        # Forward pass
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(history_occ, history_poses, future_poses)
                targets = {
                    'future_occupancy': future_occ.float(),
                    'future_poses': future_poses,
                    'global_pose': history_poses[:, -1, :7],  # Use last pose as global reference
                }
                losses = criterion(outputs, targets)

            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(history_occ, history_poses, future_poses)
            targets = {
                'future_occupancy': future_occ.float(),
                'future_poses': future_poses,
                'global_pose': history_poses[:, -1, :7],
            }
            losses = criterion(outputs, targets)

            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
            optimizer.step()

        # Accumulate losses
        for key, value in losses.items():
            if key not in total_losses:
                total_losses[key] = 0
            total_losses[key] += value.item()
        num_batches += 1

        # Log
        if batch_idx % 10 == 0:
            loss_str = " | ".join([f"{k}: {v.item():.4f}" for k, v in losses.items() if k != 'total'])
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] Total: {losses['total'].item():.4f} | {loss_str}")

        # TensorBoard
        global_step = epoch * len(dataloader) + batch_idx
        for key, value in losses.items():
            writer.add_scalar(f'Train/{key}', value.item(), global_step)

    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_losses = {}
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            history_occ = batch['history_occupancy'].to(device)
            future_occ = batch['future_occupancy'].to(device)
            history_poses = batch['history_poses'].to(device)
            future_poses = batch['future_poses'].to(device)

            outputs = model(history_occ, history_poses, future_poses)
            targets = {
                'future_occupancy': future_occ.float(),
                'future_poses': future_poses,
                'global_pose': history_poses[:, -1, :7],
            }
            losses = criterion(outputs, targets)

            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0
                total_losses[key] += value.item()
            num_batches += 1

    avg_losses = {k: v / num_batches for k, v in total_losses.items()} if num_batches > 0 else {'total': 0}
    return avg_losses


def main():
    args = parse_args()

    print("=" * 60)
    print("OccWorld 6DoF Training")
    print("=" * 60)

    # Setup
    device, work_dir = setup_environment(args)

    # Load config
    print(f"Loading config: {args.config}")
    file_config = load_config(args.config)

    # Create model config
    model_config = OccWorld6DoFConfig(
        grid_size=tuple(getattr(file_config, 'grid_size', [200, 200, 121])) if file_config else (200, 200, 121),
        history_frames=getattr(file_config, 'history_frames', 4) if file_config else 4,
        future_frames=getattr(file_config, 'future_frames', 6) if file_config else 6,
        use_transformer=args.use_transformer,
        enable_uncertainty=not args.no_uncertainty,
        enable_relocalization=not args.no_relocalization,
        enable_place_recognition=not args.no_place_recognition,
    )

    print(f"\nModel configuration:")
    print(f"  Grid size: {model_config.grid_size}")
    print(f"  History frames: {model_config.history_frames}")
    print(f"  Future frames: {model_config.future_frames}")
    print(f"  Use transformer: {model_config.use_transformer}")
    print(f"  Uncertainty: {model_config.enable_uncertainty}")
    print(f"  Relocalization: {model_config.enable_relocalization}")
    print(f"  Place recognition: {model_config.enable_place_recognition}")

    # Create dataset
    print("\nCreating dataset...")
    data_root = getattr(file_config, 'data_root', 'data/tokyo_gazebo') if file_config else 'data/tokyo_gazebo'

    ds_cfg = getattr(file_config, 'dataset_config', {}) if file_config else {}
    dataset_cfg = DatasetConfig(
        history_frames=model_config.history_frames,
        future_frames=model_config.future_frames,
        frame_skip=getattr(file_config, 'frame_skip', 1) if file_config else 1,
        split='train',
        exclude_dummy_sessions=ds_cfg.get('exclude_dummy_sessions', True),
        point_cloud_range=getattr(file_config, 'point_cloud_range', (-40, -40, -2, 40, 40, 150)) if file_config else (-40, -40, -2, 40, 40, 150),
        voxel_size=getattr(file_config, 'voxel_size', (0.4, 0.4, 1.25)) if file_config else (0.4, 0.4, 1.25),
    )

    train_dataset = GazeboOccWorldDataset(data_root, dataset_cfg)

    val_cfg = DatasetConfig(**vars(dataset_cfg))
    val_cfg.split = 'val'
    val_dataset = GazeboOccWorldDataset(data_root, val_cfg)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("\nERROR: No training data found!")
        print("Run: ./scripts/setup_and_train.sh --test --skip-train")
        sys.exit(1)

    # Create dataloaders
    batch_size = args.batch_size or (getattr(file_config, 'data', {}).get('samples_per_gpu', 1) if file_config else 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create model
    print("\nCreating model...")
    model = OccWorld6DoF(model_config)
    model = model.to(device)

    # Count and display parameters
    param_counts = count_parameters(model)
    print("\nParameter counts:")
    for name, count in param_counts.items():
        if name != 'total':
            print(f"  {name}: {count:,}")
    print(f"  TOTAL: {param_counts['total']:,}")
    print(f"  Model size (FP32): {param_counts['total'] * 4 / 1024 / 1024:.1f} MB")

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume or args.resume_from:
        ckpt_path = args.resume_from
        if args.resume and not ckpt_path:
            ckpts = sorted((work_dir / 'checkpoints').glob('epoch_*.pth'))
            if ckpts:
                ckpt_path = str(ckpts[-1])

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"\nLoading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'val_loss' in checkpoint:
                best_val_loss = checkpoint['val_loss']
            print(f"  Resumed from epoch {start_epoch}")

    # Optimizer
    lr = args.lr or (getattr(file_config, 'optimizer', {}).get('lr', 1e-4) if file_config else 1e-4)
    weight_decay = getattr(file_config, 'optimizer', {}).get('weight_decay', 0.01) if file_config else 0.01

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    max_epochs = args.epochs or (getattr(file_config, 'max_epochs', 50) if file_config else 50)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    # Loss function
    criterion = OccWorld6DoFLoss(
        occ_weight=args.occ_weight,
        pose_weight=args.pose_weight,
        uncertainty_weight=args.uncertainty_weight,
        reloc_weight=args.reloc_weight,
        place_weight=args.place_weight,
    )

    # Mixed precision
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # TensorBoard
    writer = SummaryWriter(log_dir=str(work_dir / 'logs'))

    # Training loop
    print("\n" + "=" * 60)
    print(f"Starting training for {max_epochs} epochs")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Mixed precision: {use_amp}")
    print("=" * 60)

    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()

        # Train
        train_losses = train_epoch(model, train_loader, optimizer, criterion, device, epoch, writer, scaler)

        # Validate
        val_losses = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch}/{max_epochs} - Time: {epoch_time:.1f}s")
        print(f"  Train - Total: {train_losses['total']:.4f}")
        print(f"  Val   - Total: {val_losses['total']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        for key in train_losses:
            writer.add_scalar(f'Epoch/Train_{key}', train_losses[key], epoch)
        for key in val_losses:
            writer.add_scalar(f'Epoch/Val_{key}', val_losses[key], epoch)
        writer.add_scalar('Epoch/LR', scheduler.get_last_lr()[0], epoch)

        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == max_epochs - 1:
            ckpt_path = work_dir / 'checkpoints' / f'epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_losses['total'],
                'val_loss': val_losses['total'],
                'config': model_config,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_path = work_dir / 'checkpoints' / 'best.pth'
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'val_loss': val_losses['total'],
                'config': model_config,
            }, best_path)
            print(f"  New best model! Val Loss: {val_losses['total']:.4f}")

    writer.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved to: {work_dir / 'checkpoints'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
