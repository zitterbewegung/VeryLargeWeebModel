#!/usr/bin/env python3
"""
OccWorld Training Script for Tokyo Gazebo Dataset

This script provides a training interface for OccWorld models using
data generated from Gazebo simulation.

It can either:
1. Use an existing OccWorld installation (if available)
2. Run standalone training with our custom dataset

Usage:
    # Fine-tune on Tokyo Gazebo data
    python train.py --config config/finetune_tokyo.py --work-dir out/occworld_tokyo

    # Train from scratch
    python train.py --config config/finetune_tokyo.py --work-dir out/occworld_tokyo --from-scratch

    # Resume training
    python train.py --config config/finetune_tokyo.py --work-dir out/occworld_tokyo --resume

Requirements:
    - PyTorch >= 1.9
    - mmcv-full
    - OccWorld dependencies (or standalone mode)
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import our custom datasets
from dataset.gazebo_occworld_dataset import (
    GazeboOccWorldDataset,
    DatasetConfig,
    collate_fn,
    create_dataloader,
)

# Try to import nuScenes dataset
try:
    from dataset.nuscenes_occworld_dataset import (
        NuScenesOccWorldDataset,
        NuScenesConfig,
        collate_fn as nuscenes_collate_fn,
    )
    HAS_NUSCENES = True
except ImportError:
    HAS_NUSCENES = False


def parse_args():
    parser = argparse.ArgumentParser(description='OccWorld Training for Tokyo Gazebo')

    # Config
    parser.add_argument('--config', '--py-config', type=str,
                        default='config/finetune_tokyo.py',
                        help='Path to config file')
    parser.add_argument('--work-dir', type=str, default='out/occworld_tokyo',
                        help='Directory to save outputs')

    # Training
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from specific checkpoint')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Train from scratch (no pretrained weights)')

    # Hardware
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='GPU IDs to use (comma-separated)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Overrides
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs')

    # Mode
    parser.add_argument('--eval-only', action='store_true',
                        help='Run evaluation only')
    parser.add_argument('--use-occworld', action='store_true',
                        help='Use OccWorld library if installed')

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from Python file."""
    import importlib.util

    config_path = Path(config_path).absolute()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    return config


def setup_environment(args):
    """Setup training environment."""
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create work directory
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / 'checkpoints').mkdir(exist_ok=True)
    (work_dir / 'logs').mkdir(exist_ok=True)

    return device, work_dir


def try_import_occworld():
    """Try to import OccWorld library."""
    try:
        # Try importing from installed OccWorld
        sys.path.insert(0, os.path.expanduser('~/OccWorld'))
        from models import TransVQVAE
        from utils import get_logger
        print("Using OccWorld library")
        return True, TransVQVAE
    except ImportError:
        print("OccWorld library not found, using standalone mode")
        return False, None


class SimpleOccupancyModel(nn.Module):
    """
    Simplified occupancy prediction model for standalone training.

    This is a basic encoder-decoder model for occupancy prediction.
    For full OccWorld functionality, install the OccWorld library.
    """

    def __init__(self, config):
        super().__init__()

        # Get dimensions from config
        self.history_frames = getattr(config, 'history_frames', 4)
        self.future_frames = getattr(config, 'future_frames', 6)

        grid_size = getattr(config, 'grid_size', [200, 200, 121])
        self.grid_x, self.grid_y, self.grid_z = grid_size

        # Simple 3D encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(self.history_frames, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        # Temporal modeling (simple LSTM)
        self.temporal = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, self.future_frames, kernel_size=3, padding=1),
        )

        # Pose encoder
        self.pose_encoder = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

    def forward(self, history_occupancy, history_poses=None, future_poses=None):
        """
        Forward pass.

        Args:
            history_occupancy: [B, T_h, X, Y, Z] - Past occupancy grids
            history_poses: [B, T_h, 13] - Past poses (optional)
            future_poses: [B, T_f, 13] - Future poses (optional)

        Returns:
            future_occupancy: [B, T_f, X, Y, Z] - Predicted future occupancy
        """
        batch_size = history_occupancy.shape[0]
        target_shape = history_occupancy.shape[2:]  # [X, Y, Z]

        # Encode history
        # Input: [B, T_h, X, Y, Z] -> [B, T_h, X, Y, Z] (treat T_h as channels)
        x = history_occupancy.float()

        # Encode spatial features
        encoded = self.encoder(x)  # [B, 256, X/4, Y/4, Z/4]

        # Global average pool for temporal modeling
        pooled = encoded.mean(dim=[2, 3, 4])  # [B, 256]

        # Temporal modeling (simplified)
        temporal_out, _ = self.temporal(pooled.unsqueeze(1))  # [B, 1, 256]
        temporal_features = temporal_out.squeeze(1)  # [B, 256]

        # Decode to future occupancy
        decoded_input = temporal_features.view(batch_size, 256, 1, 1, 1)
        decoded_input = decoded_input.expand(-1, -1,
                                              encoded.shape[2],
                                              encoded.shape[3],
                                              encoded.shape[4])

        future_occ = self.decoder(decoded_input)  # [B, T_f, X', Y', Z']

        # Resize to match target dimensions (handles non-divisible grid sizes)
        if future_occ.shape[2:] != target_shape:
            future_occ = torch.nn.functional.interpolate(
                future_occ,
                size=target_shape,
                mode='trilinear',
                align_corners=False
            )

        return torch.sigmoid(future_occ)


class OccupancyLoss(nn.Module):
    """
    Combined loss for sparse occupancy prediction.

    Uses weighted BCE + Dice loss to handle highly imbalanced occupancy grids
    where most voxels are empty (unoccupied).

    Args:
        bce_weight: Weight for BCE loss component
        dice_weight: Weight for Dice loss component
        pos_weight: Weight multiplier for occupied voxels in BCE (higher = more penalty for missing occupied)
        smooth: Smoothing factor for Dice loss to avoid division by zero
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, pos_weight=10.0, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.smooth = smooth

    def forward(self, pred, target):
        # Weighted BCE: occupied voxels get pos_weight, empty voxels get 1.0
        weight = target * (self.pos_weight - 1) + 1  # pos_weight for occupied, 1 for empty
        bce_loss = F.binary_cross_entropy(pred, target, weight=weight, reduction='mean')

        # Dice loss: better handles class imbalance
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, writer):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        history_occ = batch['history_occupancy'].to(device)
        future_occ = batch['future_occupancy'].to(device)
        history_poses = batch['history_poses'].to(device)
        future_poses = batch['future_poses'].to(device)

        # Forward pass
        optimizer.zero_grad()
        pred_occ = model(history_occ, history_poses, future_poses)

        # Compute loss
        loss = criterion(pred_occ, future_occ.float())

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Log
        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

        # TensorBoard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Train/Loss', loss.item(), global_step)

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            history_occ = batch['history_occupancy'].to(device)
            future_occ = batch['future_occupancy'].to(device)
            history_poses = batch['history_poses'].to(device)
            future_poses = batch['future_poses'].to(device)

            pred_occ = model(history_occ, history_poses, future_poses)
            loss = criterion(pred_occ, future_occ.float())

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


def main():
    args = parse_args()

    print("=" * 60)
    print("OccWorld Training for Tokyo Gazebo Dataset")
    print("=" * 60)

    # Setup
    device, work_dir = setup_environment(args)
    print(f"Device: {device}")
    print(f"Work directory: {work_dir}")

    # Load config
    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    # Try to use OccWorld library
    use_occworld, OccWorldModel = try_import_occworld()

    if args.use_occworld and not use_occworld:
        print("ERROR: --use-occworld specified but OccWorld library not found")
        print("Install OccWorld: git clone https://github.com/wzzheng/OccWorld.git ~/OccWorld")
        sys.exit(1)

    # Create dataset
    print("Creating dataset...")
    data_root = getattr(config, 'data_root', 'data/tokyo_gazebo')

    # Detect dataset type from config
    dataset_type = getattr(config, 'dataset_type', None)
    if dataset_type is None:
        # Auto-detect from config path or data_root
        if 'nuscenes' in str(args.config).lower() or 'nuscenes' in str(data_root).lower():
            dataset_type = 'nuscenes'
        else:
            dataset_type = 'gazebo'

    print(f"Dataset type: {dataset_type}")

    if dataset_type == 'nuscenes':
        if not HAS_NUSCENES:
            raise ImportError("nuScenes dataset requested but nuscenes_occworld_dataset.py not found. "
                            "Run ./scripts/setup_training_data.sh --nuscenes first.")

        # Create nuScenes dataset
        nuscenes_cfg = NuScenesConfig(
            version=getattr(config, 'dataset_config', {}).get('version', 'v1.0-mini'),
            history_frames=getattr(config, 'history_frames', 4),
            future_frames=getattr(config, 'future_frames', 6),
            point_cloud_range=tuple(getattr(config, 'point_cloud_range', (-40, -40, -1, 40, 40, 5.4))),
            voxel_size=tuple(getattr(config, 'voxel_size', (0.4, 0.4, 0.4))),
            split='train',
        )
        train_dataset = NuScenesOccWorldDataset(data_root, nuscenes_cfg)

        val_nuscenes_cfg = NuScenesConfig(
            version=nuscenes_cfg.version,
            history_frames=nuscenes_cfg.history_frames,
            future_frames=nuscenes_cfg.future_frames,
            point_cloud_range=nuscenes_cfg.point_cloud_range,
            voxel_size=nuscenes_cfg.voxel_size,
            split='val',
        )
        val_dataset = NuScenesOccWorldDataset(data_root, val_nuscenes_cfg)
        dataset_collate_fn = nuscenes_collate_fn
    else:
        # Create Gazebo dataset
        dataset_cfg = DatasetConfig(
            history_frames=getattr(config, 'history_frames', 4),
            future_frames=getattr(config, 'future_frames', 6),
            frame_skip=getattr(config, 'frame_skip', 1),
            agent_type=getattr(config, 'dataset_config', {}).get('agent_type', 'both'),
            split='train',
            point_cloud_range=getattr(config, 'point_cloud_range', (-40, -40, -2, 40, 40, 150)),
            voxel_size=getattr(config, 'voxel_size', (0.4, 0.4, 1.25)),
        )

        train_dataset = GazeboOccWorldDataset(data_root, dataset_cfg)

        val_cfg = DatasetConfig(**vars(dataset_cfg))
        val_cfg.split = 'val'
        val_dataset = GazeboOccWorldDataset(data_root, val_cfg)
        dataset_collate_fn = collate_fn

    # Create dataloaders
    batch_size = args.batch_size or getattr(config, 'data', {}).get('samples_per_gpu', 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=dataset_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset_collate_fn,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    print("Creating model...")
    if use_occworld and args.use_occworld:
        # Use OccWorld model
        model_cfg = getattr(config, 'model', {})
        model = OccWorldModel(**model_cfg)
    else:
        # Use simple standalone model
        model = SimpleOccupancyModel(config)

    model = model.to(device)

    # Load pretrained weights or resume checkpoint
    resume_checkpoint = None
    if not args.from_scratch:
        load_from = getattr(config, 'load_from', None)
        if args.resume_from:
            load_from = args.resume_from
            resume_checkpoint = load_from
        elif args.resume:
            # Find latest checkpoint
            ckpts = sorted((work_dir / 'checkpoints').glob('epoch_*.pth'))
            if ckpts:
                load_from = str(ckpts[-1])
                resume_checkpoint = load_from

        if load_from and os.path.exists(load_from):
            print(f"Loading weights from: {load_from}")
            checkpoint = torch.load(load_from, map_location=device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Optimizer
    lr = args.lr or getattr(config, 'optimizer', {}).get('lr', 1e-4)
    weight_decay = getattr(config, 'optimizer', {}).get('weight_decay', 0.01)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    max_epochs = args.epochs or getattr(config, 'max_epochs', 50)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    # Loss function - weighted BCE + Dice for sparse occupancy grids
    criterion = OccupancyLoss(
        bce_weight=1.0,
        dice_weight=1.0,
        pos_weight=10.0,  # 10x penalty for missing occupied voxels
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=str(work_dir / 'logs'))

    # Resume optimizer/scheduler state if resuming from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            print(f"Resuming from epoch {start_epoch}")
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("  Restored optimizer state")
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("  Restored scheduler state")
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']

    # Training loop
    print("=" * 60)
    print(f"Starting training for {max_epochs} epochs (from epoch {start_epoch})")
    print("=" * 60)

    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                  device, epoch, writer)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{max_epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}, Time: {epoch_time:.1f}s")

        writer.add_scalar('Train/EpochLoss', train_loss, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], epoch)

        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == max_epochs - 1:
            ckpt_path = work_dir / 'checkpoints' / f'epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = work_dir / 'checkpoints' / 'best.pth'
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            print(f"  New best model! Val Loss: {val_loss:.4f}")

    writer.close()
    print("=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {work_dir / 'checkpoints'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
