#!/usr/bin/env python3
"""
OccWorld 6DoF Enhanced Model

Extends the base occupancy prediction model with:
1. Pose prediction with uncertainty estimation
2. Relocalization head for global pose correction
3. Loop closure detection for place recognition
4. Multi-task learning framework

Architecture:
    Input: History occupancy [B, T_h, X, Y, Z] + History poses [B, T_h, 13]

    Outputs:
        - future_occupancy: [B, T_f, X, Y, Z] - Predicted occupancy
        - future_poses: [B, T_f, 13] - Predicted 6DoF poses + velocities
        - pose_uncertainty: [B, T_f, 6] - Pose covariance (diagonal)
        - global_pose: [B, 7] - Relocalization output (position + quaternion)
        - place_embedding: [B, D] - Place recognition embedding

Usage:
    from models.occworld_6dof import OccWorld6DoF, OccWorld6DoFLoss

    model = OccWorld6DoF(config)
    outputs = model(history_occ, history_poses, future_poses)

    criterion = OccWorld6DoFLoss()
    loss = criterion(outputs, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class OccWorld6DoFConfig:
    """Configuration for OccWorld 6DoF model."""
    # Grid dimensions
    grid_size: Tuple[int, int, int] = (200, 200, 121)

    # Temporal settings
    history_frames: int = 4
    future_frames: int = 6

    # Encoder settings
    encoder_channels: Tuple[int, ...] = (64, 128, 256)

    # Transformer settings (if using full transformer)
    use_transformer: bool = False
    num_transformer_layers: int = 4
    num_heads: int = 8
    transformer_dim: int = 256
    dropout: float = 0.1

    # 6DoF settings
    pose_dim: int = 13  # x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz
    uncertainty_dim: int = 6  # Position (3) + Orientation (3) uncertainty
    place_embedding_dim: int = 256

    # Task weights
    enable_uncertainty: bool = True
    enable_relocalization: bool = True
    enable_place_recognition: bool = True


class SpatialEncoder3D(nn.Module):
    """3D CNN encoder for occupancy grids."""

    def __init__(self, in_channels: int, channels: Tuple[int, ...] = (64, 128, 256)):
        super().__init__()

        layers = []
        prev_ch = in_channels

        for i, ch in enumerate(channels):
            stride = 2 if i > 0 else 1
            layers.extend([
                nn.Conv3d(prev_ch, ch, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm3d(ch),
                nn.ReLU(inplace=True),
            ])
            prev_ch = ch

        self.encoder = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SpatialDecoder3D(nn.Module):
    """3D CNN decoder for occupancy prediction."""

    def __init__(self, in_channels: int, out_channels: int,
                 channels: Tuple[int, ...] = (128, 64)):
        super().__init__()

        layers = []
        prev_ch = in_channels

        for ch in channels:
            layers.extend([
                nn.ConvTranspose3d(prev_ch, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(ch),
                nn.ReLU(inplace=True),
            ])
            prev_ch = ch

        layers.append(nn.Conv3d(prev_ch, out_channels, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class TemporalEncoder(nn.Module):
    """Temporal modeling with LSTM or Transformer."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 use_transformer: bool = False, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.use_transformer = use_transformer
        self.hidden_dim = hidden_dim

        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        else:
            self.temporal = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] temporal features
        Returns:
            [B, D] aggregated features
        """
        if self.use_transformer:
            out = self.temporal(x)  # [B, T, D]
            out = self.proj(out[:, -1, :])  # Take last timestep
        else:
            out, _ = self.temporal(x)
            out = out[:, -1, :]  # [B, D]

        return out


class PoseEncoder(nn.Module):
    """Encode 6DoF poses + velocities."""

    def __init__(self, pose_dim: int = 13, hidden_dim: int = 128, output_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Args:
            poses: [B, T, 13] or [B, 13]
        Returns:
            [B, T, D] or [B, D] pose embeddings
        """
        return self.encoder(poses)


class PoseDecoder(nn.Module):
    """Decode features to 6DoF poses."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, pose_dim: int = 13):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, pose_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)


class UncertaintyHead(nn.Module):
    """
    Predict pose uncertainty (covariance diagonal).

    Outputs log-variance for numerical stability, converted to variance during loss computation.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 64, uncertainty_dim: int = 6):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, uncertainty_dim),
        )

        # Initialize to predict low uncertainty
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D] or [B, T, D]
        Returns:
            log_variance: [B, 6] or [B, T, 6] - log of diagonal covariance
        """
        return self.net(features)


class RelocalizationHead(nn.Module):
    """
    Global pose regression for relocalization.

    Takes aggregated scene features and predicts absolute pose (position + quaternion).
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256):
        super().__init__()

        # Position head (x, y, z)
        self.position_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
        )

        # Orientation head (quaternion: qx, qy, qz, qw)
        self.orientation_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 4),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D] global features
        Returns:
            global_pose: [B, 7] - (x, y, z, qx, qy, qz, qw)
        """
        position = self.position_head(features)
        orientation = self.orientation_head(features)

        # Normalize quaternion
        orientation = F.normalize(orientation, p=2, dim=-1)

        return torch.cat([position, orientation], dim=-1)


class PlaceRecognitionHead(nn.Module):
    """
    Generate embeddings for place recognition / loop closure detection.

    Produces a compact embedding that can be used for nearest-neighbor search.
    """

    def __init__(self, input_dim: int = 256, embedding_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D] global features
        Returns:
            embedding: [B, embedding_dim] L2-normalized embedding
        """
        embedding = self.encoder(features)
        return F.normalize(embedding, p=2, dim=-1)


class OccWorld6DoF(nn.Module):
    """
    OccWorld with 6DoF enhancements.

    Multi-task model that predicts:
    1. Future occupancy grids
    2. Future 6DoF poses with uncertainty
    3. Global pose for relocalization
    4. Place recognition embeddings
    """

    def __init__(self, config: Optional[OccWorld6DoFConfig] = None):
        super().__init__()

        if config is None:
            config = OccWorld6DoFConfig()
        self.config = config

        # Spatial encoder
        self.spatial_encoder = SpatialEncoder3D(
            in_channels=config.history_frames,
            channels=config.encoder_channels,
        )

        # Pose encoder
        self.pose_encoder = PoseEncoder(
            pose_dim=config.pose_dim,
            hidden_dim=128,
            output_dim=config.encoder_channels[-1],
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.encoder_channels[-1] * 2, config.encoder_channels[-1]),
            nn.LayerNorm(config.encoder_channels[-1]),
            nn.ReLU(inplace=True),
        )

        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            input_dim=config.encoder_channels[-1],
            hidden_dim=config.encoder_channels[-1],
            num_layers=2,
            use_transformer=config.use_transformer,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        # Spatial decoder for occupancy
        self.spatial_decoder = SpatialDecoder3D(
            in_channels=config.encoder_channels[-1],
            out_channels=config.future_frames,
            channels=(128, 64),
        )

        # Pose decoder
        self.pose_decoder = PoseDecoder(
            input_dim=config.encoder_channels[-1],
            hidden_dim=128,
            pose_dim=config.pose_dim,
        )

        # Future pose predictor (autoregressive)
        self.future_pose_rnn = nn.GRU(
            input_size=config.pose_dim + config.encoder_channels[-1],
            hidden_size=config.encoder_channels[-1],
            num_layers=1,
            batch_first=True,
        )

        # 6DoF enhancement heads
        if config.enable_uncertainty:
            self.uncertainty_head = UncertaintyHead(
                input_dim=config.encoder_channels[-1],
                uncertainty_dim=config.uncertainty_dim,
            )

        if config.enable_relocalization:
            self.relocalization_head = RelocalizationHead(
                input_dim=config.encoder_channels[-1],
            )

        if config.enable_place_recognition:
            self.place_recognition_head = PlaceRecognitionHead(
                input_dim=config.encoder_channels[-1],
                embedding_dim=config.place_embedding_dim,
            )

        # Store grid size for output reshaping
        self.grid_size = config.grid_size

    def forward(
        self,
        history_occupancy: torch.Tensor,
        history_poses: Optional[torch.Tensor] = None,
        future_poses: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            history_occupancy: [B, T_h, X, Y, Z] - Past occupancy grids
            history_poses: [B, T_h, 13] - Past 6DoF poses (optional)
            future_poses: [B, T_f, 13] - Future poses for teacher forcing (optional)

        Returns:
            Dict containing:
                - future_occupancy: [B, T_f, X, Y, Z]
                - future_poses: [B, T_f, 13]
                - pose_uncertainty: [B, T_f, 6] (if enabled)
                - global_pose: [B, 7] (if enabled)
                - place_embedding: [B, D] (if enabled)
        """
        batch_size = history_occupancy.shape[0]
        device = history_occupancy.device
        target_shape = history_occupancy.shape[2:]  # [X, Y, Z]

        # Encode spatial features
        spatial_features = self.spatial_encoder(history_occupancy.float())  # [B, C, X', Y', Z']

        # Global average pool
        global_spatial = spatial_features.mean(dim=[2, 3, 4])  # [B, C]

        # Encode poses if provided
        if history_poses is not None:
            # Average pose embedding over history
            pose_features = self.pose_encoder(history_poses)  # [B, T_h, C]
            pose_features = pose_features.mean(dim=1)  # [B, C]

            # Fuse spatial and pose features
            fused_features = self.fusion(torch.cat([global_spatial, pose_features], dim=-1))
        else:
            fused_features = global_spatial

        # Temporal encoding (single step for now)
        temporal_features = self.temporal_encoder(fused_features.unsqueeze(1))  # [B, C]

        # === Occupancy Prediction ===
        # Expand features for decoder
        decoder_input = temporal_features.view(batch_size, -1, 1, 1, 1)
        decoder_input = decoder_input.expand(
            -1, -1,
            spatial_features.shape[2],
            spatial_features.shape[3],
            spatial_features.shape[4],
        )

        future_occ = self.spatial_decoder(decoder_input)  # [B, T_f, X', Y', Z']

        # Resize to match target dimensions
        if future_occ.shape[2:] != target_shape:
            future_occ = F.interpolate(
                future_occ,
                size=target_shape,
                mode='trilinear',
                align_corners=False,
            )

        future_occ = torch.sigmoid(future_occ)

        # === Pose Prediction ===
        outputs = {
            'future_occupancy': future_occ,
        }

        # Predict future poses autoregressively
        T_f = self.config.future_frames
        predicted_poses = []
        predicted_uncertainties = []

        # Initialize with last history pose or zeros
        if history_poses is not None:
            current_pose = history_poses[:, -1, :]  # [B, 13]
        else:
            current_pose = torch.zeros(batch_size, self.config.pose_dim, device=device)

        hidden = temporal_features.unsqueeze(0)  # [1, B, C]

        for t in range(T_f):
            # Combine current pose with temporal features
            rnn_input = torch.cat([current_pose, temporal_features], dim=-1)  # [B, 13+C]
            rnn_input = rnn_input.unsqueeze(1)  # [B, 1, 13+C]

            # GRU step
            rnn_out, hidden = self.future_pose_rnn(rnn_input, hidden)
            rnn_out = rnn_out.squeeze(1)  # [B, C]

            # Decode pose
            pose_delta = self.pose_decoder(rnn_out)  # [B, 13]
            current_pose = current_pose + pose_delta  # Residual prediction
            predicted_poses.append(current_pose)

            # Predict uncertainty
            if self.config.enable_uncertainty:
                uncertainty = self.uncertainty_head(rnn_out)
                predicted_uncertainties.append(uncertainty)

        outputs['future_poses'] = torch.stack(predicted_poses, dim=1)  # [B, T_f, 13]

        if self.config.enable_uncertainty and predicted_uncertainties:
            outputs['pose_uncertainty'] = torch.stack(predicted_uncertainties, dim=1)  # [B, T_f, 6]

        # === Relocalization ===
        if self.config.enable_relocalization:
            outputs['global_pose'] = self.relocalization_head(temporal_features)  # [B, 7]

        # === Place Recognition ===
        if self.config.enable_place_recognition:
            outputs['place_embedding'] = self.place_recognition_head(temporal_features)  # [B, D]

        return outputs


class OccWorld6DoFLoss(nn.Module):
    """
    Multi-task loss for OccWorld 6DoF.

    Combines:
    1. Occupancy loss (BCE + Dice)
    2. Pose loss (Smooth L1)
    3. Uncertainty-aware pose loss (Negative log-likelihood)
    4. Relocalization loss (Position + Quaternion)
    5. Place recognition loss (Contrastive/Triplet)
    """

    def __init__(
        self,
        occ_weight: float = 1.0,
        pose_weight: float = 0.5,
        uncertainty_weight: float = 0.1,
        reloc_weight: float = 0.2,
        place_weight: float = 0.1,
        pos_weight: float = 10.0,  # BCE weight for occupied voxels
    ):
        super().__init__()

        self.occ_weight = occ_weight
        self.pose_weight = pose_weight
        self.uncertainty_weight = uncertainty_weight
        self.reloc_weight = reloc_weight
        self.place_weight = place_weight
        self.pos_weight = pos_weight

    def occupancy_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Weighted BCE + Dice loss for sparse occupancy."""
        # Weighted BCE
        weight = target * (self.pos_weight - 1) + 1
        bce = F.binary_cross_entropy(pred, target.float(), weight=weight, reduction='mean')

        # Dice loss
        smooth = 1.0
        pred_flat = pred.view(-1)
        target_flat = target.view(-1).float()
        intersection = (pred_flat * target_flat).sum()
        dice = 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

        return bce + dice

    def pose_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Smooth L1 loss for pose prediction."""
        return F.smooth_l1_loss(pred, target, reduction='mean')

    def uncertainty_aware_pose_loss(
        self,
        pred_pose: torch.Tensor,
        target_pose: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative log-likelihood loss with learned uncertainty.

        L = 0.5 * (exp(-log_var) * (pred - target)^2 + log_var)
        """
        # Only use position and orientation (first 6 dims)
        pred_6dof = pred_pose[..., :6]  # x,y,z,qx,qy,qz
        target_6dof = target_pose[..., :6]

        precision = torch.exp(-log_variance)  # [B, T, 6]
        diff_sq = (pred_6dof - target_6dof) ** 2

        nll = 0.5 * (precision * diff_sq + log_variance)
        return nll.mean()

    def relocalization_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Combined position and orientation loss for relocalization.

        Args:
            pred: [B, 7] - predicted (x,y,z, qx,qy,qz,qw)
            target: [B, 7] - target global pose
        """
        # Position loss (L2)
        pos_loss = F.mse_loss(pred[:, :3], target[:, :3])

        # Quaternion loss (1 - |q1 · q2|)
        quat_dot = (pred[:, 3:] * target[:, 3:]).sum(dim=-1).abs()
        quat_loss = (1 - quat_dot).mean()

        return pos_loss + quat_loss

    def place_recognition_loss(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Contrastive loss for place recognition.

        If labels not provided, uses in-batch negatives (NT-Xent style).
        """
        batch_size = embeddings.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T)  # [B, B]

        # Temperature-scaled softmax
        temperature = 0.1
        sim_matrix = sim_matrix / temperature

        # In-batch negative: diagonal is positive, others are negative
        labels = torch.arange(batch_size, device=embeddings.device)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            outputs: Model outputs dict
            targets: Ground truth dict with keys:
                - future_occupancy: [B, T_f, X, Y, Z]
                - future_poses: [B, T_f, 13]
                - global_pose: [B, 7] (optional)

        Returns:
            Dict with individual losses and total loss
        """
        losses = {}
        total_loss = 0.0

        # Occupancy loss
        if 'future_occupancy' in outputs and 'future_occupancy' in targets:
            occ_loss = self.occupancy_loss(outputs['future_occupancy'], targets['future_occupancy'])
            losses['occupancy'] = occ_loss
            total_loss = total_loss + self.occ_weight * occ_loss

        # Pose loss
        if 'future_poses' in outputs and 'future_poses' in targets:
            pose_loss = self.pose_loss(outputs['future_poses'], targets['future_poses'])
            losses['pose'] = pose_loss
            total_loss = total_loss + self.pose_weight * pose_loss

            # Uncertainty-aware loss
            if 'pose_uncertainty' in outputs:
                unc_loss = self.uncertainty_aware_pose_loss(
                    outputs['future_poses'],
                    targets['future_poses'],
                    outputs['pose_uncertainty'],
                )
                losses['uncertainty'] = unc_loss
                total_loss = total_loss + self.uncertainty_weight * unc_loss

        # Relocalization loss
        if 'global_pose' in outputs and 'global_pose' in targets:
            reloc_loss = self.relocalization_loss(outputs['global_pose'], targets['global_pose'])
            losses['relocalization'] = reloc_loss
            total_loss = total_loss + self.reloc_weight * reloc_loss

        # Place recognition loss
        if 'place_embedding' in outputs:
            place_labels = targets.get('place_labels', None)
            place_loss = self.place_recognition_loss(outputs['place_embedding'], place_labels)
            losses['place_recognition'] = place_loss
            total_loss = total_loss + self.place_weight * place_loss

        losses['total'] = total_loss
        return losses


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters by component."""
    counts = {}
    total = 0

    for name, child in model.named_children():
        params = sum(p.numel() for p in child.parameters())
        counts[name] = params
        total += params

    counts['total'] = total
    return counts


# Test and parameter counting
if __name__ == '__main__':
    print("=" * 60)
    print("OccWorld 6DoF Model Test")
    print("=" * 60)

    # Create config and model
    config = OccWorld6DoFConfig(
        grid_size=(200, 200, 121),
        history_frames=4,
        future_frames=6,
        enable_uncertainty=True,
        enable_relocalization=True,
        enable_place_recognition=True,
    )

    model = OccWorld6DoF(config)

    # Count parameters
    print("\nParameter counts:")
    counts = count_parameters(model)
    for name, count in counts.items():
        print(f"  {name}: {count:,}")

    total = counts['total']
    print(f"\nTotal parameters: {total:,}")
    print(f"Model size (FP32): {total * 4 / 1024 / 1024:.1f} MB")
    print(f"Model size (FP16): {total * 2 / 1024 / 1024:.1f} MB")

    # Test forward pass
    print("\n" + "=" * 60)
    print("Testing forward pass...")

    batch_size = 2
    history_occ = torch.randn(batch_size, 4, 200, 200, 121)
    history_poses = torch.randn(batch_size, 4, 13)

    model.eval()
    with torch.no_grad():
        outputs = model(history_occ, history_poses)

    print("\nOutputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    # Test loss computation
    print("\n" + "=" * 60)
    print("Testing loss computation...")

    criterion = OccWorld6DoFLoss()

    targets = {
        'future_occupancy': torch.randint(0, 2, (batch_size, 6, 200, 200, 121)).float(),
        'future_poses': torch.randn(batch_size, 6, 13),
        'global_pose': torch.randn(batch_size, 7),
    }

    # Normalize quaternion in target
    targets['global_pose'][:, 3:] = F.normalize(targets['global_pose'][:, 3:], dim=-1)

    losses = criterion(outputs, targets)

    print("\nLosses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
