"""
OccWorld 6DoF Model

Enhanced OccWorld architecture with:
- 6DoF pose prediction (position + quaternion orientation)
- Uncertainty estimation for poses
- Relocalization head for global pose correction
- Place recognition embeddings for loop closure

Architecture:
    Input: history_occupancy [B, T_h, X, Y, Z], history_poses [B, T_h, 13]
    Output: future_occupancy [B, T_f, X, Y, Z], future_poses [B, T_f, 13],
            uncertainty [B, T_f, 6], place_embeddings [B, D]
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OccWorld6DoFConfig:
    """Configuration for OccWorld6DoF model."""
    
    # Grid dimensions
    grid_size: Tuple[int, int, int] = (200, 200, 121)
    
    # Temporal settings
    history_frames: int = 4
    future_frames: int = 6
    
    # Encoder architecture
    encoder_channels: Tuple[int, ...] = (64, 128, 256)
    
    # Temporal modeling
    use_transformer: bool = False
    num_transformer_layers: int = 4
    num_heads: int = 8
    transformer_dim: int = 256
    dropout: float = 0.1
    
    # Pose settings
    pose_dim: int = 13  # x,y,z, quat(4), linear_vel(3), angular_vel(3)
    pose_hidden_dim: int = 128
    
    # 6DoF specific
    uncertainty_dim: int = 6  # Position (3) + Orientation (3) covariance diagonal
    place_embedding_dim: int = 256
    
    # Feature dimensions
    latent_dim: int = 256
    
    # Enable/disable heads
    enable_uncertainty: bool = True
    enable_relocalization: bool = True
    enable_place_recognition: bool = True


class SpatialEncoder3D(nn.Module):
    """3D convolutional encoder for occupancy grids."""
    
    def __init__(self, in_channels: int, channels: Tuple[int, ...] = (64, 128, 256)):
        super().__init__()
        
        layers = []
        prev_ch = in_channels
        
        for ch in channels:
            layers.extend([
                nn.Conv3d(prev_ch, ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(ch),
                nn.ReLU(inplace=True),
            ])
            prev_ch = ch
        
        self.encoder = nn.Sequential(*layers)
        self.out_channels = channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, X, Y, Z]
        Returns:
            features: [B, C', X', Y', Z']
        """
        return self.encoder(x)


class SpatialDecoder3D(nn.Module):
    """3D convolutional decoder for occupancy grids."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 channels: Tuple[int, ...] = (256, 128, 64),
                 output_size: Tuple[int, int, int] = (200, 200, 121)):
        super().__init__()
        
        self.output_size = output_size
        
        layers = []
        prev_ch = in_channels
        
        for ch in channels:
            layers.extend([
                nn.ConvTranspose3d(prev_ch, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(ch),
                nn.ReLU(inplace=True),
            ])
            prev_ch = ch
        
        # Final conv to output channels
        layers.append(nn.Conv3d(prev_ch, out_channels, kernel_size=3, padding=1))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, X', Y', Z']
        Returns:
            output: [B, C_out, X, Y, Z]
        """
        out = self.decoder(x)
        
        # Resize to exact output size if needed
        if out.shape[2:] != self.output_size:
            out = F.interpolate(out, size=self.output_size, mode='trilinear', align_corners=False)
        
        return out


class PoseEncoder(nn.Module):
    """MLP encoder for 6DoF poses."""
    
    def __init__(self, pose_dim: int = 13, hidden_dim: int = 128, out_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Args:
            poses: [B, T, pose_dim]
        Returns:
            features: [B, T, out_dim]
        """
        B, T, D = poses.shape
        poses_flat = poses.view(B * T, D)
        features = self.encoder(poses_flat)
        return features.view(B, T, -1)


class PoseDecoder(nn.Module):
    """MLP decoder for 6DoF poses."""
    
    def __init__(self, in_dim: int = 256, hidden_dim: int = 128, pose_dim: int = 13):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, pose_dim),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, T, in_dim]
        Returns:
            poses: [B, T, pose_dim]
        """
        B, T, D = features.shape
        features_flat = features.view(B * T, D)
        poses = self.decoder(features_flat)
        return poses.view(B, T, -1)


class TemporalLSTM(nn.Module):
    """LSTM-based temporal encoder."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        self.out_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, T, D]
        Returns:
            output: [B, T, hidden_dim]
            (h_n, c_n): final hidden states
        """
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)


class TemporalTransformer(nn.Module):
    """Transformer-based temporal encoder."""
    
    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_dim = d_model
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            mask: optional attention mask
        Returns:
            output: [B, T, D]
        """
        return self.transformer(x, mask=mask)


class UncertaintyHead(nn.Module):
    """Predicts uncertainty (covariance diagonal) for poses."""
    
    def __init__(self, in_dim: int = 256, hidden_dim: int = 64, uncertainty_dim: int = 6):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, uncertainty_dim),
            nn.Softplus(),  # Ensure positive covariance
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, T, in_dim]
        Returns:
            uncertainty: [B, T, uncertainty_dim] - diagonal covariance elements
        """
        B, T, D = features.shape
        features_flat = features.contiguous().view(B * T, D)
        uncertainty = self.head(features_flat)
        return uncertainty.view(B, T, -1)


class RelocalizationHead(nn.Module):
    """Predicts global pose correction for relocalization."""
    
    def __init__(self, in_dim: int = 256, hidden_dim: int = 128, pose_dim: int = 7):
        super().__init__()
        
        # pose_dim = 7 for position (3) + quaternion (4)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, pose_dim),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D] - global features
        Returns:
            global_pose: [B, pose_dim] - global pose correction
        """
        return self.head(features)


class PlaceRecognitionHead(nn.Module):
    """Generates place embeddings for loop closure detection."""
    
    def __init__(self, in_dim: int = 256, embedding_dim: int = 256):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(in_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        # L2 normalize embeddings
        self.normalize = True
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D] - global features
        Returns:
            embeddings: [B, embedding_dim] - normalized place embeddings
        """
        embeddings = self.head(features)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings


class FuturePoseRNN(nn.Module):
    """Autoregressive RNN for future pose prediction."""
    
    def __init__(self, pose_dim: int = 13, hidden_dim: int = 256, context_dim: int = 256):
        super().__init__()
        
        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim
        
        # Combine pose and context
        self.input_proj = nn.Linear(pose_dim + context_dim, hidden_dim)
        
        # GRU cell for autoregressive prediction
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Output pose
        self.pose_out = nn.Linear(hidden_dim, pose_dim)
    
    def forward(self, last_pose: torch.Tensor, context: torch.Tensor, 
                num_future: int, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            last_pose: [B, pose_dim] - last observed pose
            context: [B, context_dim] - temporal context
            num_future: number of future frames to predict
            hidden: optional initial hidden state
        Returns:
            future_poses: [B, num_future, pose_dim]
        """
        B = last_pose.shape[0]
        
        if hidden is None:
            hidden = torch.zeros(B, self.hidden_dim, device=last_pose.device)
        
        future_poses = []
        current_pose = last_pose

        if num_future == 0:
            return torch.empty(B, 0, self.pose_dim, device=last_pose.device, dtype=last_pose.dtype)

        for _ in range(num_future):
            # Combine pose and context
            inp = torch.cat([current_pose, context], dim=-1)
            inp = self.input_proj(inp)
            inp = F.relu(inp)
            
            # GRU step
            hidden = self.gru_cell(inp, hidden)
            
            # Predict pose delta and add to current
            pose_delta = self.pose_out(hidden)
            current_pose = current_pose + pose_delta  # Residual prediction

            # Re-normalize quaternion to maintain unit length
            current_pose = torch.cat([
                current_pose[..., :3],
                F.normalize(current_pose[..., 3:7], p=2, dim=-1),
                current_pose[..., 7:],
            ], dim=-1)

            future_poses.append(current_pose)
        
        return torch.stack(future_poses, dim=1)


class OccWorld6DoF(nn.Module):
    """
    OccWorld with 6DoF pose prediction.
    
    Predicts:
    - Future occupancy grids
    - Future 6DoF poses with uncertainty
    - Global relocalization correction
    - Place recognition embeddings
    """
    
    def __init__(self, config: OccWorld6DoFConfig):
        super().__init__()
        
        self.config = config
        
        # Spatial encoder for occupancy
        self.spatial_encoder = SpatialEncoder3D(
            in_channels=config.history_frames,
            channels=config.encoder_channels,
        )

        # Use adaptive pooling to get fixed-size output regardless of input
        self.encoded_size = (8, 8, 8)  # Fixed small size
        self.adaptive_pool = nn.AdaptiveAvgPool3d(self.encoded_size)

        spatial_features = config.encoder_channels[-1] * self.encoded_size[0] * self.encoded_size[1] * self.encoded_size[2]

        # Spatial feature projection
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_features, config.latent_dim),
            nn.ReLU(inplace=True),
        )
        
        # Pose encoder
        self.pose_encoder = PoseEncoder(
            pose_dim=config.pose_dim,
            hidden_dim=config.pose_hidden_dim,
            out_dim=config.latent_dim,
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.latent_dim * 2, config.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        
        # Temporal encoder
        if config.use_transformer:
            self.temporal_encoder = TemporalTransformer(
                d_model=config.latent_dim,
                nhead=config.num_heads,
                num_layers=config.num_transformer_layers,
                dropout=config.dropout,
            )
        else:
            self.temporal_encoder = TemporalLSTM(
                input_dim=config.latent_dim,
                hidden_dim=config.latent_dim,
                num_layers=2,
                dropout=config.dropout,
            )
        
        # Spatial decoder for future occupancy
        self.spatial_decoder = SpatialDecoder3D(
            in_channels=config.latent_dim,
            out_channels=config.future_frames,
            channels=tuple(reversed(config.encoder_channels)),
            output_size=config.grid_size,
        )
        
        # Feature to spatial projection for decoder
        self.to_spatial = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim * self.encoded_size[0] * self.encoded_size[1] * self.encoded_size[2]),
            nn.ReLU(inplace=True),
        )
        
        # Future pose prediction (autoregressive)
        self.future_pose_rnn = FuturePoseRNN(
            pose_dim=config.pose_dim,
            hidden_dim=config.latent_dim,
            context_dim=config.latent_dim,
        )
        
        # Optional heads
        if config.enable_uncertainty:
            self.uncertainty_head = UncertaintyHead(
                in_dim=config.latent_dim,
                uncertainty_dim=config.uncertainty_dim,
            )
        else:
            self.uncertainty_head = None
        
        if config.enable_relocalization:
            self.relocalization_head = RelocalizationHead(
                in_dim=config.latent_dim,
                pose_dim=7,  # position + quaternion
            )
        else:
            self.relocalization_head = None
        
        if config.enable_place_recognition:
            self.place_recognition_head = PlaceRecognitionHead(
                in_dim=config.latent_dim,
                embedding_dim=config.place_embedding_dim,
            )
        else:
            self.place_recognition_head = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        history_occupancy: torch.Tensor,
        history_poses: torch.Tensor,
        future_poses: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            history_occupancy: [B, T_h, X, Y, Z] - past occupancy grids
            history_poses: [B, T_h, 13] - past poses
            future_poses: [B, T_f, 13] - future poses (for training, optional)
        
        Returns:
            Dict containing:
                - future_occupancy: [B, T_f, X, Y, Z]
                - future_poses: [B, T_f, 13]
                - uncertainty: [B, T_f, 6] (if enabled)
                - global_pose: [B, 7] (if relocalization enabled)
                - place_embedding: [B, D] (if place recognition enabled)
        """
        B = history_occupancy.shape[0]
        device = history_occupancy.device
        
        # Encode spatial features from occupancy
        # [B, T_h, X, Y, Z] -> [B, T_h, C, X', Y', Z']
        occ_float = history_occupancy.float()
        spatial_features = self.spatial_encoder(occ_float)  # [B, C, X', Y', Z']

        # Apply adaptive pooling to get fixed size
        spatial_features = self.adaptive_pool(spatial_features)  # [B, C, 8, 8, 8]

        # Flatten spatial features
        spatial_flat = spatial_features.view(B, -1)  # [B, C*8*8*8]
        spatial_proj = self.spatial_proj(spatial_flat)  # [B, latent_dim]
        
        # Encode poses
        pose_features = self.pose_encoder(history_poses)  # [B, T_h, latent_dim]
        
        # Use last pose features for fusion
        last_pose_feat = pose_features[:, -1, :]  # [B, latent_dim]
        
        # Fuse spatial and pose features
        fused = torch.cat([spatial_proj, last_pose_feat], dim=-1)  # [B, latent_dim*2]
        fused = self.fusion(fused)  # [B, latent_dim]
        
        # Expand for temporal processing
        # We'll use the fused features as context
        context = fused  # [B, latent_dim]
        
        # Temporal encoding
        # For simplicity, we process the pose sequence through temporal encoder
        if isinstance(self.temporal_encoder, TemporalTransformer):
            temporal_out = self.temporal_encoder(pose_features)  # [B, T_h, latent_dim]
            temporal_context = temporal_out[:, -1, :]  # [B, latent_dim]
        else:
            temporal_out, (h_n, c_n) = self.temporal_encoder(pose_features)
            temporal_context = h_n[-1]  # [B, latent_dim]
        
        # Combine contexts
        combined_context = context + temporal_context  # [B, latent_dim]
        
        # Decode future occupancy
        # Project to spatial dimensions
        spatial_for_decode = self.to_spatial(combined_context)  # [B, latent_dim * X' * Y' * Z']
        spatial_for_decode = spatial_for_decode.view(
            B, self.config.latent_dim, 
            self.encoded_size[0], self.encoded_size[1], self.encoded_size[2]
        )
        
        future_occ_logits = self.spatial_decoder(spatial_for_decode)  # [B, T_f, X, Y, Z]
        future_occupancy = torch.sigmoid(future_occ_logits)
        
        # Predict future poses autoregressively
        last_pose = history_poses[:, -1, :]  # [B, 13]
        predicted_future_poses = self.future_pose_rnn(
            last_pose, combined_context, self.config.future_frames
        )  # [B, T_f, 13]
        
        # Build output dict
        outputs = {
            'future_occupancy': future_occupancy,
            'future_poses': predicted_future_poses,
        }
        
        # Uncertainty estimation
        if self.uncertainty_head is not None:
            # Expand context for each future frame
            future_context = combined_context.unsqueeze(1).expand(-1, self.config.future_frames, -1)
            uncertainty = self.uncertainty_head(future_context)
            outputs['uncertainty'] = uncertainty
        
        # Relocalization
        if self.relocalization_head is not None:
            global_pose = self.relocalization_head(combined_context)
            outputs['global_pose'] = global_pose
        
        # Place recognition
        if self.place_recognition_head is not None:
            place_embedding = self.place_recognition_head(combined_context)
            outputs['place_embedding'] = place_embedding
        
        return outputs


class FocalLoss(nn.Module):
    """Focal Loss for sparse occupancy."""
    
    def __init__(self, alpha: float = 0.99, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(min=1e-7, max=1 - 1e-7)
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        p_t = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        return (alpha_weight * focal_weight * bce).mean()


class OccWorld6DoFLoss(nn.Module):
    """
    Combined loss for OccWorld 6DoF training.

    Includes:
    - Focal + Dice + Mean-matching for occupancy (handles sparse grids)
    - Smooth L1 for pose prediction + variance regularization (prevents constant output)
    - NLL for uncertainty-aware pose loss + bounds (prevents sigma collapse)
    - L2 for relocalization
    - Triplet loss for place recognition (prevents embedding collapse)
    """

    def __init__(
        self,
        occ_weight: float = 1.0,
        pose_weight: float = 0.5,
        uncertainty_weight: float = 0.1,
        reloc_weight: float = 0.2,
        place_weight: float = 0.1,
        focal_alpha: float = 0.99,
        focal_gamma: float = 2.0,
        dice_weight: float = 1.0,
        mean_weight: float = 10.0,
        # Anti-collapse parameters
        pose_variance_weight: float = 1.0,  # Penalize low variance in pose predictions
        min_pose_std: float = 0.01,  # Minimum expected std for poses
        uncertainty_min: float = 0.001,  # Clamp sigma lower bound
        uncertainty_max: float = 10.0,  # Clamp sigma upper bound
        triplet_margin: float = 0.2,  # Margin for triplet loss
    ):
        super().__init__()

        self.occ_weight = occ_weight
        self.pose_weight = pose_weight
        self.uncertainty_weight = uncertainty_weight
        self.reloc_weight = reloc_weight
        self.place_weight = place_weight

        # Focal loss for occupancy
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_weight = dice_weight
        self.mean_weight = mean_weight

        # Anti-collapse parameters
        self.pose_variance_weight = pose_variance_weight
        self.min_pose_std = min_pose_std
        self.uncertainty_min = uncertainty_min
        self.uncertainty_max = uncertainty_max
        self.triplet_margin = triplet_margin

        # For tracking training health
        self.debug_metrics = {}
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        Args:
            outputs: Model outputs dict
            targets: Ground truth dict with 'future_occupancy', 'future_poses', 'global_pose'
        
        Returns:
            Dict of individual losses and 'total' combined loss
        """
        losses = {}

        # Validate pose dimensions: need at least 7 (pos xyz + quat wxyz)
        for label, src in [('outputs', outputs), ('targets', targets)]:
            if 'future_poses' in src:
                p = src['future_poses']
                assert p.shape[-1] >= 7, (
                    f"{label}['future_poses'] last dim is {p.shape[-1]}, "
                    f"expected >= 7 (pos[3] + quat[4]). Shape: {list(p.shape)}"
                )

        # Occupancy loss (Focal + Dice + Mean-matching)
        pred_occ = outputs['future_occupancy']
        target_occ = targets['future_occupancy']
        
        # Focal loss
        focal = self.focal_loss(pred_occ, target_occ)
        
        # Dice loss
        pred_flat = pred_occ.view(-1)
        target_flat = target_occ.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = 1 - (2 * intersection + 1) / (pred_flat.sum() + target_flat.sum() + 1)
        
        # Mean-matching regularization
        mean_loss = F.mse_loss(pred_occ.mean(), target_occ.mean())
        
        occ_loss = focal + self.dice_weight * dice + self.mean_weight * mean_loss
        losses['occ'] = occ_loss * self.occ_weight
        
        # Pose loss
        pred_poses = outputs['future_poses']
        target_poses = targets['future_poses']
        
        # Split into position and orientation
        pred_pos = pred_poses[..., :3]
        target_pos = target_poses[..., :3]
        pred_quat = pred_poses[..., 3:7]
        target_quat = target_poses[..., 3:7]
        
        # Position loss (Smooth L1)
        pos_loss = F.smooth_l1_loss(pred_pos, target_pos)
        
        # Quaternion loss (1 - |q1 . q2| to handle double-cover)
        # Normalize quaternions to unit length before computing dot product
        pred_quat_norm = F.normalize(pred_quat, p=2, dim=-1)
        target_quat_norm = F.normalize(target_quat, p=2, dim=-1)
        quat_dot = (pred_quat_norm * target_quat_norm).sum(dim=-1).abs()
        quat_loss = (1 - quat_dot).clamp(min=0).mean()
        
        # Velocity loss
        pred_vel = pred_poses[..., 7:]
        target_vel = target_poses[..., 7:]
        vel_loss = F.smooth_l1_loss(pred_vel, target_vel)
        
        pose_loss = pos_loss + quat_loss + 0.5 * vel_loss

        # === ANTI-COLLAPSE: Pose variance regularization ===
        # Penalize if predictions have very low variance (constant output collapse)
        # Compute std per spatial dimension across (batch, time), then average
        pred_pos_std = pred_pos.reshape(-1, pred_pos.shape[-1]).std(dim=0, unbiased=False).mean()
        pred_vel_std = pred_vel.reshape(-1, pred_vel.shape[-1]).std(dim=0, unbiased=False).mean()

        # Differentiable penalty when std drops below threshold
        min_std = torch.tensor(self.min_pose_std, device=pred_poses.device)
        variance_penalty = F.relu(min_std - pred_pos_std) ** 2 + \
                           F.relu(min_std - pred_vel_std) ** 2

        pose_loss = pose_loss + self.pose_variance_weight * variance_penalty
        losses['pose'] = pose_loss * self.pose_weight

        # Track debug metrics
        self.debug_metrics['pose_pos_std'] = pred_pos_std.item()
        self.debug_metrics['pose_vel_std'] = pred_vel_std.item()
        self.debug_metrics['pose_variance_penalty'] = variance_penalty.item()

        # Uncertainty loss (if available)
        if 'uncertainty' in outputs and self.uncertainty_weight > 0:
            uncertainty = outputs['uncertainty']

            # === ANTI-COLLAPSE: Clamp uncertainty to bounds ===
            uncertainty_clamped = uncertainty.clamp(
                min=self.uncertainty_min,
                max=self.uncertainty_max
            )

            # Negative log-likelihood with learned uncertainty
            # L = 0.5 * (error^2 / sigma^2 + log(sigma^2))
            pos_error = (pred_pos - target_pos) ** 2
            pos_sigma = uncertainty_clamped[..., :3]
            eps = 1e-8
            pos_nll = 0.5 * (pos_error / (pos_sigma + eps) + torch.log(pos_sigma.clamp(min=eps)))

            # Regularize uncertainty to stay in reasonable range (not too large or small)
            uncertainty_reg = ((uncertainty - 1.0) ** 2).mean() * 0.01

            losses['uncertainty'] = (pos_nll.mean() + uncertainty_reg) * self.uncertainty_weight

            # Track debug metrics
            self.debug_metrics['uncertainty_mean'] = uncertainty.mean().item()
            self.debug_metrics['uncertainty_min'] = uncertainty.min().item()
            self.debug_metrics['uncertainty_max'] = uncertainty.max().item()

        # Relocalization loss (if available)
        if 'global_pose' in outputs and 'global_pose' in targets and self.reloc_weight > 0:
            pred_global = outputs['global_pose']
            target_global = targets['global_pose']

            reloc_loss = F.smooth_l1_loss(pred_global, target_global)
            losses['reloc'] = reloc_loss * self.reloc_weight

        # Place recognition loss (if available)
        if 'place_embedding' in outputs and self.place_weight > 0:
            embeddings = outputs['place_embedding']  # [B, D]

            # === ANTI-COLLAPSE: Triplet loss for embeddings ===
            # Require batch size >= 2 for triplet loss
            if embeddings.shape[0] >= 2:
                # In-batch hard negative mining
                # Treat sequential samples as positives (nearby in time/space)
                # and distant samples as negatives
                triplet_loss = self._compute_triplet_loss(embeddings)
            else:
                triplet_loss = torch.tensor(0.0, device=embeddings.device)

            # Also keep norm regularization
            norm_loss = (embeddings.norm(dim=-1) - 1).pow(2).mean()

            losses['place'] = (triplet_loss + 0.1 * norm_loss) * self.place_weight

            # Track debug metrics
            self.debug_metrics['embedding_std'] = embeddings.std().item()
            self.debug_metrics['embedding_mean_norm'] = embeddings.norm(dim=-1).mean().item()

        # Total loss
        losses['total'] = sum(v for v in losses.values())

        return losses

    def _compute_triplet_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss with in-batch hard negative mining.

        For each anchor, the positive is an adjacent sample (assuming temporal
        ordering in batch), and the negative is the hardest non-adjacent sample.
        """
        B, D = embeddings.shape
        if B < 4:
            # Need at least 4 samples for meaningful triplets (B=3 only yields 1/3)
            return torch.tensor(0.0, device=embeddings.device)

        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)  # [B, B]

        total_loss = torch.tensor(0.0, device=embeddings.device)
        num_triplets = 0

        for i in range(B):
            # Anchor
            anchor_idx = i

            # Positive: adjacent sample (circular)
            pos_idx = (i + 1) % B

            # Negative: hardest non-adjacent sample
            # Mask out anchor and positive
            neg_mask = torch.ones(B, dtype=torch.bool, device=embeddings.device)
            neg_mask[anchor_idx] = False
            neg_mask[pos_idx] = False
            prev_idx = (i - 1) % B
            if prev_idx != pos_idx:  # Don't double-exclude
                neg_mask[prev_idx] = False  # Also exclude other adjacent

            if neg_mask.sum() == 0:
                continue

            # Hard negative: closest non-adjacent sample
            neg_distances = distances[anchor_idx].clone()
            neg_distances[~neg_mask] = float('inf')
            neg_idx = neg_distances.argmin()

            # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
            d_ap = distances[anchor_idx, pos_idx]
            d_an = distances[anchor_idx, neg_idx]

            triplet = F.relu(d_ap - d_an + self.triplet_margin)
            total_loss = total_loss + triplet
            num_triplets += 1

        if num_triplets > 0:
            total_loss = total_loss / num_triplets

        return total_loss

    def get_debug_metrics(self) -> Dict[str, float]:
        """Return current debug metrics for logging."""
        return self.debug_metrics.copy()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters in model and its submodules."""
    counts = {}
    
    for name, module in model.named_children():
        counts[name] = sum(p.numel() for p in module.parameters())
    
    counts['total'] = sum(p.numel() for p in model.parameters())
    
    return counts
