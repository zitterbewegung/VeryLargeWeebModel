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
            # GroupNorm: batch-size independent, works with batch_size=1 and multi-GPU
            num_groups = min(32, ch)  # Standard: 32 groups, or fewer if ch < 32
            layers.extend([
                nn.Conv3d(prev_ch, ch, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups, ch),
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
            num_groups = min(32, ch)
            layers.extend([
                nn.ConvTranspose3d(prev_ch, ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(num_groups, ch),
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


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input. x: [B, T, D]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """Transformer-based temporal encoder with positional encoding."""

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()

        self.pos_encoding = SinusoidalPositionalEncoding(d_model, dropout=dropout)

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
        x = self.pos_encoding(x)
        return self.transformer(x, mask=mask)


class UncertaintyHead(nn.Module):
    """Predicts raw uncertainty logits for poses."""
    
    def __init__(self, in_dim: int = 256, hidden_dim: int = 64, uncertainty_dim: int = 6):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, uncertainty_dim),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, T, in_dim]
        Returns:
            uncertainty: [B, T, uncertainty_dim] - raw logits for bounded sigma
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


def safe_quat_normalize(q: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion to unit length, with safe handling of zero vectors.

    When the quaternion norm is near zero (e.g. from zero-initialized poses),
    falls back to identity quaternion [1, 0, 0, 0] to avoid NaN from division
    by zero.  The fallback is differentiable (uses torch.where).
    """
    norm = q.norm(p=2, dim=-1, keepdim=True)
    # Identity quaternion [1, 0, 0, 0] as fallback
    identity = torch.zeros_like(q)
    identity[..., 0] = 1.0
    # Use identity when norm is too small
    return torch.where(norm > 1e-8, q / (norm + 1e-12), identity)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions.

    Supports both [w, x, y, z] and [x, y, z, w] conventions by detecting
    which convention is used based on the pose format. In this codebase,
    UAVScenes uses [w, x, y, z] (indices 3:7 of pose = [qw, qx, qy, qz]),
    nuScenes uses [x, y, z, w] (indices 3:7 of pose = [qx, qy, qz, qw]).

    We use [w, x, y, z] (Hamilton convention) as the canonical form since
    it matches robotics convention and is used by pyquaternion/UAVScenes.

    Args:
        q1, q2: [..., 4] tensors in [w, x, y, z] order
    Returns:
        product: [..., 4] tensor in [w, x, y, z] order
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


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

        # Output pose delta: position(3) + quat_delta(4) + velocity(6)
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

            # Predict pose delta
            pose_delta = self.pose_out(hidden)

            # Position: additive residual
            new_pos = current_pose[..., :3] + pose_delta[..., :3]

            # Orientation: quaternion multiplication (proper group operation)
            # Treat delta as a small rotation quaternion, normalize to unit length
            # Use safe normalize to handle zero-initialized poses without NaN
            delta_quat = safe_quat_normalize(pose_delta[..., 3:7])
            current_quat = safe_quat_normalize(current_pose[..., 3:7])
            new_quat = quaternion_multiply(current_quat, delta_quat)
            new_quat = safe_quat_normalize(new_quat)

            # Velocity: additive residual
            new_vel = current_pose[..., 7:] + pose_delta[..., 7:]

            current_pose = torch.cat([new_pos, new_quat, new_vel], dim=-1)
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

        # Pose-conditioned occupancy: FiLM modulation
        # Encodes predicted future poses into scale/shift for spatial features
        self.pose_film = nn.Sequential(
            nn.Linear(config.pose_dim * config.future_frames, config.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.latent_dim, config.latent_dim * 2),  # gamma + beta
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
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
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
        planned_trajectory: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            history_occupancy: [B, T_h, X, Y, Z] - past occupancy grids
            history_poses: [B, T_h, 13] - past poses
            future_poses: [B, T_f, 13] - future poses (for training, optional)
            planned_trajectory: [B, T_f, 13] - planned/commanded trajectory for
                action-conditioned generation (optional). When provided, this
                trajectory is used for FiLM conditioning instead of the predicted
                poses, enabling "what will I see if I fly along this path?" queries.
                The pose predictor still runs and its output is included in results.

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

        # Predict future poses FIRST (needed for occupancy conditioning)
        last_pose = history_poses[:, -1, :]  # [B, 13]
        predicted_future_poses = self.future_pose_rnn(
            last_pose, combined_context, self.config.future_frames
        )  # [B, T_f, 13]

        # Decode future occupancy, conditioned on predicted poses via FiLM
        # Project to spatial dimensions
        spatial_for_decode = self.to_spatial(combined_context)  # [B, latent_dim * X' * Y' * Z']
        spatial_for_decode = spatial_for_decode.view(
            B, self.config.latent_dim,
            self.encoded_size[0], self.encoded_size[1], self.encoded_size[2]
        )

        # FiLM: modulate spatial features with pose information
        # Use planned trajectory for conditioning if provided (action-conditioned generation)
        film_poses = planned_trajectory if planned_trajectory is not None else predicted_future_poses
        pose_flat = film_poses.reshape(B, -1)  # [B, T_f * pose_dim]
        film_params = self.pose_film(pose_flat)  # [B, latent_dim * 2]
        gamma = film_params[:, :self.config.latent_dim].view(B, self.config.latent_dim, 1, 1, 1)
        beta = film_params[:, self.config.latent_dim:].view(B, self.config.latent_dim, 1, 1, 1)
        spatial_for_decode = spatial_for_decode * (1 + gamma) + beta

        future_occ_logits = self.spatial_decoder(spatial_for_decode)  # [B, T_f, X, Y, Z]
        future_occupancy = torch.sigmoid(future_occ_logits)
        
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
    """Focal Loss for sparse occupancy with optional dynamic alpha."""

    def __init__(self, alpha: float = 0.99, gamma: float = 2.0, dynamic_alpha: bool = False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dynamic_alpha = dynamic_alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(min=1e-7, max=1 - 1e-7)
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        p_t = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - p_t) ** self.gamma
        alpha = self._get_alpha(target)
        alpha_weight = torch.where(target == 1, alpha, 1 - alpha)
        return (alpha_weight * focal_weight * bce).mean()

    def _get_alpha(self, target: torch.Tensor) -> float:
        """Return alpha, adapting to batch occupancy ratio if dynamic."""
        if self.dynamic_alpha:
            occupied_ratio = target.sum() / target.numel()
            return (1.0 - occupied_ratio).clamp(min=0.5, max=0.999)
        return self.alpha


class LovaszBinaryLoss(nn.Module):
    """Lovász-Softmax loss for binary occupancy prediction.

    Directly optimizes the IoU (Jaccard index) using the Lovász extension,
    a tight convex surrogate that computes exact subgradients of IoU.

    Reference: Berman et al., "The Lovász-Softmax loss" (CVPR 2018)
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1).float()

        if target_flat.numel() == 0:
            return pred_flat.sum() * 0.0

        errors = (target_flat - pred_flat).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        target_sorted = target_flat[perm]

        grad = self._lovasz_grad(target_sorted)
        return torch.dot(errors_sorted, grad)

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        """Gradient of Lovász extension w.r.t sorted errors (Alg. 1 in paper)."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1.0 - gt_sorted).cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard


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
        dynamic_alpha: bool = True,
        dice_weight: float = 1.0,
        mean_weight: float = 10.0,
        lovasz_weight: float = 1.0,
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
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, dynamic_alpha=dynamic_alpha)
        self.dice_weight = dice_weight
        self.mean_weight = mean_weight
        self.lovasz_loss = LovaszBinaryLoss()
        self.lovasz_weight = lovasz_weight

        # Anti-collapse parameters
        self.pose_variance_weight = pose_variance_weight
        self.min_pose_std = min_pose_std
        self.uncertainty_min = uncertainty_min
        self.uncertainty_max = uncertainty_max
        self.triplet_margin = triplet_margin

        if self.uncertainty_max <= self.uncertainty_min:
            raise ValueError(
                f"uncertainty_max ({self.uncertainty_max}) must be > "
                f"uncertainty_min ({self.uncertainty_min})"
            )
        # Center bounded uncertainty so zero logits map to sigma ~= 1.0.
        target_sigma = min(
            max(1.0, self.uncertainty_min + 1e-6),
            self.uncertainty_max - 1e-6,
        )
        ratio = (target_sigma - self.uncertainty_min) / (self.uncertainty_max - self.uncertainty_min)
        ratio = min(max(float(ratio), 1e-6), 1 - 1e-6)
        self.uncertainty_logit_bias = math.log(ratio / (1.0 - ratio))

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

        # Lovász loss: directly optimizes IoU
        lovasz = self.lovasz_loss(pred_occ, target_occ)

        occ_loss = focal + self.dice_weight * dice + self.mean_weight * mean_loss + self.lovasz_weight * lovasz
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

            # === ANTI-COLLAPSE: Smooth bounded uncertainty ===
            # Map raw logits to (uncertainty_min, uncertainty_max). A learned
            # logit=0 corresponds to sigma~=1.0, which is a stable default.
            unc_range = self.uncertainty_max - self.uncertainty_min
            uncertainty_bounded = self.uncertainty_min + unc_range * torch.sigmoid(
                uncertainty + self.uncertainty_logit_bias
            )

            # Negative log-likelihood with learned uncertainty
            # L = 0.5 * (error^2 / sigma^2 + log(sigma^2))
            pos_error = (pred_pos - target_pos) ** 2
            pos_sigma = uncertainty_bounded[..., :3]
            eps = 1e-8
            pos_sigma_sq = pos_sigma ** 2
            pos_nll = 0.5 * (pos_error / (pos_sigma_sq + eps) + torch.log(pos_sigma_sq.clamp(min=eps)))

            # Regularize uncertainty toward 1.0 (moderate confidence)
            uncertainty_reg = ((uncertainty_bounded - 1.0) ** 2).mean() * 0.1

            losses['uncertainty'] = (pos_nll.mean() + uncertainty_reg) * self.uncertainty_weight

            # Track debug metrics
            self.debug_metrics['uncertainty_mean'] = uncertainty_bounded.mean().item()
            self.debug_metrics['uncertainty_min'] = uncertainty_bounded.min().item()
            self.debug_metrics['uncertainty_max'] = uncertainty_bounded.max().item()

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
        Compute triplet loss with in-batch hard positive/negative mining.

        Since DataLoader shuffles batches, we cannot assume adjacent samples
        are semantically related. Instead we use embedding-space similarity:
        - Positive: closest non-identical sample (hard positive)
        - Negative: farthest sample that violates the margin (hard negative)

        This encourages the embedding space to spread out while keeping
        genuinely similar embeddings close, without requiring batch ordering.
        """
        B, D = embeddings.shape
        if B < 3:
            return torch.tensor(0.0, device=embeddings.device)

        # Pairwise distances [B, B]
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Mask out self-distances
        self_mask = torch.eye(B, dtype=torch.bool, device=embeddings.device)

        # Hard positive: closest sample (smallest distance, excluding self)
        pos_distances = distances.clone()
        pos_distances[self_mask] = float('inf')
        pos_idx = pos_distances.argmin(dim=1)  # [B]
        d_ap = distances[torch.arange(B, device=embeddings.device), pos_idx]

        # Hard negative: closest sample that is NOT the positive (semi-hard mining)
        neg_distances = distances.clone()
        neg_distances[self_mask] = float('inf')
        neg_distances[torch.arange(B, device=embeddings.device), pos_idx] = float('inf')

        if (neg_distances < float('inf')).any(dim=1).all():
            neg_idx = neg_distances.argmin(dim=1)  # [B]
            d_an = distances[torch.arange(B, device=embeddings.device), neg_idx]
            triplet = F.relu(d_ap - d_an + self.triplet_margin)
            return triplet.mean()

        return torch.tensor(0.0, device=embeddings.device)

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
