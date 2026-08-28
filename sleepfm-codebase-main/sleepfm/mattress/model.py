"""Missing-modality-aware fusion model for personalized sleep assessment."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


class MattressFusionModel(nn.Module):
    """Fuse SleepFM, pressure, pose and identity features.

    Each input is a ``[batch, time, feature]`` tensor. Pressure maps may instead
    be ``[batch, time, 1, 64, 27]``. A missing modality can be omitted entirely;
    masks use ``True`` for valid time steps. The model returns one value per
    recording, suitable for a sleep-quality regression target.
    """

    def __init__(
        self,
        sleepfm_dim: int,
        pose_dim: int,
        identity_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if min(sleepfm_dim, pose_dim, identity_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        self.modalities = ("sleepfm", "pressure", "pose", "identity")
        self.pressure_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        input_dims = {
            "sleepfm": sleepfm_dim,
            "pressure": 64,
            "pose": pose_dim,
            "identity": identity_dim,
        }
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim))
                for name, dim in input_dims.items()
            }
        )
        self.modality_tokens = nn.Parameter(torch.randn(len(self.modalities), hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            nhead=8 if hidden_dim % 8 == 0 else 1,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim)
        )

    def _encode_pressure(self, pressure: Tensor) -> Tensor:
        if pressure.ndim != 5 or pressure.shape[2] != 1:
            raise ValueError("pressure must have shape [batch, time, 1, height, width]")
        batch, steps = pressure.shape[:2]
        return self.pressure_encoder(pressure.flatten(0, 1)).reshape(batch, steps, -1)

    def forward(
        self,
        features: Mapping[str, Tensor],
        masks: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        unknown = set(features) - set(self.modalities)
        if unknown:
            raise KeyError(f"unknown modalities: {sorted(unknown)}")
        if not features:
            raise ValueError("at least one modality is required")
        masks = masks or {}
        encoded, valid = [], []
        for index, name in enumerate(self.modalities):
            if name not in features:
                continue
            value = features[name]
            if name == "pressure":
                value = self._encode_pressure(value)
            if value.ndim != 3:
                raise ValueError(f"{name} must have shape [batch, time, feature]")
            projected = self.projections[name](value) + self.modality_tokens[index]
            mask = masks.get(name)
            if mask is None:
                mask = torch.ones(value.shape[:2], dtype=torch.bool, device=value.device)
            if mask.shape != value.shape[:2]:
                raise ValueError(f"{name} mask must have shape {tuple(value.shape[:2])}")
            encoded.append(projected)
            valid.append(mask.bool())

        tokens = torch.cat(encoded, dim=1)
        valid_tokens = torch.cat(valid, dim=1)
        if (~valid_tokens).all(dim=1).any():
            raise ValueError("every sample must contain at least one valid token")
        tokens = self.temporal_encoder(tokens, src_key_padding_mask=~valid_tokens)
        pooled = (tokens * valid_tokens.unsqueeze(-1)).sum(1) / valid_tokens.sum(1, keepdim=True)
        return self.head(pooled)
