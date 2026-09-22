"""Rotation-only calibrators used by the CHI 2027 experiments.

All variants consume ``[B, T, D, 12]`` tensors (scaled acceleration followed
by a flattened rotation matrix) and directly predict complete rotations.  No
residual rotation, reliability head, or downstream task loss is used.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import articulate as art


def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1
    )


class _AttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.norm1(x)
        y, _ = self.attn(
            q,
            q,
            q,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + y
        return x + self.ff(self.norm2(x))


class _BaseCalibrator(nn.Module):
    def __init__(
        self,
        combo_size: int = 3,
        input_dim_per_device: int = 12,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 3,
        nhead: int = 4,
        max_seq_len: int = 125,
    ) -> None:
        super().__init__()
        self.combo_size = combo_size
        self.input_dim_per_device = input_dim_per_device
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.nhead = nhead
        self.max_seq_len = max_seq_len

    def _check(self, x: torch.Tensor) -> tuple[int, int, int, int]:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,T,D,C], got {tuple(x.shape)}")
        shape = tuple(x.shape)
        if shape[2:] != (self.combo_size, self.input_dim_per_device):
            raise ValueError(
                f"Expected D,C={(self.combo_size, self.input_dim_per_device)}, "
                f"got {shape[2:]}"
            )
        if shape[1] > self.max_seq_len:
            raise ValueError(f"T={shape[1]} exceeds max_seq_len={self.max_seq_len}")
        return shape

    @staticmethod
    def _to_rotation(prediction_6d: torch.Tensor) -> torch.Tensor:
        return art.math.r6d_to_rotation_matrix(prediction_6d.reshape(-1, 6)).view(
            *prediction_6d.shape[:-1], 3, 3
        )


class PlainTransformerCalibrator(_BaseCalibrator):
    """Naive per-frame device concatenation followed by a temporal Transformer."""

    model_type = "plain_transformer_abs"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_proj = nn.Linear(
            self.combo_size * self.input_dim_per_device, self.hidden_dim
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.max_seq_len, self.hidden_dim)
        )
        self.blocks = nn.ModuleList(
            [
                _AttentionBlock(self.hidden_dim, self.nhead, self.dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, self.combo_size * 6)

    def forward(self, x: torch.Tensor, seq_mask: torch.Tensor | None = None):
        batch, length, _, _ = self._check(x)
        feat = self.input_proj(x.flatten(2)) + self.pos_embed[:, :length]
        padding = None if seq_mask is None else ~seq_mask
        causal = _causal_mask(length, feat.device)
        for block in self.blocks:
            feat = block(feat, attn_mask=causal, padding_mask=padding)
        pred6d = self.head(self.norm(feat)).view(
            batch, length, self.combo_size, 6
        )
        return self._to_rotation(pred6d), pred6d


class AxialTransformerCalibrator(_BaseCalibrator):
    """Factorized temporal/device attention with selectable axes.

    ``temporal=True, cross_device=True`` is the proposed alternating model.
    The single-axis settings are the two structural ablations.
    """

    model_type = "axial_cross_device_transformer_abs"

    def __init__(
        self,
        *,
        temporal: bool = True,
        cross_device: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not temporal and not cross_device:
            raise ValueError("At least one attention axis must be enabled")
        self.temporal = temporal
        self.cross_device = cross_device
        self.input_proj = nn.Linear(self.input_dim_per_device, self.hidden_dim)
        self.device_embed = nn.Embedding(self.combo_size, self.hidden_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.max_seq_len, 1, self.hidden_dim)
        )
        self.temporal_blocks = nn.ModuleList(
            [
                _AttentionBlock(self.hidden_dim, self.nhead, self.dropout)
                for _ in range(self.num_layers)
            ]
        ) if temporal else nn.ModuleList()
        self.device_blocks = nn.ModuleList(
            [
                _AttentionBlock(self.hidden_dim, self.nhead, self.dropout)
                for _ in range(self.num_layers)
            ]
        ) if cross_device else nn.ModuleList()
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, 6)

    def _temporal_step(
        self,
        feat: torch.Tensor,
        block: _AttentionBlock,
        seq_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, length, devices, hidden = feat.shape
        x = feat.permute(0, 2, 1, 3).reshape(batch * devices, length, hidden)
        padding = None
        if seq_mask is not None:
            padding = (~seq_mask).unsqueeze(1).expand(batch, devices, length)
            padding = padding.reshape(batch * devices, length)
        x = block(x, attn_mask=_causal_mask(length, feat.device), padding_mask=padding)
        return x.view(batch, devices, length, hidden).permute(0, 2, 1, 3)

    @staticmethod
    def _device_step(feat: torch.Tensor, block: _AttentionBlock) -> torch.Tensor:
        batch, length, devices, hidden = feat.shape
        x = feat.reshape(batch * length, devices, hidden)
        x = block(x)
        return x.view(batch, length, devices, hidden)

    def forward(self, x: torch.Tensor, seq_mask: torch.Tensor | None = None):
        batch, length, devices, _ = self._check(x)
        ids = torch.arange(devices, device=x.device)
        feat = self.input_proj(x)
        feat = feat + self.device_embed(ids).view(1, 1, devices, self.hidden_dim)
        feat = feat + self.pos_embed[:, :length]

        for layer in range(self.num_layers):
            if self.temporal:
                feat = self._temporal_step(feat, self.temporal_blocks[layer], seq_mask)
            if self.cross_device:
                feat = self._device_step(feat, self.device_blocks[layer])

        pred6d = self.head(self.norm(feat))
        return self._to_rotation(pred6d), pred6d


class FactorizedCrossDeviceCalibrator(_BaseCalibrator):
    """Temporal then cross-device attention with one shared FFN per layer."""

    model_type = "factorized_cross_device_transformer_abs"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_proj = nn.Linear(self.input_dim_per_device, self.hidden_dim)
        self.device_embed = nn.Embedding(self.combo_size, self.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, 1, self.hidden_dim))
        self.temporal_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.temporal_attn = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_dim, self.nhead, dropout=self.dropout, batch_first=True)
            for _ in range(self.num_layers)
        ])
        self.device_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.device_attn = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_dim, self.nhead, dropout=self.dropout, batch_first=True)
            for _ in range(self.num_layers)
        ])
        self.ff_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.ff = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 4), nn.GELU(), nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim), nn.Dropout(self.dropout),
            ) for _ in range(self.num_layers)
        ])
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, 6)

    def forward(self, x: torch.Tensor, seq_mask: torch.Tensor | None = None):
        batch, length, devices, _ = self._check(x)
        ids = torch.arange(devices, device=x.device)
        feat = self.input_proj(x)
        feat = feat + self.device_embed(ids).view(1, 1, devices, self.hidden_dim)
        feat = feat + self.pos_embed[:, :length]
        causal = _causal_mask(length, feat.device)
        padding = None
        if seq_mask is not None:
            padding = (~seq_mask).unsqueeze(1).expand(batch, devices, length)
            padding = padding.reshape(batch * devices, length)

        for layer in range(self.num_layers):
            temporal = feat.permute(0, 2, 1, 3).reshape(batch * devices, length, self.hidden_dim)
            query = self.temporal_norms[layer](temporal)
            update, _ = self.temporal_attn[layer](
                query, query, query, attn_mask=causal, key_padding_mask=padding, need_weights=False
            )
            temporal = temporal + update
            feat = temporal.view(batch, devices, length, self.hidden_dim).permute(0, 2, 1, 3)

            across = feat.reshape(batch * length, devices, self.hidden_dim)
            query = self.device_norms[layer](across)
            update, _ = self.device_attn[layer](query, query, query, need_weights=False)
            feat = (across + update).view(batch, length, devices, self.hidden_dim)
            feat = feat + self.ff[layer](self.ff_norms[layer](feat))

        pred6d = self.head(self.norm(feat))
        return self._to_rotation(pred6d), pred6d


MODEL_TYPES = {
    "plain": PlainTransformerCalibrator,
    "ours": AxialTransformerCalibrator,
    "temporal_only": AxialTransformerCalibrator,
    "cross_device_only": AxialTransformerCalibrator,
    "ours_factorized": FactorizedCrossDeviceCalibrator,
}


def build_model(model_type: str, **kwargs) -> _BaseCalibrator:
    if model_type == "plain":
        return PlainTransformerCalibrator(**kwargs)
    if model_type == "ours":
        return AxialTransformerCalibrator(
            temporal=True, cross_device=True, **kwargs
        )
    if model_type == "temporal_only":
        return AxialTransformerCalibrator(
            temporal=True, cross_device=False, **kwargs
        )
    if model_type == "cross_device_only":
        return AxialTransformerCalibrator(
            temporal=False, cross_device=True, **kwargs
        )
    if model_type == "ours_factorized":
        return FactorizedCrossDeviceCalibrator(**kwargs)
    raise ValueError(f"Unknown model type {model_type!r}; choices={sorted(MODEL_TYPES)}")
