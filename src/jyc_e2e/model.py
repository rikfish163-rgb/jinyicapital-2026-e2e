"""Two comparable score encoders over the same raw-field contract."""

import torch
from torch import nn

from .contracts import FeatureLayout


class RawMinuteEncoder(nn.Module):
    """Learn within-minute interactions while retaining field and missingness identity."""

    def __init__(self, layout: FeatureLayout, width: int = 64) -> None:
        super().__init__()
        if width < 2:
            raise ValueError("encoder width must be at least 2")
        self.n_fields = len(layout.fields)
        self.group_indices = layout.indices
        self.group_names = tuple(self.group_indices)
        self.branches = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(2 * len(indices), width), nn.GELU())
                for name, indices in self.group_indices.items()
            }
        )
        self.fuse = nn.Sequential(
            nn.Linear(width * len(self.group_names), width), nn.GELU(), nn.LayerNorm(width)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape != mask.shape or x.shape[-1] != self.n_fields:
            raise ValueError("x and mask must be [batch, lookback, fields]")
        if mask.dtype != torch.bool:
            raise TypeError("mask must be boolean")
        valid_time = mask.any(dim=-1)
        if not valid_time.any(dim=1).all():
            raise ValueError("every sample needs at least one observed minute")
        encoded = []
        for name in self.group_names:
            indices = self.group_indices[name]
            values = torch.where(mask[..., indices], x[..., indices], 0.0)
            parts = torch.cat((values, mask[..., indices].to(x.dtype)), dim=-1)
            encoded.append(self.branches[name](parts))
        tokens = self.fuse(torch.cat(encoded, dim=-1))
        return tokens * valid_time.unsqueeze(-1), valid_time


class ScoreHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2), nn.GELU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return self.net(representation).squeeze(-1)


def _last_valid(sequence: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
    last = positions.masked_fill(~valid, -1).max(dim=1).values
    if (last < 0).any():
        raise ValueError("cannot pool an empty history")
    return sequence[torch.arange(len(last), device=last.device), last]


class GRUScoreModel(nn.Module):
    """Lightweight A0/A1 baseline using one raw minute per time step."""

    def __init__(self, layout: FeatureLayout, width: int = 64, layers: int = 2) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("GRU needs at least one layer")
        self.minute = RawMinuteEncoder(layout, width)
        self.encoder = nn.GRU(width, width, num_layers=layers, batch_first=True)
        self.head = ScoreHead(width)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens, valid = self.minute(x, mask)
        batch, lookback, width = tokens.shape
        lengths = valid.sum(dim=1)
        destinations = (valid.long().cumsum(dim=1) - 1).clamp_min(0)
        compact = tokens.new_zeros(batch, lookback, width)
        compact.scatter_add_(
            1, destinations.unsqueeze(-1).expand_as(tokens), tokens
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            compact, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.encoder(packed)
        return self.head(state[-1])


class PatchTransformerScoreModel(nn.Module):
    """Nonoverlapping learned patches over already available minute records."""

    def __init__(
        self,
        layout: FeatureLayout,
        width: int = 64,
        patch_size: int = 5,
        layers: int = 2,
        heads: int = 4,
        max_patches: int = 64,
    ) -> None:
        super().__init__()
        if patch_size < 1 or max_patches < 1 or layers < 1 or heads < 1 or width < 2 or width % heads:
            raise ValueError("invalid patch size, capacity, or attention head count")
        self.minute = RawMinuteEncoder(layout, width)
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.patch = nn.Linear(patch_size * width, width)
        self.position = nn.Embedding(max_patches, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = ScoreHead(width)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens, valid = self.minute(x, mask)
        batch, lookback, width = tokens.shape
        pad = (-lookback) % self.patch_size
        if pad:
            tokens = torch.cat((tokens.new_zeros(batch, pad, width), tokens), dim=1)
            valid = torch.cat((valid.new_zeros(batch, pad), valid), dim=1)
        count = tokens.shape[1] // self.patch_size
        if count > self.max_patches:
            raise ValueError("lookback exceeds configured patch position capacity")
        patch_valid = valid.reshape(batch, count, self.patch_size).any(dim=-1)
        patches = self.patch(tokens.reshape(batch, count, self.patch_size * width))
        positions = torch.arange(
            self.max_patches - count, self.max_patches, device=x.device
        )
        patches = patches + self.position(positions)[None, :, :]
        encoded = self.encoder(patches, src_key_padding_mask=~patch_valid)
        return self.head(_last_valid(encoded, patch_valid))
