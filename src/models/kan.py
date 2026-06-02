from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.embedding import FeatureEmbedding


class SharedScalarKANLayer(nn.Module):
    """One learnable scalar KAN function shared by every embedding value.

    This is a memory-light KAN-style edge function:

        phi(x) = base_weight * SiLU(x) + linear_spline(x) + bias

    The spline is evaluated by interpolation between neighboring knots, so
    forward does not materialize a large [batch, fields, dim, grid] tensor.
    """

    def __init__(
        self,
        grid_size: int = 5,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        use_base: bool = True,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if grid_size < 2:
            raise ValueError(f"grid_size must be >= 2, got {grid_size}")
        if grid_min >= grid_max:
            raise ValueError(f"grid_min must be smaller than grid_max, got {grid_min} >= {grid_max}")

        self.grid_size = int(grid_size)
        self.use_base = bool(use_base)
        self.register_buffer("grid_min", torch.tensor(float(grid_min)))
        self.register_buffer("grid_max", torch.tensor(float(grid_max)))
        self.register_buffer("grid_step", torch.tensor(float(grid_max - grid_min) / float(grid_size - 1)))

        self.spline_weight = nn.Parameter(torch.empty(self.grid_size))
        if self.use_base:
            self.base_weight = nn.Parameter(torch.ones(()))
        else:
            self.register_parameter("base_weight", None)
        self.bias = nn.Parameter(torch.zeros(()))
        self.reset_parameters(init_scale)

    def reset_parameters(self, init_scale: float = 0.01) -> None:
        nn.init.normal_(self.spline_weight, mean=0.0, std=init_scale)
        if self.base_weight is not None:
            self.base_weight.data.fill_(1.0)
        self.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid_min = self.grid_min.to(device=x.device, dtype=x.dtype)
        grid_max = self.grid_max.to(device=x.device, dtype=x.dtype)
        grid_step = self.grid_step.to(device=x.device, dtype=x.dtype)

        clipped = torch.minimum(torch.maximum(x, grid_min), grid_max)
        position = (clipped - grid_min) / grid_step
        lower_idx = torch.floor(position).to(torch.long).clamp(min=0, max=self.grid_size - 2)
        upper_idx = lower_idx + 1
        frac = (position - lower_idx.to(position.dtype)).clamp(0.0, 1.0)

        spline_weight = self.spline_weight.to(dtype=x.dtype)
        lower = spline_weight[lower_idx]
        upper = spline_weight[upper_idx]
        out = lower * (1.0 - frac) + upper * frac

        if self.base_weight is not None:
            out = out + self.base_weight.to(dtype=x.dtype) * F.silu(x)
        return out + self.bias.to(dtype=x.dtype)


class KANBranch(nn.Module):
    """Additive branch using shared scalar KAN functions over embeddings.

    Default sharing is per field: every scalar in one field embedding uses
    the same KAN function, while different fields keep different functions.
    """

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        grid_size: int = 5,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        dropout: float = 0.0,
        use_base: bool = True,
        init_scale: float = 0.01,
        share_mode: str = "field",
    ) -> None:
        super().__init__()
        if num_fields <= 0:
            raise ValueError(f"num_fields must be positive, got {num_fields}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if share_mode not in {"field", "global"}:
            raise ValueError(f"share_mode must be 'field' or 'global', got {share_mode}")

        self.num_fields = int(num_fields)
        self.embedding_dim = int(embedding_dim)
        self.share_mode = share_mode
        num_scalar_functions = self.num_fields if share_mode == "field" else 1
        self.scalar_kans = nn.ModuleList(
            [
                SharedScalarKANLayer(
                    grid_size=grid_size,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    use_base=use_base,
                    init_scale=init_scale,
                )
                for _ in range(num_scalar_functions)
            ]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dim_weight = nn.Parameter(torch.full((self.embedding_dim,), 1.0 / float(self.embedding_dim)))
        self.field_weight = nn.Parameter(torch.ones(self.num_fields))
        self.field_bias = nn.Parameter(torch.zeros(self.num_fields))
        self.bias = nn.Parameter(torch.zeros(1))

    def compute_input_stats(
        self,
        embeddings: torch.Tensor,
        quantiles: list[float] | tuple[float, ...] | None = None,
    ) -> dict[str, object]:
        if embeddings.ndim != 3:
            raise ValueError(f"Expected embeddings with shape [batch, fields, dim], got {tuple(embeddings.shape)}")
        if embeddings.size(1) != self.num_fields:
            raise ValueError(f"Expected {self.num_fields} fields, got {embeddings.size(1)}")
        if embeddings.size(2) != self.embedding_dim:
            raise ValueError(f"Expected embedding_dim={self.embedding_dim}, got {embeddings.size(2)}")

        quantiles = quantiles or [0.001, 0.01, 0.5, 0.99, 0.999]
        flat = embeddings.detach().float().reshape(-1)
        if flat.numel() == 0:
            raise ValueError("Cannot compute KAN input stats for an empty embedding tensor")

        levels = torch.tensor(quantiles, device=flat.device, dtype=flat.dtype).clamp(0.0, 1.0)
        values = torch.quantile(flat, levels)
        grid_min = self.scalar_kans[0].grid_min.to(device=flat.device, dtype=flat.dtype)
        grid_max = self.scalar_kans[0].grid_max.to(device=flat.device, dtype=flat.dtype)
        below_grid_frac = (flat < grid_min).float().mean()
        above_grid_frac = (flat > grid_max).float().mean()

        return {
            "num_fields": self.num_fields,
            "embedding_dim": self.embedding_dim,
            "share_mode": self.share_mode,
            "num_values": int(flat.numel()),
            "min": float(flat.min().detach().cpu()),
            "max": float(flat.max().detach().cpu()),
            "mean": float(flat.mean().detach().cpu()),
            "std": float(flat.std(unbiased=False).detach().cpu()),
            "quantiles": {
                f"q{float(level.detach().cpu()):g}": float(value.detach().cpu())
                for level, value in zip(levels, values, strict=True)
            },
            "grid_min": float(grid_min.detach().cpu()),
            "grid_max": float(grid_max.detach().cpu()),
            "below_grid_frac": float(below_grid_frac.detach().cpu()),
            "above_grid_frac": float(above_grid_frac.detach().cpu()),
            "clipped_frac": float((below_grid_frac + above_grid_frac).detach().cpu()),
        }

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 3:
            raise ValueError(f"Expected embeddings with shape [batch, fields, dim], got {tuple(embeddings.shape)}")
        if embeddings.size(1) != self.num_fields:
            raise ValueError(f"Expected {self.num_fields} fields, got {embeddings.size(1)}")
        if embeddings.size(2) != self.embedding_dim:
            raise ValueError(f"Expected embedding_dim={self.embedding_dim}, got {embeddings.size(2)}")

        if self.share_mode == "global":
            transformed = self.scalar_kans[0](embeddings)
        else:
            transformed = torch.stack(
                [scalar_kan(embeddings[:, idx, :]) for idx, scalar_kan in enumerate(self.scalar_kans)],
                dim=1,
            )
        transformed = self.dropout(transformed)
        dim_weight = self.dim_weight.to(dtype=transformed.dtype).view(1, 1, -1)
        field_weight = self.field_weight.to(dtype=transformed.dtype).view(1, -1)
        field_bias = self.field_bias.to(dtype=transformed.dtype).view(1, -1)

        contributions = (transformed * dim_weight).sum(dim=2)
        contributions = contributions * field_weight + field_bias
        logits = contributions.sum(dim=1) + self.bias.to(dtype=transformed.dtype).squeeze(0)
        return logits, contributions


class KAN(nn.Module):
    """CTR additive model with shared-scalar KAN feature contributions."""

    def __init__(
        self,
        field_dims: list[int],
        embedding_dim: int = 16,
        grid_size: int = 5,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        dropout: float = 0.0,
        use_base: bool = True,
        init_scale: float = 0.01,
        share_mode: str = "field",
    ) -> None:
        super().__init__()
        self.embedding = FeatureEmbedding(field_dims, embedding_dim)
        self.kan = KANBranch(
            len(field_dims),
            embedding_dim,
            grid_size=grid_size,
            grid_min=grid_min,
            grid_max=grid_max,
            dropout=dropout,
            use_base=use_base,
            init_scale=init_scale,
            share_mode=share_mode,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits, contributions = self.kan(self.embedding(x))
        return {
            "logits": logits,
            "kan_logits": logits,
            "feature_contributions": contributions,
        }
