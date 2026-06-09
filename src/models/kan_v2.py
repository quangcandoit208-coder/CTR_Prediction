from __future__ import annotations

import torch
from torch import nn

from src.models.embedding import FeatureEmbedding
from src.models.kan import SharedScalarKANLayer


class FieldTiedKANOutputLayer(nn.Module):
    """Direct KAN layer from all embedding scalars to one output node.

    A standard one-output KAN layer sums the learned edge functions from all
    input nodes. Here, the D edges that belong to the same feature share one
    scalar function phi_f, while different features keep different functions.
    """

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        grid_size: int = 5,
        degree: int = 1,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        dropout: float = 0.0,
        use_base: bool = True,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if num_fields <= 0:
            raise ValueError(f"num_fields must be positive, got {num_fields}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

        self.num_fields = int(num_fields)
        self.embedding_dim = int(embedding_dim)
        self.share_mode = "field"
        self.topology = "direct_output"
        self.scalar_kans = nn.ModuleList(
            [
                SharedScalarKANLayer(
                    grid_size=grid_size,
                    degree=degree,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    use_base=use_base,
                    init_scale=init_scale,
                )
                for _ in range(self.num_fields)
            ]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.bias = nn.Parameter(torch.zeros(1))

    def compute_input_stats(
        self,
        embeddings: torch.Tensor,
        quantiles: list[float] | tuple[float, ...] | None = None,
    ) -> dict[str, object]:
        self._validate_embeddings(embeddings)
        quantiles = quantiles or [0.001, 0.01, 0.5, 0.99, 0.999]
        flat = embeddings.detach().float().reshape(-1)
        if flat.numel() == 0:
            raise ValueError("Cannot compute KAN input stats for an empty embedding tensor")

        levels = torch.tensor(quantiles, device=flat.device, dtype=flat.dtype).clamp(0.0, 1.0)
        values = torch.quantile(flat, levels)
        first_scalar = self.scalar_kans[0]
        grid_min = first_scalar.grid_min.to(device=flat.device, dtype=flat.dtype)
        grid_max = first_scalar.grid_max.to(device=flat.device, dtype=flat.dtype)
        below_grid_frac = (flat < grid_min).float().mean()
        above_grid_frac = (flat > grid_max).float().mean()

        return {
            "num_fields": self.num_fields,
            "embedding_dim": self.embedding_dim,
            "share_mode": self.share_mode,
            "topology": self.topology,
            "grid_size": first_scalar.grid_size,
            "degree": first_scalar.degree,
            "num_basis": first_scalar.num_basis,
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

    def _validate_embeddings(self, embeddings: torch.Tensor) -> None:
        if embeddings.ndim != 3:
            raise ValueError(f"Expected embeddings with shape [batch, fields, dim], got {tuple(embeddings.shape)}")
        if embeddings.size(1) != self.num_fields:
            raise ValueError(f"Expected {self.num_fields} fields, got {embeddings.size(1)}")
        if embeddings.size(2) != self.embedding_dim:
            raise ValueError(f"Expected embedding_dim={self.embedding_dim}, got {embeddings.size(2)}")

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_embeddings(embeddings)

        edge_contributions = torch.stack(
            [scalar_kan(embeddings[:, idx, :]) for idx, scalar_kan in enumerate(self.scalar_kans)],
            dim=1,
        )
        edge_contributions = self.dropout(edge_contributions)

        # These grouped contributions are derived attributions. The KAN output
        # node itself directly sums every transformed scalar edge.
        feature_contributions = edge_contributions.sum(dim=2)
        logits = edge_contributions.sum(dim=(1, 2)) + self.bias.to(dtype=edge_contributions.dtype).squeeze(0)
        return logits, feature_contributions, edge_contributions


class KANV2(nn.Module):
    """Standalone direct-output KAN with field-tied edge functions."""

    def __init__(
        self,
        field_dims: list[int],
        embedding_dim: int = 16,
        grid_size: int = 5,
        degree: int = 1,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        dropout: float = 0.0,
        use_base: bool = True,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.embedding = FeatureEmbedding(field_dims, embedding_dim)
        self.kan = FieldTiedKANOutputLayer(
            len(field_dims),
            embedding_dim,
            grid_size=grid_size,
            degree=degree,
            grid_min=grid_min,
            grid_max=grid_max,
            dropout=dropout,
            use_base=use_base,
            init_scale=init_scale,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits, feature_contributions, edge_contributions = self.kan(self.embedding(x))
        return {
            "logits": logits,
            "kan_logits": logits,
            "feature_contributions": feature_contributions,
            "edge_contributions": edge_contributions,
            "embedding_contributions": edge_contributions,
        }
