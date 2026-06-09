from __future__ import annotations

import torch
from torch import nn

from src.models.embedding import FeatureEmbedding
from src.models.fin import FINBranch
from src.models.kan_v2 import FieldTiedKANOutputLayer


class KANFINV2(nn.Module):
    """Direct-output field-tied KAN branch plus FIN interaction branch."""

    def __init__(
        self,
        field_dims: list[int],
        embedding_dim: int = 16,
        kan_grid_size: int = 5,
        kan_degree: int = 1,
        kan_grid_min: float = -2.0,
        kan_grid_max: float = 2.0,
        kan_dropout: float = 0.0,
        kan_use_base: bool = True,
        kan_init_scale: float = 0.01,
        fin_num_heads: int = 4,
        fin_num_layers: int = 2,
        fin_dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = FeatureEmbedding(field_dims, embedding_dim)
        self.kan = FieldTiedKANOutputLayer(
            len(field_dims),
            embedding_dim,
            grid_size=kan_grid_size,
            degree=kan_degree,
            grid_min=kan_grid_min,
            grid_max=kan_grid_max,
            dropout=kan_dropout,
            use_base=kan_use_base,
            init_scale=kan_init_scale,
        )
        self.fin = FINBranch(
            len(field_dims),
            embedding_dim,
            num_heads=fin_num_heads,
            num_layers=fin_num_layers,
            dropout=fin_dropout,
            use_residual=use_residual,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = self.embedding(x)
        kan_logits, feature_contributions, edge_contributions = self.kan(embeddings)
        fin_logits, attention_weights = self.fin(embeddings)
        logits = kan_logits + fin_logits
        return {
            "logits": logits,
            "kan_logits": kan_logits,
            "fin_logits": fin_logits,
            "attention_weights": attention_weights,
            "feature_contributions": feature_contributions,
            "edge_contributions": edge_contributions,
            "embedding_contributions": edge_contributions,
        }
