from __future__ import annotations

import torch
from torch import nn


def get_loss(name: str = "bce_with_logits") -> nn.Module:
    if name != "bce_with_logits":
        raise ValueError(f"Unsupported loss: {name}")
    return nn.BCEWithLogitsLoss()


def sigmoid_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.detach())

