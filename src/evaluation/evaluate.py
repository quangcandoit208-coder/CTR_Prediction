from __future__ import annotations

from typing import Any

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from src.training.metrics import compute_auc, compute_logloss


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, config: dict[str, Any]) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    use_amp = bool(config.get("training", {}).get("mixed_precision", True)) and device.type == "cuda"
    y_true: list[float] = []
    y_pred: list[float] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(x)["logits"]
        y_pred.extend(torch.sigmoid(logits).detach().float().cpu().numpy().tolist())
        y_true.extend(y.detach().float().cpu().numpy().tolist())
    return {"auc": compute_auc(y_true, y_pred), "logloss": compute_logloss(y_true, y_pred)}

