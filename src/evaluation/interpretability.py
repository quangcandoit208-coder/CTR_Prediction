from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_nafi_interpretability(
    model: torch.nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    max_batches: int = 50,
) -> dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    use_amp = bool(config.get("training", {}).get("mixed_precision", True)) and device.type == "cuda"

    contributions: list[np.ndarray] = []
    attention_matrices: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(x)

        logits = output["logits"]
        predictions.append(torch.sigmoid(logits).detach().float().cpu().numpy())

        if "feature_contributions" in output and output["feature_contributions"] is not None:
            contributions.append(output["feature_contributions"].detach().float().cpu().numpy())

        if "attention_weights" in output and output["attention_weights"] is not None:
            attn = output["attention_weights"].detach().float().cpu()
            # torch MultiheadAttention with batch_first returns [batch, heads, fields, fields]
            if attn.ndim == 4:
                attention_matrices.append(attn.mean(dim=(0, 1)).numpy())
            elif attn.ndim == 3:
                attention_matrices.append(attn.mean(dim=0).numpy())

    result: dict[str, np.ndarray] = {}
    if contributions:
        result["feature_contributions"] = np.concatenate(contributions, axis=0)
    if attention_matrices:
        result["attention_matrix"] = np.mean(np.stack(attention_matrices, axis=0), axis=0)
    if predictions:
        result["predictions"] = np.concatenate(predictions, axis=0)
    return result


def summarize_feature_contributions(values: np.ndarray, feature_names: list[str]) -> list[dict[str, float | str]]:
    mean_contribution = values.mean(axis=0)
    mean_abs_contribution = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs_contribution)[::-1]
    return [
        {
            "feature": feature_names[idx],
            "mean_contribution": float(mean_contribution[idx]),
            "mean_abs_contribution": float(mean_abs_contribution[idx]),
        }
        for idx in order
    ]


def summarize_attention_matrix(matrix: np.ndarray, feature_names: list[str], top_k: int = 30) -> list[dict[str, float | str]]:
    pairs = []
    for i, source in enumerate(feature_names):
        for j, target in enumerate(feature_names):
            pairs.append({"source": source, "target": target, "attention": float(matrix[i, j])})
    pairs.sort(key=lambda item: item["attention"], reverse=True)
    return pairs[:top_k]

