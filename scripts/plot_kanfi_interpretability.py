from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.data.dataloader import make_dataloader
from src.data.dataset import AvazuParquetDataset
from src.data.metadata import load_metadata
from src.evaluation.interpretability import (
    collect_nafi_interpretability,
    summarize_attention_matrix,
)
from src.evaluation.plots import plot_attention_heatmap, plot_feature_importance
from src.models.base import build_model
from src.training.checkpoint import load_checkpoint
from src.utils.config import ensure_dirs, load_config


MODEL_CHOICES = ["kan", "kanfin", "kan-fin", "kafi"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot KANFI interpretability outputs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--model", default=None, choices=MODEL_CHOICES)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--top-features", type=int, default=15)
    parser.add_argument("--function-points", type=int, default=300)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def get_kan_branch(model: nn.Module) -> nn.Module:
    base_model = unwrap_model(model)
    kan_branch = getattr(base_model, "kan", None)
    if kan_branch is None:
        raise ValueError("Model does not expose a KAN branch named 'kan'. Use --model kan or --model kanfin.")
    return kan_branch


def contribution_summary(values: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    mean_contribution = values.mean(axis=0)
    mean_abs_contribution = np.abs(values).mean(axis=0)
    total_abs = float(mean_abs_contribution.sum())
    importance = mean_abs_contribution / (total_abs + 1e-12)
    order = np.argsort(mean_abs_contribution)[::-1]
    summary = [
        {
            "feature": feature_names[idx],
            "mean_contribution": float(mean_contribution[idx]),
            "mean_abs_contribution": float(mean_abs_contribution[idx]),
            "importance_percent": float(importance[idx] * 100.0),
        }
        for idx in order
    ]
    return order, summary


def get_kan_parameters(kan_branch: nn.Module) -> dict[str, np.ndarray | float | int | str]:
    scalar_kans = getattr(kan_branch, "scalar_kans", None)
    if scalar_kans is None or len(scalar_kans) == 0:
        raise ValueError("KAN branch does not expose scalar_kans.")
    first_scalar = scalar_kans[0]
    return {
        "share_mode": str(getattr(kan_branch, "share_mode", "unknown")),
        "grid_size": int(getattr(first_scalar, "grid_size", 0)),
        "degree": int(getattr(first_scalar, "degree", 1)),
        "num_basis": int(getattr(first_scalar, "num_basis", 0)),
        "grid_min": float(first_scalar.grid_min.detach().cpu()),
        "grid_max": float(first_scalar.grid_max.detach().cpu()),
        "dim_weight": kan_branch.dim_weight.detach().float().cpu().numpy(),
        "field_weight": kan_branch.field_weight.detach().float().cpu().numpy(),
        "field_bias": kan_branch.field_bias.detach().float().cpu().numpy(),
        "branch_bias": float(kan_branch.bias.detach().float().cpu().squeeze()),
    }


def plot_kan_feature_weights(
    contributions: np.ndarray,
    feature_names: list[str],
    kan_params: dict[str, Any],
    output_path: str | Path,
    top_features: int,
) -> None:
    mean_contribution = contributions.mean(axis=0)
    mean_abs_contribution = np.abs(contributions).mean(axis=0)
    importance = mean_abs_contribution / (mean_abs_contribution.sum() + 1e-12)
    order = np.argsort(mean_abs_contribution)[::-1][:top_features]
    labels = [feature_names[idx] for idx in order]

    field_weight = np.asarray(kan_params["field_weight"])
    field_bias = np.asarray(kan_params["field_bias"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    axes[0].bar(labels, importance[order] * 100.0)
    axes[0].set_title("KAN Feature Importance")
    axes[0].set_ylabel("mean |contribution| (%)")

    colors = ["tab:blue" if value >= 0 else "tab:red" for value in mean_contribution[order]]
    axes[1].bar(labels, mean_contribution[order], color=colors)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Mean Signed Contribution")
    axes[1].set_ylabel("logit contribution")

    axes[2].bar(labels, field_weight[order])
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_title("Learned Field Weight")
    axes[2].set_ylabel("field_weight")

    axes[3].bar(labels, field_bias[order])
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set_title("Learned Field Bias")
    axes[3].set_ylabel("field_bias")

    for axis in axes:
        axis.tick_params(axis="x", rotation=60)
        for label in axis.get_xticklabels():
            label.set_ha("right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_kan_schematic(
    contributions: np.ndarray,
    feature_names: list[str],
    kan_params: dict[str, Any],
    output_path: str | Path,
    top_features: int,
) -> None:
    mean_contribution = contributions.mean(axis=0)
    mean_abs_contribution = np.abs(contributions).mean(axis=0)
    importance = mean_abs_contribution / (mean_abs_contribution.sum() + 1e-12)
    order = np.argsort(mean_abs_contribution)[::-1][:top_features]
    field_weight = np.asarray(kan_params["field_weight"])
    share_mode = str(kan_params["share_mode"])

    height = max(6.0, 0.48 * len(order) + 2.0)
    fig, ax = plt.subplots(figsize=(13, height))
    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(-1, len(order))
    ax.axis("off")

    ax.text(0.0, len(order) - 0.15, "Feature embedding", ha="center", va="bottom", fontsize=11, weight="bold")
    ax.text(1.35, len(order) - 0.15, "KAN scalar function", ha="center", va="bottom", fontsize=11, weight="bold")
    ax.text(2.75, len(order) - 0.15, "Feature contribution", ha="center", va="bottom", fontsize=11, weight="bold")

    max_importance = float(importance[order].max()) if len(order) else 1.0
    for row, idx in enumerate(order):
        y = len(order) - row - 1
        width = 1.0 + 7.0 * float(importance[idx] / (max_importance + 1e-12))
        color = "tab:blue" if mean_contribution[idx] >= 0 else "tab:red"
        phi_name = "phi" if share_mode == "global" else f"phi_{feature_names[idx]}"

        ax.scatter([0.0, 1.35, 2.75], [y, y, y], s=[220, 260, 220], color=["#d9e8ff", "#fff3bf", "#d7f5dd"], edgecolor="black")
        ax.plot([0.17, 1.17], [y, y], color=color, linewidth=width, alpha=0.55)
        ax.plot([1.53, 2.57], [y, y], color=color, linewidth=width, alpha=0.55)
        ax.text(-0.08, y, feature_names[idx], ha="right", va="center", fontsize=9)
        ax.text(1.35, y, phi_name, ha="center", va="center", fontsize=8)
        ax.text(
            2.92,
            y,
            f"{importance[idx] * 100:.2f}% | mean={mean_contribution[idx]:+.4f} | w={field_weight[idx]:+.3f}",
            ha="left",
            va="center",
            fontsize=8,
        )

    title = f"KAN Branch Schematic ({share_mode=}, top {len(order)} by mean |contribution|)"
    ax.set_title(title, pad=20)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


@torch.no_grad()
def plot_kan_scalar_functions(
    kan_branch: nn.Module,
    feature_names: list[str],
    feature_order: np.ndarray,
    output_path: str | Path,
    top_features: int,
    function_points: int,
) -> dict[str, object]:
    scalar_kans = getattr(kan_branch, "scalar_kans", None)
    if scalar_kans is None or len(scalar_kans) == 0:
        raise ValueError("KAN branch does not expose scalar_kans.")

    share_mode = str(getattr(kan_branch, "share_mode", "unknown"))
    first_scalar = scalar_kans[0]
    grid_min = float(first_scalar.grid_min.detach().cpu())
    grid_max = float(first_scalar.grid_max.detach().cpu())
    device = next(kan_branch.parameters()).device
    xs = torch.linspace(grid_min, grid_max, function_points, device=device)
    xs_np = xs.detach().float().cpu().numpy()

    if share_mode == "field":
        selected = feature_order[:top_features].tolist()
    else:
        selected = [0]

    fig, ax = plt.subplots(figsize=(10, 6))
    summaries: list[dict[str, float | str]] = []
    for idx in selected:
        scalar_idx = idx if share_mode == "field" else 0
        ys = scalar_kans[scalar_idx](xs).detach().float().cpu().numpy()
        label = feature_names[idx] if share_mode == "field" else "global_phi"
        ax.plot(xs_np, ys, label=label)
        summaries.append(
            {
                "feature": feature_names[idx] if share_mode == "field" else "global",
                "function_min": float(ys.min()),
                "function_max": float(ys.max()),
                "function_mean": float(ys.mean()),
            }
        )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("embedding scalar")
    ax.set_ylabel("KAN scalar output")
    ax.set_title("KAN Scalar Functions" if share_mode == "field" else "Global KAN Scalar Function")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return {"share_mode": share_mode, "grid_min": grid_min, "grid_max": grid_max, "functions": summaries}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.processed_dir:
        config.setdefault("paths", {})["processed_dir"] = args.processed_dir
    if args.output_dir:
        config.setdefault("paths", {})["output_dir"] = args.output_dir
    ensure_dirs(config)

    metadata = load_metadata(config["paths"]["processed_dir"])
    feature_cols = metadata["feature_cols"]
    dataset = AvazuParquetDataset(metadata["split_files"][args.split], feature_cols, metadata["target_col"])
    loader = make_dataloader(
        dataset,
        batch_size=int(args.batch_size or config.get("training", {}).get("batch_size", 2048)),
        num_workers=0,
        pin_memory=bool(config.get("environment", {}).get("pin_memory", True)),
    )

    model_name = args.model or "kanfin"
    model = build_model(model_name, metadata["field_dims"], config)
    load_checkpoint(args.checkpoint, model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    kan_branch = get_kan_branch(model)
    kan_params = get_kan_parameters(kan_branch)

    collected = collect_nafi_interpretability(model, loader, config, max_batches=args.max_batches)
    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs")) / "figures" / "kanfi_interpretability"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(config.get("paths", {}).get("output_dir", "outputs")) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "split": args.split,
        "model": model_name,
        "max_batches": args.max_batches,
        "kan": {
            "share_mode": kan_params["share_mode"],
            "grid_size": kan_params["grid_size"],
            "degree": kan_params["degree"],
            "num_basis": kan_params["num_basis"],
            "grid_min": kan_params["grid_min"],
            "grid_max": kan_params["grid_max"],
            "branch_bias": kan_params["branch_bias"],
        },
    }

    feature_order = np.arange(len(feature_cols))
    if "feature_contributions" in collected:
        contributions = collected["feature_contributions"]
        feature_order, feature_summary = contribution_summary(contributions, feature_cols)
        plot_feature_importance(contributions, feature_cols, output_dir / "kan_feature_importance.png")
        plot_kan_feature_weights(
            contributions,
            feature_cols,
            kan_params,
            output_dir / "kan_feature_weights.png",
            args.top_features,
        )
        plot_kan_schematic(
            contributions,
            feature_cols,
            kan_params,
            output_dir / "kan_branch_schematic.png",
            args.top_features,
        )
        np.save(output_dir / "kan_feature_contributions.npy", contributions)
        summary["kan_feature_contributions"] = feature_summary

    summary["kan_scalar_functions"] = plot_kan_scalar_functions(
        kan_branch,
        feature_cols,
        feature_order,
        output_dir / "kan_scalar_functions.png",
        args.top_features,
        args.function_points,
    )

    if "attention_matrix" in collected:
        attention_matrix = collected["attention_matrix"]
        plot_attention_heatmap(attention_matrix, feature_cols, output_dir / "fin_attention_heatmap.png")
        np.save(output_dir / "fin_attention_matrix.npy", attention_matrix)
        summary["top_fin_attention_pairs"] = summarize_attention_matrix(attention_matrix, feature_cols)

    if "predictions" in collected:
        preds = collected["predictions"]
        summary["prediction_summary"] = {
            "rows": int(len(preds)),
            "mean": float(preds.mean()),
            "std": float(preds.std()),
            "min": float(preds.min()),
            "max": float(preds.max()),
        }

    summary_path = metrics_dir / f"{args.split}_{model_name}_kanfi_interpretability.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({"summary": str(summary_path), "figures": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
