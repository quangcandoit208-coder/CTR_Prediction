from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.dataloader import make_dataloader
from src.data.dataset import AvazuParquetDataset
from src.data.metadata import load_metadata
from src.evaluation.interpretability import (
    collect_nafi_interpretability,
    summarize_attention_matrix,
    summarize_feature_contributions,
)
from src.evaluation.plots import plot_attention_heatmap, plot_feature_importance
from src.models.base import build_model
from src.training.checkpoint import load_checkpoint
from src.utils.config import ensure_dirs, load_config


MODEL_CHOICES = ["nafi", "kanfin", "kanfin_v2", "kd_nafi", "autoint"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot NAFI interpretability outputs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--model", default=None, choices=MODEL_CHOICES)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


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

    model_name = args.model or config.get("model", {}).get("name", "nafi")
    model = build_model(model_name, metadata["field_dims"], config)
    load_checkpoint(args.checkpoint, model)

    collected = collect_nafi_interpretability(model, loader, config, max_batches=args.max_batches)
    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs")) / "figures" / "interpretability"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(config.get("paths", {}).get("output_dir", "outputs")) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "split": args.split,
        "model": model_name,
        "max_batches": args.max_batches,
    }

    if "feature_contributions" in collected:
        contributions = collected["feature_contributions"]
        plot_feature_importance(contributions, feature_cols, output_dir / "feature_importance.png")
        np.save(output_dir / "feature_contributions.npy", contributions)
        summary["feature_contributions"] = summarize_feature_contributions(contributions, feature_cols)

    if "attention_matrix" in collected:
        attention_matrix = collected["attention_matrix"]
        plot_attention_heatmap(attention_matrix, feature_cols, output_dir / "attention_heatmap.png")
        np.save(output_dir / "attention_matrix.npy", attention_matrix)
        summary["top_attention_pairs"] = summarize_attention_matrix(attention_matrix, feature_cols)

    if "predictions" in collected:
        preds = collected["predictions"]
        summary["prediction_summary"] = {
            "rows": int(len(preds)),
            "mean": float(preds.mean()),
            "std": float(preds.std()),
            "min": float(preds.min()),
            "max": float(preds.max()),
        }

    summary_path = metrics_dir / f"{args.split}_{model_name}_interpretability.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({"summary": str(summary_path), "figures": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
