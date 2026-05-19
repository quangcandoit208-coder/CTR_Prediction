from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.dataloader import make_dataloader
from src.data.dataset import AvazuParquetDataset
from src.evaluation.evaluate import evaluate_model
from src.evaluation.report import save_metrics_report
from src.models.base import build_model
from src.training.checkpoint import load_checkpoint
from src.utils.config import ensure_dirs, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CTR model checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    metadata_path = Path(config["paths"]["processed_dir"]) / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    dataset = AvazuParquetDataset(metadata["split_files"][args.split], metadata["feature_cols"], metadata["target_col"])
    loader = make_dataloader(
        dataset,
        int(config.get("training", {}).get("batch_size", 2048)),
        num_workers=int(config.get("environment", {}).get("num_workers", 0)),
        pin_memory=bool(config.get("environment", {}).get("pin_memory", True)),
    )
    model_name = config.get("model", {}).get("name", "nafi")
    model = build_model(model_name, metadata["field_dims"], config)
    load_checkpoint(args.checkpoint, model)
    metrics = evaluate_model(model, loader, config)
    report_path = Path(config.get("paths", {}).get("output_dir", "outputs")) / "metrics" / f"{args.split}_metrics.json"
    save_metrics_report(metrics, report_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

