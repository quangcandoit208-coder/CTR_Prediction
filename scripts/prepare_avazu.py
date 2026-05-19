from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.parquet_converter import convert_gz_to_parquet
from src.utils.config import ensure_dirs, load_config
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Avazu gzip CSV into parquet partitions.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    seed_everything(int(config.get("project", {}).get("seed", 42)))
    raw_path = Path(config["paths"]["raw_train_gz"])
    processed_dir = Path(config["paths"]["processed_dir"])
    metadata = convert_gz_to_parquet(raw_path, processed_dir, config)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

