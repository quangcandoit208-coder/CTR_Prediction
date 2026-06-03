from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.preprocess import MISSING, parse_hour_column
from src.data.schema import AVAZU_COLUMNS, DTYPE_MAP
from src.features.feature_map import get_feature_cols
from src.utils.config import ensure_dirs, load_config


HASH_SPACE = float(1 << 64)


class KMVSketch:
    """Approximate distinct counter using k minimum deterministic hashes."""

    def __init__(self, k: int = 4096) -> None:
        if k < 128:
            raise ValueError(f"KMV sketch size should be >= 128, got {k}")
        self.k = int(k)
        self.hashes: set[int] = set()

    def update(self, values: pd.Series) -> None:
        for value in values.drop_duplicates().astype(str):
            self.hashes.add(stable_hash64(value))
        if len(self.hashes) > self.k * 2:
            self._prune()

    def _prune(self) -> None:
        if len(self.hashes) > self.k:
            self.hashes = set(sorted(self.hashes)[: self.k])

    def estimate(self) -> int:
        if len(self.hashes) < self.k:
            return len(self.hashes)
        kth = max(self.hashes) / HASH_SPACE
        if kth <= 0:
            return len(self.hashes)
        return int(round((self.k - 1) / kth))


class NumericStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = math.inf
        self.max = -math.inf

    def update(self, values: pd.Series) -> None:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return
        numeric_float = numeric.astype("float64")
        self.count += int(numeric_float.size)
        self.sum += float(numeric_float.sum())
        self.sumsq += float((numeric_float * numeric_float).sum())
        self.min = min(self.min, float(numeric_float.min()))
        self.max = max(self.max, float(numeric_float.max()))

    def as_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = self.sum / self.count
        variance = max(self.sumsq / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.min,
            "max": self.max,
        }


def stable_hash64(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming EDA for raw Avazu CSV before hashing tricks.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--input", default=None, help="Override raw CSV/train.gz path")
    parser.add_argument("--compression", default=None, help="CSV compression: gzip, infer, or none")
    parser.add_argument("--chunksize", type=int, default=None, help="Rows per pandas chunk")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick EDA")
    parser.add_argument("--top-k", type=int, default=20, help="Top values to write per column")
    parser.add_argument("--top-per-chunk", type=int, default=200, help="Candidate top values retained from each chunk")
    parser.add_argument("--kmv-size", type=int, default=4096, help="KMV sketch size for approximate cardinality")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    return parser.parse_args()


def normalize_for_counts(series: pd.Series) -> pd.Series:
    return series.fillna(MISSING).astype(str).replace("", MISSING)


def prune_counter(counter: Counter[str], max_size: int) -> None:
    if len(counter) <= max_size:
        return
    kept = Counter(dict(counter.most_common(max_size)))
    counter.clear()
    counter.update(kept)


def update_top_stats(
    df: pd.DataFrame,
    col: str,
    target_col: str,
    value_counts: dict[str, Counter[str]],
    value_click_sums: dict[str, Counter[str]],
    top_per_chunk: int,
    max_counter_size: int,
) -> None:
    values = normalize_for_counts(df[col])
    counts = values.value_counts(dropna=False).head(top_per_chunk)
    value_counts[col].update({str(key): int(value) for key, value in counts.items()})

    if target_col in df.columns:
        target = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype("int8")
        temp = pd.DataFrame({"value": values, "target": target})
        click_sums = temp.groupby("value", observed=True)["target"].sum()
        for key in counts.index:
            value_click_sums[col][str(key)] += int(click_sums.get(key, 0))

    prune_counter(value_counts[col], max_counter_size)
    prune_counter(value_click_sums[col], max_counter_size)


def update_group_ctr(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    group_counts: dict[str, Counter[str]],
    group_click_sums: dict[str, Counter[str]],
) -> None:
    if target_col not in df.columns:
        return
    target = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype("int8")
    for col in group_cols:
        if col not in df.columns:
            continue
        values = normalize_for_counts(df[col])
        temp = pd.DataFrame({"value": values, "target": target})
        counts = temp.groupby("value", observed=True)["target"].count()
        sums = temp.groupby("value", observed=True)["target"].sum()
        group_counts[col].update({str(key): int(value) for key, value in counts.items()})
        group_click_sums[col].update({str(key): int(value) for key, value in sums.items()})


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_report(
    output_path: Path,
    summary: dict[str, Any],
    column_summary: pd.DataFrame,
    top_values: pd.DataFrame,
    time_ctr: pd.DataFrame,
) -> None:
    lines = [
        "# Raw CSV EDA Report",
        "",
        "This report is computed by streaming CSV chunks before hashing tricks.",
        "",
        "## Dataset",
        "",
        f"- input_path: `{summary['input_path']}`",
        f"- rows_scanned: `{summary['rows_scanned']}`",
        f"- chunks_scanned: `{summary['chunks_scanned']}`",
        f"- target_col: `{summary['target_col']}`",
    ]
    if summary.get("click_rate") is not None:
        lines.extend(
            [
                f"- click_count: `{summary['click_count']}`",
                f"- click_rate: `{summary['click_rate']:.6f}`",
            ]
        )

    lines.extend(["", "## Highest Missing Rate Columns", ""])
    missing_view = column_summary.sort_values("missing_rate", ascending=False).head(15)
    lines.append(markdown_table(missing_view[["column", "missing_count", "missing_rate", "approx_distinct"]]))

    lines.extend(["", "## Highest Approximate Cardinality Columns", ""])
    cardinality_view = column_summary.sort_values("approx_distinct", ascending=False).head(15)
    lines.append(markdown_table(cardinality_view[["column", "approx_distinct", "missing_rate"]]))

    if not top_values.empty:
        lines.extend(["", "## Top Raw Values", ""])
        lines.append(markdown_table(top_values.head(40)))

    if not time_ctr.empty:
        lines.extend(["", "## Time CTR", ""])
        lines.append(markdown_table(time_ctr.head(80)))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config.setdefault("paths", {})["output_dir"] = args.output_dir
    ensure_dirs(config)

    data_cfg = config.get("data", {})
    paths_cfg = config.get("paths", {})
    target_col = str(data_cfg.get("target_col", "click"))
    raw_path = Path(args.input or paths_cfg.get("raw_train_gz", "train.gz"))
    chunksize = int(args.chunksize or data_cfg.get("chunksize", 250000))
    max_rows = args.max_rows if args.max_rows is not None else data_cfg.get("max_rows")
    max_rows = int(max_rows) if max_rows else None
    compression = args.compression if args.compression is not None else data_cfg.get("compression", "infer")
    compression = "infer" if compression == "none" else compression
    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs")) / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_cols(config)
    base_columns = [col for col in AVAZU_COLUMNS if col != "id"]
    expected_cols = [col for col in base_columns + feature_cols if col not in {"id"}]

    dtype_map = {key: value for key, value in DTYPE_MAP.items() if key != target_col}
    if target_col in DTYPE_MAP:
        dtype_map[target_col] = DTYPE_MAP[target_col]

    rows_scanned = 0
    chunks_scanned = 0
    click_sum = 0
    click_count = 0
    missing_counts: Counter[str] = Counter()
    column_counts: Counter[str] = Counter()
    distinct_sketches: dict[str, KMVSketch] = defaultdict(lambda: KMVSketch(args.kmv_size))
    value_counts: dict[str, Counter[str]] = defaultdict(Counter)
    value_click_sums: dict[str, Counter[str]] = defaultdict(Counter)
    numeric_stats: dict[str, NumericStats] = defaultdict(NumericStats)
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_click_sums: dict[str, Counter[str]] = defaultdict(Counter)

    reader = pd.read_csv(raw_path, compression=compression, chunksize=chunksize, dtype=dtype_map)

    for raw_chunk in reader:
        if max_rows is not None and rows_scanned >= max_rows:
            break
        if max_rows is not None and rows_scanned + len(raw_chunk) > max_rows:
            raw_chunk = raw_chunk.iloc[: max_rows - rows_scanned]
        if raw_chunk.empty:
            continue

        chunk = parse_hour_column(raw_chunk.copy()) if data_cfg.get("parse_hour", True) else raw_chunk.copy()
        rows = len(chunk)
        rows_scanned += rows
        chunks_scanned += 1

        columns = [col for col in expected_cols if col in chunk.columns]
        for col in columns:
            series = chunk[col]
            column_counts[col] += rows
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                missing = series.isna() | series.astype("string").fillna("").str.len().eq(0)
            else:
                missing = series.isna()
            missing_counts[col] += int(missing.sum())
            normalized = normalize_for_counts(series)
            distinct_sketches[col].update(normalized)
            update_top_stats(
                chunk,
                col,
                target_col,
                value_counts,
                value_click_sums,
                top_per_chunk=args.top_per_chunk,
                max_counter_size=max(args.top_k * 20, args.top_k),
            )

        for col in columns:
            if col == target_col or pd.api.types.is_numeric_dtype(chunk[col]):
                numeric_stats[col].update(chunk[col])

        if target_col in chunk.columns:
            target = pd.to_numeric(chunk[target_col], errors="coerce").fillna(0).astype("int8")
            click_sum += int(target.sum())
            click_count += int(target.size)

        update_group_ctr(
            chunk,
            ["hour", "day", "hour_of_day", "weekday"],
            target_col,
            group_counts,
            group_click_sums,
        )

    column_rows: list[dict[str, Any]] = []
    for col in sorted(column_counts):
        numeric = numeric_stats[col].as_dict() if col in numeric_stats else {}
        column_rows.append(
            {
                "column": col,
                "rows_seen": int(column_counts[col]),
                "missing_count": int(missing_counts[col]),
                "missing_rate": float(missing_counts[col] / max(column_counts[col], 1)),
                "approx_distinct": int(distinct_sketches[col].estimate()),
                **{f"numeric_{key}": value for key, value in numeric.items()},
            }
        )
    column_summary = pd.DataFrame(column_rows)

    top_rows: list[dict[str, Any]] = []
    for col, counter in sorted(value_counts.items()):
        for rank, (value, count) in enumerate(counter.most_common(args.top_k), start=1):
            clicks = int(value_click_sums[col].get(value, 0))
            top_rows.append(
                {
                    "column": col,
                    "rank": rank,
                    "value": value,
                    "count": int(count),
                    "frequency": float(count / max(rows_scanned, 1)),
                    "click_sum": clicks if click_count else None,
                    "ctr": float(clicks / count) if click_count and count else None,
                }
            )
    top_values = pd.DataFrame(top_rows)

    time_rows: list[dict[str, Any]] = []
    for col in ["hour", "day", "hour_of_day", "weekday"]:
        for value, count in sorted(group_counts[col].items(), key=lambda item: item[0]):
            clicks = int(group_click_sums[col].get(value, 0))
            time_rows.append(
                {
                    "column": col,
                    "value": value,
                    "count": int(count),
                    "click_sum": clicks,
                    "ctr": float(clicks / count) if count else None,
                }
            )
    time_ctr = pd.DataFrame(time_rows)

    summary = {
        "input_path": str(raw_path),
        "rows_scanned": rows_scanned,
        "chunks_scanned": chunks_scanned,
        "target_col": target_col,
        "click_count": click_sum if click_count else None,
        "click_rate": float(click_sum / click_count) if click_count else None,
        "top_k": args.top_k,
        "top_per_chunk": args.top_per_chunk,
        "kmv_size": args.kmv_size,
        "columns": dataframe_to_records(column_summary) if not column_summary.empty else [],
    }

    column_summary.to_csv(output_dir / "raw_column_summary.csv", index=False)
    top_values.to_csv(output_dir / "raw_top_values.csv", index=False)
    time_ctr.to_csv(output_dir / "raw_time_ctr.csv", index=False)
    with (output_dir / "raw_eda_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **summary,
                "top_values": dataframe_to_records(top_values.head(200)) if not top_values.empty else [],
                "time_ctr": dataframe_to_records(time_ctr) if not time_ctr.empty else [],
            },
            f,
            indent=2,
        )
    write_markdown_report(output_dir / "raw_eda_report.md", summary, column_summary, top_values, time_ctr)

    print(
        json.dumps(
            {
                "rows_scanned": rows_scanned,
                "click_rate": summary["click_rate"],
                "outputs": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
