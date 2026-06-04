from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.features.hashing import hash_categorical_column

UNKNOWN_TOKEN = "__UNKNOWN__"
MISSING_TOKEN = "__MISSING__"
UNKNOWN_INDEX = 0


def init_category_maps(feature_cols: Iterable[str], hash_cols: Iterable[str]) -> dict[str, dict[str, int]]:
    hash_set = set(hash_cols)
    return {str(col): {UNKNOWN_TOKEN: UNKNOWN_INDEX} for col in feature_cols if col not in hash_set}


def _normalize_values(series: pd.Series) -> pd.Series:
    return series.fillna(MISSING_TOKEN).astype(str).replace("", MISSING_TOKEN)


def _update_category_map(values: pd.Series, category_map: dict[str, int]) -> None:
    for value in values.drop_duplicates():
        value = str(value)
        if value not in category_map:
            category_map[value] = len(category_map)


def encode_features(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    hash_buckets: dict[str, int],
    default_bucket: int,
    seed: int,
    hash_cols: Iterable[str],
    category_maps: dict[str, dict[str, int]] | None = None,
    update_category_maps: bool = True,
) -> pd.DataFrame:
    """Encode feature columns for embedding lookup.

    Only columns listed in ``hash_cols`` use the MD5 hashing trick. Other
    categorical columns are mapped to contiguous field-specific ids using
    ``category_maps``.
    """
    hash_set = set(hash_cols)
    if category_maps is None:
        category_maps = init_category_maps(feature_cols, hash_set)

    for col in feature_cols:
        if col not in df.columns:
            continue
        if col in hash_set:
            bucket = int(hash_buckets.get(col, default_bucket))
            df[col] = hash_categorical_column(df[col], bucket, seed=seed)
            continue

        values = _normalize_values(df[col])
        category_map = category_maps.setdefault(str(col), {UNKNOWN_TOKEN: UNKNOWN_INDEX})
        if update_category_maps:
            _update_category_map(values, category_map)
        df[col] = values.map(category_map).fillna(UNKNOWN_INDEX).astype("int64")

    return df


def rare_category_encode(series: pd.Series, min_count: int = 10, rare_token: str = "__RARE__") -> pd.Series:
    counts = series.value_counts(dropna=False)
    return series.where(series.map(counts) >= min_count, rare_token)
