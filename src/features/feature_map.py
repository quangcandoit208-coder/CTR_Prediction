from __future__ import annotations

from typing import Any

DEFAULT_HASH_COLS = ("device_ip", "device_id")


def get_feature_cols(config: dict[str, Any]) -> list[str]:
    features = config.get("features", {})
    cols = list(features.get("categorical_cols", []))
    for col in features.get("derived_cols", []):
        if col not in cols:
            cols.append(col)
    return cols


def get_hash_cols(config: dict[str, Any], feature_cols: list[str] | None = None) -> list[str]:
    features = config.get("features", {})
    configured = features.get("hash_cols", DEFAULT_HASH_COLS)
    cols = [str(col) for col in configured]
    if feature_cols is None:
        return cols
    allowed = set(feature_cols)
    return [col for col in cols if col in allowed]


def build_field_dims(
    feature_cols: list[str],
    config: dict[str, Any],
    category_maps: dict[str, dict[str, int]] | None = None,
) -> list[int]:
    features = config.get("features", {})
    use_hashing = bool(config.get("data", {}).get("use_hashing", True))
    hash_cols = set(get_hash_cols(config, feature_cols)) if use_hashing else set()
    default_bucket = int(features.get("hash_bucket_default", 100000))
    buckets = features.get("hash_buckets", {})
    field_dims: list[int] = []
    for col in feature_cols:
        if col in hash_cols:
            field_dims.append(int(buckets.get(col, default_bucket)))
        elif category_maps is not None and col in category_maps:
            field_dims.append(max(len(category_maps[col]), 1))
        else:
            field_dims.append(int(buckets.get(col, default_bucket)))
    return field_dims
