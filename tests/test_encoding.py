import pandas as pd

from src.features.encoding import UNKNOWN_INDEX, encode_features, init_category_maps
from src.features.hashing import stable_hash


def test_encode_features_hashes_only_selected_columns() -> None:
    df = pd.DataFrame(
        {
            "device_ip": ["ip_a", "ip_b"],
            "site_id": ["site_a", "site_b"],
            "C1": ["1005", "1005"],
        }
    )
    feature_cols = ["device_ip", "site_id", "C1"]
    hash_cols = ["device_ip"]
    category_maps = init_category_maps(feature_cols, hash_cols)

    encoded = encode_features(
        df,
        feature_cols=feature_cols,
        hash_buckets={"device_ip": 100},
        default_bucket=10,
        seed=42,
        hash_cols=hash_cols,
        category_maps=category_maps,
        update_category_maps=True,
    )

    assert encoded["device_ip"].tolist() == [stable_hash("ip_a", 100, seed=42), stable_hash("ip_b", 100, seed=42)]
    assert encoded["site_id"].tolist() == [1, 2]
    assert encoded["C1"].tolist() == [1, 1]


def test_encode_features_maps_unseen_categories_to_unknown() -> None:
    category_maps = {"site_id": {"__UNKNOWN__": UNKNOWN_INDEX, "seen": 1}}
    df = pd.DataFrame({"site_id": ["seen", "new"]})

    encoded = encode_features(
        df,
        feature_cols=["site_id"],
        hash_buckets={},
        default_bucket=10,
        seed=42,
        hash_cols=[],
        category_maps=category_maps,
        update_category_maps=False,
    )

    assert encoded["site_id"].tolist() == [1, UNKNOWN_INDEX]
