from __future__ import annotations

from typing import Any

from torch import nn

from src.models.autoint import AutoInt
from src.models.deepfm import DeepFM
from src.models.fm import FM
from src.models.kan import KAN
from src.models.kan_v2 import KANV2
from src.models.kanfin import KANFIN
from src.models.kanfin_v2 import KANFINV2
from src.models.kd_nafi import KDNAFI
from src.models.lr import LR
from src.models.nafi import NAFI
from src.models.nam import NAM
from src.models.xdeepfm import xDeepFM


def build_model(name: str, field_dims: list[int], config: dict[str, Any]) -> nn.Module:
    name = name.lower()
    model_cfg = config.get("model", {})
    nam_cfg = config.get("nam", {})
    fin_cfg = config.get("fin", {})
    kan_cfg = config.get("kan", {})
    kan_v2_cfg = config.get("kan_v2", kan_cfg)
    embedding_dim = int(model_cfg.get("embedding_dim", 16))
    hidden_units = list(model_cfg.get("hidden_units", [128, 64, 32]))
    dropout = float(model_cfg.get("dropout", 0.2))
    nam_activation = str(nam_cfg.get("activation", "relu"))
    exu_max_value = float(nam_cfg.get("exu_max_value", 1.0))
    exu_weight_clip = float(nam_cfg.get("exu_weight_clip", 10.0))
    xdeepfm_cfg = config.get("xdeepfm", {})

    if name == "lr":
        return LR(field_dims)
    if name == "fm":
        return FM(field_dims, embedding_dim=embedding_dim)
    if name == "deepfm":
        return DeepFM(field_dims, embedding_dim=embedding_dim, hidden_units=hidden_units, dropout=dropout)
    if name == "xdeepfm":
        return xDeepFM(
            field_dims,
            embedding_dim=embedding_dim,
            cin_layers=list(xdeepfm_cfg.get("cin_layers", [16, 16])),
            hidden_units=hidden_units,
            dropout=dropout,
        )
    if name == "autoint":
        return AutoInt(
            field_dims,
            embedding_dim=embedding_dim,
            num_heads=int(fin_cfg.get("num_heads", 4)),
            num_layers=int(fin_cfg.get("num_layers", 2)),
            dropout=float(fin_cfg.get("attention_dropout", 0.1)),
        )
    if name == "nam":
        return NAM(
            field_dims,
            embedding_dim=embedding_dim,
            hidden_units=list(nam_cfg.get("hidden_units", [32, 16])),
            dropout=float(nam_cfg.get("dropout", 0.1)),
            activation=nam_activation,
            exu_max_value=exu_max_value,
            exu_weight_clip=exu_weight_clip,
        )
    if name == "kan":
        return KAN(
            field_dims,
            embedding_dim=embedding_dim,
            grid_size=int(kan_cfg.get("grid_size", 5)),
            degree=int(kan_cfg.get("degree", 1)),
            grid_min=float(kan_cfg.get("grid_min", -2.0)),
            grid_max=float(kan_cfg.get("grid_max", 2.0)),
            dropout=float(kan_cfg.get("dropout", 0.0)),
            use_base=bool(kan_cfg.get("use_base", True)),
            init_scale=float(kan_cfg.get("init_scale", 0.01)),
            share_mode=str(kan_cfg.get("share_mode", "field")),
        )
    if name in {"kan_v2", "kan-v2"}:
        return KANV2(
            field_dims,
            embedding_dim=embedding_dim,
            grid_size=int(kan_v2_cfg.get("grid_size", 5)),
            degree=int(kan_v2_cfg.get("degree", 1)),
            grid_min=float(kan_v2_cfg.get("grid_min", -2.0)),
            grid_max=float(kan_v2_cfg.get("grid_max", 2.0)),
            dropout=float(kan_v2_cfg.get("dropout", 0.0)),
            use_base=bool(kan_v2_cfg.get("use_base", True)),
            init_scale=float(kan_v2_cfg.get("init_scale", 0.01)),
        )
    if name == "nafi":
        return NAFI(
            field_dims,
            embedding_dim=embedding_dim,
            nam_hidden_units=list(nam_cfg.get("hidden_units", [32, 16])),
            fin_num_heads=int(fin_cfg.get("num_heads", 4)),
            fin_num_layers=int(fin_cfg.get("num_layers", 2)),
            nam_dropout=float(nam_cfg.get("dropout", 0.1)),
            fin_dropout=float(fin_cfg.get("attention_dropout", 0.1)),
            use_residual=bool(fin_cfg.get("use_residual", True)),
            nam_activation=nam_activation,
            exu_max_value=exu_max_value,
            exu_weight_clip=exu_weight_clip,
        )
    if name in {"kanfin", "kan-fin", "kafi"}:
        return KANFIN(
            field_dims,
            embedding_dim=embedding_dim,
            kan_grid_size=int(kan_cfg.get("grid_size", 5)),
            kan_degree=int(kan_cfg.get("degree", 1)),
            kan_grid_min=float(kan_cfg.get("grid_min", -2.0)),
            kan_grid_max=float(kan_cfg.get("grid_max", 2.0)),
            kan_dropout=float(kan_cfg.get("dropout", 0.0)),
            kan_use_base=bool(kan_cfg.get("use_base", True)),
            kan_init_scale=float(kan_cfg.get("init_scale", 0.01)),
            kan_share_mode=str(kan_cfg.get("share_mode", "field")),
            fin_num_heads=int(fin_cfg.get("num_heads", 4)),
            fin_num_layers=int(fin_cfg.get("num_layers", 2)),
            fin_dropout=float(fin_cfg.get("attention_dropout", 0.1)),
            use_residual=bool(fin_cfg.get("use_residual", True)),
        )
    if name in {"kanfin_v2", "kanfin-v2", "kafi_v2"}:
        return KANFINV2(
            field_dims,
            embedding_dim=embedding_dim,
            kan_grid_size=int(kan_v2_cfg.get("grid_size", 5)),
            kan_degree=int(kan_v2_cfg.get("degree", 1)),
            kan_grid_min=float(kan_v2_cfg.get("grid_min", -2.0)),
            kan_grid_max=float(kan_v2_cfg.get("grid_max", 2.0)),
            kan_dropout=float(kan_v2_cfg.get("dropout", 0.0)),
            kan_use_base=bool(kan_v2_cfg.get("use_base", True)),
            kan_init_scale=float(kan_v2_cfg.get("init_scale", 0.01)),
            fin_num_heads=int(fin_cfg.get("num_heads", 4)),
            fin_num_layers=int(fin_cfg.get("num_layers", 2)),
            fin_dropout=float(fin_cfg.get("attention_dropout", 0.1)),
            use_residual=bool(fin_cfg.get("use_residual", True)),
        )
    if name in {"kd_nafi", "kd-nafi"}:
        return KDNAFI(
            field_dims,
            embedding_dim=embedding_dim,
            nam_hidden_units=list(nam_cfg.get("hidden_units", [32, 16])),
            fin_num_heads=int(fin_cfg.get("num_heads", 4)),
            fin_num_layers=int(fin_cfg.get("num_layers", 2)),
            nam_dropout=float(nam_cfg.get("dropout", 0.1)),
            fin_dropout=float(fin_cfg.get("attention_dropout", 0.1)),
            use_residual=bool(fin_cfg.get("use_residual", True)),
            nam_activation=nam_activation,
            exu_max_value=exu_max_value,
            exu_weight_clip=exu_weight_clip,
        )
    raise ValueError(f"Unknown model: {name}")
