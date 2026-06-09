import pytest
import torch

from src.models.base import build_model


@pytest.mark.parametrize(
    "name",
    ["lr", "fm", "deepfm", "autoint", "nam", "kan", "kan_v2", "nafi", "kanfin", "kanfin_v2"],
)
def test_model_forward(name: str) -> None:
    config = {
        "model": {"embedding_dim": 4, "hidden_units": [8], "dropout": 0.0},
        "nam": {"hidden_units": [4], "dropout": 0.0},
        "kan": {"grid_size": 5, "degree": 1, "grid_min": -2.0, "grid_max": 2.0, "dropout": 0.0},
        "fin": {"num_heads": 2, "num_layers": 1, "attention_dropout": 0.0, "use_residual": True},
    }
    model = build_model(name, [10, 20, 30], config)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    out = model(x)
    assert out["logits"].shape == (2,)


def test_kan_branch_is_smaller_than_nam_branch() -> None:
    config = {
        "model": {"embedding_dim": 16, "hidden_units": [128, 64, 32], "dropout": 0.2},
        "nam": {"hidden_units": [32, 16], "dropout": 0.1},
        "kan": {"grid_size": 5, "degree": 1, "grid_min": -2.0, "grid_max": 2.0, "dropout": 0.0},
        "fin": {"num_heads": 4, "num_layers": 2, "attention_dropout": 0.1, "use_residual": True},
    }
    field_dims = [10, 20, 30, 40]
    nam = build_model("nam", field_dims, config)
    kan = build_model("kan", field_dims, config)

    nam_branch_params = sum(param.numel() for param in nam.branch.parameters())
    kan_branch_params = sum(param.numel() for param in kan.kan.parameters())

    assert kan_branch_params < nam_branch_params


def test_kan_cubic_degree_forward() -> None:
    config = {
        "model": {"embedding_dim": 4, "hidden_units": [8], "dropout": 0.0},
        "kan": {"grid_size": 5, "degree": 3, "grid_min": -2.0, "grid_max": 2.0, "dropout": 0.0},
        "fin": {"num_heads": 2, "num_layers": 1, "attention_dropout": 0.0, "use_residual": True},
    }
    model = build_model("kan", [10, 20, 30], config)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    out = model(x)

    assert out["logits"].shape == (2,)


def test_kanfin_v2_direct_output_decomposition() -> None:
    config = {
        "model": {"embedding_dim": 4, "hidden_units": [8], "dropout": 0.0},
        "kan_v2": {
            "grid_size": 5,
            "degree": 2,
            "grid_min": -2.0,
            "grid_max": 2.0,
            "dropout": 0.0,
            "use_base": True,
            "init_scale": 0.01,
        },
        "fin": {"num_heads": 2, "num_layers": 1, "attention_dropout": 0.0, "use_residual": True},
    }
    model = build_model("kanfin_v2", [10, 20, 30], config)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    out = model(x)

    edge_contributions = out["embedding_contributions"]
    feature_contributions = out["feature_contributions"]
    assert out["edge_contributions"] is edge_contributions
    assert edge_contributions.shape == (2, 3, 4)
    assert feature_contributions.shape == (2, 3)
    assert torch.allclose(feature_contributions, edge_contributions.sum(dim=2))
    assert torch.allclose(
        out["kan_logits"],
        edge_contributions.sum(dim=(1, 2)) + model.kan.bias.squeeze(0),
    )
    assert torch.allclose(out["logits"], out["kan_logits"] + out["fin_logits"])

    assert len(model.kan.scalar_kans) == 3
    assert not hasattr(model.kan, "dim_weight")
    assert not hasattr(model.kan, "field_weight")
    assert not hasattr(model.kan, "field_bias")
    assert sum(param.numel() for param in model.kan.parameters()) == 3 * (6 + 2) + 1

    embeddings = model.embedding(x)
    expected_first_field = model.kan.scalar_kans[0](embeddings[:, 0, :])
    assert torch.allclose(edge_contributions[:, 0, :], expected_first_field)
    assert model.kan.scalar_kans[0].spline_weight.data_ptr() != model.kan.scalar_kans[1].spline_weight.data_ptr()
