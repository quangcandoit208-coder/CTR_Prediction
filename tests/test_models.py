import pytest
import torch

from src.models.base import build_model


@pytest.mark.parametrize("name", ["lr", "fm", "deepfm", "autoint", "nam", "kan", "nafi", "kanfin"])
def test_model_forward(name: str) -> None:
    config = {
        "model": {"embedding_dim": 4, "hidden_units": [8], "dropout": 0.0},
        "nam": {"hidden_units": [4], "dropout": 0.0},
        "kan": {"grid_size": 5, "grid_min": -2.0, "grid_max": 2.0, "dropout": 0.0},
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
        "kan": {"grid_size": 5, "grid_min": -2.0, "grid_max": 2.0, "dropout": 0.0},
        "fin": {"num_heads": 4, "num_layers": 2, "attention_dropout": 0.1, "use_residual": True},
    }
    field_dims = [10, 20, 30, 40]
    nam = build_model("nam", field_dims, config)
    kan = build_model("kan", field_dims, config)

    nam_branch_params = sum(param.numel() for param in nam.branch.parameters())
    kan_branch_params = sum(param.numel() for param in kan.kan.parameters())

    assert kan_branch_params < nam_branch_params
