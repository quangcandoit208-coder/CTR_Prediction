from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models.base import build_model
from src.training.checkpoint import load_checkpoint


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    checkpoint: Path


class TeacherEnsemble(nn.Module):
    """Frozen teacher ensemble for binary CTR distillation."""

    def __init__(
        self,
        teachers: list[nn.Module],
        names: list[str],
        mode: str = "uniform",
        weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        if not teachers:
            raise ValueError("TeacherEnsemble requires at least one teacher")
        self.teachers = nn.ModuleList(teachers)
        self.names = names
        self.mode = mode
        if weights is not None:
            if len(weights) != len(teachers):
                raise ValueError("Number of teacher weights must match number of teachers")
            weight_tensor = torch.as_tensor(weights, dtype=torch.float32)
            weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-12)
        else:
            weight_tensor = torch.full((len(teachers),), 1.0 / len(teachers), dtype=torch.float32)
        self.register_buffer("weights", weight_tensor)
        for teacher in self.teachers:
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> dict[str, torch.Tensor]:
        logits = []
        for teacher in self.teachers:
            logits.append(teacher(x)["logits"])
        stacked_logits = torch.stack(logits, dim=0)
        teacher_probs = torch.sigmoid(stacked_logits / temperature)

        if self.mode == "uniform" or self.mode == "fixed":
            weights = self.weights.to(device=teacher_probs.device, dtype=teacher_probs.dtype).view(-1, 1)
            ensemble_probs = (teacher_probs * weights).sum(dim=0)
        elif self.mode == "confidence":
            scores = torch.abs(teacher_probs - 0.5)
            weights = torch.softmax(scores, dim=0)
            ensemble_probs = (teacher_probs * weights).sum(dim=0)
        else:
            raise ValueError(f"Unsupported teacher ensemble mode: {self.mode}")

        ensemble_probs = ensemble_probs.clamp(1e-6, 1.0 - 1e-6)
        ensemble_logits = torch.logit(ensemble_probs)
        return {
            "logits": ensemble_logits,
            "probs": ensemble_probs,
            "teacher_logits": stacked_logits,
        }


def parse_teacher_spec(value: str) -> TeacherSpec:
    if ":" not in value:
        raise ValueError("Teacher spec must have format model_name:/path/to/checkpoint.pt")
    name, checkpoint = value.split(":", 1)
    return TeacherSpec(name=name.strip().lower(), checkpoint=Path(checkpoint.strip()))


def load_teacher_ensemble(
    teacher_specs: list[TeacherSpec],
    field_dims: list[int],
    default_config: dict[str, Any],
    device: torch.device,
    mode: str = "uniform",
    weights: list[float] | None = None,
) -> TeacherEnsemble:
    teachers: list[nn.Module] = []
    names: list[str] = []
    for spec in teacher_specs:
        if not spec.checkpoint.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {spec.checkpoint}")
        checkpoint = torch.load(spec.checkpoint, map_location="cpu")
        teacher_config = checkpoint.get("config", default_config)
        teacher = build_model(spec.name, field_dims, teacher_config)
        load_checkpoint(spec.checkpoint, teacher, map_location="cpu")
        teacher.to(device)
        teacher.eval()
        teachers.append(teacher)
        names.append(spec.name)
    return TeacherEnsemble(teachers=teachers, names=names, mode=mode, weights=weights).to(device)

