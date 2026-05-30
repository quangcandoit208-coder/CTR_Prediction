from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.nafi import NAFI


class KDNAFI(NAFI):
    """Student NAFI model for knowledge distillation experiments."""


def binary_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, hard_labels)
    teacher_probs = torch.sigmoid(teacher_logits / temperature)
    soft_loss = F.binary_cross_entropy_with_logits(student_logits / temperature, teacher_probs)
    return alpha * hard_loss + (1.0 - alpha) * (temperature**2) * soft_loss


def binary_kd_loss_from_teacher_probs(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, hard_labels)
    soft_loss = F.binary_cross_entropy_with_logits(student_logits / temperature, teacher_probs)
    loss = alpha * hard_loss + (1.0 - alpha) * (temperature**2) * soft_loss
    return loss, hard_loss, soft_loss

