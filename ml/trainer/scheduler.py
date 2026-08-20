"""Learning-rate schedules for training."""

from __future__ import annotations

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ml.models import TrainingConfig


def build_scheduler(
    optimizer: Optimizer,
    training: TrainingConfig,
) -> LambdaLR:
    """Cosine schedule with linear warmup over ``warmup_steps``.

    LR is ramped from ``0`` to ``learning_rate`` over warmup, then decays
    cosinely to ``learning_rate / 10`` at ``max_steps``.
    """

    def lr_lambda(step: int) -> float:
        if step < training.warmup_steps:
            if training.warmup_steps == 0:
                return 1.0
            return (step + 1) / training.warmup_steps
        progress = (step - training.warmup_steps) / max(
            training.max_steps - training.warmup_steps, 1
        )
        return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(progress) * 3.141592653589793)).item())

    return LambdaLR(optimizer, lr_lambda)


def group_parameters(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into weight-decay and no-weight-decay groups.

    Embeddings, biases, and norm weights are excluded from weight decay.
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name or "embed_tokens" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]