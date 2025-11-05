"""BrierLM evaluation metric for CALM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from .autoencoder import ChunkVAE


@dataclass
class BrierConfig:
    num_samples: int = 8


def brier_score(probs: Tensor, targets: Tensor) -> Tensor:
    one_hot = torch.nn.functional.one_hot(targets, num_classes=probs.size(-1)).float()
    return ((probs - one_hot) ** 2).sum(dim=-1)


def evaluate_brierlm(
    model: ChunkVAE,
    latents: Tensor,
    decode_fn: Callable[[Tensor], Tensor],
    targets: Tensor,
    config: BrierConfig | None = None,
) -> Tensor:
    if config is None:
        config = BrierConfig()
    samples = model.sample_latent(latents, torch.zeros_like(latents), config.num_samples)
    logits = model.decode_latent(samples.view(-1, samples.size(-1)))
    probs = logits.softmax(dim=-1).view(samples.size(0), samples.size(1), model.config.K, -1)
    probs = probs.mean(dim=1)
    scores = brier_score(probs, targets)
    return scores.mean()


__all__ = ["BrierConfig", "evaluate_brierlm"]
