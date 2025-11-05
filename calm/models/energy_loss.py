"""Energy Score objective for likelihood-free training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


@dataclass
class EnergyScoreConfig:
    norm: Literal["l2", "l1"] = "l2"
    temperature: float = 1.0


def _reduce_pairwise(dist: Tensor) -> Tensor:
    # dist shape [..., A, B]
    return dist.mean(dim=(-2, -1))


def pairwise_distance(a: Tensor, b: Tensor, norm: str = "l2") -> Tensor:
    if norm == "l2":
        return torch.cdist(a, b, p=2)
    if norm == "l1":
        return torch.cdist(a, b, p=1)
    raise ValueError(f"Unsupported norm: {norm}")


def energy_score(target_samples: Tensor, model_samples: Tensor, config: EnergyScoreConfig | None = None) -> Tensor:
    """Monte Carlo Energy Score estimator.

    Args:
        target_samples: Tensor of shape [batch, M, latent_dim].
        model_samples: Tensor of shape [batch, N, latent_dim].
    """

    if config is None:
        config = EnergyScoreConfig()
    targets = target_samples / config.temperature
    models = model_samples / config.temperature

    cross = pairwise_distance(targets, models, config.norm)
    term1 = _reduce_pairwise(cross)

    intra = pairwise_distance(models, models, config.norm)
    batch = intra.shape[0]
    n = model_samples.size(1)
    mask = 1 - torch.eye(n, device=intra.device, dtype=intra.dtype)
    intra = (intra * mask).sum(dim=(-2, -1)) / (n * (n - 1) + 1e-8)
    loss = term1.mean() - 0.5 * intra.mean()
    return loss


__all__ = ["EnergyScoreConfig", "energy_score"]
