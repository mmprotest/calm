"""Likelihood-free temperature sampling algorithms."""

from __future__ import annotations

import torch
from torch import Tensor


def _normalize(energies: Tensor, T: float) -> Tensor:
    scaled = -energies / max(T, 1e-6)
    scaled = scaled - scaled.max(dim=-1, keepdim=True).values
    weights = scaled.exp()
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights


def sample_at_temperature(candidates: Tensor, energies: Tensor, T: float) -> Tensor:
    """Exact temperature sampling following Algorithm 1 in the paper."""

    batch, num = energies.shape
    probs = _normalize(energies, T)
    idx = torch.multinomial(probs, num_samples=1)
    result = torch.gather(candidates, 1, idx.unsqueeze(-1).expand(-1, -1, candidates.size(-1)))
    return result.squeeze(1)


def batch_temperature_sample(candidates: Tensor, energies: Tensor, T: float) -> Tensor:
    weights = _normalize(energies, T)
    choice = torch.argmax(weights, dim=-1)
    return candidates[torch.arange(candidates.size(0)), choice]


__all__ = ["sample_at_temperature", "batch_temperature_sample"]
