from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from calm.models.autoencoder import AutoencoderConfig, ChunkVAE
from calm.models.brierlm import BrierConfig, evaluate_brierlm


def test_brierlm_returns_scalar():
    model = ChunkVAE(AutoencoderConfig(vocab_size=16, K=2, d_model=8, latent_dim=4))
    latents = torch.zeros(1, 4)
    targets = torch.zeros(1, 2, dtype=torch.long)
    score = evaluate_brierlm(model, latents, model.decode, targets, BrierConfig(num_samples=2))
    assert torch.isfinite(score)
