from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from calm.models.energy_transformer import EnergyTransformer, EnergyTransformerConfig


def test_energy_transformer_shapes():
    config = EnergyTransformerConfig(latent_dim=8, d_model=16, num_layers=1, num_heads=2)
    model = EnergyTransformer(config)
    latents = torch.randn(2, 5, config.latent_dim)
    hidden = model.forward(latents)
    assert hidden.shape == (2, 5, config.d_model)
    state = model.step(latents)
    assert state.shape == (2, config.d_model)
    candidates = model.sample_next(state, 3)
    assert candidates.shape == (2, 3, config.latent_dim)
    energies = model.energy(state, candidates)
    assert energies.shape == (2, 3)
