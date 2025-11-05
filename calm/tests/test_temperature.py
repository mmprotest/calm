from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from calm.models.temperature import batch_temperature_sample, sample_at_temperature


def test_temperature_sampling_boundaries():
    candidates = torch.randn(2, 4, 3)
    energies = torch.linspace(0, 1, 4).repeat(2, 1)
    exact = sample_at_temperature(candidates, energies, 1.0)
    approx = batch_temperature_sample(candidates, energies, 1.0)
    assert exact.shape == approx.shape


def test_temperature_zero_limit():
    candidates = torch.randn(1, 3, 2)
    energies = torch.tensor([[0.1, 0.0, 0.2]])
    sample = batch_temperature_sample(candidates, energies, 1e-4)
    assert sample.shape == (1, 2)
