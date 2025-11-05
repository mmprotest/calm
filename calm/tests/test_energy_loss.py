from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from calm.models.energy_loss import EnergyScoreConfig, energy_score


def test_energy_score_positive():
    targets = torch.randn(2, 4, 8)
    models = torch.randn(2, 3, 8)
    score = energy_score(targets, models, EnergyScoreConfig())
    assert torch.isfinite(score)


def test_energy_score_temperature_scaling():
    targets = torch.ones(1, 2, 4)
    models = torch.zeros(1, 2, 4)
    score_high = energy_score(targets, models, EnergyScoreConfig(temperature=1.0))
    score_low = energy_score(targets, models, EnergyScoreConfig(temperature=0.5))
    assert score_low >= score_high
