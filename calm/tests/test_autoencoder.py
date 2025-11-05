from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from calm.models.autoencoder import AutoencoderConfig, ChunkVAE


def test_autoencoder_forward_and_loss():
    config = AutoencoderConfig(vocab_size=32, K=4, d_model=16, latent_dim=8)
    model = ChunkVAE(config)
    tokens = torch.randint(0, 32, (2, 4))
    outputs = model(tokens)
    assert outputs["loss"].item() > 0
    outputs2 = model(tokens)
    assert outputs2["loss"].item() >= 0


def test_autoencoder_serialization(tmp_path):
    config = AutoencoderConfig(vocab_size=32, K=4, d_model=16, latent_dim=8)
    model = ChunkVAE(config)
    path = tmp_path / "ae.pt"
    torch.save(model.state_dict(), path)
    new_model = ChunkVAE(config)
    new_model.load_state_dict(torch.load(path))
