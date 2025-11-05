"""Evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from calm.models.autoencoder import AutoencoderConfig, ChunkVAE
from calm.models.brierlm import BrierConfig, evaluate_brierlm
from calm.models.utils import SeedConfig, seed_everything
from .utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "config" / "defaults.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(SeedConfig(seed=config.get("seed", 42)))
    dummy_vocab = 50257
    ae_cfg = config["autoencoder"]
    model = ChunkVAE(
        AutoencoderConfig(
            vocab_size=dummy_vocab,
            K=ae_cfg["K"],
            d_model=ae_cfg["d_model"],
            latent_dim=ae_cfg["latent_dim"],
            beta=ae_cfg["beta"],
        )
    )
    latents = torch.zeros(2, ae_cfg["latent_dim"])
    targets = torch.zeros(2, ae_cfg["K"], dtype=torch.long)
    score = evaluate_brierlm(model, latents, model.decode, targets, BrierConfig(num_samples=config["calm"].get("brier_samples", 8)))
    print({"brier": score.item()})


if __name__ == "__main__":
    main()
