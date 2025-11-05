"""Profile inference latency for CALM generation."""

from __future__ import annotations

import argparse
import time

import torch

from calm.models.energy_transformer import EnergyTransformer, EnergyTransformerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    model = EnergyTransformer(EnergyTransformerConfig())
    model.eval()
    latents = torch.randn(1, 32, model.config.latent_dim)
    with torch.no_grad():
        start = time.time()
        for _ in range(args.steps):
            model.forward(latents)
        elapsed = time.time() - start
    print({"steps": args.steps, "seconds": elapsed})


if __name__ == "__main__":
    main()
