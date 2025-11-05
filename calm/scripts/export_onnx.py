"""Export CALM models to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from calm.models.autoencoder import AutoencoderConfig, ChunkVAE
from calm.models.energy_transformer import EnergyTransformer, EnergyTransformerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="exports/calm.onnx")
    args = parser.parse_args()

    dummy_vocab = 50257
    ae = ChunkVAE(AutoencoderConfig(vocab_size=dummy_vocab))
    transformer = EnergyTransformer(EnergyTransformerConfig())
    latents = torch.randn(1, 4, transformer.config.latent_dim)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(transformer, latents, output_path)
    print(f"Exported {output_path}")


if __name__ == "__main__":
    main()
