"""Generate text with CALM."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from calm.data.chunking import pack_to_chunks, unpack_chunks
from calm.data.tokenization import TokenizerConfig, TokenizerWrapper
from calm.models.autoencoder import AutoencoderConfig, ChunkVAE
from calm.models.energy_transformer import EnergyTransformer, EnergyTransformerConfig
from calm.models.temperature import batch_temperature_sample, sample_at_temperature
from .utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "config" / "defaults.yaml"))
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--exact_temperature", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer = TokenizerWrapper(TokenizerConfig())
    ae_cfg = config["autoencoder"]
    calm_cfg = config["calm"]
    ae = ChunkVAE(
        AutoencoderConfig(
            vocab_size=tokenizer.tokenizer.vocab_size,
            K=ae_cfg["K"],
            d_model=ae_cfg["d_model"],
            latent_dim=ae_cfg["latent_dim"],
            beta=ae_cfg["beta"],
        )
    )
    transformer = EnergyTransformer(
        EnergyTransformerConfig(
            latent_dim=calm_cfg["latent_dim"],
            d_model=calm_cfg["d_model"],
            num_layers=calm_cfg["num_layers"],
            num_heads=calm_cfg["num_heads"],
            ff_multiplier=calm_cfg["ff_multiplier"],
            dropout=calm_cfg["dropout"],
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae.to(device)
    transformer.to(device)

    tokens = tokenizer.encode(args.prompt)[: args.max_tokens]
    chunks = pack_to_chunks(tokens, ae_cfg["K"], drop_last=False, pad_id=tokenizer.pad_id, eos_id=tokenizer.eos_id)
    if chunks.numel() == 0:
        print(args.prompt)
        return
    chunks = chunks.to(device)
    mu, logvar = ae.encode(chunks)
    latents = ae.reparameterize(mu, logvar)
    generated_tokens = tokens.copy()

    while len(generated_tokens) < args.max_tokens:
        context = latents.unsqueeze(0)
        hidden = transformer.forward(context)
        state = hidden[:, -1]
        candidates = transformer.sample_next(state, args.num_candidates)
        energies = transformer.energy(state, candidates)
        if args.exact_temperature:
            chosen = sample_at_temperature(candidates, energies, args.temperature)
        else:
            chosen = batch_temperature_sample(candidates, energies, args.temperature)
        next_chunk = ae.decode(chosen.unsqueeze(0))[0].tolist()
        generated_tokens.extend(next_chunk)
        if tokenizer.eos_id in next_chunk:
            break
        latents = torch.cat([latents, chosen.unsqueeze(0)], dim=0)

    text = tokenizer.decode(generated_tokens)
    print(text)


if __name__ == "__main__":
    main()
