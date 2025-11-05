"""Train the CALM Energy Transformer head."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from calm.data.adapters import load_hf_dataset
from calm.data.chunking import pack_to_chunks
from calm.data.tokenization import TokenizerConfig, TokenizerWrapper
from calm.models.autoencoder import AutoencoderConfig, ChunkVAE
from calm.models.energy_loss import energy_score
from calm.models.energy_transformer import EnergyTransformer, EnergyTransformerConfig
from calm.models.utils import SeedConfig, seed_everything
from .utils import load_config


def build_sequences(config: Dict[str, Any], tokenizer: TokenizerWrapper) -> list[torch.Tensor]:
    dataset = load_hf_dataset(config.get("dataset", "wikitext-2"))
    K = config["autoencoder"]["K"]
    sequences = []
    for text in dataset:
        ids = tokenizer.encode(text)
        chunks = pack_to_chunks(ids, K, drop_last=True)
        if chunks.numel() == 0:
            continue
        sequences.append(chunks)
        if config["calm"].get("max_steps") and len(sequences) >= config["calm"]["max_steps"]:
            break
    if not sequences:
        sequences.append(torch.full((1, K), tokenizer.eos_id, dtype=torch.long))
    return sequences


def train(config: Dict[str, Any], tokenizer: TokenizerWrapper | None = None, autoencoder: ChunkVAE | None = None) -> EnergyTransformer:
    seed_everything(SeedConfig(seed=config.get("seed", 42)))
    tokenizer = tokenizer or TokenizerWrapper(TokenizerConfig())
    sequences = build_sequences(config, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_cfg = config["autoencoder"]
    ae = autoencoder or ChunkVAE(
        AutoencoderConfig(
            vocab_size=tokenizer.tokenizer.vocab_size,
            K=ae_cfg["K"],
            d_model=ae_cfg["d_model"],
            latent_dim=ae_cfg["latent_dim"],
            beta=ae_cfg["beta"],
        )
    )
    ae.to(device)
    ae.eval()
    calm_cfg = config["calm"]
    transformer = EnergyTransformer(
        EnergyTransformerConfig(
            latent_dim=calm_cfg["latent_dim"],
            d_model=calm_cfg["d_model"],
            num_layers=calm_cfg["num_layers"],
            num_heads=calm_cfg["num_heads"],
            ff_multiplier=calm_cfg["ff_multiplier"],
            dropout=calm_cfg["dropout"],
        )
    ).to(device)
    optimizer = AdamW(transformer.parameters(), lr=calm_cfg["lr"], weight_decay=calm_cfg.get("weight_decay", 0.0))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(calm_cfg.get("scheduler", {}).get("cosine_steps", 100), 1))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    transformer.train()
    for epoch in range(calm_cfg.get("epochs", 1)):
        progress = tqdm(sequences, desc=f"Epoch {epoch+1}")
        for chunks in progress:
            chunks = chunks.to(device)
            mu, logvar = ae.encode(chunks)
            latents = ae.reparameterize(mu, logvar)
            context = latents.unsqueeze(0)
            hidden = transformer.forward(context)
            state = hidden[:, -1]
            targets = ae.sample_latent(mu[-1:], logvar[-1:], calm_cfg["targets_M"])
            model_samples = transformer.sample_next(state, calm_cfg["samples_N"])
            targets = targets.view(1, calm_cfg["targets_M"], -1)
            model_samples = model_samples.view(1, calm_cfg["samples_N"], -1)
            loss = energy_score(targets, model_samples)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(transformer.parameters(), calm_cfg.get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            progress.set_postfix({"loss": loss.item()})
    return transformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "config" / "defaults.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
