"""Train the chunk autoencoder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from calm.data.adapters import load_hf_dataset
from calm.data.chunking import pack_to_chunks
from calm.data.tokenization import TokenizerConfig, TokenizerWrapper
from calm.models.autoencoder import AutoencoderConfig, ChunkVAE
from calm.models.utils import SeedConfig, seed_everything
from .utils import load_config, merge_dict


def build_dataloader(config: Dict[str, Any], tokenizer: TokenizerWrapper) -> DataLoader:
    dataset_cfg = config["autoencoder"]
    dataset = load_hf_dataset(config.get("dataset", "wikitext-2"))
    chunks = []
    for text in dataset:
        ids = tokenizer.encode(text)
        chunk_tensor = pack_to_chunks(ids, dataset_cfg["K"], drop_last=True)
        if chunk_tensor.numel() == 0:
            continue
        chunks.extend(chunk_tensor.tolist())
        if dataset_cfg.get("max_steps") and len(chunks) >= dataset_cfg["max_steps"] * dataset_cfg["batch_size"]:
            break
    if not chunks:
        chunks.append([tokenizer.eos_id] * dataset_cfg["K"])
    tensor_dataset = torch.tensor(chunks, dtype=torch.long)
    return DataLoader(tensor_dataset, batch_size=dataset_cfg["batch_size"], shuffle=True)


def train(config: Dict[str, Any], overrides: Dict[str, Any] | None = None, tokenizer: TokenizerWrapper | None = None) -> ChunkVAE:
    if overrides:
        config = merge_dict(config, overrides)
    seed_everything(SeedConfig(seed=config.get("seed", 42)))
    tokenizer = tokenizer or TokenizerWrapper(TokenizerConfig())
    dataloader = build_dataloader(config, tokenizer)
    vocab_size = tokenizer.tokenizer.vocab_size
    ae_cfg = config["autoencoder"]
    model = ChunkVAE(
        AutoencoderConfig(
            vocab_size=vocab_size,
            K=ae_cfg["K"],
            d_model=ae_cfg["d_model"],
            latent_dim=ae_cfg["latent_dim"],
            beta=ae_cfg["beta"],
            stop_grad_steps=ae_cfg.get("stop_grad_steps", 0),
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=ae_cfg["lr"], weight_decay=ae_cfg.get("weight_decay", 0.0))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(ae_cfg.get("scheduler", {}).get("cosine_steps", 100), 1))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    step = 0
    model.train()
    for epoch in range(ae_cfg.get("epochs", 1)):
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(batch, step=step)
                loss = outputs["loss"]
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), ae_cfg.get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            progress.set_postfix({"loss": loss.item(), "acc": outputs["accuracy"].item()})
            if ae_cfg.get("max_steps") and step >= ae_cfg["max_steps"]:
                break
        if ae_cfg.get("max_steps") and step >= ae_cfg["max_steps"]:
            break
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "config" / "defaults.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
