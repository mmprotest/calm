"""Variational autoencoder mapping token chunks to latent vectors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .embeddings import TokenEmbedding


@dataclass
class AutoencoderConfig:
    vocab_size: int
    K: int = 4
    d_model: int = 512
    latent_dim: int = 128
    beta: float = 0.001
    stop_grad_steps: int = 0


class ChunkVAE(nn.Module):
    """Variational autoencoder over token chunks with robust latent space."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = TokenEmbedding(config.vocab_size, config.d_model)
        self.encoder_ffn = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
        )
        self.encoder_proj = nn.Linear(config.d_model * config.K, config.d_model)
        self.encoder_out = nn.Linear(config.d_model, 2 * config.latent_dim)

        self.decoder_in = nn.Linear(config.latent_dim, config.d_model)
        self.decoder_ffn = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model * config.K),
            nn.GELU(),
        )
        self.decoder_proj = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
        )
        self.beta = config.beta
        self.stop_grad_steps = config.stop_grad_steps

    def encode_tokens(self, token_ids: Tensor) -> Tensor:
        emb = self.embeddings(token_ids)
        emb = self.encoder_ffn(emb)
        flat = emb.view(emb.size(0), -1)
        hidden = self.encoder_proj(flat)
        mu_logvar = self.encoder_out(hidden)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return torch.stack([mu, logvar], dim=0)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_latent(self, z: Tensor) -> Tensor:
        hidden = self.decoder_in(z)
        decoded = self.decoder_ffn(hidden)
        decoded = decoded.view(z.size(0), self.config.K, self.config.d_model)
        decoded = self.decoder_proj(decoded)
        logits = self.embeddings.project(decoded)
        return logits

    def forward(self, token_ids: Tensor, step: int | None = None) -> Dict[str, Tensor]:
        enc = self.encode_tokens(token_ids)
        mu, logvar = enc[0], enc[1]
        if step is not None and step < self.stop_grad_steps:
            mu = mu.detach()
            logvar = logvar.detach()
        z = self.reparameterize(mu, logvar)
        logits = self.decode_latent(z)
        recon_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), token_ids.view(-1), reduction="mean"
        )
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kl
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == token_ids).float().mean()
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl": kl,
            "accuracy": accuracy,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "logits": logits,
        }

    def encode(self, token_ids: Tensor) -> Tuple[Tensor, Tensor]:
        stats = self.encode_tokens(token_ids)
        return stats[0], stats[1]

    def sample_latent(self, mu: Tensor, logvar: Tensor, num_samples: int = 1) -> Tensor:
        if num_samples == 1:
            return self.reparameterize(mu, logvar)
        mu = mu.unsqueeze(1).expand(-1, num_samples, -1)
        logvar = logvar.unsqueeze(1).expand_as(mu)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        logits = self.decode_latent(z)
        return logits.argmax(dim=-1)


__all__ = ["AutoencoderConfig", "ChunkVAE"]
