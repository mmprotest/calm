"""Single-step Energy Transformer head for CALM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn


@dataclass
class EnergyTransformerConfig:
    latent_dim: int = 128
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ff_multiplier: int = 4
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.register_buffer("mask", torch.triu(torch.ones(1024, 1024), diagonal=1).bool(), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        L = x.size(1)
        mask = self.mask[:L, :L]
        out, _ = self.attn(x, x, x, attn_mask=mask)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: EnergyTransformerConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config.d_model, config.num_heads, config.dropout)
        self.attn_ln = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.ff_multiplier * config.d_model),
            nn.GELU(),
            nn.Linear(config.ff_multiplier * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_ln(x))
        x = x + self.ffn(x)
        return x


class EnergyTransformer(nn.Module):
    """Autoregressive Transformer that scores candidate latents."""

    def __init__(self, config: EnergyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.latent_dim, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.output_proj = nn.Linear(config.d_model, config.latent_dim)
        self.energy_head = nn.Sequential(
            nn.LayerNorm(config.d_model + config.latent_dim),
            nn.Linear(config.d_model + config.latent_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )
        self.sample_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.ff_multiplier * config.latent_dim),
            nn.GELU(),
            nn.Linear(config.ff_multiplier * config.latent_dim, config.latent_dim),
        )

    def forward(self, latents: Tensor) -> Tensor:
        hidden = self.input_proj(latents)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def step(self, context: Tensor) -> Tensor:
        hidden = self.forward(context)
        return hidden[:, -1]

    def energy(self, context_state: Tensor, candidates: Tensor) -> Tensor:
        expanded_state = context_state.unsqueeze(1).expand_as(candidates)
        concat = torch.cat([expanded_state, candidates], dim=-1)
        energy = self.energy_head(concat).squeeze(-1)
        return energy

    def sample_next(self, context_state: Tensor, num_samples: int) -> Tensor:
        noise = torch.randn(context_state.size(0), num_samples, self.config.latent_dim, device=context_state.device)
        base = self.sample_head(context_state).unsqueeze(1)
        return base + noise


__all__ = ["EnergyTransformerConfig", "EnergyTransformer"]
