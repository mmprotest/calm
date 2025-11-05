"""Embedding utilities for CALM models."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Tied input/output token embeddings."""

    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=embedding_dim ** -0.5)
        self.padding_idx = padding_idx

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.embedding(self.weight, token_ids, padding_idx=self.padding_idx)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.weight.t()


class PositionalEncoding(nn.Module):
    """Deterministic sinusoidal positional encoding."""

    def __init__(self, dim: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, start: int = 0) -> torch.Tensor:
        return x + self.pe[start : start + x.size(1)]


__all__ = ["TokenEmbedding", "PositionalEncoding"]
