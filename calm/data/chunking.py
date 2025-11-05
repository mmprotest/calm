"""Utilities for converting token sequences into fixed length chunks."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch


def pack_to_chunks(token_ids: Sequence[int], K: int, *, pad_id: int | None = None, eos_id: int | None = None,
                   drop_last: bool = True) -> torch.Tensor:
    """Pack a sequence of token ids into [num_chunks, K] tensor.

    Args:
        token_ids: Sequence of token ids.
        K: Chunk size.
        pad_id: Padding token id to use for the final chunk when ``drop_last`` is False.
        eos_id: EOS id used when padding.
        drop_last: Whether to drop incomplete final chunks.
    """

    ids = list(token_ids)
    if drop_last:
        length = (len(ids) // K) * K
        ids = ids[:length]
    elif pad_id is not None:
        remainder = len(ids) % K
        if remainder:
            pad_value = eos_id if eos_id is not None else pad_id
            ids.extend([pad_value] * (K - remainder))
    if len(ids) == 0:
        return torch.empty(0, K, dtype=torch.long)
    tensor = torch.tensor(ids, dtype=torch.long).view(-1, K)
    return tensor


def unpack_chunks(chunks: torch.Tensor) -> List[int]:
    """Flatten chunked tokens into a list."""

    if chunks.numel() == 0:
        return []
    return chunks.view(-1).tolist()


def sliding_chunk_view(latents: torch.Tensor, context_length: int) -> torch.Tensor:
    """Create an overlapping context view for autoregressive training."""

    if latents.size(0) < context_length:
        pad = latents.new_zeros(context_length - latents.size(0), *latents.shape[1:])
        latents = torch.cat([pad, latents], dim=0)
    unfolded = latents.unfold(dimension=0, size=context_length, step=1)
    return unfolded


__all__ = ["pack_to_chunks", "unpack_chunks", "sliding_chunk_view"]
