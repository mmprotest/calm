"""High level data module utilities for CALM training."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .adapters import TextDatasetAdapter, load_hf_dataset
from .chunking import pack_to_chunks
from .tokenization import TokenizerConfig, TokenizerWrapper


@dataclass
class ChunkDatasetConfig:
    dataset_name: str = "wikitext-2"
    split: str = "train"
    text_field: str = "text"
    max_examples: int | None = None
    K: int = 4
    tokenizer: TokenizerConfig = TokenizerConfig()


class ChunkDataset(Dataset[List[int]]):
    """Dataset that yields token chunks."""

    def __init__(self, adapter: TextDatasetAdapter, tokenizer: TokenizerWrapper, config: ChunkDatasetConfig) -> None:
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.config = config
        self.examples: List[List[int]] = []
        for idx, text in enumerate(adapter):
            if config.max_examples is not None and idx >= config.max_examples:
                break
            token_ids = tokenizer.encode(text)
            chunks = pack_to_chunks(token_ids, config.K, drop_last=True)
            if chunks.numel() == 0:
                continue
            self.examples.extend(chunks.tolist())

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> List[int]:
        return self.examples[idx]


def create_dataloader(config: ChunkDatasetConfig, batch_size: int, shuffle: bool = True) -> DataLoader:
    tokenizer = TokenizerWrapper(config.tokenizer)
    adapter = load_hf_dataset(config.dataset_name, config.split, config.text_field)
    dataset = ChunkDataset(adapter, tokenizer, config)

    def collate_fn(batch: Sequence[Sequence[int]]) -> torch.Tensor:
        return torch.tensor(batch, dtype=torch.long)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


__all__ = ["ChunkDatasetConfig", "ChunkDataset", "create_dataloader"]
