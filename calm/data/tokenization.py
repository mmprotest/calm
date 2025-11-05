"""Tokenizer utilities built on top of Hugging Face tokenizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer creation."""

    name_or_path: str = "gpt2"
    padding_side: str = "left"
    truncation_side: str = "right"
    use_fast: bool = True


class TokenizerWrapper:
    """Lightweight wrapper that exposes the minimal API we require."""

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            config.name_or_path,
            padding_side=config.padding_side,
            truncation_side=config.truncation_side,
            use_fast=config.use_fast,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_id(self) -> int:
        return self.tokenizer.eos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: Iterable[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def batch_encode(self, texts: List[str], max_length: int | None = None) -> List[List[int]]:
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=bool(max_length),
            max_length=max_length,
        )
        return [list(ids) for ids in encoded["input_ids"]]


__all__ = ["TokenizerConfig", "TokenizerWrapper"]
