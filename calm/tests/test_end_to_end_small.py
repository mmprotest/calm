from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

pytest.importorskip("torch")

from calm.train.train_autoencoder import train as train_ae
from calm.train.train_calm import train as train_calm
from calm.train.utils import load_config


class DummyTokenizer:
    def __init__(self):
        self.pad_id = 0
        self.eos_id = 1
        self.tokenizer = SimpleNamespace(vocab_size=32)

    def encode(self, text: str):
        return [ord(c) % 30 + 2 for c in text.split()]

    def decode(self, ids):  # pragma: no cover
        return " ".join(str(i) for i in ids)


class DummyAdapter:
    def __iter__(self):
        yield "hello world"
        yield "calm testing"

    def __len__(self):  # pragma: no cover
        return 2


def test_end_to_end_smoke(monkeypatch):
    config = load_config("calm/config/tiny.yaml")
    tokenizer = DummyTokenizer()

    monkeypatch.setattr("calm.train.train_autoencoder.load_hf_dataset", lambda *a, **k: DummyAdapter())
    monkeypatch.setattr("calm.train.train_calm.load_hf_dataset", lambda *a, **k: DummyAdapter())

    ae = train_ae(config, tokenizer=tokenizer)
    train_calm(config, tokenizer=tokenizer, autoencoder=ae)
