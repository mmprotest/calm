"""Utility helpers for model training."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn


@dataclass
class SeedConfig:
    seed: int = 42
    deterministic: bool = True


def seed_everything(config: SeedConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if config.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def save_manifest(path: str | Path, metadata: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


__all__ = ["SeedConfig", "seed_everything", "count_parameters", "save_manifest"]
