"""Checkpoint utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: str | Path, state: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


__all__ = ["save_checkpoint", "load_checkpoint"]
