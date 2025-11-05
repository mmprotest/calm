"""Training callbacks for logging and early stopping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MetricHistory:
    name: str
    best: float | None = None
    mode: str = "min"
    history: List[float] = field(default_factory=list)

    def update(self, value: float) -> bool:
        self.history.append(value)
        improved = False
        if self.best is None:
            improved = True
        elif self.mode == "min" and value < self.best:
            improved = True
        elif self.mode == "max" and value > self.best:
            improved = True
        if improved:
            self.best = value
        return improved


__all__ = ["MetricHistory"]
