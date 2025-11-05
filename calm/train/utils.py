"""Training utility functions."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    inherit = data.pop("inherit", None)
    if inherit:
        base_path = path.parent / f"{inherit}.yaml"
        base = load_config(base_path)
        base.update(data)
        return base
    return data


def merge_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result


__all__ = ["load_config", "merge_dict"]
