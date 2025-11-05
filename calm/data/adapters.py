"""Dataset adapters for Hugging Face datasets and plain text folders."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List

from datasets import Dataset, load_dataset


class TextDatasetAdapter:
    """Iterator over text samples for training."""

    def __init__(self, dataset: Dataset, text_field: str = "text") -> None:
        self.dataset = dataset
        self.text_field = text_field

    def __iter__(self) -> Iterator[str]:
        for sample in self.dataset:
            yield str(sample[self.text_field])

    def __len__(self) -> int:  # pragma: no cover - dataset may not support len
        return len(self.dataset)


def load_hf_dataset(name: str, split: str = "train", text_field: str = "text") -> TextDatasetAdapter:
    dataset = load_dataset(name, split=split)
    return TextDatasetAdapter(dataset, text_field=text_field)


def load_text_folder(path: str | Path) -> TextDatasetAdapter:
    path = Path(path)
    texts: List[str] = []
    for file in sorted(path.glob("**/*.txt")):
        texts.append(file.read_text(encoding="utf-8"))
    dataset = Dataset.from_dict({"text": texts})
    return TextDatasetAdapter(dataset)


__all__ = ["TextDatasetAdapter", "load_hf_dataset", "load_text_folder"]
