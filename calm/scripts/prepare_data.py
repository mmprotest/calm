"""Prepare datasets for CALM training."""

from __future__ import annotations

import argparse
from pathlib import Path

from calm.data.adapters import load_hf_dataset, load_text_folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext-2")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, default="data/prepared.txt")
    parser.add_argument("--text_folder", type=str, default=None)
    args = parser.parse_args()

    if args.text_folder:
        adapter = load_text_folder(args.text_folder)
    else:
        adapter = load_hf_dataset(args.dataset, split=args.split)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for text in adapter:
            f.write(text.replace("\n", " ").strip() + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
