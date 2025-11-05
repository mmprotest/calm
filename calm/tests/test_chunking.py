from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from calm.data.chunking import pack_to_chunks, unpack_chunks


def test_pack_unpack_roundtrip():
    ids = list(range(10))
    chunks = pack_to_chunks(ids, 4, drop_last=False, pad_id=0, eos_id=9)
    flat = unpack_chunks(chunks)
    assert flat[:10] == ids


def test_pack_drop_last():
    ids = [1, 2, 3]
    chunks = pack_to_chunks(ids, 4, drop_last=True)
    assert chunks.numel() == 0
