"""Tensors in `W`: packing, and decoding one tensor from only the segments it overlaps."""

from __future__ import annotations

import random

import pytest

from verity.core.protocols.pous.layout import Layout, decode_tensor
from verity.core.protocols.pous.params import Params
from verity.core.protocols.pous.primitives import reference
from verity.core.protocols.pous.scheme import encode

BASE = Params(8, 64, 3)
SIZES = [("embed", 700), ("norm", 3), ("empty", 0), ("proj", 1100), ("head", 512)]


@pytest.mark.parametrize("align", [False, True])
def test_decode_per_tensor(align):
    layout = Layout.of(BASE, SIZES, align=align)
    rng = random.Random(2)
    data = {name: rng.randbytes(size) for name, size in SIZES}
    w = layout.pack(data)
    assert len(w) == layout.params.w_bytes and layout.params.segments == (7 if align else 5)
    assert all(layout.unpack(w, name) == data[name] for name in data)
    prims = reference(layout.params)
    enc = encode(layout.params, prims, w, b"\x09" * 24)
    for name in data:
        read = set()

        def block(i):
            read.add(i)
            return enc.block(i)

        assert decode_tensor(layout, prims, enc.salt, name, block) == data[name]
        assert read == set(layout.blocks(name))
    if align:
        assert all(p.offset % BASE.segment_bytes == 0 for p in layout.tensors if p.size)


def test_layout_rejects():
    with pytest.raises(ValueError):
        Layout.of(BASE, [("a", 1), ("a", 2)])
    layout = Layout.of(BASE, [("a", 4)])
    with pytest.raises(ValueError):
        layout.pack({"a": b"123"})
    with pytest.raises(KeyError):
        layout.placement("b")
