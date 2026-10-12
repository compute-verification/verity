"""Encode and decode: bit-exact round trips on every backend, determinism, the salt and segment tags, the bit layout."""

from __future__ import annotations

import hashlib
import os
import random

import pytest

from verity.core.protocols.pous.labels import component, from_bytes, join, split, tag, to_bytes, transpose, w_labels, w_value
from verity.core.protocols.pous.oracle import run
from verity.core.protocols.pous.params import LEAN_P3_MEETS, OPERATING_POINT, Params
from verity.core.protocols.pous.primitives import heuristic_arx, ideal, reference
from verity.core.protocols.pous.scheme import decode, decode_prog, decode_segment, encode, encode_prog

BACKENDS = {"reference": reference, "heuristic-arx": heuristic_arx, "ideal": lambda p: ideal(p, b"test-seed")}
SHAPES = [Params(8, 64, 3), Params(6, 64, 0), Params(4, 96, 12, 3), Params(1, 384, 1)]


def _w(params: Params, seed: int) -> bytes:
    return random.Random(seed).randbytes(params.w_bytes)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("params", SHAPES, ids=str)
def test_round_trip(backend, params):
    prims = BACKENDS[backend](params)
    w, salt = _w(params, 1), bytes(range(24))
    enc = encode(params, prims, w, salt)
    blocks = [enc.block(i) for i in range(params.blocks)]
    assert len(blocks) == params.blocks and all(len(b) == params.m // 8 for b in blocks)
    assert decode(params, prims, salt, blocks) == w
    for s in range(params.segments):
        seg = decode_segment(params, prims, salt, s, blocks[s * params.n:(s + 1) * params.n])
        assert seg == w[s * params.segment_bytes:(s + 1) * params.segment_bytes]


def test_deterministic_and_salted():
    params = Params(8, 64, 3, 2)
    prims = reference(params)
    w = bytes(params.w_bytes)
    a, b = encode(params, prims, w, b"\x01" * 24), encode(params, prims, w, b"\x01" * 24)
    assert a == b
    c = encode(params, prims, w, b"\x02" * 24)
    assert not set(a.blocks) & set(c.blocks)
    assert not set(a.blocks[:8]) & set(a.blocks[8:]), "equal-W segments must not share labels (attack 3)"


def test_decode_depth_and_work():
    """Decode is two passes, each one key per label (`j + 1` calls of Π₂) beside one `P⁻¹` round."""
    params = Params(8, 64, 3)
    prims = reference(params)
    enc = encode(params, prims, _w(params, 2), b"\x03" * 24)
    salt = from_bytes(enc.salt, 192)
    _, cost = run(decode_prog(params, prims.h, salt, enc.blocks), prims)
    per_pass = 1 + 3
    assert cost.depth == 2 * per_pass
    keys = sum(1 + len(params.parents(v)) for v in range(8))
    assert cost.kinds == {"PiQuery": 2 * keys, "PinvQuery": 16}
    _, cost = run(encode_prog(params, prims.h, from_bytes(_w(params, 2), params.w_bits), salt), prims)
    assert cost.kinds["PQuery"] == 16 and cost.kinds["PiQuery"] == 2 * keys


def test_bit_layout():
    """Lean's `codeEquiv`: bit `j` of block `i` is bit `i * width + j`; little-endian bytes."""
    rng = random.Random(3)
    parts = [rng.getrandbits(5) for _ in range(7)]
    x = join(parts, 5)
    assert split(x, 7, 5) == parts and all((x >> (5 * i + j)) & 1 == (parts[i] >> j) & 1 for i in range(7) for j in range(5))
    assert to_bytes(0x0102, 16) == b"\x02\x01" and from_bytes(b"\x02\x01", 16) == 0x0102
    with pytest.raises(ValueError):
        from_bytes(b"\xff", 7)
    labels = [rng.getrandbits(12) for _ in range(4)]
    t = transpose(labels, 4, 3)
    assert all(component(t[v], u, 3) == component(labels[u], v, 3) for u in range(4) for v in range(4))
    assert transpose(t, 4, 3) == labels
    params = Params(3, 16, 1, 2)
    w = rng.getrandbits(params.w_bits)
    assert w_value(params, w_labels(params, w)) == w


def test_parameters():
    assert OPERATING_POINT.m == 524480 and OPERATING_POINT.m // 8 == 65560 and OPERATING_POINT.wc // 8 == 298
    assert OPERATING_POINT.segment_bytes == 220 * 65560
    assert LEAN_P3_MEETS.m == 8140
    p = Params(20, 8, 12)
    assert list(p.parents(0)) == [] and list(p.parents(5)) == [0, 1, 2, 3, 4] and list(p.parents(19)) == list(range(7, 19))
    assert list(Params(5, 8, 0).parents(3)) == [2], "Lean's bandGraph keeps the spine at k = 0"
    assert tag(1, 2) == 1 | (2 << 192)
    for bad in (dict(n=0, wc=8, k=1), dict(n=4, wc=0, k=1), dict(n=4, wc=8, k=-1)):
        with pytest.raises(ValueError):
            Params(**bad)
    with pytest.raises(ValueError):
        encode(Params(8, 64, 3), reference(Params(8, 64, 3)), bytes(512), b"short")


def test_vectors():
    """Known-answer vectors pinned in `fixtures/vectors.json` (regenerate with `tests/pous_vectors.py`)."""
    import json
    from pathlib import Path

    import pous_vectors

    pinned = json.loads((Path(__file__).parent / "fixtures" / "vectors.json").read_text())
    slow = {"segment_64kb", "band/64kb/segment"}
    assert pous_vectors.vectors(operating_point_segment=False) == {k: v for k, v in pinned.items() if k not in slow}


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("VERITY_POUS_SLOW"), reason="about a minute: set VERITY_POUS_SLOW=1")
def test_operating_point_segment():
    """One whole 64 KB-label segment: its pinned digest and a bit-exact decode (about a minute)."""
    import json
    from pathlib import Path

    import pous_vectors

    pinned = json.loads((Path(__file__).parent / "fixtures" / "vectors.json").read_text())
    params, prims = OPERATING_POINT, reference(OPERATING_POINT)
    w, salt = pous_vectors.segment_input(params)
    enc = encode(params, prims, w, salt)
    assert hashlib.sha256(b"".join(enc.block(i) for i in range(params.blocks))).hexdigest() == pinned["segment_64kb"]
    assert decode(params, prims, salt, [enc.block(i) for i in range(params.blocks)]) == w
