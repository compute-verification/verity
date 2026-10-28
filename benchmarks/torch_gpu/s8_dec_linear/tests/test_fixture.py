"""The staging reference (`fixture.py`) against the P2 spec's deployment vector, and its placement of payload bytes."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fixture as F


def deployment_vector() -> dict:
    doc = json.loads(F.R.VECTORS.read_bytes())
    (v,) = [v for v in doc["vectors"] if int(v["w"]) == F.PARAMS.w and int(v["c0"]) == F.PARAMS.c0]
    return v


@pytest.fixture(scope="module")
def vec() -> tuple[bytes, int, bytes, bytes]:
    v = deployment_vector()
    pfx = F.R.expand_prefix(bytes.fromhex(v["setup_salt"]), F.SPEC, bytes.fromhex(v["model_commitment"]))
    j = int(v["segment"]) * F.PARAMS.blocks_per_segment + int(v["block"])
    return pfx, j, bytes.fromhex(v["c"]), bytes.fromhex(v["m"])


def test_the_vector_decodes_to_its_payload(vec):
    pfx, j, c, m = vec
    got, flags = F.decode_row(pfx, j, c)
    assert got == m and flags == 0
    k, t = F.keys(pfx, j)
    v = deployment_vector()
    assert (k.to_bytes(F.ROW, "big").hex(), t.to_bytes(F.ROW, "big").hex()) == (v["K"], v["t"])


@pytest.mark.parametrize("shift,length", [(0, F.PAYLOAD), (10, 100), (-50, 60), (-7, F.PAYLOAD + 20), (2000, 54)])
def test_staged_bytes_sit_at_their_w_offset(vec, shift, length):
    pfx, j, c, m = vec
    base = F.PAYLOAD * j + shift
    assert j in F.blocks_of(base, length)
    got = F.staged(pfx, base, length, {j: c})
    w = {F.PAYLOAD * j + q: m[2 + q] for q in range(F.PAYLOAD)}
    want = bytes(w[base + i] for i in range(length) if base + i in w)
    (at, b), = got.items()
    assert b == want and at == max(0, -shift)
    buf = bytearray(length)
    buf[at:at + len(b)] = b
    assert F.check_rows(pfx, base, length, {j: c}, buf) == []
    buf[at] ^= 1
    assert F.check_rows(pfx, base, length, {j: c}, buf) == [j]


def test_blocks_of():
    assert F.blocks_of(0, 0) == range(0)
    assert F.blocks_of(0, 1) == range(1)
    assert F.blocks_of(F.PAYLOAD - 1, 2) == range(2)
    assert F.blocks_of(F.PAYLOAD, F.PAYLOAD) == range(1, 2)
    assert F.blocks_of(5, 3 * F.PAYLOAD) == range(4)


def test_refusals(vec):
    pfx, j, c, _ = vec
    assert F.refusals(pfx, {j: c}) == 0
    assert F.refusals(pfx, {j: b"\xff" * F.ROW}) == F.FLAG_C
    assert F.staged(pfx, 0, F.PAYLOAD * (j + 1), {j: b"\xff" * F.ROW}) == {}
    noncode = bytes([0x12]) + c[1:]     # canonical, and decodes to a full-width value with overwhelming probability
    assert F.refusals(pfx, {j: noncode}) == F.FLAG_WIDE


def test_the_h200_layer_packs_back_to_back():
    t = F.packed(F.layer_tensors(16384, 64, 65536))
    assert [(n, k) for _, n, k, _, _ in t] == [(16512, 16384), (16384, 16384), (65536, 16384), (16384, 65536)]
    assert sum(length for *_, length in t) == 2_686_451_712
    for (_, _, _, b0, l0), (_, _, _, b1, _) in itertools.pairwise(t):
        assert b0 + l0 == b1
        assert F.blocks_of(b0, l0).stop - 1 <= F.blocks_of(b1, 1).start
