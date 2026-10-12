"""The vllm-v1 position-leaf gadget without torch: the native digests against the core ``vllm-v1`` vectors and ``pos_leaf``,
and a whole chain of column gadgets evaluated from its witness program (honest -> every constraint holds and the last
column's digest expressions are ``pos_leaf(row)``; a moved hold byte, a wrong length prefix or another row -> the digest
differs or a constraint fails)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from verity.ml.commitments import vllm_v1 as core

from ..compile import Expr, LigeroCtx, P
from ..relations import FP8_ADA_RELATION
from . import vllm_v1 as V
from .sha256_test import _run, _violations

L = V.VllmV1Leaf()
VECTORS = json.loads((Path(core.__file__).parent / "vectors.json").read_text())


def test_prefix_and_leaves_match_the_core_vectors() -> None:
    for c in VECTORS["layouts"]:
        if c["leaf"] == "pos":
            value = bytes.fromhex(c["value"])
            assert V.prefix(len(value)).hex() == c["prefix"]
            assert core.pos_leaf(value).hex() == c["digest"]
    for c in VECTORS["pos_leaf"]:
        value = bytes.fromhex(c["value"])
        assert core.pos_leaf(value).hex() == c["leaf"]
        m = V.message(value)
        assert len(m) % 64 == 0 and m.startswith(V.prefix(len(value)) + value)


@pytest.mark.parametrize("word_bits", (8, 16))
def test_native_is_pos_leaf_and_carries_chain(word_bits: int) -> None:
    rows = np.random.default_rng(word_bits).integers(0, 1 << word_bits, size=(5, V.K_VU))
    d = L.native(rows, word_bits=word_bits, role=1)
    assert [L.leaf_bytes(r) for r in d] == L.leaf_bytes_many(d) == L.reference_digest(rows, word_bits=word_bits, role=2)
    car = L.carry_values(rows, word_bits=word_bits, role=1, steps=12)
    assert not car[:, 0].any()                      # a chain start carries 0 (H = IV, hold = prefix)
    bad = d.copy()
    bad[1, 3] = 1 << 16
    assert L.leaf_bytes(bad[1]) != L.leaf_bytes(d[1]) and len(L.leaf_bytes(bad[1])) == 32
    assert L.leaf_bytes_many(bad) == [L.leaf_bytes(r) for r in bad]


def _column_system(k: int, row_words: int):
    ctx = LigeroCtx(FP8_ADA_RELATION.params)
    bits = [[ctx.hint(f"w{i}.b{j}") for j in range(8)] for i in range(k)]
    for i, bs in enumerate(bits):
        for j, b in enumerate(bs):
            ctx.boolean(f"w{i}.bool{j}", b)
    words = [sum((b * (1 << j) for j, b in enumerate(bs)), Expr(ctx)) for bs in bits]
    carry = [ctx.hint(f"cap{x}") for x in range(V.CARRY_ELEMS)]
    digest, carry_out = L.gadget(ctx, words, word_bits=8, role=1, row_words=row_words, carry_in=carry)
    return ctx.sys, digest, carry_out


def _ev(e, W):
    return (e.c + sum(v * W[j] for j, v in e.lin.items())) % P


def _chain(row: np.ndarray, steps: int, tamper=None):
    """Evaluate every column of ``row`` (8-bit words) from the honest carries; returns (violations, last digest bytes)."""
    k = row.size // steps
    sys, digest, carry_out = _column_system(k, row.size)
    car = L.carry_values(row[None, :], word_bits=8, role=1, steps=steps)[0]
    bad, last, prev = 0, None, [0] * V.CARRY_ELEMS
    for c in range(steps):
        hints = {f"w{i}.b{j}": (int(row[c * k + i]) >> j) & 1 for i in range(k) for j in range(8)}
        hints.update({f"cap{x}": int(car[c, x]) for x in range(V.CARRY_ELEMS)})
        if tamper is not None:
            tamper(c, hints)
        # the chain test: carry_in = 0 at the start, the previous column's carry_out elsewhere
        bad += sum(hints[f"cap{x}"] % P != prev[x] % P for x in range(V.CARRY_ELEMS))
        W = _run(sys, hints)
        bad += _violations(sys, W)
        prev = [_ev(e, W) for e in carry_out]
        if c + 1 < steps and tamper is None:
            assert prev == [int(v) for v in car[c + 1]], f"carry out of column {c}"
        last = [_ev(e, W) for e in digest]
    return bad, V.digest_of_limbs(last) if all(v < 1 << 16 for v in last) else None


def test_chain_of_column_gadgets_is_pos_leaf() -> None:
    row = np.random.default_rng(7).integers(0, 256, size=256)
    bad, dig = _chain(row, steps=2)
    assert bad == 0
    assert dig == core.pos_leaf(bytes(row.astype(np.uint8)))


def test_tampered_hold_or_prefix_fails() -> None:
    row = np.random.default_rng(8).integers(0, 256, size=256)
    good = core.pos_leaf(bytes(row.astype(np.uint8)))

    def hold(c, h):
        if c == 1:
            h[f"cap{16 + 3}"] = (h[f"cap{16 + 3}"] + 1) % P

    bad, dig = _chain(row, steps=2, tamper=hold)
    assert bad > 0 or dig != good

    def start(c, h):
        if c == 0:
            h["cap16"] = 1                           # a chain start whose hold is not the prefix: another length prefix

    bad, dig = _chain(row, steps=2, tamper=start)
    assert bad > 0 or dig != good


def test_wrong_value_length_is_another_leaf() -> None:
    """The length prefix is a circuit constant (PROTOCOL.md section 9.4): the same bytes under another declared length give
    another leaf, so a prover cannot open a 1536-byte position with a 1535- or 1537-byte value."""
    row = bytes(range(256)) * 6
    assert core.pos_leaf(row) != core.pos_leaf(row[:-1]) and core.pos_leaf(row) != core.pos_leaf(row + b"\0")
    assert V.prefix(1536) != V.prefix(1535)
    with pytest.raises(ValueError):
        V._column_shape(100, 8, 1536)


@pytest.mark.parametrize("word_bits,steps", ((8, 12), (16, 12)))
def test_row_sponges_are_carry_values_and_native(word_bits: int, steps: int) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    rows = np.random.default_rng(steps).integers(0, 1 << word_bits, size=(37, V.K_VU))
    words = torch.as_tensor(rows, device="cuda").reshape(37 * steps, V.K_VU // steps)
    car, dg = L.row_sponges(words, word_bits=word_bits, role=1, row_words=V.K_VU, n=37, steps=steps)
    assert np.array_equal(car.cpu().numpy(), L.carry_values(rows, word_bits=word_bits, role=1, steps=steps))
    assert np.array_equal(dg.cpu().numpy(), L.native(rows, word_bits=word_bits, role=1))
