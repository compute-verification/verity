"""The XOR-output-bits compression (``blake3_xob.py``) against the native BLAKE3 compression, through a numpy interpreter
of its program (pin | hint | prod | bits | sels | xadd), and the documented census."""
from __future__ import annotations

import numpy as np
import pytest

from ..compile import P
from . import blake3 as L
from . import blake3_xob as X
from .blake3_test import _eval, failing_constraints


def _signed(x: np.ndarray) -> np.ndarray:
    return np.where(x > P // 2, x - P, x)


def run_program_np(sys, hints: dict) -> np.ndarray:
    l = len(next(iter(hints.values())))
    W = np.zeros((sys.m, l), dtype=np.int64)
    for op in sys.program:
        kind = op[0]
        if kind == "hint":
            W[op[1]] = np.asarray(hints[op[2]], dtype=np.int64) % P
        elif kind == "prod":
            W[op[1]] = (_eval(op[2], W) * _eval(op[3], W)) % P
        elif kind == "bits":
            _, row0, nbits, e = op
            x = _eval(e, W)
            for i in range(nbits):
                W[row0 + i] = (x >> i) & 1
        elif kind == "sels":
            _, row0, k0s, k1s, key = op
            x = _signed(_eval(key, W))
            for i, (k0, k1) in enumerate(zip(k0s, k1s)):
                W[row0 + i] = ((x >= k0) & (x <= k1)).astype(np.int64)
        elif kind == "xadd":
            _, row0, lo, hi, olo, ohi = op
            W[row0:row0 + 32] = X.run_xadd(_eval(lo, W), _eval(hi, W), _eval(olo, W), _eval(ohi, W))
        else:
            raise ValueError(kind)
    return W


def _word(ctx, name: str, as_bits: bool):
    if as_bits:
        return L._W32([ctx.hint(f"{name}.b{i}") for i in range(32)])
    return L._W32(limbs=(ctx.hint(f"{name}.lo"), ctx.hint(f"{name}.hi")))


def _system(limb_cv: bool, const_counter: bool):
    """One compression on hint inputs: ``cv`` words 0..3 as limbs (the column gadget's shape) or bits, the counter a
    constant 0 or one hint bit, the chunk-start / chunk-end flags hint bits."""
    ctx = L._fresh_ctx()
    cv = [_word(ctx, f"cv{w}", not (limb_cv and w < 4)) for w in range(8)]
    msg = [_word(ctx, f"m{w}", False) for w in range(16)]
    ctr = L._W32.const(ctx, 0) if const_counter else L._W32([ctx.hint("ctr.b0")] + [L._const(ctx, 0)] * 31)
    one, zero = L._const(ctx, 1), L._const(ctx, 0)
    fl = L._W32([ctx.hint("cs"), ctx.hint("ce"), zero, zero, one] + [zero] * 27)
    h = X.compress_xob(ctx, cv, msg, ctr, L._W32.const(ctx, L.BLOCK_LEN), fl, "t")
    return ctx.sys, h


def _inputs(seed: int, l: int, const_counter: bool):
    rng = np.random.default_rng(seed)
    cv = rng.integers(0, 1 << 32, size=(8, l), dtype=np.uint64).astype(np.uint32)
    msg = rng.integers(0, 1 << 32, size=(16, l), dtype=np.uint64).astype(np.uint32)
    ctr = np.zeros(l, dtype=np.int64) if const_counter else rng.integers(0, 2, size=l)
    cs, ce = rng.integers(0, 2, size=l), rng.integers(0, 2, size=l)
    hints: dict = {}
    for nm, words in (("cv", cv), ("m", msg)):
        for w in range(words.shape[0]):
            x = words[w].astype(np.int64)
            hints[f"{nm}{w}.lo"], hints[f"{nm}{w}.hi"] = x & 0xFFFF, x >> 16
            for i in range(32):
                hints[f"{nm}{w}.b{i}"] = (x >> i) & 1
    hints["ctr.b0"], hints["cs"], hints["ce"] = ctr, cs, ce
    flags = (cs | (ce << 1) | L.KEYED_HASH).astype(np.uint32)
    want = L._cv_of(L.compress_np(cv, msg, ctr.astype(np.uint64), L.BLOCK_LEN, flags))
    return hints, want


def _value(word, W) -> np.ndarray:
    return sum((_eval(b, W).astype(np.int64) << i) for i, b in enumerate(word.bits))


@pytest.mark.parametrize("limb_cv,const_counter", [(True, False), (False, True)])
def test_xob_compression_is_blake3(limb_cv, const_counter):
    sys, h = _system(limb_cv, const_counter)
    hints, want = _inputs(7 + 2 * limb_cv + const_counter, 6, const_counter)
    hints = {n: v for n, v in hints.items() if any(nm == n for _, nm in sys.hints)}
    W = run_program_np(sys, hints)
    assert failing_constraints(sys, W, {}) == []
    got = np.stack([_value(w, W) for w in h])
    assert np.array_equal(got, want.astype(np.int64))


def test_tampered_rows_break_a_constraint():
    """A flipped XOR-output bit, a flipped carry, a changed square-pair product each fail some constraint."""
    sys, _ = _system(True, False)
    hints, _ = _inputs(3, 4, False)
    hints = {n: v for n, v in hints.items() if any(nm == n for _, nm in sys.hints)}
    W0 = run_program_np(sys, hints)
    assert failing_constraints(sys, W0, {}) == []
    xadd = [op[1] + 7 for op in sys.program if op[0] == "xadd"]
    carry = [r.idx for r in sys.rows if r.kind == "sel"]
    sq = [r.idx for r in sys.rows if r.kind == "prod" and ".sq" in r.name]
    for rows in (xadd, carry, sq):
        W = W0.copy()
        j = rows[len(rows) // 2]
        W[j, 1] = (W[j, 1] + 1) % P if W[j, 1] == 0 else (W[j, 1] - 1) % P
        assert failing_constraints(sys, W, {}) != []


def test_roots_exist():
    assert X.R2 * X.R2 % P == 2 and X.U * X.U % P == P - 1


def test_census_numbers_are_the_documented_ones():
    c = X.census(8)
    assert (c["per_block_pinned"], c["per_block_xob"]) == (PINNED_PER_BLOCK, XOB_PER_BLOCK)
    assert c["hash_rows_per_vu_xob"] == 48 * XOB_PER_BLOCK


PINNED_PER_BLOCK, XOB_PER_BLOCK = None, None      # the column gadget's input shapes (a counter bit, the chunk flags)
