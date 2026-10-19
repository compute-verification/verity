"""C-Flock's pieces for the int8 row's LUT softmax (`verity_vllm.program.registry.int8_exact`: e = LUT[LutIndex(m - S, A)]):
`Exp2Lut127_v1`, the 256-entry table read one-hot in the circuit, and `LutIndex_v1`, the clamped high word of d A.  Lowered by
`ir_lower.lower` into a C-Flock circuit and evaluated there, each agrees with its Definition's reference: the table on every
index, where it is fattn_ref.py's byte for byte; the index on uniform words, on the served domain (d in [0, 2^23)), on words
whose index spreads over every byte or straddles the clamp, and on the edges: a negative d, 0, the clamp at 255 from both
sides, the product's bits from 32 one at a time, and A's extremes.

The two Definitions are this file's copies of `int8_exact`'s, id, signature and evaluator, unregistered: a backend's test
imports no integration (`tests/test_backend_boundaries.py`). circuit-check holds each piece to the registered Definition."""
from __future__ import annotations

import hashlib

import numpy as np

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.types import Value
from verity_flock import ir_lower as IL
from verity_flock import tail_pieces as TP
from verity_flock.gf2 import Circuit

S32_MIN, S32_MAX, U32_MAX = -(1 << 31), (1 << 31) - 1, (1 << 32) - 1
#: fattn_ref.py's table: LUT[t] = rint(127 2^(-t/16))
LUT = tuple(int(v) for v in np.rint(127.0 * np.exp2(-np.arange(256) / 16.0)))


def _lut_index(d: int, a: int) -> int:
    s = (d & U32_MAX) - ((d & 0x80000000) << 1)
    return min((max(s, 0) * (a & U32_MAX)) >> 32, 255)


LutIndex = PrimitiveDefinition("LutIndex", 1, (("d", Value(32)), ("a", Value(32))), Value(8), _lut_index, register=False)
Exp2Lut127 = PrimitiveDefinition("Exp2Lut127", 1, (("t", Value(8)),), Value(32), lambda t: LUT[t & 0xFF], register=False,
                                 tables=(("exp2_lut127", "registered"),))
D_EDGES = (S32_MIN, S32_MIN + 1, -(1 << 23), -1, 0, 1, 2, 255, 256, 257, 1 << 16, (1 << 23) - 1, 1 << 23, 1 << 30, S32_MAX - 1, S32_MAX)
A_EDGES = (0, 1, 2, 1 << 24, (1 << 31) - 1, 1 << 31, (1 << 31) + 1, U32_MAX - 1, U32_MAX)


def _agree(prim, *inputs: np.ndarray) -> np.ndarray:
    """The piece's outputs on ``inputs`` (one array a parameter), held to the Definition's reference; the reference."""
    params, _ = E.batch.signature_of(prim)
    widths = [w for pt in params for w in pt.leaf_widths()]
    C = Circuit()
    ins = [C.input(f"in{j}", w) for j, w in enumerate(widths)]
    (out,) = IL.lower(C, prim, ins)
    C.output("out", out)
    T = len(inputs[0])
    got_s, fail = C.evaluate({f"in{j}": IL.slice_words(x.astype(np.uint64), w) for j, (x, w) in enumerate(zip(inputs, widths))}, T)
    assert not fail
    got = IL.unslice_words(got_s["out"], T)
    ref = E.evaluate_batch(prim, *inputs).astype(np.uint64).reshape(T, -1)[:, 0]
    bad = np.flatnonzero(got != ref)
    assert not bad.size, f"{prim.id}: {bad.size} of {T} instances differ from the reference, first {bad[:5].tolist()}"
    return ref


def _words(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).astype(np.uint32)


def test_the_exp2_lut127_piece_reads_fattns_table_at_every_index() -> None:
    """All 256 indices; the table is fattn_ref.py's ``rint(127 2^(-t/16))``, its words' sha256 the pin."""
    ref = _agree(Exp2Lut127, np.arange(256, dtype=np.uint8))
    fattn = np.rint(127.0 * np.exp2(-np.arange(256) / 16.0)).astype(np.int64)
    assert ref.tolist() == fattn.tolist() == list(TP.exp2_lut127_table())
    assert hashlib.sha256(fattn.astype("<u4").tobytes()).hexdigest() == TP.EXP2_LUT127_SHA256
    assert ref[0] == 127 and ref[127] == 1 and not ref[128:].any()


def test_the_lut_index_piece_computes_the_clamped_index() -> None:
    """Every pair of d's and A's edges; the clamp from both sides (d A >> 32 at 254, 255 and 256, through A = 2^24 and through
    A's top, where product bit 62 is set); each product bit 32..61 set alone (from 40 the clamp's); uniform words, the served d
    in [0, 2^23), words whose index spreads over 0..300, and words whose index is within 2 of 255."""
    pairs = [(d, a) for d in D_EDGES for a in A_EDGES]
    clamp = [(255 * 256 - 1, 1 << 24), (255 * 256, 1 << 24), (256 * 256 - 1, 1 << 24), (256 * 256, 1 << 24),
             (255, U32_MAX), (256, U32_MAX), (257, U32_MAX), (S32_MAX, U32_MAX)]
    bits = [(1 << max(0, m - 31), 1 << (m - max(0, m - 31))) for m in range(32, 62)]
    edges = pairs + clamp + bits
    rng = np.random.default_rng(59)
    n = 4000
    uniform = rng.integers(0, 1 << 32, (2, n), dtype=np.uint64).astype(np.uint32)
    served = [rng.integers(0, 1 << 23, n).astype(np.uint32), rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)]
    spread_a = rng.integers(1 << 20, 1 << 32, n, dtype=np.uint64)
    spread_d = [int(rng.integers(0, (300 << 32) // int(x))) for x in spread_a]
    near_a = rng.integers(1 << 20, 1 << 32, n, dtype=np.uint64)
    near_d = [(255 << 32) // int(x) + int(k) for x, k in zip(near_a, rng.integers(-2, 3, n))]
    d = np.concatenate([_words([e for e, _ in edges]), uniform[0], served[0], _words(spread_d), _words(near_d)])
    a = np.concatenate([_words([e for _, e in edges]), uniform[1], served[1], spread_a.astype(np.uint32), near_a.astype(np.uint32)])
    ref = _agree(LutIndex, d, a).tolist()
    e = len(pairs)
    assert ref[e:e + len(clamp)] == [254, 255, 255, 255, 254, 255, 255, 255]
    assert ref[e + len(clamp):len(edges)] == [1 << k for k in range(8)] + [255] * 22
    assert all(r == 0 for r, (dd, _a) in zip(ref, pairs) if dd <= 0)
    s = len(edges) + 2 * n
    assert len(set(ref[s:s + n])) == 256 and set(ref[s + n:]) == {252, 253, 254, 255}
