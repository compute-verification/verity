"""`RMSNormFusedCudaVector8_v3`, whose sum of squares is batches (squares, pair sums, per-lane scans, one batch per CUB
shuffle level), computes `RMSNormFusedCudaVector8_v1`'s f32 sum of squares and returns its two outputs, word for word: at the served statics (hidden widths
6144 and 7168 at both launch blocks) and at statics that reach every branch of the reduction (a block wider than the
row's vectors, threads holding one more vector than others, a last partial warp, one warp, a block of three), on random
words of every kind `twins.rand_bf16` draws and on rows whose squares overflow, underflow or cancel. Its primitive gates
are v1's, and its body's nodes don't grow with N."""
from __future__ import annotations

from collections import Counter
from functools import cache

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.evaluation.reference import evaluate_call, standalone_call
from verity.ml.definitions.numerics import twins as TW
from verity.ml.definitions.registry import cuda_rmsnorm as C

#: (N, EPS, BLOCK): V = N / 8 vectors over min(BLOCK, V) threads
SMALL = [(16, 1e-6, 16), (64, 1e-5, 3), (72, 1e-5, 4), (320, 1e-5, 40), (400, 1e-5, 16), (512, 1e-5, 33), (768, 1e-5, 64),
         (1024, 1e-5, 512), pytest.param(1600, 1e-5, 64, marks=pytest.mark.slow), pytest.param(2048, 1e-5, 64, marks=pytest.mark.slow)]
SERVED = [(6144, 1e-5, 1024), (6144, 1e-5, 256), (7168, 1e-6, 1024), (7168, 1e-6, 256)]
KINDS = ("gauss", "spread", "wide", "mixed", "special", "extreme", "subnormal")
MAX, BIG, TINY = 0x7F7F, 0x5F80, 0x0001  # bf16: the largest finite, 2^64, 2^-133


def _powers(rng, N: int) -> np.ndarray:
    """Half the words ±2^k for k in -8..12, the rest 0: squares 2^-16..2^24 whose sums round differently in any other
    order, which a Gaussian row's rarely do."""
    words = (rng.integers(127 - 8, 127 + 13, N) << 7 | rng.integers(0, 2, N) << 15).astype(np.uint16)
    return np.where(rng.random(N) < 0.5, words, 0).astype(np.uint16)


def _one_lane(rng, N: int, BLOCK: int) -> np.ndarray:
    """Powers of two only in the vectors one thread reads, so the sum of squares is that thread's partial exactly."""
    vectors = N // C.VECTOR
    t = int(rng.integers(min(BLOCK, vectors)))
    own = np.zeros(N, bool)
    for v in range(t, vectors, BLOCK):
        own[C.VECTOR * v:C.VECTOR * (v + 1)] = True
    return np.where(own, _powers(rng, N), 0).astype(np.uint16)


def _orders(N: int, BLOCK: int) -> dict[str, np.ndarray]:
    """One row for each level of the reduction with at least three summands: 2^12 then three 1s in four of its summands
    (a vector's pairs, a thread's vectors, a warp's lanes, the warps), in its order, and 0 elsewhere. The sum of squares
    is then that level's 2^24 + 1 + 1 + 1: 2^24 where each 1 meets 2^24 alone (2^24 + 1 rounds to even) and 2^24 + 2 where
    two 1s meet first, so an order that groups the 1s other than v1 does gives another word."""
    vectors = N // C.VECTOR
    threads = min(BLOCK, vectors)
    levels = {"pairs": [0, 2, 4, 6], "vectors": [C.VECTOR * v for v in range(0, vectors, BLOCK)][:4],
              "lanes": [C.VECTOR * t for t in range(min(threads, 4))],
              "warps": [C.VECTOR * C.WARP * k for k in range((threads + C.WARP - 1) // C.WARP)][:4]}
    rows = {}
    for level, at in levels.items():
        if len(at) >= 3:
            rows[level] = np.zeros(N, np.uint16)
            rows[level][at] = 0x3F80
            rows[level][at[0]] = 0x4580
    return rows


def _rows(rng, N: int, BLOCK: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x, res and w as (rows, N) words: two rows of each random kind, four of powers of two, two of one thread's powers
    of two, a row for each level's order, uniformly random words, then a zero row, x = -res, an add that overflows,
    squares that overflow and squares that underflow."""
    def full(v: int) -> np.ndarray:
        return np.full(N, v, np.uint16)

    g = TW.rand_bf16(rng, N, "gauss")
    rows = [tuple(TW.rand_bf16(rng, N, kind) for _ in range(3)) for kind in KINDS for _ in range(2)]
    rows += [(_powers(rng, N), full(0), g) for _ in range(4)]
    rows += [(_one_lane(rng, N, BLOCK), full(0), g) for _ in range(2)]
    rows += [(x, full(0), g) for x in _orders(N, BLOCK).values()]
    rows += [tuple(rng.integers(0, 1 << 16, (3, N)).astype(np.uint16)), (full(0), full(0), g), (g, g ^ 0x8000, g),
             (full(MAX), full(MAX), g), (full(BIG), full(0), g), (full(TINY), full(0), g)]
    return tuple(np.stack([row[i] for row in rows]).astype(np.uint64) for i in range(3))


def _sums_of_squares(spec, x, res, w) -> list[int]:
    """Each row's f32 sum of squares, the first operand of the body's one ``F32Div``. A change of add order moves it by
    an ulp or so, which the approximate rsqrt and the bf16 outputs rarely show, so it is compared itself."""
    prog, call = standalone_call(spec)
    div, = [c for c in call.children() if c.fn.id == "F32Div_v1"]
    out = []
    for row in zip(x, res, w):
        trace: dict[int, int] = {}
        evaluate_call(prog.circuit, call, [int(v) for a in row for v in a], trace)
        out.append(trace[div.args[0]])
    return out


def _same(N: int, EPS: float, BLOCK: int) -> None:
    v1 = bind(C.RMSNormFusedCudaVector8V1, N=N, EPS=EPS, BLOCK=BLOCK)
    v3 = bind(C.RMSNormFusedCudaVector8, N=N, EPS=EPS, BLOCK=BLOCK)
    x, res, w = _rows(np.random.default_rng(N * 4099 + BLOCK), N, BLOCK)
    assert _sums_of_squares(v3, x, res, w) == _sums_of_squares(v1, x, res, w)
    want, got = evaluate_batch(v1, x, res, w, kernel="reference"), evaluate_batch(v3, x, res, w, kernel="reference")
    assert want.shape == got.shape == (len(x), 2 * N)
    bad = np.argwhere(want != got)
    assert not len(bad), f"{v3.id}: {len(bad)} words differ from v1, first (row, word) {bad[:4].tolist()}"


@pytest.mark.parametrize("N,EPS,BLOCK", SMALL)
def test_v3_is_v1_word_for_word(N, EPS, BLOCK):
    _same(N, EPS, BLOCK)


@pytest.mark.slow
@pytest.mark.parametrize("N,EPS,BLOCK", SERVED)
def test_v3_is_v1_word_for_word_at_the_served_statics(N, EPS, BLOCK):
    _same(N, EPS, BLOCK)


@pytest.mark.parametrize("N,EPS,BLOCK", [(2048, 1e-5, 64), (1600, 1e-5, 64), (1024, 1e-5, 512)])
def test_each_order_rows_sum_of_squares_is_its_levels_sum(N, EPS, BLOCK):
    """In v1's order a vector's pairs, a thread's vectors and the warps meet 2^24 one 1 at a time; CUB's lanes 2 and 3
    meet each other first."""
    rows = _orders(N, BLOCK)
    zero = np.zeros((len(rows), N), np.uint64)
    got = _sums_of_squares(bind(C.RMSNormFusedCudaVector8V1, N=N, EPS=EPS, BLOCK=BLOCK), np.stack(list(rows.values())), zero, zero)
    two_24, two_24_plus_2 = 0x4B800000, 0x4B800001
    assert dict(zip(rows, got)) == {level: two_24_plus_2 if level == "lanes" else two_24 for level in rows}
    assert {"pairs", "lanes"} <= set(rows)


@cache
def _gates(fn) -> Counter:
    """``fn``'s primitive gates by id, a batch member or scan step counted once each."""
    body = getattr(fn, "body", None)
    if body is None:
        return Counter({fn.id: 1})
    out: Counter = Counter()
    for node in body.nodes:
        for k, v in _gates(node.fn).items():
            out[k] += v * node.n
    return out


@pytest.mark.parametrize("N,EPS,BLOCK", SMALL + SERVED)
def test_v3_has_v1s_gates_in_a_body_that_does_not_grow_with_N(N, EPS, BLOCK):
    v1 = bind(C.RMSNormFusedCudaVector8V1, N=N, EPS=EPS, BLOCK=BLOCK)
    v3 = bind(C.RMSNormFusedCudaVector8, N=N, EPS=EPS, BLOCK=BLOCK)
    assert _gates(v3) == _gates(v1)
    assert len(v3.body.nodes) <= 24 + C.WARP
