"""`registry.rmsnorm_fused_split`: `RMSNormFusedCuda_v2{N, EPS}`'s parts and combine, wired as a window wires them (part g's x
the norm's x on `part_ranges`, its res the whole residual row, the combine reading the parts' r and s in part order), return
the norm's two outputs word for word: on Gaussian rows at three scales, on uniformly random words (NaN, ±inf, ±0 and
subnormals among them), and on rows whose residual add, squares or variance overflow, underflow or cancel. Each part and the
combine fits a unit's 256 rows where the norm after Llama-3.2-1B's down_proj does not, and every gate of theirs is one of the
norm's but the combine's widening of r, which it reads as words."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import rmsnorm_fused_split as RS
from verity.ml.windows import interface as IF

#: (N, EPS, PARTS): the norm after Llama-3.2-1B's down_proj in its four parts and in two; four strides (three FFMAs a thread)
#: in sixteen parts; one stride of 18 warps in two; a lone partial warp
CASES = [(2048, 1e-5, 4), (2048, 1e-6, 2), pytest.param(4096, 1e-5, 16, marks=pytest.mark.slow), (576, 1e-5, 2), (16, 1e-6, 1)]
MAX, INF, NAN, TINY, BIG = 0x7F7F, 0x7F80, 0x7FC1, 0x0001, 0x5F80  # bf16: the largest finite, +inf, a NaN, 2^-133, 2^64


def _flat(v) -> list[int]:
    return [w for u in v for w in _flat(u)] if isinstance(v, (tuple, list)) else [int(v)]


def _gauss(rng, n: int, scale: float) -> np.ndarray:
    return ((rng.standard_normal(n) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _rows(rng, N: int) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """(kind, x, res, w) rows of bf16 words."""
    def g(scale: float = 1.0) -> np.ndarray:
        return _gauss(rng, N, scale)

    def full(v: int) -> np.ndarray:
        return np.full(N, v, np.uint16)

    def put(a: np.ndarray, at: dict[int, int]) -> np.ndarray:
        a = a.copy()
        for j, v in at.items():
            a[j % N] = v
        return a

    zero, h = full(0), g()
    return [
        *((f"gauss x{s}", g(s), g(s), g()) for s in (0.02, 1.0, 40.0)),
        ("random words", *(rng.integers(0, 1 << 16, (3, N)).astype(np.uint16))),
        ("zero row: var 0, s = rsqrtf(EPS)", zero, zero, g()),
        ("x = -res: every r a cancelled +0", h, h ^ 0x8000, g()),
        ("-0 + -0", full(0x8000), full(0x8000), g()),
        ("bf16 add overflows to inf", full(MAX), full(MAX), g()),
        ("squares overflow f32", full(BIG), zero, g()),
        ("squares underflow", full(TINY), zero, g()),
        ("one NaN element, in the last stride", put(g(), {-1: NAN}), g(), g()),
        ("one inf element", put(g(), {N // 3: INF}), g(), g()),
        ("w with NaN, inf and 0", g(), g(), put(g(), {1: NAN, 2: INF, 3: 0, 4: 0x8000})),
    ]


def _split(N: int, EPS: float, PARTS: int, x: np.ndarray, res: np.ndarray, w: np.ndarray) -> list[int]:
    """The parts' and the combine's evaluations, each unit's ports its writers' rows as the window wires them."""
    n, r, s = N // PARTS, [], []
    for g in range(PARTS):
        xg = np.concatenate([x[lo:hi] for lo, hi in RS.part_ranges(N, PARTS, g)])
        out = _flat(evaluate(bind(RS.RMSNormFusedCudaPart, N=N, PARTS=PARTS, PART=g), [int(v) for v in xg], [int(v) for v in res]))
        r, s = r + out[:n], s + out[n:]
    return _flat(evaluate(bind(RS.RMSNormFusedCudaCombine, N=N, EPS=EPS, PARTS=PARTS), r, s, [int(v) for v in w]))


@pytest.mark.parametrize("N,EPS,PARTS", CASES)
def test_the_parts_and_combine_are_the_norm(N, EPS, PARTS):
    rng = np.random.default_rng(N + PARTS)
    norm = bind(b1.RMSNormFusedCuda, N=N, EPS=EPS)
    for kind, x, res, w in _rows(rng, N):
        want = _flat(evaluate(norm, *([int(v) for v in a] for a in (x, res, w))))
        got = _split(N, EPS, PARTS, x, res, w)
        assert len(got) == 2 * N and got == want, (kind, [j for j, (a, b) in enumerate(zip(got, want)) if a != b][:8])


def _gates(fn) -> Counter:
    """Its primitive gates by id, a batch counted once a member."""
    out: Counter = Counter()
    for node in fn.body.nodes:
        out[node.fn.id] += node.n
    return out


@pytest.mark.parametrize("N,EPS,PARTS", [(2048, 1e-5, 4), (576, 1e-5, 2)])
def test_every_gate_is_one_of_the_norms_but_the_combines_widening_of_r(N, EPS, PARTS):
    split = sum((_gates(bind(RS.RMSNormFusedCudaPart, N=N, PARTS=PARTS, PART=g)) for g in range(PARTS)),
                _gates(bind(RS.RMSNormFusedCudaCombine, N=N, EPS=EPS, PARTS=PARTS)))
    norm = _gates(bind(b1.RMSNormFusedCuda, N=N, EPS=EPS))
    assert split - norm == Counter({"Bf16ToF32_v1": N}) and not norm - split


def _unit_rows(fn, cuts: dict[str, int]) -> int:
    return sum(map(len, IF.unit_rows(fn, cuts)))


def test_at_llama_1b_down_proj_each_unit_holds_at_most_256_rows():
    """The norm's x as down_proj's 512 slices of 4 columns (`Gemm_v2{K=8192, N=4}`) is 516 rows in and out; a part reads 128
    of them and the residual row and writes r and s (131), and the combine reads 4 r rows, 4 s rows and w and writes
    normed and res' (11)."""
    N, EPS, PARTS, slice_words = 2048, 1e-5, 4, 4
    assert _unit_rows(bind(b1.RMSNormFusedCuda, N=N, EPS=EPS), {"x": N // slice_words}) == 516
    for g in range(PARTS):
        part = bind(RS.RMSNormFusedCudaPart, N=N, PARTS=PARTS, PART=g)
        assert _unit_rows(part, {"x": N // PARTS // slice_words}) == 131
        assert [hi - lo for lo, hi in RS.part_ranges(N, PARTS, g)] == [256, 256]
    assert _unit_rows(bind(RS.RMSNormFusedCudaCombine, N=N, EPS=EPS, PARTS=PARTS), {"r": PARTS, "s": PARTS}) == 11


@pytest.mark.parametrize("defn,kw,why", [
    (RS.RMSNormFusedCudaPart, {"N": 1536, "PARTS": 1, "PART": 0}, "N=1536 is not a multiple of the kernel's 1024 threads"),
    (RS.RMSNormFusedCudaPart, {"N": 2048, "PARTS": 3, "PART": 0}, "PARTS=3 does not cut the kernel's 1024 threads into whole warps"),
    (RS.RMSNormFusedCudaPart, {"N": 2048, "PARTS": 4, "PART": 4}, "PART=4 is not one of PARTS=4"),
    (RS.RMSNormFusedCudaCombine, {"N": 16, "EPS": 1e-5, "PARTS": 2}, "PARTS=2 does not cut the kernel's 16 threads"),
])
def test_a_binding_with_no_cut_is_refused_by_name(defn, kw, why):
    with pytest.raises(ValueError, match=why):
        bind(defn, **kw)
