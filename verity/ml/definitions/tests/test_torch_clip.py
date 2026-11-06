"""`registry.torch_clip`: each Boolean Definition computes, word for word, the numpy model of its torch CUDA kernel
(`numerics.torch_kernels`, written from ForeachReduceOp.cu, Reduce.cuh and clip_grad.py as they run): chunk partials on both of
LpNormFunctor's paths, at sizes around the 4-, 512-, 2048- and 65536-element boundaries, with NaN, ±inf, -0, subnormals, squares
below f32's normal range and squares that overflow; the cleanup with one and several slots a thread; the vector norm on both of
reduce_kernel's input paths; the coefficient on every bf16 total; and the whole clip. The models then match torch's CPU clip to
bf16 rounding. The CUDA kernels themselves are `test_torch_training_gpu.py`'s."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.definitions.numerics import torch_blocks as TB
from verity.ml.definitions.numerics import torch_kernels as TK
from verity.ml.definitions.registry import torch_clip as TC

ONE = 0x3FF0000000000000


def _words(rng, n: int, lo: int = 97, span: int = 54, specials: bool = True) -> np.ndarray:
    """n bf16 words of exponent fields lo .. lo + span - 1, both signs, with NaN, ±inf, -0, a subnormal and the largest finite
    word at fixed places when ``specials``."""
    w = ((rng.integers(0, 2, n) << 15) | (rng.integers(lo, lo + span, n) << 7) | rng.integers(0, 128, n)).astype(np.uint16)
    if specials:
        for i, s in zip((1, 3, 5, 7, 9, 11, 13), (0x7FC1, 0xFF80, 0x8000, 0x0001, 0x7F7F, 0x8003, 0x7F80)):
            if i < n:
                w[i] = s
    return w


def _finite(w: np.ndarray) -> np.ndarray:
    return np.where((w & 0x7F80) == 0x7F80, np.uint16(0x3F80), w).astype(np.uint16)


def _same(got, want) -> None:
    got, want = np.asarray(got, np.uint64).reshape(-1), np.asarray(want, np.uint64).reshape(-1)
    assert got.shape == want.shape and (got == want).all(), np.flatnonzero(got != want)[:5]


def _f64(x: float) -> np.ndarray:
    return np.array([x], np.float64).view(np.uint64)


@pytest.mark.parametrize("N,aligned,lo,span", [(1, 0, 97, 54), (3, 0, 97, 54), (4, 1, 97, 54), (4, 0, 0, 72), (576, 1, 97, 54),
                                               (2047, 0, 112, 16), (2051, 0, 0, 72), (2052, 1, 180, 75), (6000, 0, 97, 54)])
def test_a_chunk_is_the_kernels_partial(N, aligned, lo, span) -> None:
    """One chunk on the 4-vector path (ALIGNED = 1) or the strided one: lanes of one and of two elements, a partial last
    thread, and words whose squares underflow (exponent fields from 0) or overflow (from 180)."""
    rng = np.random.default_rng(N + aligned)
    x = _finite(_words(rng, N, lo, span, specials=False)) if lo else _words(rng, N, lo, span, specials=False)
    st = dict(N=N, ALIGNED=aligned)
    _same(evaluate_words(bind(TC.LpNormChunk, **st), x[None]), TB.MODELS["LpNormChunk_v1"](st, x[None]))


@pytest.mark.slow
@pytest.mark.parametrize("N,aligned", [(TC.CHUNK, 1), (TC.CHUNK - 1, 0)])
def test_a_whole_chunk_is_the_kernels_partial(N, aligned) -> None:
    """A full chunk, 32 elements a lane, on each path: what the smaller chunks can't reach, Bf16SumSquares_v1{J=32}'s 31
    FFMAs and every thread's four lanes. About 5 s each."""
    rng = np.random.default_rng(N)
    x = _words(rng, N)
    st = dict(N=N, ALIGNED=aligned)
    _same(evaluate_words(bind(TC.LpNormChunk, **st), x[None]), TB.MODELS["LpNormChunk_v1"](st, x[None]))


@pytest.mark.parametrize("C", [1, 3, 512, 513, 1100])
def test_the_cleanup_adds_each_threads_slots(C) -> None:
    """lpnorm_cleanup over C chunk partials: one slot a thread, the first 3 threads, and two and three slots a thread; one
    partial +inf and, at 1100, one NaN."""
    rng = np.random.default_rng(C)
    o = TK.f32(_finite(_words(rng, C, 100, 40, specials=False)) & np.uint16(0x7FFF))
    if C > 2:
        o[2] = 0x7F800000
    if C > 1000:
        o[1000] = TK.GPU_NAN
    _same(evaluate_words(bind(TC.LpNormCleanup, C=C), o[None]), TB.MODELS["LpNormCleanup_v1"]({"C": C}, o[None]))


@pytest.mark.parametrize("N", [1, 7, 577, 2051, 2052])
def test_a_tensors_norm_is_its_chunk_then_the_cleanup(N) -> None:
    rng = np.random.default_rng(N)
    x = _words(rng, N, specials=N > 600)
    _same(evaluate_words(bind(TC.ForeachNorm, N=N), x[None]), TB.MODELS["ForeachNorm_v1"]({"N": N}, x[None]))


@pytest.mark.slow
def test_a_tensor_of_three_chunks() -> None:
    """65,536 x 2 + 5 elements on the strided path: what one chunk can't reach, a batch of chunks and the cleanup's three
    slots. About 4 s."""
    N = 2 * TC.CHUNK + 5
    x = _finite(_words(np.random.default_rng(3), N))
    _same(evaluate_words(bind(TC.ForeachNorm, N=N), x[None]), TB.MODELS["ForeachNorm_v1"]({"N": N}, x[None]))


@pytest.mark.parametrize("L", [1, 2, 3, 5, 16, 20, 33, 127, 128, 131, 272, 513])
def test_the_vector_norm_is_one_cta(L) -> None:
    """Both input paths of reduce_kernel (L >= 128 in 4-vectors, a tail of L mod 4) and block widths 1 .. 128."""
    rng = np.random.default_rng(L)
    h = _words(rng, L, 100, 40, specials=False) & np.uint16(0x7FFF)
    _same(evaluate_words(bind(TC.VectorNormBf16, L=L), h[None]), TB.MODELS["VectorNormBf16_v1"]({"L": L}, h[None]))


def test_the_vector_norm_keeps_nan_and_inf() -> None:
    h = np.array([0x3F80, 0x7F80, 0x4000, 0x7FFF, 0x0000], np.uint16)
    for L, want in ((3, 0x7F80), (5, 0x7FFF)):
        got = evaluate_words(bind(TC.VectorNormBf16, L=L), h[None, :L])
        _same(got, [want])
        _same(got, TB.MODELS["VectorNormBf16_v1"]({"L": L}, h[None, :L]))


@pytest.mark.parametrize("L", [1, 2, 7, 64, 127, 128, 129, 131, 272, 1000, 4096, TC.VECTOR_NORM_MAX])
def test_vector_norm_slots_take_each_element_once(L) -> None:
    slots = TC.vector_norm_slots(L)
    assert len(slots) & (len(slots) - 1) == 0 and len(slots) <= TC.THREADS
    assert sorted(e for t in slots for acc in t for e in acc) == list(range(L))
    assert max(len(acc) for t in slots for acc in t) <= TC.SQUARES_MAX


def test_the_coefficient_on_every_bf16_total() -> None:
    w = np.arange(1 << 16, dtype=np.uint16)
    m = np.full(w.size, ONE, np.uint64)
    _same(evaluate_words(TC.ClipCoef, w, m), TB.MODELS["ClipCoef_v1"]({}, w, m))


def test_the_coefficient_at_other_max_norms() -> None:
    """max_norm 0.5, 3, 1e-30 (a coefficient that underflows), 1e39 (past FLT_MAX: f32 infinity, so 1.0 or NaN), -2, -0 and
    NaN, on 512 totals."""
    w = np.unique(np.concatenate([_words(np.random.default_rng(9), 500, 0, 255), [0, 0x8000, 0x3F80, 0x7F80, 0xFF80, 0x7FC0]]))
    for x in (0.5, 3.0, 1e-30, 1e39, -2.0, -0.0, float("nan")):
        m = np.full(w.size, _f64(x)[0], np.uint64)
        _same(evaluate_words(TC.ClipCoef, w.astype(np.uint16), m), TB.MODELS["ClipCoef_v1"]({}, w.astype(np.uint16), m))


@pytest.mark.parametrize("max_norm,specials,coef", [(1.0, False, "below 1"), (1e6, False, 1.0), (1.0, True, 0.0)])
def test_the_clip_whole(max_norm, specials, coef) -> None:
    """Four tensors (3, 576, 5 and 2052 elements, both paths), clipped (a coefficient below 1), not (1.0), and with specials,
    among them the largest finite word, whose square is inf: the total is inf and every finite gradient becomes ±0."""
    sizes = (3, 576, 5, 2052)
    rng = np.random.default_rng(int(max_norm) + specials)
    g = np.concatenate([_finite(_words(rng, n, 100, 30, specials)) for n in sizes])
    st = {"SIZES": sizes}
    got = np.asarray(evaluate_words(bind(TC.ClipGradNorm, **st), g[None], _f64(max_norm))).reshape(-1)
    total, clipped = TB.MODELS["ClipGradNorm_v1"](st, g[None], _f64(max_norm))
    _same(got, np.concatenate([total.reshape(-1), clipped.reshape(-1)]))
    c = float(TK.f32(TK.clip_coef(total.reshape(-1), max_norm)).view(np.float32)[0])
    assert 0 < c < 1 if coef == "below 1" else c == coef


def test_the_model_matches_torch_to_rounding() -> None:
    """torch's CPU clip of bf16 grads sums in another order: its total is within two bf16 steps of the model's."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    sizes = (576, 3, 70000, 2052)
    gs = [_finite(_words(rng, n, specials=False)) for n in sizes]
    total, _ = TK.clip_grad_norm(gs, 1.0)
    ts = [torch.from_numpy(g.view(np.int16).copy()).view(torch.bfloat16) for g in gs]
    for t in ts:
        t.requires_grad_(True)
        t.grad = t.detach().clone()
    ref = float(torch.nn.utils.clip_grad_norm_(ts, 1.0, foreach=False).float())
    got = float(TK.f32(np.array([total], np.uint16)).view(np.float32)[0])
    assert abs(got - ref) <= 2 * 2.0 ** -7 * ref


@pytest.mark.parametrize("d,statics", [
    (TC.Bf16SumSquares, {"J": 0}), (TC.Bf16SumSquares, {"J": TC.SQUARES_MAX + 1}), (TC.NormThreadSum, {"K": 1}),
    (TC.NormThreadSum, {"K": 65}), (TC.LpNormChunkSum, {"P": 0, "ALIGNED": 0}), (TC.LpNormChunkSum, {"P": 2049, "ALIGNED": 0}),
    (TC.LpNormChunkSum, {"P": 6, "ALIGNED": 1}), (TC.LpNormChunk, {"N": 3, "ALIGNED": 1}),
    (TC.LpNormChunk, {"N": TC.CHUNK + 1, "ALIGNED": 0}), (TC.LpNormChunk, {"N": 4, "ALIGNED": 2}),
    (TC.LpNormCleanup, {"C": 0}), (TC.ForeachNorm, {"N": 0}), (TC.ForeachNorm, {"N": TC.N_MAX + 1}),
    (TC.VectorNormBf16, {"L": 0}), (TC.VectorNormBf16, {"L": TC.VECTOR_NORM_MAX + 1}), (TC.ClipGradNorm, {"SIZES": ()}),
    (TC.ClipGradNorm, {"SIZES": (4, 0)}), (TC.ClipGradNorm, {"SIZES": [4, 5]}),
], ids=lambda a: getattr(a, "name", repr(a)))
def test_bind_refuses_statics_outside_the_kernels_range(d, statics) -> None:
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(d, **statics)
