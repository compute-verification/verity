"""The call boundaries' host evaluation off the per-instance path: the same words, computed in batches.

A gemma2-9b B8 Commit evaluates its call boundaries on the host while it holds the leased GPU, and the GPU idles at 0% until each step's
words are hashed.  cov-gm192 (B8 1024/128 stoch) spent 1,747 s of its 1,919 s lease in `work.pair0.instrumented` (other models take
7-16 s): about 1,250 s in the logits `Gemm_v2{K=3584,N=256000,DOT=HopperBF16WgmmaDot16_v1}`, kernel-only, evaluated one row at a time
(2.9 s a row, the 917,504,000-word W read and scanned for finiteness on every row, numpy's k16 chain keeping ~3 of 8 threads busy), and
about 475 s in the 168 GemmaRMSNorm bodies a step, each instance a Python call (`MeanTriton` per row, the IR reference for the scalar
add and the rsqrt).  Now the chain is `cpp/k16_chain.cpp` in integers (each W row read once for all of a step's rows, the GIL
released), and every dense kernel and `Gemm_v2` has a batch form; on 4 CPUs a decode step's call boundaries take 3.3 s instead of 13.2.

Here: the C++ chain hands back exactly the coordinates numpy's does and equals it on the rest; `chain_rows` equals the int64 twin on
every row at any thread count; each batch form gives the instance form's words and declines the instances it declines (non-finite
operands and results, broadcast operands); and `_kernel_rows` takes the batch form and names a declined instance's reason."""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from verity.ml.circuits.evaluation import evaluate_batch, kernels
from verity.ml.circuits.evaluation.batch import type_shape
from verity.ml.circuits.evaluation.reference import signature_of, statics_of
from verity.ml.circuits.defs import bind
from verity_vllm.acquire.sources import call_boundary_plan as CP
from verity_vllm.program.kernels import dense_rows as DR
from verity_vllm.program.kernels import derived_rows as D
from verity_vllm.program.kernels import kernel_registry as KR
from verity_vllm.program.kernels import rows
from verity_vllm.program.registry import b1, hopper

K = 64
DOTS = {"AmpereBF16TcDot16_v2": b1.P.AmpereBF16TcDot16, "HopperBF16WgmmaDot16_v1": hopper.HopperBF16WgmmaDot16}
IN_CHECK = bool(os.environ.get("VERITY_SUITE_ROOT"))
needs_cxx = pytest.mark.skipif(shutil.which(os.environ.get("CXX", "g++")) is None and not IN_CHECK, reason="needs a C++ compiler")


def _bf16(a) -> np.ndarray:
    return (np.asarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _xs(rng: np.random.Generator) -> np.ndarray:
    """Token rows: typical activations, wide exponents, bf16 subnormals, cancelling +-pairs, a zero row and a row of huge words."""
    xs = np.stack([_bf16(rng.standard_normal(K)), _bf16(rng.standard_normal(K) * 2.0 ** rng.integers(-30, 30, K)),
                   rng.integers(0, 0x80, K).astype(np.uint16) | (rng.integers(0, 2, K).astype(np.uint16) << 15),
                   _bf16(np.repeat(rng.standard_normal(K // 2), 2) * np.tile([1, -1], K // 2)), np.zeros(K, dtype=np.uint16),
                   _bf16(rng.standard_normal(K) * 2.0 ** 100)])
    return xs


def _ws(rng: np.random.Generator, n: int = 97) -> np.ndarray:
    """Weight rows: typical, wide, random finite words, subnormal, huge (overflowing steps), tiny (subnormal steps), and three rows
    with an infinity or a NaN."""
    words = rng.integers(0, 1 << 16, (n, K)).astype(np.uint16)
    w = np.concatenate([
        _bf16(rng.standard_normal((n, K)) * 0.02),
        _bf16(rng.standard_normal((n, K)) * 2.0 ** rng.integers(-40, 40, (n, K))),
        np.where((words & 0x7F80) == 0x7F80, words ^ 0x4000, words),
        rng.integers(0, 0x80, (8, K)).astype(np.uint16),
        _bf16(rng.standard_normal((8, K)) * 2.0 ** 112),
        _bf16(rng.standard_normal((8, K)) * 2.0 ** -120),
        _bf16(rng.standard_normal((3, K))),
    ])
    w[-3, 5], w[-2, 40], w[-1, 63] = 0x7F80, 0x7FC1, 0xFF80
    return w


def _mild_w(rng: np.random.Generator, n: int) -> np.ndarray:
    """Finite weight rows no token row of `_xs` but the huge one overflows: typical, bf16 subnormals, tiny (subnormal steps)."""
    return np.concatenate([_bf16(rng.standard_normal((n, K)) * 0.02), rng.integers(0, 0x80, (4, K)).astype(np.uint16),
                           _bf16(rng.standard_normal((4, K)) * 2.0 ** -120)])


@needs_cxx
@pytest.mark.parametrize("dot_id", sorted(DR.STEPS))
def test_the_cpp_chain_hands_back_what_numpy_hands_back_and_equals_it_elsewhere(dot_id, monkeypatch):
    rng = np.random.default_rng(8301)
    X, w = _xs(rng), _ws(rng)
    assert DR.k16_lib() is not None, "cpp/k16_chain.cpp does not build"
    cpp, cpp_back = DR._claimed(X, w, DR.STEPS[dot_id], 3)
    monkeypatch.setattr(DR, "k16_lib", lambda: None)
    ref, ref_back = DR._claimed(X, w, DR.STEPS[dot_id], 3)
    assert np.array_equal(cpp_back, ref_back), np.argwhere(cpp_back != ref_back)[:8]
    assert 0 < ref_back.sum() < ref_back.size
    assert np.array_equal(cpp[~ref_back], ref[~ref_back]), np.argwhere((cpp != ref) & ~ref_back)[:8]


@pytest.mark.parametrize("path", ["cpp", "numpy"])
@pytest.mark.parametrize("dot_id", sorted(DR.STEPS))
def test_chain_rows_equals_the_int64_twin_on_every_row_at_any_thread_count(dot_id, path, monkeypatch):
    """One W for all of a step's rows, one of them with a NaN word (every coordinate of that row is the twin's), over 1, 3 and 8
    coordinate spans (97 * 3 + 27 coordinates: no span count divides them)."""
    if path == "numpy":
        monkeypatch.setattr(DR, "k16_lib", lambda: None)
    elif shutil.which(os.environ.get("CXX", "g++")) is None and not IN_CHECK:
        pytest.skip("needs a C++ compiler")
    rng = np.random.default_rng(8302)
    X, w = _xs(rng), _ws(rng)
    X = np.concatenate([X, X[:1]])
    X[-1, 7] = 0x7FC0
    twin = D.dot_bf16_for(dot_id)
    ref = [twin(x[None, :], w) for x in X]
    for n in (1, 3, 8):
        out, nonfin = DR.chain_rows(X, w, dot_id, n)
        for r, (want, want_nf) in enumerate(ref):
            assert np.array_equal(out[r], want), (n, r, np.flatnonzero(out[r] != want)[:8])
            assert np.array_equal(nonfin[r], want_nf), (n, r)
    assert nonfin[-1].all() and nonfin[0][-3:].all()


def test_chain_rows_takes_the_twin_where_the_cpp_chain_is_not_built(monkeypatch):
    monkeypatch.setattr(DR, "k16_lib", lambda: None)
    rng = np.random.default_rng(8303)
    X, w = _xs(rng)[:2], _ws(rng, 9)
    out, _ = DR.chain_rows(X, w, "HopperBF16WgmmaDot16_v1", 2)
    assert np.array_equal(out[1], D.dot_bf16_for("HopperBF16WgmmaDot16_v1")(X[1][None, :], w)[0])


# ---- the dense chain's batch forms -----------------------------------------------------------------------------------------

_BAD = {32: np.array([0x7F800000, 0xFF800000, 0x7FC00000, 0x7F7FFFFF, 0xFF7FFFFF], dtype=np.uint32),
        16: np.array([0x7F80, 0xFF80, 0x7FC0, 0x7F7F, 0xFF7F], dtype=np.uint16)}
DENSE = [(family, t) for family in DR.KERNELS for t in DR._targets()[family]]


def _words(rng: np.random.Generator, t, n: int) -> list[np.ndarray]:
    """The kernel's self-check words, one word of about one instance in eight per operand non-finite or the largest finite (an
    overflowing result), and instance 0's first word an infinity."""
    params, _ = signature_of(t)
    out = []
    for a, pt in zip(kernels(t)[DR.KERNEL].sample(rng, n, params, **statics_of(t)), params):
        flat = a.reshape(n, -1).copy()
        hit = np.flatnonzero(rng.random(n) < 1 / 8)
        flat[hit, rng.integers(0, flat.shape[1], hit.size)] = rng.choice(_BAD[flat.dtype.itemsize * 8], hit.size)
        out.append(flat.reshape(n, *type_shape(pt)))
    out[0].reshape(n, -1)[0, 0] = _BAD[out[0].dtype.itemsize * 8][0]
    return out


def _instances(t, args: list[np.ndarray]) -> tuple[list[np.ndarray | None], np.ndarray]:
    fn, statics = KR.instance(t, DR.KERNEL), statics_of(t)
    got: list[np.ndarray | None] = []
    for i in range(len(args[0])):
        try:
            got.append(np.asarray(fn(*(a[i] for a in args), **statics)).reshape(-1))
        except KR.Declined:
            got.append(None)
    return got, np.array([g is None for g in got])


def _agree(t, args: list[np.ndarray]) -> np.ndarray:
    want, want_declined = _instances(t, args)
    out, declined = kernels(t)[DR.KERNEL].fn(*args, **statics_of(t))
    assert np.array_equal(np.asarray(declined), want_declined), (t.id, np.flatnonzero(declined != want_declined)[:8])
    for i, g in enumerate(want):
        if g is not None:
            assert np.array_equal(np.asarray(out[i]).astype(np.uint64), g.astype(np.uint64)), (t.id, i)
    return want_declined


@pytest.mark.parametrize("family,t", DENSE, ids=[t.id for _f, t in DENSE])
def test_a_dense_batch_form_gives_the_instance_forms_words_and_declines(family, t):
    rng = np.random.default_rng(8310 + len(t.id))
    n = 96
    args = _words(rng, t, n)
    declined = _agree(t, args)
    assert 0 < declined.sum() < n, (t.id, int(declined.sum()))
    for j, a in enumerate(args):
        # an operand broadcast over the batch (a norm weight, a row's scale): finite, then a NaN
        b = list(args)
        i0 = int(np.flatnonzero(~DR._nonfinite_rows(a))[0])
        b[j] = np.broadcast_to(a[i0:i0 + 1], a.shape)
        _agree(t, b)
        b[j] = np.broadcast_to(np.full(a[:1].shape, _BAD[a.dtype.itemsize * 8][2]), a.shape)
        assert _agree(t, b).all()


def test_registry_runs_the_batch_form_first_and_falls_back_to_the_instance_form():
    calls = []

    def fn(x, **_s):
        calls.append("instance")
        if x[0] == 7:
            raise KR.Declined("seven")
        return x + 1

    def batch(x, **_s):
        calls.append("batch")
        return None if x[0, 0] == 9 else (x + 1, x[:, 0] == 7)

    core = KR._batch(fn, lambda statics: "unclaimed" if statics.get("off") else None, batch)
    x = np.array([[1], [7], [3]], dtype=np.uint64)
    out, declined = core(x)
    assert calls == ["batch"] and list(declined) == [False, True, False] and list(out[:, 0]) == [2, 8, 4]
    calls.clear()
    out, declined = core(np.array([[9], [7]], dtype=np.uint64))
    assert calls == ["batch", "instance", "instance"] and list(declined) == [False, True] and out[0, 0] == 10
    calls.clear()
    assert core(x, off=True) is NotImplemented and not calls


# ---- Gemm_v2's batch form --------------------------------------------------------------------------------------------------

def _gemm(dot_id: str, N: int):
    return bind(b1.GemmV2, K=K, N=N, DOT=DOTS[dot_id])


@pytest.mark.parametrize("dot_id", sorted(DR.STEPS))
def test_gemm_v2_batch_form_reads_one_w_for_every_row(dot_id, monkeypatch):
    """The logits rows of a step, the weight field broadcast over them: one `gemm_rows` for all of them, each row the instance kernel's
    words, declined where it declines (a non-finite x word, an overflowing coordinate); a NaN anywhere in W declines every row."""
    monkeypatch.setattr(DR, "FAST_GEMM_WORDS", 1)
    rng = np.random.default_rng(8320)
    X, w = _xs(rng), _mild_w(rng, 41)
    w[0] = _bf16(rng.standard_normal(K) * 2.0 ** 80)
    X = np.concatenate([X, X[:1]])
    X[-1, 3] = 0xFF80
    t, M, N = _gemm(dot_id, len(w)), len(X), len(w)
    wb = np.broadcast_to(w.reshape(1, -1), (M, N * K))
    calls = []
    monkeypatch.setattr(DR, "gemm_rows", lambda *a, _g=DR.gemm_rows: calls.append(len(a[0])) or _g(*a))
    want = []
    for i in range(M):
        try:
            want.append(KR.instance(t, rows.KERNEL)(X[i], w.reshape(-1), **statics_of(t)))
        except KR.Declined:
            want.append(None)
    calls.clear()
    out, declined = kernels(t)[rows.KERNEL].fn(X, wb, **statics_of(t))
    assert calls == [M], "one gemm_rows over every row"
    assert list(declined) == [g is None for g in want]
    assert list(declined) == [False] * 5 + [True, True], "the huge row overflows on W's huge row; the last has an infinity"
    for i, g in enumerate(want):
        if g is not None:
            assert np.array_equal(out[i], np.asarray(g).astype(np.uint64)), i
    assert np.array_equal(evaluate_batch(t, X[:5], wb[:5], kernel="reference"), out[:5])
    w_nan = w.copy()
    w_nan[N // 2, 0] = 0x7FC0
    _, declined = kernels(t)[rows.KERNEL].fn(X[:5], np.broadcast_to(w_nan.reshape(1, -1), (5, N * K)), **statics_of(t))
    assert declined.all()


def test_gemm_v2_batch_form_leaves_other_batches_to_the_instance_form(monkeypatch):
    rng = np.random.default_rng(8321)
    X, w = _xs(rng)[:3], _mild_w(rng, 5)
    t = _gemm("HopperBF16WgmmaDot16_v1", len(w))
    wb = np.broadcast_to(w.reshape(1, -1), (3, w.size))
    assert DR.gemm_dot_rows(rows.gemm_dot_prim, X, wb, **statics_of(t)) is None, "below FAST_GEMM_WORDS: the int64 twin per row"
    monkeypatch.setattr(DR, "FAST_GEMM_WORDS", 1)
    assert DR.gemm_dot_rows(rows.gemm_dot_prim, X, np.ascontiguousarray(wb), **statics_of(t)) is None, "a W per row is not one W"
    assert DR.gemm_dot_rows(rows.gemm_dot_prim, X[:1], np.ascontiguousarray(wb[:1]), **statics_of(t)) is not None, "one row reads one W"
    out, declined = kernels(t)[rows.KERNEL].fn(X, np.ascontiguousarray(wb), **statics_of(t))
    assert not declined.any() and np.array_equal(out, evaluate_batch(t, X, wb, kernel="reference")), "the instance form, per row"


def test_kernel_rows_takes_the_batch_form_and_names_a_declined_instance(monkeypatch):
    monkeypatch.setattr(DR, "FAST_GEMM_WORDS", 1)
    rng = np.random.default_rng(8322)
    X, w = _xs(rng)[:4], _mild_w(rng, 9)
    t = _gemm("HopperBF16WgmmaDot16_v1", len(w))
    spec = f"Gemm_v2{{K={K},N={len(w)},DOT=HopperBF16WgmmaDot16_v1}}"
    wb = np.broadcast_to(w.reshape(1, -1), (4, w.size))
    calls = []
    monkeypatch.setattr(DR, "gemm_rows", lambda *a, _g=DR.gemm_rows: calls.append(len(a[0])) or _g(*a))
    got = CP._kernel_rows(t, spec, [X, wb])
    assert calls == [4] and np.array_equal(got, evaluate_batch(t, X, wb, kernel="reference"))
    X[2, 0] = 0x7F80
    with pytest.raises(ValueError, match=r"the row kernel of Gemm_v2\{.*\} declines an instance \(a bf16 operand word is not finite\)"):
        CP._kernel_rows(t, spec, [X, wb])
    X[2, 0], w[3] = 0, _bf16(rng.standard_normal(K) * 2.0 ** 126)
    X[1] = _bf16(rng.standard_normal(K) * 2.0 ** 10)
    with pytest.raises(ValueError, match=r"declines an instance \(Gemm_v2: \d+ output coordinates have a non-finite bf16 operand"):
        CP._kernel_rows(t, spec, [X, np.broadcast_to(w.reshape(1, -1), (4, w.size))])
