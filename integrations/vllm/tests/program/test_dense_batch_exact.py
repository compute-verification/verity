"""The k16 chain of a step's rows against one W, in integers: the same words as numpy's float64 chain and the int64 twin.

A gemma2-9b B8 Commit evaluates its call boundaries on the host while it holds the leased GPU, and the GPU idles at 0% until each step's
words are hashed.  cov-gm192 (B8 1024/128 stoch) spent 1,747 s of its 1,919 s lease in `work.pair0.instrumented` (other models take
7-16 s), about 1,250 s of it in the logits `Gemm_v2{K=3584,N=256000,DOT=HopperBF16WgmmaDot16_v1}` one row at a time: 2.9 s a row,
numpy's k16 chain keeping ~3 of 8 threads busy.  `cpp/k16_chain.cpp` computes the chain in integers (1.8-2.0 CPU-s a row instead of
6), each W row read once for all of a step's rows, with the GIL released.

Here: the C++ chain hands back exactly the coordinates numpy's does and equals it on the rest, and `chain_rows` equals the int64 twin
on every row at any thread count, on either path."""
from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from verity_vllm.program.kernels import dense_rows as DR
from verity_vllm.program.kernels import derived_rows as D

K = 64
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
