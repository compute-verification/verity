"""h_falsify's reference: the zero instance's tiled transcript, route (a)'s identity, and the duo receipt the arms fold
into.

The GPU arms (`h_falsify/hfalsify.cu`) dump their receipts and are judged bit for bit against `ref.receipt` on the card's
run; these tests hold the reference itself: the transcript against a plain loop, T_ℓ = P_ℓ Uᵀ at B* = 0, the vectorized
fold against the stdlib step, and every segment's key under key_ok.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

POUW = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ref = _load("h_falsify_ref", POUW / "h_falsify" / "ref.py")


def _loop_transcript(At: np.ndarray, Bt: np.ndarray) -> np.ndarray:
    m, k = At.shape
    n, L = Bt.shape[0], k // ref.SEG
    T = np.zeros((L, m, n), dtype=np.int64)
    for i in range(m):
        for j in range(n):
            acc = 0
            for t in range(k):
                acc += int(At[i, t]) * int(Bt[j, t])
                if t % ref.SEG == ref.SEG - 1:
                    T[t // ref.SEG, i, j] = acc
    return T


def test_transcript_is_every_prefix_word() -> None:
    A, Bt = ref.generic(5, 96, 3, seed=7)
    assert np.array_equal(ref.transcript(A, Bt), _loop_transcript(A, Bt))


def test_zero_instance_domain() -> None:
    R, U = ref.zero_instance(16, 64, 8, seed=3)
    assert set(np.unique(R)) <= {0, 1}
    assert np.abs(U).min() >= 1 and np.abs(U).max() <= 64
    St = ref.s_transposed(U, 64)
    assert all(np.array_equal(St[:, t], U[:, t % ref.SEG]) for t in range(64))


def test_route_a_identity() -> None:
    """At B* = 0 every segment shares U, so the prefix words are the residue prefix sums of R times Uᵀ."""
    R, U = ref.zero_instance(16, 256, 8, seed=11)
    assert np.array_equal(ref.transcript(R, ref.s_transposed(U, 256)), ref.transcript_via_prefix(R, U))


def test_vectorized_fold_is_the_stdlib_step() -> None:
    R, U = ref.zero_instance(4, 128, 3, seed=5)
    T = ref.transcript(R, ref.s_transposed(U, 128))
    rc = ref.receipt(T)
    assert rc.dtype == np.uint32
    assert all(int(rc[i, j]) == ref.receipt_scalar(T, i, j) for i in range(4) for j in range(3))


def test_receipt_binds_every_word() -> None:
    R, U = ref.zero_instance(4, 128, 3, seed=5)
    T = ref.transcript(R, ref.s_transposed(U, 128))
    base = ref.receipt(T)
    for ell in range(T.shape[0]):
        moved = T.copy()
        moved[ell, 2, 1] += 1
        assert ref.receipt(moved)[2, 1] != base[2, 1]


def test_keys_pass_key_ok() -> None:
    a, kx, r = ref.keys(448)
    assert all(ref.rf.key_ok(x) for x in a)
    assert all(1 <= x <= 31 for x in r)
