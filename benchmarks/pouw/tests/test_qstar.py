"""QSTAR's word programs as arithmetic: each scalar arm of `qstar/qstar.cu` makes the zero instance's prefix words by an
identity, and these tests hold the identity against `h_falsify/ref.py`'s transcript with the kernel's own table layouts.
The arms' receipts are judged bit for bit on the card (`qstar --mode gate`); `qstar/expect.json` pins each timed loop.
"""
from __future__ import annotations

import importlib.util
import json
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
SEG = ref.SEG


def _instance(m=8, k=128, n=16, seed=13):
    R, U = ref.zero_instance(m, k, n, seed=seed)
    return R, U, ref.transcript(R, ref.s_transposed(U, k))


def _rw(R: np.ndarray) -> np.ndarray:
    """pack_rw: bit p of Rw[ℓ, i] is R[i, 32ℓ + p]."""
    m, k = R.shape
    bits = R.reshape(m, k // SEG, SEG).astype(np.uint64) << np.arange(SEG, dtype=np.uint64)
    return bits.sum(axis=2).T.astype(np.uint32)


def test_fr8_tables_make_every_word() -> None:
    """k_fr8: tab[g][q][p] holds columns 4q..4q+3 of Σ_b p_b U[j, 8g + b]; a word is acc + four entries."""
    R, U, T = _instance()
    n = U.shape[0]
    p = np.arange(256)
    bits = (p[:, None] >> np.arange(8)) & 1                       # [p][b]
    tab = np.einsum("pb,jgb->gpj", bits, U.astype(np.int64).reshape(n, 4, 8))  # [g][p][j]
    Rw = _rw(R)
    acc = np.zeros((R.shape[0], n), dtype=np.int64)
    for ell in range(Rw.shape[0]):
        for g in range(4):
            acc += tab[g, (Rw[ell] >> (8 * g)) & 255]
        assert np.array_equal(acc, T[ell])


def test_fr16_halves_make_every_word() -> None:
    """k_fr16_half/expand: T[g][p] = H[g][0][p & 255] + H[g][1][p >> 8]; a word is acc + e0 + e1 (one IADD3)."""
    R, U, T = _instance()
    n = U.shape[0]
    b = np.arange(256)
    bits = (b[:, None] >> np.arange(8)) & 1
    H = np.einsum("pb,jghb->ghpj", bits, U.astype(np.int64).reshape(n, 2, 2, 8))  # [g][h][b][j]
    p = np.arange(65536)
    tab = H[:, 0, p & 255] + H[:, 1, p >> 8]                                        # [g][p][j]
    Rw = _rw(R)
    acc = np.zeros((R.shape[0], n), dtype=np.int64)
    for ell in range(Rw.shape[0]):
        acc += tab[0, Rw[ell] & 0xFFFF] + tab[1, Rw[ell] >> 16]
        assert np.array_equal(acc, T[ell])


def test_bit_planes_make_every_word() -> None:
    """k_bits: Σ_{b<7} 2^b popc(R & P_b) − 2^7 popc(R & P_7), with P_b bit p = bit b of U[j, p] as a byte."""
    R, U, T = _instance()
    n = U.shape[0]
    ub = U.astype(np.uint8).astype(np.uint64)
    P = ((ub[:, :, None] >> np.arange(8, dtype=np.uint64)) & 1) << np.arange(SEG, dtype=np.uint64)[None, :, None]
    P = P.sum(axis=1)                                                               # [j][b]
    w = np.array([1 << b for b in range(7)] + [-(1 << 7)], dtype=np.int64)
    Rw = _rw(R).astype(np.uint64)
    acc = np.zeros((R.shape[0], n), dtype=np.int64)
    popc = np.vectorize(lambda x: bin(int(x)).count("1"))
    for ell in range(Rw.shape[0]):
        acc += (popc(Rw[ell][:, None, None] & P[None, :, :]) * w).sum(axis=2)
        assert np.array_equal(acc, T[ell])


def test_every_timed_kernel_is_pinned_and_folds_every_word() -> None:
    """expect.json pins each kernel count.py prices, and each pinned loop holds the duo fold of each of its words."""
    count = _load("qstar_count", POUW / "qstar" / "count.py")
    pins = json.loads((POUW / "qstar" / "expect.json").read_text())
    assert set(pins) == set(count.ARMS)
    for name, hist in pins.items():
        words = count.ARMS[name][0]
        assert all(hist.get(op, 0) >= n * words for op, n in count.FOLD.items()), name
