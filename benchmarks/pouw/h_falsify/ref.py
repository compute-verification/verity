"""The reference for H-FALSIFY: the zero instance's tiled transcript and the duo receipt every arm folds it into.

The zero instance is A* = B* = 0, so Ã = R ∈ {0,1}^{m×k} and B̃ = S with S[t, j] = U[j, t mod 32], U ∈ {±1..±64}^{n×32}.
The transcript is every 32-step prefix word

    T_ℓ[i, j] = Σ_{t < 32(ℓ+1)} Ã[i, t] B̃[t, j],   ℓ = 0 .. L-1,  L = k / 32,

exact in s32 (|T| ≤ 64 k < 2^31 here, and every arm's accumulator is s32). The honest arm computes the same words for
generic int8 Ã and B̃; this module takes any (Ã, B̃).

Route (a)'s identity: with P_ℓ[i, r] = Σ_{l ≤ ℓ} R[i, 32 l + r], the prefix column sums of R by residue,
T_ℓ = P_ℓ Uᵀ, because at B* = 0 every segment shares U. `transcript_via_prefix` computes it that way.

The receipt binds every word: a duo chain per (i, j) absorbs T_0[i, j], ..., T_{L-1}[i, j] in order with segment ℓ's
public keys, from a start derived from the chain's index, and the receipt is the m × n chains' `duo_final` words. The step
is `receipts/receipt_folds.py`'s `duo_word` (the CUDA arms use `receipt_folds.cuh`'s `rf_duo_word`).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SEG = 32
SEED = 0x48464C53  # "HFLS"
_RECEIPTS = Path(__file__).resolve().parents[1] / "receipts" / "receipt_folds.py"


def _folds():
    name = "receipt_folds"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _RECEIPTS)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


rf = _folds()


def keys(L: int) -> tuple[list[int], list[int], list[int]]:
    """Segment ℓ's duo keys (a odd under key_ok, k, rotation r), public and independent of the data."""
    a, kx, r = [], [], []
    for ell in range(L):
        i = 0
        while not rf.key_ok(c := rf.mix32(0xA0000000 ^ (ell << 8) ^ i) | 1):
            i += 1
        a.append(c)
        kx.append(rf.mix32(0xB0000000 + ell))
        r.append((ell * 7 + 5) % 31 + 1)
    return a, kx, r


def chain_start(m: int, n: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Chain (i, j)'s start: y0 = mix32(seed ^ mix32(i n + j)), z0 = mix32(y0 + GOLDEN), as m × n uint32."""
    idx = (np.arange(m, dtype=np.uint64)[:, None] * n + np.arange(n, dtype=np.uint64)[None, :]).astype(np.uint32)
    y0 = rf.mix32(np.uint32(seed) ^ rf.mix32(idx))
    return y0, rf.mix32(y0 + np.uint32(rf.GOLDEN))


def zero_instance(m: int, k: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """R (m × k, 0/1) and U (n × 32, ±1..±64), as int8."""
    g = np.random.default_rng(seed)
    R = g.integers(0, 2, size=(m, k), dtype=np.int8)
    U = (g.integers(1, 65, size=(n, SEG)) * g.choice(np.array([-1, 1]), size=(n, SEG))).astype(np.int8)
    return R, U


def generic(m: int, k: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The honest arm's generic int8 Ã (m × k) and B̃ᵀ (n × k)."""
    g = np.random.default_rng(seed)
    return (g.integers(-128, 128, size=(m, k), dtype=np.int8), g.integers(-128, 128, size=(n, k), dtype=np.int8))


def s_transposed(U: np.ndarray, k: int) -> np.ndarray:
    """B̃ᵀ = Sᵀ (n × k) of the zero instance: Sᵀ[j, t] = U[j, t mod 32]."""
    return np.tile(U, (1, k // SEG))


def transcript(At: np.ndarray, Bt: np.ndarray) -> np.ndarray:
    """T (L × m × n, int64): every 32-step prefix word of Ã B̃, with B̃ᵀ = Bt."""
    m, k = At.shape
    L = k // SEG
    a = At.astype(np.int64).reshape(m, L, SEG)
    b = Bt.astype(np.int64).reshape(Bt.shape[0], L, SEG)
    seg = np.einsum("ilr,jlr->lij", a, b)
    return np.cumsum(seg, axis=0)


def transcript_via_prefix(R: np.ndarray, U: np.ndarray) -> np.ndarray:
    """T_ℓ = P_ℓ Uᵀ with P_ℓ the prefix column sums of R by residue: route (a)'s identity at B* = 0."""
    m, k = R.shape
    P = np.cumsum(R.astype(np.int64).reshape(m, k // SEG, SEG), axis=1)
    return np.einsum("ilr,jr->lij", P, U.astype(np.int64))


def receipt(T: np.ndarray, seed: int = SEED) -> np.ndarray:
    """The m × n duo receipt of a transcript T (L × m × n)."""
    L, m, n = T.shape
    a, kx, r = keys(L)
    y, z = chain_start(m, n, seed)
    for ell in range(L):
        d = T[ell].astype(np.int64).astype(np.uint32)
        y, z = rf.duo_word(y, z, d, np.uint32(kx[ell]), np.uint32(a[ell]), r[ell])
    return rf.duo_final(y, z).astype(np.uint32)


def receipt_scalar(T: np.ndarray, i: int, j: int, seed: int = SEED) -> int:
    """Chain (i, j)'s receipt word by the stdlib step alone, to hold the vectorized fold to it."""
    L, m, n = T.shape
    a, kx, r = keys(L)
    y = rf.mix32(seed ^ rf.mix32(i * n + j))
    z = rf.mix32(y + rf.GOLDEN)
    for ell in range(L):
        y, z = rf.duo_word(y, z, int(T[ell, i, j]) & rf.M32, kx[ell], a[ell], r[ell])
    return rf.duo_final(y, z)
