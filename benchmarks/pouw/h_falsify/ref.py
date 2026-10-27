"""The reference for h_falsify: the zero instance's tiled transcript and the duo receipt every arm folds it into.

The zero instance is A* = B* = 0, so Ã = R ∈ {0,1}^{m×k} and B̃ = S with S[t, j] = U[j, t mod 32], U ∈ {±1..±64}^{n×32}.
The transcript is every 32-step prefix word

    T_ℓ[i, j] = Σ_{t < 32(ℓ+1)} Ã[i, t] B̃[t, j],   ℓ = 0 .. L-1,  L = k / 32,

exact in s32 (|T| ≤ 64 k < 2^31 here, and every arm's accumulator is s32). The honest arm computes the same words for
generic int8 Ã and B̃; this module takes any (Ã, B̃).

Route (a)'s identity: with P_ℓ[i, r] = Σ_{l ≤ ℓ} R[i, 32 l + r], the prefix column sums of R by residue,
T_ℓ = P_ℓ Uᵀ, because at B* = 0 every segment shares U. `transcript_via_prefix` computes it that way.

The receipt absorbs every word: a duo chain per (i, j) absorbs T_0[i, j], ..., T_{L-1}[i, j] in order with segment ℓ's
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
        d = (T[ell].astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
        y, z = rf.duo_word(y, z, d, np.uint32(kx[ell]), np.uint32(a[ell]), r[ell])
    return rf.duo_final(y, z).astype(np.uint32)


# ---- the m16n8k32 s8 fragments, packed so a lane loads its whole fragment in one vector load ------------------------
#
# PTX ISA, mma.m16n8k32 with .s8: lane = 4 g + t (g = groupID, t = threadID_in_group). A (16 × 32, row) is four .b32,
# rows (g, g+8, g, g+8) and columns (4t, 4t, 16+4t, 16+4t), four bytes each, lowest address in the low byte. B (32 × 8,
# col) is two .b32: column g, rows 4t and 16+4t. C (16 × 8) is c0..c3 at rows (g, g, g+8, g+8), columns (2t, 2t+1, 2t,
# 2t+1). Both arms of a comparison use the same packed layout (a host reorder, outside the timing), so the layout favours
# neither; it gives each lane its fragment in one LDG.128 (A) or LDG.64 (B), as ldmatrix would.

_LANE = np.arange(32)
_G, _T = _LANE // 4, _LANE % 4


def pack_a(At: np.ndarray) -> np.ndarray:
    """A fragments, uint32 (m/16, L, 32 lanes, 4 regs)."""
    m, k = At.shape
    X = np.ascontiguousarray(At).view(np.uint8).reshape(m // 16, 16, k // SEG, SEG).transpose(0, 2, 1, 3)
    rows = np.stack([_G, _G + 8, _G, _G + 8], axis=1)                       # (32, 4)
    cols = np.stack([4 * _T, 4 * _T, 16 + 4 * _T, 16 + 4 * _T], axis=1)     # (32, 4)
    by = X[:, :, rows[:, :, None], cols[:, :, None] + np.arange(4)]          # (MB, L, 32, 4 regs, 4 bytes)
    return np.ascontiguousarray(by).view(np.uint32)[..., 0]


def pack_b(Bt: np.ndarray) -> np.ndarray:
    """B fragments from B̃ᵀ, uint32 (n/8, L, 32 lanes, 2 regs)."""
    n, k = Bt.shape
    Y = np.ascontiguousarray(Bt).view(np.uint8).reshape(n // 8, 8, k // SEG, SEG).transpose(0, 2, 1, 3)
    cols = np.stack([4 * _T, 16 + 4 * _T], axis=1)                            # (32, 2)
    by = Y[:, :, _G[:, None, None], cols[:, :, None] + np.arange(4)]         # (NB, L, 32, 2 regs, 4 bytes)
    return np.ascontiguousarray(by).view(np.uint32)[..., 0]


def pack_u(U: np.ndarray) -> np.ndarray:
    """The zero instance's B fragments, the same for every segment: uint32 (n/8, 32 lanes, 2 regs)."""
    return pack_b(U)[:, 0]


def mma_emulate(af: np.ndarray, bf: np.ndarray) -> np.ndarray:
    """One m16n8k32 s8 MMA from packed fragments (32 × 4 and 32 × 2 uint32), as each lane's c0..c3 (32 × 4 int64)."""
    ab = af.astype(np.uint32).view(np.uint8).reshape(32, 4, 4).view(np.int8).astype(np.int64)
    bb = bf.astype(np.uint32).view(np.uint8).reshape(32, 2, 4).view(np.int8).astype(np.int64)
    A, B = np.zeros((16, SEG), np.int64), np.zeros((SEG, 8), np.int64)
    for lane in range(32):
        g, t = lane // 4, lane % 4
        A[g, 4 * t:4 * t + 4], A[g + 8, 4 * t:4 * t + 4] = ab[lane, 0], ab[lane, 1]
        A[g, 16 + 4 * t:20 + 4 * t], A[g + 8, 16 + 4 * t:20 + 4 * t] = ab[lane, 2], ab[lane, 3]
        B[4 * t:4 * t + 4, g], B[16 + 4 * t:20 + 4 * t, g] = bb[lane, 0], bb[lane, 1]
    C = A @ B
    return np.array([[C[g, 2 * t], C[g, 2 * t + 1], C[g + 8, 2 * t], C[g + 8, 2 * t + 1]]
                     for g, t in zip(_G, _T)])


def chain_words(At: np.ndarray, Bt: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """T_ℓ[i, j] for the chains `pairs` (P × 2) alone, O(k) a chain: L × P int64."""
    i, j = pairs[:, 0], pairs[:, 1]
    L = At.shape[1] // SEG
    a = At[i].astype(np.int64).reshape(len(i), L, SEG)
    b = Bt[j].astype(np.int64).reshape(len(j), L, SEG)
    return np.cumsum(np.einsum("plr,plr->pl", a, b), axis=1).T


def receipt_at(At: np.ndarray, Bt: np.ndarray, pairs: np.ndarray, n: int, seed: int = SEED) -> np.ndarray:
    """The receipt words of the chains `pairs`, without the whole transcript: the gate's sample at the timed shapes."""
    W = chain_words(At, Bt, pairs)
    a, kx, r = keys(W.shape[0])
    idx = (pairs[:, 0].astype(np.uint64) * n + pairs[:, 1].astype(np.uint64)).astype(np.uint32)
    y = rf.mix32(np.uint32(seed) ^ rf.mix32(idx))
    z = rf.mix32(y + np.uint32(rf.GOLDEN))
    for ell in range(W.shape[0]):
        y, z = rf.duo_word(y, z, (W[ell] & 0xFFFFFFFF).astype(np.uint32), np.uint32(kx[ell]), np.uint32(a[ell]),
                           r[ell])
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
