"""Dither's instance, its words and its receipts, with `segment.py` as the reference.

At = A + R and Bt = B + S as int8 encodings (wrapping as `segment.seg_check3` does), R in {0, 1}^(n x k), and
S_tj = U_(t mod L, j) with U in [-64, 64] (H1's keyed word, `h1tile.PouwU129`). A and B are any int8 matrices; A = B = 0 is
admissible. A word is T^(l)_ij = sum_{t < L (l + 1)} At_it Bt_tj mod 2^32 (`segment.prefixes`).

Lanes follow the H200 reference kernel: a wgmma m64n256k32 gives each of a warpgroup's 128 threads 128 accumulators, so
a lane holds `WPL` = 128 words a segment, absorbed as four `segment.absorb` sets of 32, and after each segment the warp's
checkpoint (`segment.warp_checkpoint`: the XOR over lanes and chains of each part of the state) is one receipt word a
part.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from verity.core.protocols.pouw.circuit import segment as SG

L = 32
M32 = 0xFFFFFFFF
U_LO, U_HI = -64, 64
WPL = 128
SET = 32


def sx8(x) -> np.ndarray:
    return ((np.asarray(x, dtype=np.int64) + 128) & 0xFF) - 128


@dataclass(frozen=True)
class Instance:
    R: np.ndarray
    U: np.ndarray
    A: np.ndarray
    B: np.ndarray

    @property
    def n(self) -> int:
        return self.R.shape[0]

    @property
    def k(self) -> int:
        return self.R.shape[1]

    @property
    def ks(self) -> int:
        return self.k // L

    @property
    def S(self) -> np.ndarray:
        return self.U[np.arange(self.k) % L]

    @property
    def At(self) -> np.ndarray:
        return sx8(self.A + self.R)

    @property
    def Bt(self) -> np.ndarray:
        return sx8(self.B + self.S)

    @property
    def zero(self) -> bool:
        return not self.A.any() and not self.B.any()


def instance(n: int, k: int, seed: int, general: bool = False) -> Instance:
    """n x k by k x n; ``general``: A and B uniform over int8, else A = B = 0."""
    if k % L or (n * n) % (WPL * 32):
        raise ValueError(f"k a multiple of {L} and n^2 of {WPL * 32}")
    rng = np.random.default_rng(seed)
    R = rng.integers(0, 2, (n, k), dtype=np.int64)
    U = rng.integers(U_LO, U_HI + 1, (L, n), dtype=np.int64)
    if general:
        A = rng.integers(-128, 128, (n, k), dtype=np.int64)
        B = rng.integers(-128, 128, (k, n), dtype=np.int64)
    else:
        A = np.zeros((n, k), dtype=np.int64)
        B = np.zeros((k, n), dtype=np.int64)
    return Instance(R, U, A, B)


def words(inst: Instance) -> np.ndarray:
    """T^(0) .. T^(ks - 1), each n x n, uint32: the reference transcript (exact int64 products, then mod 2^32)."""
    At, Bt = inst.At, inst.Bt
    seg = np.stack([At[:, l * L:(l + 1) * L] @ Bt[l * L:(l + 1) * L] for l in range(inst.ks)])
    return (np.cumsum(seg, axis=0) & M32).astype(np.uint32)


def lane_sets(W: np.ndarray) -> np.ndarray:
    """A segment's words as (lanes, WPL): lane g holds the n^2 / WPL words g WPL .. g WPL + WPL - 1 (row-major)."""
    return W.reshape(-1, WPL)


def start(fold: str, lanes: int, seed: int) -> list[list[int]]:
    """Each lane's fold state at the window's start, `segment.fold_keys` (li: y; duo: y then z)."""
    return [SG.fold_keys(fold, seed, 0, g, 0, 1) for g in range(lanes)]


def receipt(W: np.ndarray, fold: str, seed: int) -> np.ndarray:
    """The receipt by `segment.py`: per segment and warp, `warp_checkpoint` of the lanes' states after the segment's
    WPL / SET absorbs. Shape (ks, warps, parts), uint32."""
    lanes = W[0].size // WPL
    states = start(fold, lanes, seed)
    out = []
    for Wl in W:
        sets = lane_sets(Wl)
        for g in range(lanes):
            for s in range(WPL // SET):
                states[g] = SG.absorb(fold, states[g], [int(v) for v in sets[g, s * SET:(s + 1) * SET]])
        out.append([SG.warp_checkpoint(states[w * 32:(w + 1) * 32]) for w in range(lanes // 32)])
    return np.array(out, dtype=np.uint32)
