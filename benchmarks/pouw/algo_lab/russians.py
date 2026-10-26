"""(d) Four Russians for the binary R, through shared memory, A = B = 0.

The segment's 32 steps are cut into 32 / g groups of g. Group gamma's table holds, for each g-bit pattern p and column j,
sum_e p_e S_(gamma g + e, j): built once, since S repeats every segment (2^g n 32 / g entries, one IADD each in Gray-code
order, and an STS each: phase ``setup``). A warp takes one row i and 32 consecutive columns, so a group's pattern is
warp-uniform (USHF and ULOP3 on the uniform datapath a group, row and segment) and the 32 lanes read 32 consecutive
entries of one table row: one conflict-free LDS a group a word, through [R + UR] addressing, then IADD3s add the
lookups and the previous word. Entries are 32-bit, or 16-bit (``entry=2``: |sum| <= 64 g fits int16 for g <= 8, and
LDS.S16 sign-extends).
"""
from __future__ import annotations

import numpy as np

from . import dither as D
from . import reference as REF
from .isa import Kernel, M32, U32


def tables(U: np.ndarray, g: int) -> np.ndarray:
    """(32 / g, 2^g, n) int64: group gamma, pattern p, column j."""
    pats = (np.arange(1 << g)[:, None] >> np.arange(g)[None, :]) & 1                  # (2^g, g)
    return np.stack([pats @ U[gm * g:(gm + 1) * g] for gm in range(D.L // g)])


def run(inst: D.Instance, g: int = 8, entry: int = 4, cpl: int = 1, fold: str = "li", seed: int = 1) -> REF.Run:
    """``cpl`` columns a lane: one LDS of cpl x entry bytes (LDS.64 or LDS.128) reads a group's entries for all of
    them, and the warp-uniform pattern serves 32 cpl columns."""
    assert inst.zero and D.L % g == 0 and cpl * entry <= 16
    k = Kernel()
    n = inst.n
    T = tables(inst.U, g)
    if entry == 2:
        assert np.abs(T).max() < 1 << 15
    with k.phase("setup"):
        k.add("IADD3", T.size)
        k.sts_bulk(T.size, T.size * entry)
    ng = D.L // g
    prev = np.zeros((n, n), dtype=U32)
    W = []
    weights = 1 << np.arange(g)
    for l in range(inst.ks):
        acc = [prev]
        for gm in range(ng):
            bits = inst.R[:, l * D.L + gm * g:l * D.L + (gm + 1) * g]
            p = bits @ weights                                                       # (n,) a row's pattern
            k.uni(2 * n * (n // (32 * cpl)))                                         # USHF + ULOP3 a row, warp
            look = T[gm][p[:, None], np.arange(n)[None, :]]
            k.lds_bulk(n * n // cpl, n * n * entry)
            acc.append((look & M32).astype(U32))
        while len(acc) > 1:
            a, b, c = (acc + [0, 0])[:3]
            acc = [k.iadd3(a, b, c)] + acc[3:]
        prev = acc[0]
        W.append(prev)
    W = np.stack(W)
    return REF.Run(W, REF.receipt(k, W, fold, seed), k)
