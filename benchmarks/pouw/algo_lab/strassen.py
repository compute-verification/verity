"""(a) Strassen on Dither's segment products, one or two levels, counted.

Each segment's product At[:, seg] Bt[seg, :] (k = L = 32) is split 2^d ways in m, k and n, Strassen's 7^d products are
each accumulated across segments in their own accumulator (mod 2^32), and each segment's words are the C blocks formed
from them. The k split must stay inside the segment, since the words are prefix sums at every L-step boundary: so a
leaf product has k = 32 / 2^d, which no wgmma takes (s8's wgmma is k32). It runs as mma.sync m16n8k16 (`MMA.S8.K16`,
the same tensor rate on DS-90, but one issue slot a 2,048 MACs), and at depth 2 (k = 8) padded to k16.

P1 at each leaf: an operand inside s8 goes straight in. One with rare entries outside (A = B = 0: U's combinations
reach +128 when two 64s meet) is clipped to s8 and corrected sparsely on the lanes (an IADD3 or IMAD a row or column
of each outside entry). A dense wide operand (general A and B) is split into s8 slices, v = 128 hi + lo, each slice
product in its own accumulator and recombined by LEA.

Counted beside the MMAs: the operands' combinations (phase ``setup``, n k work: R's per segment, and U's once when B = 0
since S repeats every segment), the operand loads (ldmatrix at a 64 x 64 warp tile's bytes a MAC, phase ``word``), the
words' assembly from the products (IADD3 with its negation flags, LEA for a shifted slice), and the receipt (the
reference's). ``delivery="smem"`` adds P2's cost when the receipt's lane of a C word isn't the thread holding its
products: an STS and an LDS a word for the three of each four quadrant words held elsewhere.
"""
from __future__ import annotations

import numpy as np

from . import dither as D
from . import reference as REF
from .isa import Kernel, M32, U32

SA = np.array([[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]])
SB = np.array([[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]])
SC = np.array([[1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, 1, 0]])
SPARSE = 1 / 64
WARP_TILE = 64


def blocks(M: np.ndarray) -> list[np.ndarray]:
    r, c = M.shape[0] // 2, M.shape[1] // 2
    return [M[:r, :c], M[:r, c:], M[r:, :c], M[r:, c:]]


def unblocks(b: list[np.ndarray]) -> np.ndarray:
    return np.block([[b[0], b[1]], [b[2], b[3]]])


def combine(k: Kernel, coef, parts: list[np.ndarray], phase: str) -> np.ndarray:
    """sum coef_i parts_i, counting (terms - 1) packed-int8 adds (4 entries a lane instruction) in ``phase``."""
    nz = [(int(c), p) for c, p in zip(coef, parts) if c]
    with k.phase(phase):
        k.add("IADD3", (len(nz) - 1) * nz[0][1].size // 4)
    return sum(c * p for c, p in nz)


def in8(x: np.ndarray) -> bool:
    return x.size == 0 or (x.min() >= -128 and x.max() <= 127)


def slices(x: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """x as s8 pieces with their shifts: x itself, or lo + 128 hi with lo in [0, 127] (hi must be s8)."""
    if in8(x):
        return [(x, 0)]
    lo, hi = x & 127, x >> 7
    if not in8(hi):
        raise ValueError("operand past 15 bits")
    return [(lo, 0), (hi, 7)]


class Leaf:
    """One Strassen product's accumulators across segments: a slice pair each, and its sparse corrections."""

    def __init__(self) -> None:
        self.acc: dict[int, np.ndarray] = {}

    def add(self, k: Kernel, X: np.ndarray, Y: np.ndarray) -> None:
        kk = X.shape[1]
        corr = np.zeros((X.shape[0], Y.shape[1]), dtype=np.int64)
        if not in8(Y) and np.mean((Y < -128) | (Y > 127)) <= SPARSE:
            Yc = np.clip(Y, -128, 127)
            for t, j in zip(*np.nonzero(Y - Yc)):
                corr[:, j] += X[:, t] * (Y[t, j] - Yc[t, j])
                k.add("IADD3" if abs(Y[t, j] - Yc[t, j]) == 1 else "IMAD", X.shape[0])
            Y = Yc
        if not in8(X) and np.mean((X < -128) | (X > 127)) <= SPARSE:
            Xc = np.clip(X, -128, 127)
            for i, t in zip(*np.nonzero(X - Xc)):
                corr[i, :] += (X[i, t] - Xc[i, t]) * Y[t, :]
                k.add("IADD3" if abs(X[i, t] - Xc[i, t]) == 1 else "IMAD", Y.shape[1])
            X = Xc
        for xa, sa in slices(X):
            for yb, sb in slices(Y):
                s = sa + sb
                prev = self.acc.get(s, np.zeros(corr.shape, dtype=U32))
                self.acc[s] = k.mma_s8(xa, yb, prev, cls="MMA.S8.K16", k_instr=16)
                nbytes = xa.shape[0] * (-(-kk // 16) * 16) * yb.shape[1] * 2 // WARP_TILE
                k.lds_bulk(nbytes // 16, nbytes)      # ldmatrix.x4: 512 bytes a warp instruction
        if corr.any():
            self.acc[0] = (self.acc[0] + (corr & M32).astype(U32)).astype(U32)

    def value(self, k: Kernel) -> np.ndarray:
        """The product's running value: its slices recombined (a LEA a shifted group, IADD3 within a group)."""
        out = self.acc[0]
        for s in sorted(x for x in self.acc if x):
            out = k.lea(self.acc[s], out, s)
        return out


def _assemble(k: Kernel, vals: list[np.ndarray]) -> list[np.ndarray]:
    """The four C blocks from the 7 product values: IADD3 (three inputs, negation flags) a word, ceil((terms - 1) / 2)."""
    out = []
    for c in range(4):
        terms = [(int(SC[c, p]), vals[p]) for p in range(7) if SC[c, p]]
        k.add("IADD3", -(-(len(terms) - 1) // 2) * terms[0][1].size)
        out.append(sum(((v.astype(np.int64) * s) for s, v in terms)) & M32)
    return [x.astype(U32) for x in out]


class Strassen:
    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.leaves = [Strassen(depth - 1) if depth > 1 else Leaf() for _ in range(7)]

    def add(self, k: Kernel, X: np.ndarray, Y: np.ndarray, ycache: dict | None, key=()) -> None:
        a, b = blocks(X), blocks(Y)
        for p in range(7):
            x = combine(k, SA[p], a, "setup")
            if ycache is not None and key + (p,) in ycache:
                y = ycache[key + (p,)]
            else:
                y = combine(k, SB[p], b, "setup")
                if ycache is not None:
                    ycache[key + (p,)] = y
            leaf = self.leaves[p]
            if isinstance(leaf, Leaf):
                leaf.add(k, x, y)
            else:
                leaf.add(k, x, y, ycache, key + (p,))

    def value(self, k: Kernel) -> np.ndarray:
        return unblocks(_assemble(k, [leaf.value(k) for leaf in self.leaves]))


def run(inst: D.Instance, depth: int, fold: str = "li", seed: int = 1, delivery: str = "registers") -> REF.Run:
    k = Kernel()
    REF.setup_operands(k, inst)
    At, Bt = inst.At, inst.Bt
    tree = Strassen(depth)
    ycache: dict | None = {} if inst.zero else None
    W = []
    for l in range(inst.ks):
        tree.add(k, At[:, l * D.L:(l + 1) * D.L], Bt[l * D.L:(l + 1) * D.L], ycache)
        W.append(tree.value(k))
    W = np.stack(W)
    if delivery == "smem":
        moved = W.size * 3 // 4
        k.sts_bulk(moved, moved * 4)
        k.lds_bulk(moved, moved * 4)
    return REF.Run(W, REF.receipt(k, W, fold, seed), k)
