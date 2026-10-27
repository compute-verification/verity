"""All schedules: the least tensor-core cost of an exact product of 2^l x 2^l blocks, over every recursion tree.

A node multiplies two block matrices whose entries range over V_A and V_B consecutive integers, at depth budget l. It is
  * an atom: one MMA format holding both operands exactly (cost f.cost per MAC),
  * a split of one operand into two digit matrices, x - c = x_0 + d x_1 with x_0 in d values and x_1 in ceil(V/d)
    (each part its own subtree, so the two parts may be sliced differently against the other operand), or
  * a scheme (l >= 1): Strassen, Strassen-Winograd, or the trivial block split, each child at depth l - 1 on the
    +-1 combination its coefficients make, which ranges over (V - 1) N + 1 values (N the coefficient vector's l1 norm).
W_l(V_A, V_B) = 8^(l+1) x (cost per MAC) is an integer. Naive tiled is the atom s8 at (256, 256): W_l = 8^(l+1).

The table lives on a grid of range values. lo[i, j] is a proved lower bound on W_l at every (V_A, V_B) that rounds down
to (g[i], g[j]) (W_l is monotone); up[i, j] is the cost of an explicit schedule at (g[i], g[j]). noscheme marks
schedules that apply no scheme; has[i, j] is the least cost of a schedule that applies at least one non-trivial scheme.
"""
from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np

from .costs import Costs, inst

STRASSEN = [(2, 2), (2, 1), (1, 2), (1, 2), (2, 1), (2, 2), (2, 2)]
WINOGRAD = [(1, 1), (1, 1), (4, 1), (1, 4), (2, 2), (3, 3), (2, 2)]
TRIVIAL = [(1, 1)] * 8
BIG = np.int64(2 ** 58)


def _kernel():
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits.c")
    with open(src, "rb") as fh:
        tag = hashlib.sha256(fh.read()).hexdigest()[:16]
    so = os.path.join(tempfile.gettempdir(), f"strassen-splits-{tag}.so")
    if not os.path.exists(so):
        tmp = so + f".{os.getpid()}"
        subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", tmp, src], check=True)
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    p = ctypes.POINTER(ctypes.c_int64)
    lib.splits.argtypes = [ctypes.c_int, p, p, p, p, ctypes.c_int64]
    return lib.splits


def _splits(T, H, rest, g):
    for a in (T, H, rest, g):
        assert a.dtype == np.int64 and a.flags["C_CONTIGUOUS"]
    p = ctypes.POINTER(ctypes.c_int64)
    _splits.fn = getattr(_splits, "fn", None) or _kernel()
    _splits.fn(len(g), T.ctypes.data_as(p), H.ctypes.data_as(p), rest.ctypes.data_as(p), g.ctypes.data_as(p), int(BIG))


def grid(Vmax: int, extra=()) -> np.ndarray:
    pts = set(range(1, 65)) | {256, 257, 513, 1025, 2049, 4097} | set(extra)
    x = 0
    while True:
        v = int(np.ceil(2 ** (x / 16)))
        if v > Vmax:
            break
        pts.add(v)
        x += 1
    for x in range(64 * 11, 64 * 12 + 1):  # 1/64-bit resolution where FP16's coupling M_A M_B <= L binds
        pts.add(int(np.ceil(2 ** (x / 64))))
    n = 1
    while 255 * n + 1 <= Vmax:
        m = n
        while 255 * m + 1 <= Vmax:
            pts.add(255 * m + 1)
            m *= 3
        n *= 2
    return np.array(sorted(p for p in pts if p <= Vmax), dtype=np.int64)


@functools.lru_cache(maxsize=None)
def _costs(name: str) -> Costs:
    return Costs(inst(name))


class Tables:
    def __init__(self, name: str, lmax: int, schemes, clamp: int | None = None, extra=()):
        """clamp: the product mod clamp (a CRT residue), every combination reduced, so ranges never exceed clamp.
        extra: further grid points (operand ranges to read exactly)."""
        self.C = _costs(name)
        self.schemes = schemes
        self.clamp = clamp
        Vmax = clamp if clamp else 255 * 4 ** lmax + 1
        self.g = g = grid(Vmax, extra=tuple(self.C.closure) + ((clamp,) if clamp else ()) + tuple(extra))
        G = len(g)
        self.down = lambda v: np.searchsorted(g, v, side="right") - 1
        self.upi = lambda v: np.minimum(np.searchsorted(g, v, side="left"), G - 1)
        self.over = lambda v: v > g[-1]
        atom = np.full((G, G), BIG, dtype=np.int64)
        dmax = self.C.dmax
        small = [i for i in range(G) if g[i] <= dmax]
        for i in small:
            for j in small:
                c = self.C.c(int(g[i]), int(g[j]))
                if c < 10 ** 8:
                    atom[i, j] = int(c * 8)
        atom[0, :] = 0
        atom[:, 0] = 0
        self.atom = atom
        # digit split of g[i] at digit index e: the rest's range, rounded for each table
        self.rest_lo = np.zeros((G, G), dtype=np.int64)
        self.rest_up = np.zeros((G, G), dtype=np.int64)
        hi = np.append(g[1:] - 1, g[-1])
        for i in range(G):
            es = np.arange(i)
            ok = g[es] >= 2
            self.rest_lo[i, es[ok]] = self.down(-(-g[i] // hi[es[ok]]))
            self.rest_up[i, es[ok]] = self.upi(-(-g[i] // g[es[ok]]))

    def child(self, N: int, rnd):
        v = (self.g - 1) * N + 1
        if self.clamp:
            v = np.minimum(v, self.clamp)
        idx = rnd(v)
        return idx, self.over(v)

    def level(self, prev_lo, prev_up, prev_has_lo, prev_has_up, l, full=False):
        """W_l's tables from W_{l-1}'s (prev_* None at l = 0). full: every leaf at depth l (no atom above depth 0, no
        trivial split), so every root-to-leaf path applies l non-trivial schemes."""
        g, G = self.g, len(self.g)
        lo = self.atom.copy()
        up = self.atom.copy()
        has_lo = np.full((G, G), BIG, dtype=np.int64)
        has_up = np.full((G, G), BIG, dtype=np.int64)
        if l > 0:
            lo *= 8 ** l
            up *= 8 ** l
            lo[self.atom >= BIG] = BIG
            up[self.atom >= BIG] = BIG
            if full:
                lo[:], up[:] = BIG, BIG
                lo[0, :] = lo[:, 0] = up[0, :] = up[:, 0] = 0
            for sch in self.schemes:
                trivial = sch is TRIVIAL
                if full and trivial:
                    continue
                s_lo = np.zeros((G, G), dtype=np.int64)
                s_up = np.zeros((G, G), dtype=np.int64)
                kids = []
                for na, nb in sch:
                    ia, _ = self.child(na, self.down)
                    ib, _ = self.child(nb, self.down)
                    ja, pa = self.child(na, self.upi)
                    jb, pb = self.child(nb, self.upi)
                    klo = prev_lo[np.ix_(ia, ib)]
                    kup = prev_up[np.ix_(ja, jb)].copy()
                    kup[np.ix_(pa, np.ones(G, bool))] = BIG
                    kup[np.ix_(np.ones(G, bool), pb)] = BIG
                    khlo = prev_has_lo[np.ix_(ia, ib)]
                    khup = prev_has_up[np.ix_(ja, jb)].copy()
                    khup[np.ix_(pa, np.ones(G, bool))] = BIG
                    khup[np.ix_(np.ones(G, bool), pb)] = BIG
                    kids.append((klo, kup, khlo, khup))
                    s_lo = np.minimum(s_lo + klo, BIG)
                    s_up = np.minimum(s_up + kup, BIG)
                lo = np.minimum(lo, s_lo)
                up = np.minimum(up, s_up)
                if not trivial:
                    has_lo = np.minimum(has_lo, s_lo)
                    has_up = np.minimum(has_up, s_up)
                else:
                    for klo, kup, khlo, khup in kids[:1]:
                        # the trivial split's children are equal; one child with a scheme, the others its best
                        has_lo = np.minimum(has_lo, np.minimum(s_lo - klo + khlo, BIG))
                        has_up = np.minimum(has_up, np.minimum(s_up - kup + khup, BIG))
        # splits, in increasing i then j (each reads only finished entries): splits.c
        for T, H, rest in ((lo, has_lo, self.rest_lo), (up, has_up, self.rest_up)):
            _splits(T, H, rest, g)
        return lo, up, has_lo, has_up


def at(T, tab, VA, VB):
    i, j = int(T.down(VA)), int(T.down(VB))
    assert T.g[i] == VA and T.g[j] == VB
    return int(tab[i, j])


def run(name: str, lmax: int):
    """Print: the per-product cost (depth 0) at operands of 8 + i and 8 + j bits; uniform-depth Strassen and Winograd
    with those leaves; and, at each depth budget l, the least cost of any schedule, of one with a scheme, and of one
    applying l schemes on every path."""
    from .depth import profile

    T = Tables(name, lmax, [STRASSEN, WINOGRAD, TRIVIAL])
    g = T.g
    print(f"{name}: grid {len(g)} points up to {g[-1]}", flush=True)
    adaptive = T.level(None, None, None, None, 0)
    full = adaptive
    lo0, up0 = adaptive[0], adaptive[1]
    print("per-product cost x 8, operand ranges 255*2^i+1 by 255*2^j+1 (lower/upper):")
    for i in range(13):
        row = []
        for j in range(13):
            a, b = at(T, lo0, 255 * 2 ** i + 1, 255 * 2 ** j + 1), at(T, up0, 255 * 2 ** i + 1, 255 * 2 ** j + 1)
            row.append(f"{a}" if a == b else f"{a}/{b}")
        print(f"  {i:2d} " + " ".join(f"{x:>7}" for x in row))
    for sname, sch in (("strassen", STRASSEN), ("winograd", WINOGRAD)):
        for r in range(1, lmax + 1):
            kl = ku = 0
            for (na, nb), cnt in profile(sch, r).items():
                kl += cnt * at(T, lo0, 255 * na + 1, 255 * nb + 1)
                ku += cnt * at(T, up0, 255 * na + 1, 255 * nb + 1)
            d = 8 * 8 ** r
            print(f"{name} uniform {sname} r={r:2d}: [{kl / d:.6f}, {ku / d:.6f}]" + ("" if kl == ku else " GAP"),
                  flush=True)
    for l in range(1, lmax + 1):
        adaptive = T.level(*adaptive, l)
        full = T.level(*full, l, full=True)
        n = 8 ** (l + 1)
        lo, up, hlo, hup = adaptive
        flo, fup = full[0], full[1]
        print(f"{name} l={l:2d} any: [{at(T, lo, 256, 256) / n:.6f}, {at(T, up, 256, 256) / n:.6f}]"
              f"  with a scheme, minus naive, in 8^-(l+1) s8 MACs per block MAC: "
              f"[{at(T, hlo, 256, 256) - n}, {at(T, hup, 256, 256) - n}]"
              f"  depth {l} on every path: [{at(T, flo, 256, 256) / n:.6f}, {at(T, fup, 256, 256) / n:.6f}]",
              flush=True)


def narrow(name: str, lmax: int, pairs=((2, 128), (2, 129), (128, 255), (256, 256))):
    """Operands narrower than int8 (Dither's chosen data): at each depth budget, the least cost of any schedule and of
    one with a scheme, at each pair of operand ranges, against naive int8 (8^(l+1))."""
    T = Tables(name, lmax, [STRASSEN, WINOGRAD, TRIVIAL],
               extra=[(v - 1) * k + 1 for p in pairs for v in p for k in range(1, 33)]
               + [2 ** a + 1 for a in range(1, 13)] + [2 ** a for a in range(13)])
    tabs = T.level(None, None, None, None, 0)
    for l in range(lmax + 1):
        if l:
            tabs = T.level(*tabs, l)
        lo, up, hlo, hup = tabs
        n = 8 ** (l + 1)
        cells = []
        for VA, VB in pairs:
            a = (at(T, lo, VA, VB), at(T, up, VA, VB), at(T, hlo, VA, VB), at(T, hup, VA, VB))
            cells.append(f"({VA},{VB}) any [{a[0] / n:.6f}, {a[1] / n:.6f}] scheme [{a[2] / n:.6f}, {a[3] / n:.6f}]"
                         f" minus naive [{a[2] - n}, {a[3] - n}]")
        print(f"{name} l={l:2d}: " + "; ".join(cells), flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "narrow":
    if len(sys.argv) > 4:
        narrow(sys.argv[2], int(sys.argv[3]),
               tuple(tuple(int(v) for v in p.split(":")) for p in sys.argv[4].split(",")))
    else:
        narrow(sys.argv[2], int(sys.argv[3]))
    sys.exit(0)

if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
