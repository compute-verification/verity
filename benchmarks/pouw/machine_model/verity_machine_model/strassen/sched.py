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

import sys

import numpy as np

from .costs import Costs, inst

STRASSEN = [(2, 2), (2, 1), (1, 2), (1, 2), (2, 1), (2, 2), (2, 2)]
WINOGRAD = [(1, 1), (1, 1), (4, 1), (1, 4), (2, 2), (3, 3), (2, 2)]
TRIVIAL = [(1, 1)] * 8
BIG = np.int64(2 ** 58)


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


class Tables:
    def __init__(self, name: str, lmax: int, schemes):
        self.C = Costs(inst(name))
        self.schemes = schemes
        Vmax = 255 * 4 ** lmax + 1
        self.g = g = grid(Vmax, extra=self.C.closure)
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
        idx = rnd(v)
        return idx, self.over(v)

    def level(self, prev_lo, prev_up, prev_has_lo, prev_has_up, l):
        """W_l's tables from W_{l-1}'s (prev_* None at l = 0)."""
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
            for sch in self.schemes:
                trivial = sch is TRIVIAL
                s_lo = np.zeros((G, G), dtype=np.int64)
                s_up = np.zeros((G, G), dtype=np.int64)
                h_lo = np.full((G, G), BIG, dtype=np.int64)  # the best with one child holding a non-trivial scheme
                h_up = np.full((G, G), BIG, dtype=np.int64)
                kids = []
                for na, nb in sch:
                    ia, oa = self.child(na, self.down)
                    ib, ob = self.child(nb, self.down)
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
        # splits, in increasing i then j: A-splits of row i vectorised over j, B-splits sequential in j
        for T, H, rest in ((lo, has_lo, self.rest_lo), (up, has_up, self.rest_up)):
            for i in range(G):
                es = np.nonzero((g[:i] >= 2))[0]
                if len(es):
                    r = rest[i, es]
                    a = T[es, :] + T[r, :]
                    T[i, :] = np.minimum(T[i, :], np.minimum(a.min(axis=0), BIG))
                    ha = np.minimum(H[es, :] + T[r, :], T[es, :] + H[r, :])
                    H[i, :] = np.minimum(H[i, :], np.minimum(ha.min(axis=0), BIG))
                for j in range(G):
                    fs = np.nonzero(g[:j] >= 2)[0]
                    if not len(fs):
                        continue
                    r = rest[j, fs]
                    b = T[i, fs] + T[i, r]
                    T[i, j] = min(T[i, j], min(b.min(), BIG))
                    hb = np.minimum(H[i, fs] + T[i, r], T[i, fs] + H[i, r])
                    H[i, j] = min(H[i, j], min(hb.min(), BIG))
        return lo, up, has_lo, has_up


def run(name: str, lmax: int, schemes, at=((256, 256),)):
    T = Tables(name, lmax, schemes)
    g = T.g
    print(f"{name}: grid {len(g)} points up to {g[-1]}", flush=True)
    prev = (None, None, None, None)
    out = []
    for l in range(lmax + 1):
        prev = T.level(*prev, l)
        lo, up, hlo, hup = prev
        for VA, VB in at:
            i, j = int(T.down(VA)), int(T.down(VB))
            assert g[i] == VA and g[j] == VB
            n = 8 ** (l + 1)
            print(f"{name} l={l:2d} ({VA},{VB}) any: [{lo[i, j] / n:.6f}, {up[i, j] / n:.6f}]"
                  f"  with a scheme: [{hlo[i, j] / n:.6f}, {hup[i, j] / n:.6f}]", flush=True)
        out.append(prev)
    return T, out


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), [STRASSEN, WINOGRAD, TRIVIAL])
