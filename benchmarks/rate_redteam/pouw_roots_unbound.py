"""T4 witness: if `pouw.roots` is not the root the H1 tile's `a` opens under, A can be chosen after the noise.

The prover fixes any roots r0, derives the tile's noise from (salt, position, r0), sets A = -b (its dither bits) and
serves that A. Then X = A + b = 0 and H1's prefix sets T_0 .. T_(KS-1) are zero at every B: the coin x coin blocks the
floor credits (b . c, b . U) are never computed, and the whole receipt is the same at every B. With `pouw.roots` the root
the tile's `a` opens under, this needs A = -b(seed(salt, position, root(A), root(B))), a fixed point of SHA-512.
Runs on main (7d291b14d): ``uv run python benchmarks/rate_redteam/pouw_roots_unbound.py``.
"""
import random

from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import seeds as SD
from verity.core.protocols.pouw.circuit import segment as SG

R, C, K = 4, 8, 64
salt, pos, r0 = bytes(range(64)), 7, (b"\x00" * 64, b"\x00" * 64)
lay = H.Layout(R, C, K)
rows, cols = H.noise(SD.seed(salt, pos, *r0), lay)
a = [[(-(bit & 1)) & 0xFF for bit in rows[i][0]] for i in range(R)]


def prefix_sets(b):
    seg = [range(32 * s, 32 * s + 32) for s in range(lay.ks)]
    Rr, Cs = lay.rows(0), lay.cols(0)
    xr = [[[a[i][t] for t in ts] for i in Rr] for ts in seg]
    bb = [[[rows[i][0][t] for t in ts] for i in Rr] for ts in seg]
    wr = [[[b[j][t] for t in ts] for j in Cs] for ts in seg]
    cc = [[[cols[j][0][t] for t in ts] for j in Cs] for ts in seg]
    words = SG.call_words(xr, bb, wr, cc, [cols[j][1] for j in Cs],
                          [[rows[i][1][d] for i in Rr] for d in range(lay.m - 1)])
    return words[:lay.ks]


rng = random.Random(1)
B1 = [[rng.randrange(256) for _ in range(K)] for _ in range(C)]
B2 = [[rng.randrange(256) for _ in range(K)] for _ in range(C)]
t1, t2 = prefix_sets(B1), prefix_sets(B2)
print("X = A + b is zero:", all(((a[i][t] + (rows[i][0][t] & 1)) & 0xFF) == 0 for i in range(R) for t in range(K)))
print("prefix sets T_s all zero at B1:", all(v == 0 for s in t1 for v in s))
print("prefix sets equal at B1 and B2:", t1 == t2)
print("receipt at B1 == receipt at B2 :", H.reference(a, B1, salt, pos, r0)[1] == H.reference(a, B2, salt, pos, r0)[1])
