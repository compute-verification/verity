"""SERVE-H1's wrap: W' = B + c + U and X = A + b mod 2^8 at int8-exact's |A|, |B| <= 127 (h1tile, main).

Checks, on `segment.call_words` (the words H1 absorbs), that the wrap keeps what H-mu and the receipt-luck premises read:
1. wraps happen at |B| > 63 (and at A = 127, b = 1);
2. per entry, U -> W' is injective at every (B, c): the noise keeps its point mass;
3. flipping one fresh coin (c_k or b_k) moves the prefix word it enters by (+1 or -255) times the other operand's
   entry, mod 2^32: nonzero exactly when that entry is nonzero, as without the wrap;
4. the words agree mod 2^8 with the unwrapped integer words: the wrap only adds -256 X wb (+ the X side's).

Runs on main (7d291b14d): ``uv run python benchmarks/rate_redteam/h1_wrap_check.py``.
"""
import random

from verity.core.protocols.pouw.circuit import segment as SG

rng = random.Random(7)
KS = 2
sx8 = lambda v: v - 256 if v & 0x80 else v


def words(xr, b, wr, c, u, z):
    return SG.call_words(xr, b, wr, c, u, z)


def tile():
    xr = [[[rng.choice([127, -127, rng.randint(-127, 127)]) & 0xFF for _ in range(32)] for _ in range(4)] for _ in range(KS)]
    wr = [[[rng.choice([127, -127, 100, -100, rng.randint(-127, 127)]) & 0xFF for _ in range(32)] for _ in range(8)]
          for _ in range(KS)]
    b = [[[rng.randint(0, 1) for _ in range(32)] for _ in range(4)] for _ in range(KS)]
    c = [[[rng.randint(0, 1) for _ in range(32)] for _ in range(8)] for _ in range(KS)]
    u = [[rng.randint(-64, 64) & 0xFF for _ in range(32)] for _ in range(8)]
    m = SG.n_digits(KS)
    z = [[[rng.randint(0, 1) for _ in range(32)] for _ in range(4)] for _ in range(m - 1)]
    return xr, b, wr, c, u, z


wraps = inj = flips = flip_ok = low_ok = 0
for _ in range(40):
    xr, b, wr, c, u, z = tile()
    for s in range(KS):
        for k in range(8):
            for t in range(32):
                B, U = sx8(wr[s][k][t]), sx8(u[k][t])
                wraps += not -128 <= B + c[s][k][t] + U <= 127
                inj += len({(B + c[s][k][t] + uu) & 0xFF for uu in range(-64, 65)}) == 129
    base = words(xr, b, wr, c, u, z)
    for _ in range(20):
        s, k, t = rng.randrange(KS), rng.randrange(8), rng.randrange(32)
        c2 = [[row[:] for row in seg] for seg in c]
        c2[s][k][t] ^= 1
        w2 = words(xr, b, wr, c2, u, z)
        flips += 1
        ok = True
        for q in range(32):
            if SG.col_slot(q) != k:
                continue
            r = SG.row_slot(q)
            X = sx8((xr[s][r][t] + b[s][r][t]) & 0xFF)
            d = [(w2[j][q] - base[j][q]) & 0xFFFFFFFF for j in range(s, KS)]
            Wp = lambda cc: sx8((wr[s][k][t] + cc + u[k][t]) & 0xFF)
            want = ((Wp(c2[s][k][t]) - Wp(c[s][k][t])) * X) & 0xFFFFFFFF
            ok &= all(x == want for x in d) and ((want != 0) == (X != 0))
        flip_ok += ok
    for j in range(KS):
        for q in range(32):
            r, k = SG.row_slot(q), SG.col_slot(q)
            exact = sum((sx8(xr[s][r][t]) + b[s][r][t]) * (sx8(wr[s][k][t]) + c[s][k][t] + sx8(u[k][t]))
                        for s in range(j + 1) for t in range(32))
            low_ok += (exact - base[j][q]) % 256 == 0

print("entries that wrap (|B + c + U| > 127):", wraps)
print("entries where U -> W' is injective:", inj, "of", 40 * KS * 8 * 32)
print("c flips whose word moves by Δ·X (Δ in {1, -255}), nonzero iff X != 0:", flip_ok, "of", flips)
print("prefix words equal to the unwrapped words mod 2^8:", low_ok, "of", 40 * KS * 32)
