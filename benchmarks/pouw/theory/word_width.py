"""ACCEL ALGO-COST item 6: is an int8 product mod 2^8 or mod 2^16 cheaper than the full one, on v1/v2's operands?

Mod 2^8: wrap8(F) = F − 256·F_hi ≡ F (mod 256) for every form F, so a bilinear scheme's summed forms go into the s8 MMA
wrapped, with no overflow correction, and each word comes out right mod 2^8. Mod 2^16 keeps 256·(F_hi·G + F·G_hi), so
the corrections (A1's 2:4 passes) stay. The check below runs depth-1 and depth-2 Strassen on v1 operands (±80 noise on
both sides, period L, U's columns permuted every segment) with every form wrapped, against every segment's prefix word,
mod 2^8 (must hold) and mod 2^16 (must fail).

The price (mod 2^8): tensor (7/8)^r, deliveries D_r per output element a segment with D_1 = 5/4 and
D_r = 5/4 + (7/4)·D_(r−1) (each level's 5 extra block feeds, Theorem O's first-feed-free count, done level by level),
at c_op MAC-times a lane-op and d lane-ops a delivery (1 IADD, or 0.5 when an IADD3 takes two), plus operand forms
(5 block sums a level a side, amortized over the other side's width).
"""

from __future__ import annotations

import numpy as np

PAIRS = [((1, 0, 0, 1), (1, 0, 0, 1)), ((0, 0, 1, 1), (1, 0, 0, 0)), ((1, 0, 0, 0), (0, 1, 0, -1)),
         ((0, 0, 0, 1), (-1, 0, 1, 0)), ((1, 1, 0, 0), (0, 0, 0, 1)), ((-1, 0, 1, 0), (1, 1, 0, 0)),
         ((0, 1, 0, -1), (0, 0, 1, 1))]
OUT = [(1, 0, 0, 1, -1, 0, 1), (0, 0, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 0, 0), (1, -1, 1, 0, 0, 1, 0)]


def wrap8(v):
    return ((v + 128) & 0xFF) - 128


def quads(x):
    m, k = x.shape
    return [x[: m // 2, : k // 2], x[: m // 2, k // 2 :], x[m // 2 :, : k // 2], x[m // 2 :, k // 2 :]]


def strassen(a, b, depth):
    """a (m x k) @ b (k x n) with every form wrapped to s8 before it is multiplied (int64 accumulation)."""
    if depth == 0:
        return a @ b
    A, B = quads(a), quads(b)
    prods = []
    for ca, cb in PAIRS:
        fa = wrap8(sum(c * q for c, q in zip(ca, A, strict=True)))
        fb = wrap8(sum(c * q for c, q in zip(cb, B, strict=True)))
        prods.append(strassen(fa, fb, depth - 1))
    c = [sum(w * p for w, p in zip(row, prods, strict=True)) for row in OUT]
    return np.block([[c[0], c[1]], [c[2], c[3]]])


def check(depth: int, m=32, n=32, L=64, S=4, u=80, seed=1) -> tuple[bool, bool]:
    rng = np.random.default_rng(seed)
    k = S * L
    A = rng.integers(-128, 128, (m, k))
    B = rng.integers(-128, 128, (n, k))
    V = rng.integers(-u, u + 1, (m, L))
    U = rng.integers(-u, u + 1, (n, L))
    X = wrap8(A + rng.integers(0, 2, (m, k)) + np.tile(V, S))
    W = wrap8(B + rng.integers(0, 2, (n, k)) + np.concatenate([U[:, rng.permutation(L)] for _ in range(S)], 1))
    acc = np.zeros((m, n), np.int64)
    ok8 = ok16 = True
    for s in range(S):
        acc = acc + strassen(X[:, s * L:(s + 1) * L], W[:, s * L:(s + 1) * L].T, depth)
        want = X[:, :(s + 1) * L] @ W[:, :(s + 1) * L].T
        ok8 &= bool(np.array_equal(acc % 256, want % 256))
        ok16 &= bool(np.array_equal(acc % 65536, want % 65536))
    return ok8, ok16


def deliveries(r: int) -> float:
    d = 0.0
    for _ in range(r):
        d = 5 / 4 + 7 / 4 * d
    return d


def forms(r: int) -> float:
    """Block-sum lane-ops a side, per operand entry a segment: level j has 7^(j-1) sub-problems of 5 sums of 1/4^j."""
    return sum(7 ** (j - 1) * 5 / 4 ** j for j in range(1, r + 1))


def main() -> None:
    for depth in (1, 2):
        ok8, ok16 = check(depth)
        print(f"Strassen depth {depth}, forms wrapped to s8, v1 operands: words mod 2^8 exact={ok8}, mod 2^16 exact={ok16}")
    cards = {"H200": 32, "RTX PRO 6000": 8}
    print("price mod 2^8 (tensor + deliveries + forms at width 4096 both sides), d = 1 / d = 0.5:")
    for card, cop in cards.items():
        for L in (640, 1024, 2688):
            row = []
            for r in range(1, 5):
                t = (7 / 8) ** r
                f = 2 * cop * forms(r) / 4096
                full = t + cop * deliveries(r) / L + f
                half = t + 0.5 * cop * deliveries(r) / L + f
                row.append(f"r={r} {full:.3f}/{half:.3f}")
            print(f"  {card:12s} L={L:4d}: " + "  ".join(row))


if __name__ == "__main__":
    main()
