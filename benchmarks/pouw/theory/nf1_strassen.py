"""NF1: depth-1 Strassen on H1's operands at A = B = 0 and a large segment, exact.

H1's operands are X = A + b and W' = B + c + U_rep, U uniform on {+-1..+-64} with period L (one column per position of
the segment), b, c fair bits. At A = B = 0 every Strassen form is a sum of at most two entries, and on the weight side
that is U + c terms in [-128, 130]. This counts the forms that leave int8 and checks that the seven int8 products give
every segment's word exactly, with five block deliveries a segment (against m n L MACs).
"""

from __future__ import annotations

import numpy as np


def strassen(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int, int]:
    """a (m x k) times b (k x n) by one Strassen level; returns the product, int8 overflows, forms checked."""
    m, k = a.shape
    n = b.shape[1]
    a11, a12, a21, a22 = a[: m // 2, : k // 2], a[: m // 2, k // 2 :], a[m // 2 :, : k // 2], a[m // 2 :, k // 2 :]
    b11, b12, b21, b22 = b[: k // 2, : n // 2], b[: k // 2, n // 2 :], b[k // 2 :, : n // 2], b[k // 2 :, n // 2 :]
    pairs = [
        (a11 + a22, b11 + b22), (a21 + a22, b11), (a11, b12 - b22), (a22, b21 - b11),
        (a11 + a12, b22), (a21 - a11, b11 + b12), (a12 - a22, b21 + b22),
    ]
    over = sum(int(((f < -128) | (f > 127)).sum()) for p in pairs for f in p)
    total = sum(f.size for p in pairs for f in p)
    m1, m2, m3, m4, m5, m6, m7 = (x @ y for x, y in pairs)
    c = np.block([[m1 + m4 - m5 + m7, m3 + m5], [m2 + m4, m1 - m2 + m3 + m6]])
    return c, over, total


def main(m: int = 64, n: int = 64, L: int = 1024, segs: int = 4, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    u = rng.integers(1, 65, size=(n, L)) * rng.choice([-1, 1], size=(n, L))
    X = rng.integers(0, 2, size=(m, segs * L)).astype(np.int64)
    W = rng.integers(0, 2, size=(n, segs * L)).astype(np.int64) + np.tile(u, segs)
    over = total = 0
    acc = np.zeros((m, n), dtype=np.int64)
    for s in range(segs):
        xs, ws = X[:, s * L : (s + 1) * L], W[:, s * L : (s + 1) * L]
        c, o, t = strassen(xs, ws.T)
        over, total = over + o, total + t
        acc = acc + c
        assert np.array_equal(acc, X[:, : (s + 1) * L] @ W[:, : (s + 1) * L].T), s
    deliveries = 5 * (m // 2) * (n // 2) * segs
    macs = m * n * L * segs
    print(f"m={m} n={n} L={L} segments={segs}: every word exact; int8 overflows {over} of {total} form entries "
          f"({over / total:.2e}); tensor MACs 7/8 of naive; deliveries {deliveries} adds against {macs} MACs, "
          f"{32 * deliveries / macs:.4f} MAC-times a MAC at 32 MAC-times a lane-op (40/L = {40 / L:.4f})")


if __name__ == "__main__":
    main()
