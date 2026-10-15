"""The combined-room LP on sm_120 (CMP-12, W10 round 2), solved exactly.

Per SMSP per clock: the tensor pipe with the register reads A5a measured (T), the FMA pipe's heavy and lite halves (H,
L), the ALU (A), MUFU (U), issue (I) and the memory-side adder (M, M5-v2's 2.4 adds a SM a tick). Variables are warp
instructions a clock (t is the tensor share, 16 MMAs a clock of m16n8k32 at t = 1; memory-side adds are elements).
The objective is MAC-equivalents at M5-v2's credits (dp4a 4 a lane, packed and integer 2, FFMA and MUFU 1, a
memory-side add 1 or 2) over FP8 capacity (256 MACs a SMSP a clock). Tensor costs w (cycles a warp instruction) are
the minimum measured: A5a's column for the multi-register forms, run r20261015-083314-3d5a for the one-register ones
(0 at or below the measured value).

    python3 combined_lp.py      # prints the optimum, the dual and S(t0) for the honest mixes
"""

from fractions import Fraction as F

ROWS = ["T", "H", "L", "A", "U", "I", "M"]
CAP = {"T": F(1), "H": F(1), "L": F(1), "A": F(1), "U": F(1), "I": F(1), "M": F(3, 5)}

# name: (credit share a unit, {row: cost a unit})
VARS = {
    "mma": (F(1), {"T": F(1), "I": F(1, 16)}),
    "dp4a.1reg": (F(1, 2), {"H": F(2), "I": F(1)}),
    "dp4a.reuse": (F(1, 2), {"T": F(1, 10), "H": F(2), "I": F(1)}),
    "dp4a.3reg": (F(1, 2), {"T": F(46, 100), "H": F(2), "I": F(1)}),
    "imad": (F(1, 4), {"H": F(2), "I": F(1)}),
    "hfma2": (F(1, 4), {"T": F(19, 100), "H": F(2), "I": F(1)}),
    "ffma.heavy": (F(1, 8), {"H": F(2), "I": F(1)}),
    "ffma.lite": (F(1, 8), {"L": F(2), "I": F(1)}),
    "ffma.2reg": (F(1, 8), {"T": F(43, 100), "L": F(2), "I": F(1)}),
    "ffma2": (F(1, 4), {"L": F(2), "I": F(1)}),
    "iop.1reg": (F(1, 4), {"A": F(2), "I": F(1)}),
    "iop.2reg": (F(1, 4), {"T": F(25, 100), "A": F(2), "I": F(1)}),
    "iop.3reg": (F(1, 4), {"T": F(27, 100), "A": F(2), "I": F(1)}),
    "mufu": (F(1, 8), {"U": F(8), "I": F(1)}),
    "redAdd": (F(1, 256), {"M": F(1), "I": F(1, 32)}),
    "redAdd2": (F(1, 128), {"M": F(1), "I": F(1, 32)}),
    "bulkRedAdd": (F(1, 256), {"M": F(1)}),
}


def simplex(c, A, b):
    """max c.x, A x <= b, x >= 0, b >= 0; Bland's rule, exact. Returns (value, x, y)."""
    m, n = len(A), len(c)
    tab = [list(A[i]) + [F(int(i == j)) for j in range(m)] + [b[i]] for i in range(m)]
    obj = [-ci for ci in c] + [F(0)] * m + [F(0)]
    basis = [n + i for i in range(m)]
    while True:
        col = next((j for j in range(n + m) if obj[j] < 0), None)
        if col is None:
            break
        ratios = [(tab[i][-1] / tab[i][col], basis[i], i) for i in range(m) if tab[i][col] > 0]
        _, _, row = min(ratios)
        piv = tab[row][col]
        tab[row] = [v / piv for v in tab[row]]
        for i in range(m):
            if i != row and tab[i][col] != 0:
                f = tab[i][col]
                tab[i] = [a - f * p for a, p in zip(tab[i], tab[row])]
        f = obj[col]
        obj = [a - f * p for a, p in zip(obj, tab[row])]
        basis[row] = col
    x = [F(0)] * (n + m)
    for i, j in enumerate(basis):
        x[j] = tab[i][-1]
    return obj[-1], x[:n], obj[n : n + m]


def solve(fix=None):
    names = list(VARS)
    c = [VARS[v][0] for v in names]
    A = [[VARS[v][1].get(r, F(0)) for v in names] for r in ROWS]
    b = [CAP[r] for r in ROWS]
    if fix is not None:  # t = fix: move the tensor column to the right-hand side
        k = names.index("mma")
        b = [bi - A[i][k] * fix for i, bi in enumerate(b)]
        c[k] = F(0)
        for i in range(len(A)):
            A[i][k] = F(0)
    val, x, y = simplex(c, A, b)
    if fix is not None:
        x[names.index("mma")] = fix
        val += fix
    return val, dict(zip(names, x)), dict(zip(ROWS, y))


def check_dual(y):
    for v, (gain, cost) in VARS.items():
        assert sum(y[r] * cost.get(r, F(0)) for r in ROWS) >= gain, v
    return sum(y[r] * CAP[r] for r in ROWS)


if __name__ == "__main__":
    val, x, y = solve()
    print(f"optimum S* = {val} = {float(val):.6f}")
    print("  primal:", {k: str(v) for k, v in x.items() if v})
    print("  dual:  ", {k: str(v) for k, v in y.items() if v}, "bound", check_dual(y))
    for t0 in [F(1), F(998, 1000), F(998, 1000) * F(97, 100), F(998, 1000) * F(94, 100), F(998, 1000) * F(87, 100)]:
        v, _, _ = solve(t0)
        print(f"  S(t = {float(t0):.4f}) = {float(v):.6f}, room beside it {float(v - t0):.6f}")
