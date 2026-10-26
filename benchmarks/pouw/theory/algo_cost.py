"""ACCEL ALGO-COST (CMP-114): known exact bilinear matmul schemes priced on v1's operands, against the standard product.

v1's operands are X = wrap8(A + b + V) and W' = wrap8(B + c + U): every entry is an int8 whose noise is uniform on ±80
and independent of every other entry of its segment. A scheme ⟨m,k,n;R⟩ applied inside a segment forms, per product,
an A-side form (a signed sum of t_A blocks of X) and a B-side form; a form with t > 1 leaves int8 and its overflow part
F_hi (F = wrap8(F) + 256 F_hi) is multiplied on the 2:4 sparse path, at half a product a pass. A pass takes at most two
nonzeros of every group of four consecutive k entries of a row, so a form costs 0.5 x (passes): one pass when no group
holds three or more nonzeros, two when some does (a tile of 16 rows x 64 k needs the second pass with probability
1 - (1 - q)^256, q the group probability).

Prints, for each scheme read from Sedoglavic's catalogue (LRP files under SCHEMES, fetched from fmm.univ-lille.fr):
its check (the scheme reproduces random integer products exactly), rank, the number of summed forms per side and their
term counts, Σ_p (f_p − 1) on the output side, and its tensor price (R + Σ c_form)/(mkn) at v1's noise. The overflow
rates are exact (convolutions of the entry distributions), at data 0 and minimized over a data grid. `paper_family`
reprices NAIVE-STRASSEN's schedule family (CMP-110) at v1's range with this 2:4 route added, depths 1 to 3.
"""

from __future__ import annotations

import itertools
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

U = 80


def dist(d: int, u: int = U) -> np.ndarray:
    p = np.zeros(256)
    for e in (0, 1):
        for nv in range(-u, u + 1):
            p[(d + e + nv + 128) % 256] += 0.5 / (2 * u + 1)
    return p


def overflow(ps, signs) -> float:
    s, lo = np.array([1.0]), 0
    for p, sg in zip(ps, signs, strict=True):
        s = np.convolve(s, p if sg > 0 else p[::-1])
        lo += -128 if sg > 0 else -127
    v = np.arange(lo, lo + len(s))
    return float(s[(v < -128) | (v > 127)].sum())


def rates(tmax: int = 8) -> dict[int, tuple[float, float]]:
    """t -> (overflow at data 0, min over a data grid) for an all-plus t-term sum (signs only reflect the grid)."""
    D = {d: dist(d) for d in range(-128, 128, 8)}
    out = {}
    for t in range(2, tmax + 1):
        z = overflow([D[0]] * t, (1,) * t)
        grid = range(-128, 128, 32 if t > 4 else 16)
        mn = min(overflow([D[d] for d in ds], (1,) * t) for ds in itertools.combinations_with_replacement(grid, t))
        out[t] = (z, min(z, mn))
    return out


def form_cost(p: float) -> float:
    """0.5 a 2:4 pass; the second pass's share is the chance a 16 x 64 tile has a group of 4 with >= 3 nonzeros."""
    q = 4 * p**3 * (1 - p) + p**4
    return 0.5 * (1 + (1 - (1 - q) ** 256))


def best_form_cost(p: float, c_op: float) -> float:
    """Additive price of one summed form: the cheapest of two 2:4 passes, one pass with each group's third and fourth
    nonzeros added on the lanes, and every nonzero on the lanes (c_op MAC-times a lane add)."""
    leftover = (4 * p**3 * (1 - p) + 2 * p**4) / 4
    return min(form_cost(p), 0.5 + c_op * leftover, c_op * p)


def strassen_overlapped(lane_load: float, pX: float, pW: float, lane_per_add: float, c_op: float = 8,
                        port_per_add: float = 8, staging: float = 0.22) -> tuple[float, int, int]:
    """Depth-1 Strassen under G2's per-resource maximum, in units of the standard product's tensor time (A8).

    lane_load is the honest program's own lane time (hashing) over its tensor time. Each of the five summed forms a side
    goes either to one 2:4 pass with lane leftovers (tensor) or to sparse adds on the lanes (lane_per_add MAC-times an
    add, port_per_add MAC-times of shared-memory port an add). Returns the cheapest (time, X forms on 2:4,
    W forms on 2:4).
    """
    def leftover(p: float) -> float:
        return (4 * p**3 * (1 - p) + 2 * p**4) / 4

    best = (float("inf"), 0, 0)
    for jx in range(6):
        for jw in range(6):
            tensor = (7 + 0.5 * (jx + jw)) / 8
            adds = ((5 - jx) * pX + (5 - jw) * pW) / 8
            lanes = lane_load + lane_per_add * adds + c_op * (jx * leftover(pX) + jw * leftover(pW)) / 8
            port = port_per_add * adds + staging * tensor
            best = min(best, (max(tensor, lanes, port), jx, jw))
    return best


def scheme_overlapped(terms: list[int], rank: int, mkn: int, R: dict, lane_load: float, lane_per_add: float,
                      c_op: float = 8, port_per_add: float = 8, staging: float = 0.22) -> tuple[float, dict]:
    """One level of a scheme under G2's per-resource maximum (time in standard tensor units). terms lists every summed
    form's term count (both sides); forms of equal count are interchangeable, so the split is chosen per count."""
    def lo(p: float) -> float:
        return (4 * p**3 * (1 - p) + 2 * p**4) / 4

    counts = {t: terms.count(t) for t in sorted(set(terms)) if t > 1}
    ps = {t: R[min(t, 8)][0] for t in counts}
    best = (float("inf"), {})
    for js in itertools.product(*[range(c + 1) for c in counts.values()]):
        split = dict(zip(counts, js, strict=True))
        tensor = (rank + 0.5 * sum(js)) / mkn
        adds = sum((counts[t] - j) * ps[t] for t, j in split.items()) / mkn
        lanes = lane_load + lane_per_add * adds + c_op * sum(j * lo(ps[t]) for t, j in split.items()) / mkn
        port = port_per_add * adds + staging * tensor
        best = min(best, (max(tensor, lanes, port), split), key=lambda x: x[0])
    return best


def wrapped_pair_overflow(u: int = U) -> float:
    """Overflow of a two-term sum of level-1 wrapped forms f = wrap8(x1 + x2), x = b + N at data 0 (depth-2 routes)."""
    s = np.convolve(dist(0, u), dist(0, u))
    f = np.zeros(256)
    for i, w in enumerate(s):
        f[(i - 256 + 128) % 256] += w
    return overflow([f, f], (1, 1))


def strassen2_overlapped(lane_load: float, lane_per_add: float, c_op: float = 8, port_per_add: float = 8,
                         staging: float = 0.22, u: int = U) -> dict[str, float]:
    """Depth-2 Strassen under G2's per-resource maximum (time in standard tensor units), two correction plans.

    direct: every level-2 form corrected once (A and B forms of 1, 2 or 4 terms, 49 leaves of mkn/64).
    hierarchical: level-1 forms wrapped and corrected at level 1 (10 forms of mkn/8), then level-2 two-term forms of
    raw blocks or of wrapped bytes. Each summed form goes to one 2:4 pass with lane leftovers or to lane adds.
    """
    def lo(p: float) -> float:
        return (4 * p**3 * (1 - p) + 2 * p**4) / 4

    R = rates(4)
    p2, p4, pf = R[2][0], R[4][0], wrapped_pair_overflow(u)
    # (count, weight a form, overflow) per class; weight = the form's product size over mkn
    plans = {
        "direct": [(20, 1 / 64, p2), (25, 1 / 64, p4)] * 2,
        "hierarchical": [(5, 1 / 8, p2), (10, 1 / 64, p2), (25, 1 / 64, pf)] * 2,
    }
    out = {}
    for name, classes in plans.items():
        best = float("inf")
        for js in itertools.product(*[range(0, n + 1, max(1, n // 10)) for n, _, _ in classes]):
            pairs = list(zip(js, classes, strict=True))
            tensor = 49 / 64 + sum(0.5 * j * w for j, (_, w, _) in pairs)
            adds = sum((n - j) * w * p for j, (n, w, p) in pairs)
            lanes = lane_load + lane_per_add * adds + c_op * sum(j * w * lo(p) for j, (_, w, p) in pairs)
            port = port_per_add * adds + staging * tensor
            best = min(best, max(tensor, lanes, port))
        out[name] = best
    return out


STRASSEN_NU = [(2, 2), (2, 1), (1, 2), (1, 2), (2, 1), (2, 2), (2, 2)]
WINOGRAD_NU = [(1, 1), (1, 1), (4, 1), (1, 4), (2, 2), (3, 3), (2, 2)]


def paper_family(R: dict, depth: int = 3) -> None:
    """NAIVE-STRASSEN's family at v1's range with the 2:4 overflow route added (A5).

    A leaf whose forms have (nu_A, nu_B) terms costs min(2, 1 + c(nu_A) + c(nu_B)): 2 is the family's price (f16 at
    weight 2, or two INT8 digits) while f16 holds the form (161 nu <= 4097), and a lower bound past that.
    """
    def c(t: int) -> float:
        return 0.0 if t == 1 else form_cost(R[t][1]) if t in R else 1.0

    def leaf(a: int, b: int, route: bool) -> float:
        return 1.0 if (a, b) == (1, 1) else min(2.0, 1 + c(a) + c(b)) if route else 2.0

    def uniform(nus, r: int, route: bool) -> float:
        tot = 0.0
        for path in itertools.product(nus, repeat=r):
            a = int(np.prod([x for x, _ in path]))
            b = int(np.prod([y for _, y in path]))
            tot += leaf(a, b, route)
        return tot / 8**r

    def mix(r: int, a: int = 1, b: int = 1) -> float:
        if r == 0:
            return leaf(a, b, True)
        return min(sum(mix(r - 1, a * x, b * y) for x, y in nus) / 8 for nus in (STRASSEN_NU, WINOGRAD_NU))

    for r in range(1, depth + 1):
        print(f"depth {r}: Strassen {uniform(STRASSEN_NU, r, True):.4f} (family {uniform(STRASSEN_NU, r, False):.5f}), "
              f"Winograd {uniform(WINOGRAD_NU, r, True):.4f} (family {uniform(WINOGRAD_NU, r, False):.5f}), "
              f"per-node mix {mix(r):.4f}")


def parse(path: Path):
    """The catalogue's ⟨3,3,3;23⟩ is a family in a parameter l; it is read at l = 1."""
    txt = path.read_text().replace("1/l", "1").replace("l", "1").replace("LRP:=", "")
    mats = []
    for m in re.finditer(r"Matrix\((\d+), (\d+), \[(.*?)\]\)", txt, re.S):
        r, c, body = int(m.group(1)), int(m.group(2)), m.group(3)
        rows = re.findall(r"\[([^\[\]]*)\]", body)
        mats.append([[Fraction(x.strip()) for x in row.split(",")] for row in rows])
        assert len(mats[-1]) == r and all(len(x) == c for x in mats[-1])
    return mats


def check(Lm, Rm, Pm, m, k, n) -> str | None:
    """Find the flattening convention under which the scheme computes AB; return its name or None."""
    rng = np.random.default_rng(0)
    A = rng.integers(-9, 10, (m, k))
    B = rng.integers(-9, 10, (k, n))
    La, Ra, Pa = (np.array(x, dtype=object) for x in (Lm, Rm, Pm))
    for ta, tb, tc in itertools.product((0, 1), repeat=3):
        a, b = (A.T if ta else A).flatten(), (B.T if tb else B).flatten()
        if La.shape[1] != a.size or Ra.shape[1] != b.size or Pa.shape[0] != m * n:
            return None
        prod = (La @ a.astype(object)) * (Ra @ b.astype(object))
        C = A @ B
        if np.array_equal(Pa @ prod, (C.T if tc else C).flatten().astype(object)):
            return f"A{'T' if ta else ''} B{'T' if tb else ''} C{'T' if tc else ''}"
    return None


def describe(path: Path, m: int, k: int, n: int, R: dict) -> None:
    Lm, Rm, Pm = parse(path)
    rank = len(Lm)
    conv = check(Lm, Rm, Pm, m, k, n)
    coeffs = {x for M in (Lm, Rm, Pm) for row in M for x in row if x}
    tA = [sum(1 for x in row if x) for row in Lm]
    tB = [sum(1 for x in row if x) for row in Rm]
    fp = [sum(1 for row in Pm if row[j]) for j in range(rank)]
    sA, sB = sum(t > 1 for t in tA), sum(t > 1 for t in tB)
    both = sum(1 for a, b in zip(tA, tB, strict=True) if a > 1 and b > 1)
    cost = rank + sum(form_cost(R[min(t, 8)][1]) for t in tA + tB if t > 1)
    crit = rank + 0.5 * (sA + sB)
    print(f"{path.stem:14s} <{m},{k},{n};{rank}> check={conv} coeffs={sorted(coeffs)}")
    print(f"   s_A={sA} terms {sorted(tA)}; s_B={sB} terms {sorted(tB)}; products with both sides summed {both}")
    print(f"   sum(f_p - 1)={sum(fp) - rank}; criterion R + 0.5(s_A+s_B) = {crit:.1f} vs 1.2 mkn = {1.2 * m * k * n:.1f} "
          f"(standard {m * k * n}); tensor price at +-{U} with 2:4 passes {cost:.2f} = {cost / (m * k * n):.3f}x standard")
    best = {card: (rank + sum(best_form_cost(R[min(t, 8)][1], c) for t in tA + tB if t > 1)) / (m * k * n)
            for card, c in (("H200", 32), ("RTX PRO 6000", 8))}
    print(f"   additive, best route per form (2:4 twice | 2:4 + lane leftovers | lanes): "
          f"H200 {best['H200']:.3f}x, RTX PRO 6000 {best['RTX PRO 6000']:.3f}x")
    adds = sum(R[min(t, 8)][1] for t in tA + tB if t > 1) / (m * k * n)
    print(f"   lanes route (idle lanes, A8): tensor {rank / (m * k * n):.3f}, {adds:.4f} correction adds a MAC; "
          f"RTX PRO 6000 port {8 * adds:.2f} lanes {6 * adds:.2f}; H200 port {32 * adds:.2f} lanes {24 * adds:.2f} "
          f"(before staging and divergence)")


def main(argv: list[str]) -> None:
    R = rates()
    for t, (z, mn) in R.items():
        print(f"t={t}: overflow at data 0 {z:.4f}, min over data grid {mn:.4f}; form cost {form_cost(mn):.3f} products")
    paper_family(R)
    d = Path(argv[1]) if len(argv) > 1 else Path("/tmp/fmm")
    for name, shape in [("2x2x2", (2, 2, 2)), ("2x3x3", (2, 3, 3)), ("3x3x3", (3, 3, 3)), ("4x4x4", (4, 4, 4))]:
        f = d / f"{name}_LRP.mpl"
        if f.exists():
            describe(f, *shape, R)


if __name__ == "__main__":
    main(sys.argv)
