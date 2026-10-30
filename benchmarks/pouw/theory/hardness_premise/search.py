"""Addition-priced search over exact bilinear schedules for E·Rᵀ mod 2^w (HT-PREMISE, M3 and M9).

The family is `schedules.py`'s: atoms of the chip's MMA formats, digit splits of either operand, and Strassen or
Winograd levels (roles AB), to a depth L. E ranges over `a` consecutive integers and R over `b`; the output is needed
mod 2^w. Unlike `schedules.Model`, every node is charged its additions in W1 at the shape it runs at, so the optimum is
a lower bound on `execute.py`'s count of every schedule in the family:

- MMA MACs at the format's price;
- per scheme node, the cheapest of `execute.PROGRAMS` given which products can chain (`execute.chainable`), at IADD3
  (`fast`) or IADD (`plain`), 32 a write per sub-output;
- per scheme node, the noise-side combinations, 4 int8 lanes a register (`fast`) or 2 (`plain`);
- per digit split one IMAD an output, and for a noise-side split its two digits;
- for a float atom, its fp32 drains, with the operands centred (the fewest drains).

Not charged, so the bound stays a bound: shift corrections, E's affine map, the spilled reading's reloads, and wrapping
mod 2^w (free here; a mask in practice). Mod 2^w, an operand wider than 2^w values wraps to 2^w, and the high digit of a split at a
power of two d needs only w − log₂d bits (zero bits: free). `run_mod` executes any returned tree exactly mod 2^w.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

from execute import PROGRAMS, SCHEMES, _eval
from schedules import Model, rng

# name: (values held, W1 a MAC, accumulator (k, bits) for a float format, None for an integer one)
CHIPS = {
    "h200": {
        "s8": (256, 1.0, None), "b1": (2, 0.125, None), "e4m3": (33, 1.0, (32, 14)), "e5m2": (17, 1.0, (32, 14)),
        "f16": (4097, 2.0, (16, 24)), "bf16": (513, 2.0, (16, 24)), "tf32": (4097, 4.0, (8, 24)),
    },
    # sm_120: the card's own atoms (no native b1); MMA prices from the sheet ratio, unmeasured
    "sm120": {
        "s8": (256, 1.0, None), "e4m3": (33, 1.0, (32, 24)), "e5m2": (17, 1.0, (32, 24)), "e2m1": (9, 1.0, (32, 24)),
        "nvf4": (9, 0.5, (64, 24)), "f16": (4097, 2.0, (16, 24)), "bf16": (513, 2.0, (16, 24)),
        "tf32": (4097, 4.0, (8, 24)),
    },
}
# The H200 with a hypothetical int4 path at 2× s8 (A100's ratio): mma.sync .s4 on sm_90 is unmeasured.
CHIPS["h200+s4"] = dict(CHIPS["h200"], s4=(16, 0.5, None))

FORMS = {name: (U, V) for name, (U, V, _W) in SCHEMES.items()}
INF = float("inf")

# Range grid: every integer to 64, every format's run and its neighbour, powers of two and their successors, and
# 2^(k/16). Cost is monotone in the ranges, so the search with ranges rounded down is a lower bound and rounded up an
# upper bound whose argmin tree runs on the true ranges (`run_mod` checks it).
_CAPS = sorted({c for chip in CHIPS.values() for c, _p, _a in chip.values()})
GRID = sorted(set(range(1, 65)) | set(_CAPS) | {c + 1 for c in _CAPS} | {2**j for j in range(40)}
              | {2**j + 1 for j in range(40)} | {round(2 ** (k / 16)) for k in range(16 * 40)})


def _grid(v: int, mode: str | None) -> int:
    if mode is None:
        return v
    import bisect
    i = bisect.bisect_left(GRID, v)
    if mode == "up" or GRID[i] == v:
        return GRID[i]
    return GRID[i - 1]


def _prog_cost(prog, iadd3: bool):
    """(writes from sum steps, the products it chains)."""
    n, chained = 0, []
    for step in prog:
        if step[0] == "sum":
            n += -(-(len(step[2]) - 1) // 2) if iadd3 else len(step[2]) - 1
        else:
            chained.append(step[2])
    return n, chained


class Search:
    def __init__(self, chip: str, K: int, n: int, depth: int, reading: str = "fast", grid: str | None = None):
        self.atoms = CHIPS[chip]
        self.K, self.n, self.depth = K, n, depth
        self.iadd3 = reading == "fast"
        self.lanes = 4 if self.iadd3 else 2
        self.grid = grid
        self.memo = {}  # a dict, not lru_cache: its C wrapper hits Python 3.12's fixed C recursion limit

    def F(self, lvl: int, a: int, b: int, w: int):
        if w > 0:
            a, b = min(a, 2**w), min(b, 2**w)
        key = (lvl, _grid(a, self.grid), _grid(b, self.grid), w)
        r = self.memo.get(key)
        if r is None:
            r = self.memo[key] = self._F(*key)
        return r

    def atom(self, lvl: int, a: int, b: int):
        K = self.K / 2**lvl
        best = (INF, None)
        for name, (cap, price, acc) in self.atoms.items():
            if a > cap or b > cap:
                continue
            cost = price
            if acc is not None:
                k, bits = acc
                p = max(1, (a // 2) * (b // 2))
                if k * p > 2**bits:
                    continue
                chunk = max(k, (2**bits // p) // k * k)
                cost += 32 * 2 * math.ceil(K / chunk) / K
            best = min(best, (cost, name))
        return best

    def _F(self, lvl: int, a: int, b: int, w: int):
        """((cost, choice) any schedule, (cost, choice) one whose result can chain), W1 a MAC at level lvl."""
        if w <= 0:
            return (0.0, ("zero",)), (0.0, ("zero",))
        a, b = min(a, 2**w), min(b, 2**w)
        K, n = self.K / 2**lvl, self.n / 2**lvl
        c, name = self.atom(lvl, a, b)
        best_any = best_ch = (c, ("atom", name, a, b, w))
        for side in "ab":
            v = a if side == "a" else b
            for d in Model.cands(v):
                hi = -(-v // d)
                if max(_grid(d, self.grid), _grid(hi, self.grid)) >= v:
                    continue
                w_hi = w - int(math.log2(d)) if d & (d - 1) == 0 else w
                lo_ab, hi_ab = ((d, b), (hi, b)) if side == "a" else ((a, d), (a, hi))
                lo = self.F(lvl, *lo_ab, w)
                hi_c = self.F(lvl, *hi_ab, w_hi)[0][0]
                extra = 32 / K + (64 / (self.lanes * n) if side == "a" else 0)
                ch = ("split", side, d, a, b, w)
                best_any = min(best_any, (lo[0][0] + hi_c + extra, ch))
                best_ch = min(best_ch, (lo[1][0] + hi_c + extra, ch))
        if lvl < self.depth:
            for sname, (U, V) in FORMS.items():
                kids = [self.F(lvl + 1, rng(a, sum(map(abs, u))), rng(b, sum(map(abs, v))), w) for u, v in zip(U, V)]
                e_adds = sum(sum(map(abs, u)) - 1 for u in U)
                for pi, prog in enumerate(PROGRAMS[sname]):
                    writes, chained = _prog_cost(prog, self.iadd3)
                    tot, flags = 0.0, []
                    for t, kid in enumerate(kids):
                        if t in chained and kid[1][0] / 8 <= kid[0][0] / 8 + 8 / K:
                            tot += kid[1][0] / 8
                            flags.append(True)
                        else:
                            tot += kid[0][0] / 8 + (8 / K if t in chained else 0)
                            flags.append(False)
                    tot += 8 * writes / K + 32 * e_adds / (4 * self.lanes * n)
                    best_any = min(best_any, (tot, ("scheme", sname, pi, tuple(flags), a, b, w)))
        return best_any, best_ch

    def tree(self, lvl, a, b, w, chain=False):
        a, b = (min(a, 2**w), min(b, 2**w)) if w > 0 else (a, b)
        a, b = _grid(a, self.grid), _grid(b, self.grid)
        cost, ch = self.F(lvl, a, b, w)[1 if chain else 0]
        if ch[0] == "zero":
            return {"zero": True, "w": w, "cost": 0.0}
        if ch[0] == "atom":
            return {"atom": ch[1], "a": a, "b": b, "w": w, "cost": cost}
        if ch[0] == "split":
            _, side, d, a, b, w = ch
            hi = -(-(a if side == "a" else b) // d)
            w_hi = w - int(math.log2(d)) if d & (d - 1) == 0 else w
            lo_ab, hi_ab = ((d, b), (hi, b)) if side == "a" else ((a, d), (a, hi))
            return {"split": side, "d": d, "a": a, "b": b, "w": w, "cost": cost,
                    "kids": [self.tree(lvl, *lo_ab, w, chain), self.tree(lvl, *hi_ab, w_hi)]}
        _, sname, pi, flags, a, b, w = ch
        U, V = FORMS[sname]
        kids = [self.tree(lvl + 1, rng(a, sum(map(abs, u))), rng(b, sum(map(abs, v))), w, f)
                for (u, v), f in zip(zip(U, V), flags)]
        return {"scheme": sname, "roles": "AB", "program": pi, "a": a, "b": b, "w": w, "cost": cost, "kids": kids}

    def root(self, a: int, b: int, w: int):
        (cost, _), _ = self.F(0, a, b, w)
        return cost


def stats(node, lvl=0, acc=None):
    """Scheme depth, and the MAC mix by format (fractions of the root's MACs)."""
    acc = acc if acc is not None else {"depth": 0, "mix": {}}
    if "atom" in node:
        acc["mix"][node["atom"]] = acc["mix"].get(node["atom"], 0) + 8.0**-lvl
        acc["depth"] = max(acc["depth"], lvl)
    for k in node.get("kids", []):
        stats(k, lvl + ("scheme" in node), acc)
    return acc


def _wrap(X, w):
    if w is None:
        return X
    M = 1 << w
    return ((X + M // 2) % M) - M // 2


def run_mod(node, A, B, chip: str):
    """A·B mod 2^w(node), executed exactly; every atom operand is shifted into its format's window and checked."""
    w = node["w"]
    M = 1 << w
    if node.get("zero"):
        return np.zeros((A.shape[0], B.shape[1]), dtype=np.int64)
    if A.max() - A.min() + 1 > M:
        A = _wrap(A, w)
    if B.max() - B.min() + 1 > M:
        B = _wrap(B, w)
    if "atom" in node:
        cap, _price, acc = CHIPS[chip][node["atom"]]
        ra, rb = int(A.max() - A.min()) + 1, int(B.max() - B.min()) + 1
        assert ra <= cap and rb <= cap, node
        if node["atom"] == "b1":
            sa, sb = int(A.min()), int(B.min())
        else:  # centred: [−⌈(r−1)/2⌉, ⌊(r−1)/2⌋] lies in [−⌊cap/2⌋, cap − 1 − ⌊cap/2⌋]
            sa, sb = int(A.min()) + ra // 2, int(B.min()) + rb // 2
        A2, B2 = A - sa, B - sb
        if acc is not None:
            k, bits = acc
            assert k * int(np.abs(A2).max()) * int(np.abs(B2).max()) <= 2**bits, node
        Y = A2 @ B2 + sa * B2.sum(axis=0)[None, :] + sb * A2.sum(axis=1)[:, None] + sa * sb * A.shape[1]
        return np.mod(Y, M)
    if "split" in node:
        d = node["d"]
        lo_n, hi_n = node["kids"]
        if node["split"] == "a":
            c = int(A.min())
            x0, x1 = (A - c) % d, (A - c) // d
            Y = run_mod(lo_n, x0, B, chip) + d * run_mod(hi_n, x1, B, chip) + c * B.sum(axis=0)[None, :]
        else:
            c = int(B.min())
            x0, x1 = (B - c) % d, (B - c) // d
            Y = run_mod(lo_n, A, x0, chip) + d * run_mod(hi_n, A, x1, chip) + c * A.sum(axis=1)[:, None]
        return np.mod(Y, M)
    U, V, _W = SCHEMES[node["scheme"]]
    h, k2, n2 = A.shape[0] // 2, A.shape[1] // 2, B.shape[1] // 2
    Ab = [A[:h, :k2], A[:h, k2:], A[h:, :k2], A[h:, k2:]]
    Bb = [B[:k2, :n2], B[:k2, n2:], B[k2:, :n2], B[k2:, n2:]]
    Ms = []
    for t, (u, v) in enumerate(zip(U, V)):
        a = sum(c * Ab[p] for p, c in enumerate(u) if c)
        b = sum(c * Bb[q] for q, c in enumerate(v) if c)
        Ms.append(run_mod(node["kids"][t], a, b, chip))
    C = _eval(PROGRAMS[node["scheme"]][node["program"]], Ms)
    return np.mod(np.block([[C[0], C[1]], [C[2], C[3]]]), M)


def check(node, a: int, b: int, w: int, chip: str, seeds: int = 2) -> bool:
    L = stats(node)["depth"]
    side = 2**L * 4
    ok = True
    for s in range(seeds):
        r = np.random.default_rng(s)
        E = r.integers(0, a, (side, side), dtype=np.int64)
        R = r.integers(-(b // 2), b - b // 2, (side, side), dtype=np.int64)
        E[0, 0], E[-1, -1], R[0, 0], R[-1, -1] = 0, a - 1, -(b // 2), b - b // 2 - 1
        ok &= bool(np.array_equal(run_mod(node, E, R.T, chip), np.mod(E @ R.T, 1 << w)))
    return ok


def executed(tree, a: int, shape) -> dict:
    """`execute.py`'s three readings of an exact (w = 16, R int8) tree at `shape`, E's affine map included."""
    import execute

    m, K, n = shape
    t = json.loads(json.dumps(tree))
    execute.heights(t)
    o = execute.Ops()
    execute.run(t, None, None, (0, a - 1), (-128, 127), shape, o)
    o.add_out(m * n)
    return {r: o.w1(r) / (m * K * n) for r in ("fast", "plain", "spilled")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chips", default="h200 sm120")
    ap.add_argument("--cases", default="4,256,16",
                    help="space-separated a,b,w: E over a values, R over b, output mod 2^w")
    ap.add_argument("--shapes", default="8192,2048,8192 8192,8192,8192 8192,16384,8192 8192,28672,8192 16384,53248,16384")
    ap.add_argument("--depths", default="4 6", help="depth caps to search at")
    ap.add_argument("--readings", default="fast plain")
    ap.add_argument("--grids", default="down up",
                    help="range rounding: `down` (a lower bound), `up` (an upper bound with a checked witness), `none`")
    ap.add_argument("--trees", type=Path, help="directory for the argmin trees (JSON)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    sys.setrecursionlimit(100000)
    out, ok = [], True
    for chip in args.chips.split():
        for case in args.cases.split():
            a, b, w = (int(v) for v in case.split(","))
            for shape in args.shapes.split():
                m, K, n = (int(v) for v in shape.split(","))
                for reading in args.readings.split():
                    for L in (int(v) for v in args.depths.split()):
                        for grid in args.grids.split():
                            s = Search(chip, K, n, L, reading, None if grid == "none" else grid)
                            cost = s.root(a, b, w)
                            t = s.tree(0, a, b, w)
                            st = stats(t)
                            tensor = sum(CHIPS[chip][f][1] * x for f, x in st["mix"].items())
                            row = {"chip": chip, "a": a, "b": b, "w": w, "m": m, "K": K, "n": n, "reading": reading,
                                   "depth_cap": L, "grid": grid, "w1": cost, "depth_used": st["depth"], "tensor": tensor,
                                   "mix": st["mix"], "states": len(s.memo)}
                            if grid != "down":
                                row["exact"] = check(t, a, b, w, chip, seeds=1 if st["depth"] >= 5 else 2)
                                ok &= row["exact"]
                                if b == 256 and w == 16:
                                    row["executed"] = executed(t, a, (m, K, n))
                                if args.trees:
                                    args.trees.mkdir(parents=True, exist_ok=True)
                                    f = args.trees / f"{chip}_a{a}_b{b}_w{w}_K{K}_n{n}_{reading}_L{L}.json.gz"
                                    f.write_bytes(gzip.compress(json.dumps(dict(t, a=a, b=b)).encode()))
                            print(json.dumps(row), flush=True)
                            out.append(row)
    res = {"ok": bool(ok), "rows": out}
    if args.json:
        args.json.write_text(json.dumps(res, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    import threading

    threading.stack_size(1 << 29)
    rc = []
    th = threading.Thread(target=lambda: rc.append(main()))
    th.start()
    th.join()
    sys.exit(rc[0] if rc else 1)
