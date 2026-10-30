"""Exact bilinear schedules for E·Rᵀ on a card's MMA formats (HT-PREMISE, P2).

A Python mirror of `Security/Definitions/Core/NaiveStrassen.lean`'s cost model, extended in three ways:

- the operands' ranges are parameters: E (the noise side) ranges over `a` consecutive integers and R (the public
  random matrix) over `b`. The Lean guarantee `NaiveBeatsStrassenServed` is the case `(256, 256)`; the structural
  design's forced product is `(2, 256)` for ±1 noise (E = 2e − 1, the −1·Rᵀ term precomputed), `(3, 256)` for ternary
  and `(4, 256)` for 2-bit noise;
- any bilinear scheme, not only Strassen and Winograd: every scheme of the Kauers–Moosbauer flip-graph catalogue whose
  file is given, under all six assignments of (E, R, output) to the scheme's three factors;
- costs are per leaf MAC (eighths of an s8 MAC), so a scheme of format ⟨p, q, r⟩ costs the sum of its products'
  children over p·q·r. Naive is 8.

The model is the Lean's: tensor units only, additions free, an operand combination with integer coefficients `u`
over operands of `a` values ranges over `(a − 1)·‖u‖₁ + 1` values, a digit split `x − c = x₀ + d·x₁` puts `x₀` over
`d` values and `x₁` over `⌈a/d⌉`, an atom holds `(a, b)` if both fit its run and, for a float format, its `k` products
sum exactly within `w` bits. Splits are searched over d in SPLIT_D and d = ⌈a/x⌉ for x in SPLIT_D, so a cost found
here is achievable (a witness), and the min over that set is what is reported; check 1 compares it with the Lean's
exact tables at (256, 256).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import namedtuple
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

Atom = namedtuple("Atom", "name w8 cap coupling")

S8 = Atom("s8", 8, 256, None)
B1 = Atom("b1", 1, 2, None)

H200_MEASURED = [
    S8,
    Atom("e4m3", 8, 33, (32, 14)),
    Atom("e5m2", 8, 17, (32, 14)),
    Atom("f16", 16, 4097, (16, 24)),
    Atom("bf16", 16, 513, (16, 24)),
    Atom("tf32", 32, 4097, (8, 24)),
    B1,
]
H200_SHEET = [
    S8,
    Atom("e4m3", 8, 33, (32, 14)),
    Atom("e5m2", 8, 17, (32, 14)),
    Atom("f16", 16, 4097, (8, 24)),
    Atom("bf16", 16, 513, (8, 24)),
    Atom("tf32", 32, 4097, (4, 24)),
    B1,
]
RTX_SHEET = [
    S8,
    Atom("e4m3", 8, 33, (32, 24)),
    Atom("e5m2", 8, 17, (32, 24)),
    Atom("e2m1", 8, 9, (32, 24)),
    Atom("nvf4", 4, 9, (64, 24)),
    Atom("f16", 16, 4097, (8, 24)),
    Atom("bf16", 16, 513, (8, 24)),
    Atom("tf32", 32, 4097, (4, 24)),
    B1,
]
RTX_MEASURED = [
    S8,
    Atom("e4m3", 8, 33, (32, 24)),
    Atom("e5m2", 8, 17, (32, 24)),
    Atom("e2m1", 8, 9, (32, 24)),
    Atom("nvf4", 4, 9, (64, 24)),
    Atom("f16", 16, 4097, (16, 24)),
    Atom("f16k8", 32, 4097, (8, 24)),
    Atom("bf16", 16, 513, (16, 24)),
    Atom("tf32", 32, 4097, (4, 24)),
    B1,
]
INSTANCES = {
    "h200Measured": H200_MEASURED,
    "h200Sheet": H200_SHEET,
    "rtx6000Sheet": RTX_SHEET,
    "rtx6000Measured": RTX_MEASURED,
    # sm_120 has no native b1 MMA (ptxas emulates it; NaiveStrassen's model note M13), so the card's own atoms:
    "rtx6000MeasuredNoB1": [f for f in RTX_MEASURED if f.name != "b1"],
}

# Digit sizes tried in a split: every atom's run (2, 9, 17, 33, 256, 513, 4097), its neighbours and powers of two, and
# the quotients ⌈a/x⌉ for the same x. Costs are monotone in the ranges, so a digit sized to a run is where a split pays.
SPLIT_D = (2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 32, 33, 64, 128, 255, 256, 257, 512, 513, 1024, 2048, 4096, 4097)
INF = float("inf")


def holds(f: Atom, a: int, b: int) -> bool:
    if a > f.cap or b > f.cap:
        return False
    if f.coupling is None:
        return True
    k, w = f.coupling
    return k * (a // 2) * (b // 2) <= 2**w


def rng(a: int, nu: int) -> int:
    return (a - 1) * nu + 1


# --- schemes -------------------------------------------------------------------------------------------------------

Scheme = namedtuple("Scheme", "name dims prods")  # prods: list of (l1_A, l1_B, l1_C) of primitive integer forms

STRASSEN = Scheme("strassen", (2, 2, 2), [(2, 2, 2), (2, 1, 2), (1, 2, 2), (1, 2, 2), (2, 1, 2), (2, 2, 1), (2, 2, 1)])
WINOGRAD = Scheme("winograd", (2, 2, 2), [(1, 1, 4), (1, 1, 1), (4, 1, 1), (1, 4, 1), (2, 2, 2), (3, 3, 3), (2, 2, 2)])

_TERM = re.compile(r"([+-]?)\s*(?:(\d+)\s*\*\s*)?([abc])(\d)(\d)")


def _form(text: str) -> dict:
    """Coefficients of one linear form, as Fractions; `text` may carry an integer factor `k*(…)` or `/k`."""
    coeffs: dict = {}
    for sign, k, var, i, j in _TERM.findall(text):
        c = Fraction(int(k) if k else 1) * (-1 if sign == "-" else 1)
        key = (var, int(i), int(j))
        coeffs[key] = coeffs.get(key, 0) + c
    return {k: v for k, v in coeffs.items() if v != 0}


def _split_factors(line: str) -> list[str]:
    """Split `(…)*(…)*(…)[/k]` at top-level `*`."""
    out, depth, cur = [], 0, ""
    for ch in line:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "*" and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def _primitive_l1(coeffs: dict) -> int:
    vals = list(coeffs.values())
    den = math.lcm(*[v.denominator for v in vals])
    ints = [int(v * den) for v in vals]
    g = math.gcd(*ints)
    return sum(abs(x) // g for x in ints)


def parse_scheme(path: Path) -> Scheme | None:
    """A catalogue `.exp` file: one product a line, `(A-form)*(B-form)*(C-form)`, maybe with a scalar factor."""
    prods = []
    dims = [0, 0, 0]
    for line in path.read_text().splitlines():
        line = line.strip().replace(" ", "")
        if not line:
            continue
        line = re.sub(r"/\d+$", "", line)  # an overall scalar lands on the output side, which is free here
        parts = [p for p in _split_factors(line) if re.search(r"[abc]\d\d", p)]
        if len(parts) != 3:
            return None
        forms = [_form(p) for p in parts]
        order = []
        for f in forms:
            vs = {k[0] for k in f}
            if len(vs) != 1:
                return None
            order.append(vs.pop())
        if sorted(order) != ["a", "b", "c"]:
            return None
        byvar = dict(zip(order, forms))
        for f in byvar["a"]:
            dims[0] = max(dims[0], f[1]); dims[1] = max(dims[1], f[2])
        for f in byvar["b"]:
            dims[1] = max(dims[1], f[1]); dims[2] = max(dims[2], f[2])
        prods.append(tuple(_primitive_l1(byvar[v]) for v in "abc"))
    if not prods:
        return None
    return Scheme(path.stem, tuple(dims), prods)


def roles(s: Scheme) -> list[tuple[str, int, list[tuple[int, int]]]]:
    """Every assignment of (E, R) to two of the scheme's factors, the third being the output: (label, naive, pairs)."""
    naive = s.dims[0] * s.dims[1] * s.dims[2]
    out = []
    for x, y in [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]:
        out.append((f"{'ABC'[x]}{'ABC'[y]}", naive, [(p[x], p[y]) for p in s.prods]))
    return out


# --- the cost model ------------------------------------------------------------------------------------------------


class Model:
    def __init__(self, atoms: list[Atom], schemes: list[Scheme]):
        self.atoms = atoms
        self.opts = []  # (scheme name, role label, naive, pairs), fast ones only
        for s in schemes:
            naive = s.dims[0] * s.dims[1] * s.dims[2]
            if len(s.prods) >= naive:
                continue
            for label, n, pairs in roles(s):
                self.opts.append((s.name, label, n, tuple(pairs)))
        self.lo = lru_cache(maxsize=None)(self._lo)
        self.has = lru_cache(maxsize=None)(self._has)

    def atom(self, a: int, b: int) -> float:
        return min((f.w8 for f in self.atoms if holds(f, a, b)), default=INF)

    @staticmethod
    @lru_cache(maxsize=None)
    def cands(a: int):
        seen = {d for d in SPLIT_D if d < a}
        seen |= {-(-a // x) for x in SPLIT_D if 2 <= -(-a // x) < a}
        return tuple(sorted(seen))

    def _lo(self, L: int, a: int, b: int) -> float:
        """Least cost (eighths a leaf MAC) with at most L nested scheme levels."""
        best = self.atom(a, b)
        if L > 0:
            best = min(best, self._scheme(L, a, b, self.lo))
        for d in self.cands(a):
            best = min(best, self.lo(L, d, b) + self.lo(L, -(-a // d), b))
        for d in self.cands(b):
            best = min(best, self.lo(L, a, d) + self.lo(L, a, -(-b // d)))
        return best

    def _scheme(self, L: int, a: int, b: int, kid) -> float:
        best = INF
        for _name, _label, naive, pairs in self.opts:
            tot = 0.0
            for ua, vb in pairs:
                tot += kid(L - 1, rng(a, ua), rng(b, vb))
                if tot >= best * naive:
                    break
            best = min(best, tot / naive)
        return best

    def _has(self, L: int, a: int, b: int) -> float:
        """Least cost of a schedule with at least one fast scheme node."""
        if L == 0:
            return INF
        best = self._scheme(L, a, b, self.lo)
        for d in self.cands(a):
            hi = -(-a // d)
            best = min(best, self.has(L, d, b) + self.lo(L, hi, b), self.lo(L, d, b) + self.has(L, hi, b))
        for d in self.cands(b):
            hi = -(-b // d)
            best = min(best, self.has(L, a, d) + self.lo(L, a, hi), self.lo(L, a, d) + self.has(L, a, hi))
        return best

    def best_scheme_option(self, L: int, a: int, b: int):
        """The five cheapest scheme options at `(a, b)`, children at depth `L − 1`: (cost, scheme, roles, rank, naive)."""
        rows = []
        for name, label, naive, pairs in self.opts:
            tot = sum(self.lo(L - 1, rng(a, ua), rng(b, vb)) for ua, vb in pairs)
            rows.append((tot / naive, name, label, len(pairs), naive))
        rows.sort()
        return rows[:5]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogue", type=Path, help="directory of flip-graph `.exp` schemes (searched recursively)")
    ap.add_argument("--max-files", type=int, default=0, help="cap on catalogue files (0: all)")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--pairs", default="2:256,3:256,4:256,256:256",
                    help="catalogue (a, b) pairs, e.g. 2:128 for the R_hi product after peeling R's low bit")
    ap.add_argument("--instances", default="h200Measured,rtx6000Sheet,rtx6000MeasuredNoB1")
    ap.add_argument("--catalogue-only", action="store_true", help="skip the Strassen–Winograd base sweep")
    args = ap.parse_args(argv)
    pairs = [tuple(int(v) for v in p.split(":")) for p in args.pairs.split(",")]

    out: dict = {"model": "NaiveStrassen mirror, per leaf MAC in eighths; naive = 8", "checks": {}, "results": {}}
    ok = True

    # Check 1: the mirror reproduces the Lean's stated facts at (256, 256) with Strassen and Winograd.
    base = [STRASSEN, WINOGRAD]
    m = Model(RTX_SHEET, base)
    d1 = m.has(1, 256, 256)
    out["checks"]["lean_witness_l1_256_256"] = d1
    ok &= d1 == 12.0  # Winograd once: 8^2 + 32 = 96 at l = 1, i.e. 12 eighths a leaf MAC
    for inst in [] if args.catalogue_only else ("h200Measured", "h200Sheet", "rtx6000Sheet", "rtx6000Measured"):
        mm = Model(INSTANCES[inst], base)
        vals = [mm.has(l, 256, 256) for l in range(1, 4)]
        out["checks"][f"lean_thm1_{inst}_256_256_l1_3"] = vals
        ok &= all(v >= 8 + 32 / 8**l for l, v in zip(range(1, 4), vals))

    # The question: E's ranges against R's 256 values, Strassen and Winograd to depth `--depth`.
    for inst in [] if args.catalogue_only else INSTANCES:
        mm = Model(INSTANCES[inst], base)
        res = {}
        for a in (2, 3, 4, 256):
            res[str(a)] = {
                "lo": [mm.lo(l, a, 256) for l in range(0, args.depth + 1)],
                "has": [mm.has(l, a, 256) for l in range(1, args.depth + 1)],
            }
        out["results"][f"strassen_winograd/{inst}"] = res
        print(f"[{inst}] done", file=sys.stderr, flush=True)
        if args.json:
            args.json.write_text(json.dumps(out, indent=1, default=str))

    if args.catalogue:
        files = sorted(args.catalogue.rglob("*.exp"))
        if args.max_files:
            files = files[: args.max_files]
        schemes = []
        bad = 0
        for f in files:
            s = parse_scheme(f)
            if s is None:
                bad += 1
                continue
            schemes.append(s)
        out["catalogue"] = {"files": len(files), "parsed": len(schemes), "unparsed": bad}
        # One level of each catalogue scheme, every role assignment, each product by the best flat schedule (atoms and
        # digit splits, no further scheme). Root splits before the scheme are not searched here.
        for inst in args.instances.split(","):
            mm = Model(INSTANCES[inst], base + schemes)
            res = {}
            for a, b in pairs:
                rows = mm.best_scheme_option(1, a, b)
                res[f"{a}:{b}"] = {"flat": mm.lo(0, a, b), "best_one_level": [list(r) for r in rows]}
            out["results"][f"catalogue/{inst}"] = res
            print(f"[catalogue {inst}] done", file=sys.stderr, flush=True)
            if args.json:
                args.json.write_text(json.dumps(out, indent=1, default=str))

    out["ok"] = bool(ok)
    text = json.dumps(out, indent=1, default=str)
    if args.json:
        args.json.write_text(text)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
