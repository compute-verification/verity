"""FP-STEP: every MMA an exact-integer FP8 route issues lies inside premise (a)'s small-grid regime, on every input.

Premise (a), `Accounting.Assumptions.TS.Fp8GridExact p` over `FpGridSmall` (Security/Definitions/Accounting/Assumptions/
FpMma.lean), grants one k32 MMA step its exact sum when the accumulator's value c and the step's 32 products are integer
multiples of one 2^g (-126 <= g, g + p <= 128) and |c| + sum |t| < 2^(g+p). It is claimed at p = 24 for sm_120's QMMA
and B200's kind::f8f6f4 and at p = 14 for H200's wgmma E4M3. A route reads nothing else of the FP adder when every MMA it
issues is inside that regime on every input, by induction over its MMAs: each accumulator is then the exact sum.

For each route as its kernel issues it (its planes; the MMAs one k32 slice chains into each accumulator; the slices a
word chains; the words an accumulator carries before its restart; its start), the script computes the exact maximum
of |c| + sum |t| at every MMA, over every input and every issue order. A slice's 32 lanes draw their operand pairs
independently, and so do the slices, while one instruction's lanes share its order. So for a sign s of c the maximum is

    s C + 32 ((carried slices) F_s + (other slices of the word) G_s + U_s)

with per-pair maxima over the operand pairs (65,536 for int8): F_s of s times a slice's sum, G_s of s times the sum of
the MMAs a slice has issued, U_s of s times the sum issued before an MMA plus that MMA's |t|. Over all orders these are
closed forms in the pair's products: G_+ the sum of its positive products, U_+ that plus its most negative product's
magnitude. g is the largest exponent dividing C and every product the route can form, a witness for every input at once.
A uniform input (every lane the maximizing pair, one order) attains each maximum, and the script builds it and checks it.

It also checks what the word reads besides the MMAs:
  - every plane value is an E4M3 value, so each operand has a code;
  - the recombination is the dot product on every operand pair, and a bits route's constants cancel mod 2^32;
  - a route that reads its word from the accumulator's bits (k3, k4, k4s, k4t) keeps every partial sum inside its start's
    binade, whose ulp divides C and every product, so bits - bits(C) is (c - C) / ulp: IEEE 754 binary32's encoding;
  - a route that folds by FFMA (FADD for k4z) into the 1.5 2^23 binade (k3z, k4z, k3h, e4r) has each fold's exact result
    an integer of that binade, so the round-to-nearest fma.rn.f32 (PTX ISA: the exact a b + c, rounded once) returns it;
  - a replay of each route's word in this model (exact accumulators, FP32 encoding, round-to-nearest folds) on random and
    extreme inputs gives the int64 dot product.

Controls, in each kernel's issue order:
  - On H200 (p = 14), the routes LOWBIT-DC measured wrong there fall outside the regime for an explicit input, at that
    input's own largest grid: the binade starts k3 and k4 (fpx's k3b, k4b) and k4s, and k3z and k4z, whose k32 sums pass
    2^14 beside small terms. So premise (a) claims nothing those measurements contradict.
  - On B200, k4t's accumulator one word past its rebase spacing falls outside: RB = 7 at L = 32, and 3 at L = 64, is the
    most words premise (a) admits.

Run: `python fp_step_bounds.py [--mutations] [--verdict PATH]`. Stdlib + exact_kit only; reads nothing.
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, replace
from fractions import Fraction as Q

from exact_kit.checks import Checks

P = {"sm_120": 24, "sm_100a": 24, "sm_90a": 14}  # premise (a)'s p: QMMA, kind::f8f6f4, wgmma E4M3
LANES = 32  # an FP8 MMA's k32 slice
C0_BITS, C1_BITS, C2_BITS = 0x4B400000, 0x4B7BAB6C, 0x49E07A56  # emu8.cu's k4 binade and k3's acc1 and acc2 starts
MAGIC_BITS = C0_BITS  # the folds' 1.5 2^23


# ---- FP32 and E4M3, from their encodings --------------------------------------------------------------------------------

def fp32(bits: int) -> Q:
    e, f = (bits >> 23) & 0xFF, bits & 0x7FFFFF
    if e == 0xFF:
        raise ValueError(f"{bits:#x} is not finite")
    v = Q(f, 1 << 23) * Q(1, 1 << 126) if e == 0 else (1 + Q(f, 1 << 23)) * Q(2) ** (e - 127)
    return -v if bits >> 31 else v


def binade(v: Q) -> int:
    """E with 2^E <= |v| < 2^(E+1)."""
    a = abs(v)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Q(2) ** e > a:
        e -= 1
    while Q(2) ** (e + 1) <= a:
        e += 1
    return e


def fp32_rn(v: Q) -> int:
    """The FP32 word of v rounded to nearest, ties to even (normal range)."""
    if v == 0:
        return 0
    e = binade(v)
    m = abs(v) / Q(2) ** (e - 23)
    n, r = divmod(m.numerator, m.denominator)
    r = Q(r, m.denominator)
    if r > Q(1, 2) or (r == Q(1, 2) and n & 1):
        n += 1
    if n == 1 << 24:
        n, e = 1 << 23, e + 1
    if not -126 <= e <= 127:
        raise ValueError(f"{v} is outside FP32's normal range")
    return (1 << 31 if v < 0 else 0) | (e + 127) << 23 | (n - (1 << 23))


def e4m3_values() -> set[Q]:
    out = set()
    for c in range(256):
        e, f = (c >> 3) & 15, c & 7
        if e == 15 and f == 7:
            continue
        v = Q(f, 8) * Q(1, 64) if e == 0 else (1 + Q(f, 8)) * Q(2) ** (e - 7)
        out.add(-v if c >> 7 else v)
    return out


def v2(v) -> int | None:
    """The 2-adic valuation: v = odd 2^v2 (None for 0)."""
    if v == 0:
        return None
    if isinstance(v, int):
        n = abs(v)
        return (n & -n).bit_length() - 1
    n, d = abs(v.numerator), v.denominator
    return ((n & -n).bit_length() - 1) - ((d & -d).bit_length() - 1)


def num(v: Q):
    """v as an int when it is one (the int8 routes' sums), else the Fraction."""
    return v.numerator if v.denominator == 1 else v


# ---- the routes, as their kernels issue them ------------------------------------------------------------------------------

def limbs(x: int) -> tuple[int, int]:
    """emu8's balanced base-16 limbs: x = 16 h + l, h = (x + 8) >> 4 in [-8, 8], l in [-8, 7]."""
    h = (x + 8) >> 4
    return h, x - 16 * h


def k3_planes(x, side):  # A [4h, m, -l], B [4h', m', l']
    h, l = limbs(x)
    return (4 * h, h + l, l if side else -l)


def k4_planes(x, side):  # [16h, l] both sides
    h, l = limbs(x)
    return (16 * h, l)


def k3h_planes(x, side):  # [h, m, l] both sides
    h, l = limbs(x)
    return (h, h + l, l)


def grid_planes(x, side):  # e4r: the operand is its own plane
    return (x,)


INT8 = tuple(range(-128, 128))


def e4m3_grid(s: int, vmax: int = 16) -> tuple[Q, ...]:
    """fpx's E4M3 grid: v / s for integer |v| <= 16 (s = 1, 2, 4: the integers, halves to 8, quarters to 4)."""
    return tuple(Q(v, s) for v in range(-vmax, vmax + 1))


@dataclass(frozen=True)
class Acc:
    mmas: tuple[tuple[int, int], ...]  # (A plane, B plane) of each MMA one k32 slice chains into it, in issue order
    start: int  # FP32 bits of its start: a constant the kernel restores at each restart, or +0


@dataclass(frozen=True)
class Route:
    name: str
    source: str
    planes: object  # (operand, side) -> plane values
    domain: tuple  # operand values
    accs: tuple[Acc, ...]
    read: tuple  # ("bits", weights, offset): word += sum w bits(acc) - offset; ("fold", ((acc, coef), ...)) into 1.5 2^23
    unit: Q = Q(1)  # the word counts the dot product in units of `unit` (e4r: 1 / s^2)
    slices: int = 1  # k32 slices a word chains into each accumulator
    words: int = 1  # words an accumulator carries before its restart


K4 = Acc(((0, 0), (0, 1), (1, 0), (1, 1)), C0_BITS)
K4_BITS = ("bits", (1,), C0_BITS)
ROUTES = {r.name: r for r in [
    Route("k3", "emu8/emu8.cu (sm_120 QMMA); lowbit_dc/fpx.cu k3b", k3_planes, INT8,
          (Acc(((0, 0), (2, 2)), C1_BITS), Acc(((1, 1),), C2_BITS)), ("bits", (15, 2), 0)),
    Route("k4", "emu8/emu8.cu (sm_120 QMMA); fpx.cu k4b", k4_planes, INT8, (K4,), K4_BITS),
    Route("k3z", "lowbit_dc/egemm.cu and fpx.cu k3z", k3_planes, INT8,
          (Acc(((0, 0), (2, 2)), 0), Acc(((1, 1),), 0)), ("fold", ((0, 15), (1, 16)))),
    Route("k4z", "fpx.cu k4z (one FADD)", k4_planes, INT8, (replace(K4, start=0),), ("fold", ((0, 1),))),
    Route("k4s-L32", "egemm.cu k4s; pgemm.cu K4S, K4P and K4B at L = 32", k4_planes, INT8, (K4,), K4_BITS),
    Route("k4s-L64", "pgemm.cu K4S, K4P and K4B at L = 64 (8 MMAs a word)", k4_planes, INT8, (K4,), K4_BITS, slices=2),
    Route("k4t-L32", "pgemm.cu K4T at L = 32 (RB = 7 words a rebase)", k4_planes, INT8, (K4,), K4_BITS, words=7),
    Route("k4t-L64", "pgemm.cu K4T at L = 64 (RB = 3)", k4_planes, INT8, (K4,), K4_BITS, slices=2, words=3),
    Route("k3h", "egemm.cu eg90 and fpx.cu k3h (wgmma)", k3h_planes, INT8,
          (Acc(((0, 0),), 0), Acc(((1, 1),), 0), Acc(((2, 2),), 0)), ("fold", ((0, 240), (1, 16), (2, -15)))),
    *(Route(f"e4r-s{s}", f"fpx.cu e4r grid {i} (v / {s}, |v| <= 16)", grid_planes, e4m3_grid(s),
            (Acc(((0, 0),), 0),), ("fold", ((0, s * s),)), unit=Q(1, s * s)) for i, s in enumerate((1, 2, 4))),
]}
ISSUED = {  # each device's routes, as the kernels above issue them there
    "sm_120": ("k3", "k4"),
    "sm_100a": ("k3", "k4", "k3z", "k4z", "k4s-L32", "k4s-L64", "k4t-L32", "k4t-L64", "e4r-s1", "e4r-s2", "e4r-s4"),
    "sm_90a": ("k3h", "e4r-s1", "e4r-s2", "e4r-s4"),
}
H200_WRONG = ("k3", "k4", "k4s-L32", "k3z", "k4z")  # LOWBIT-DC: wrong on H200's 14-bit adder (lowbit_dc/README.md)


# ---- per-pair tables and the exact maxima -------------------------------------------------------------------------------

_CACHE: dict = {}


def _key(route: Route) -> tuple:
    return route.planes.__name__, route.domain, tuple(a.mmas for a in route.accs)


def table(route: Route) -> list[tuple]:
    """Per operand pair: (x, y, per accumulator the tuple of its MMAs' products on one lane)."""
    key = ("table",) + _key(route)
    if key not in _CACHE:
        pa = [route.planes(x, 0) for x in route.domain]
        pb = [route.planes(y, 1) for y in route.domain]
        _CACHE[key] = [(x, y, tuple(tuple(a[i] * b[j] for i, j in acc.mmas) for acc in route.accs))
                       for x, a in zip(route.domain, pa) for y, b in zip(route.domain, pb)]
    return _CACHE[key]


def lane(ps: tuple) -> dict:
    """One lane's extremes over every issue order of its MMAs' products ps: F (a slice's sum), G (what a slice has
    issued), U (what was issued before an MMA plus that MMA's |t|), each times + or -, and H, L (a partial sum's max, min)."""
    pos, neg = sum(p for p in ps if p > 0), sum(p for p in ps if p < 0)
    return {"F+": sum(ps), "F-": -sum(ps), "G+": pos, "G-": -neg,
            "U+": pos + max(0, -min(ps)), "U-": -neg + max(0, max(ps)),
            "H": pos if pos > 0 else max(ps), "L": neg if neg < 0 else min(ps)}


def extremes(route: Route, i: int) -> dict:
    """Accumulator i's per-pair maxima (L: its minimum), each with a pair that attains it."""
    key = ("extremes", i) + _key(route)
    if key not in _CACHE:
        best: dict = {}
        for x, y, prods in table(route):
            for k, v in lane(prods[i]).items():
                if k not in best or (v < best[k][0] if k == "L" else v > best[k][0]):
                    best[k] = (v, (x, y))
        _CACHE[key] = best
    return _CACHE[key]


def grid(route: Route, i: int, upto: int | None = None) -> int:
    """The largest g with C and every product accumulator i can form (in its first `upto` MMAs) a multiple of 2^g."""
    key = ("grid", i, upto, route.accs[i].start) + _key(route)
    if key not in _CACHE:
        vals = {v2(p) for _, _, prods in table(route) for p in prods[i][:upto] if p}
        c = fp32(route.accs[i].start)
        _CACHE[key] = min(vals | ({v2(c)} if c else set()))
    return _CACHE[key]


def worst(route: Route, i: int, p: int) -> dict:
    """The exact maximum of |c| + sum |t| over every MMA of accumulator i, every input and every issue order, against
    2^(g+p); the sign of c that attains it; and the extremes of every partial sum."""
    ex, C = extremes(route, i), fp32(route.accs[i].start)
    carried, others = (route.words - 1) * route.slices, route.slices - 1
    tot = {s: s * C + LANES * (carried * ex["F" + sg][0] + others * ex["G" + sg][0] + ex["U" + sg][0])
           for s, sg in ((1, "+"), (-1, "-"))}
    g = min(grid(route, i), 128 - p)
    hi = C + LANES * (carried * ex["F+"][0] + others * ex["G+"][0] + ex["H"][0])
    lo = C - LANES * (carried * ex["F-"][0] + others * ex["G-"][0]) + LANES * ex["L"][0]
    s = 1 if tot[1] >= tot[-1] else -1
    return {"worst": tot[s], "sign": s, "g": g, "bound": Q(2) ** (g + p), "hi": hi, "lo": lo}


def uniform_witness(route: Route, i: int, s: int) -> tuple[Q, list]:
    """The input and order that attain `worst` for sign s: every carried slice's lanes the F_s pair; every other slice
    of the word the G_s pair, with its sign-s products issued; the last slice the U_s pair, with its sign-s products
    issued and then its opposite-sign product of largest magnitude. Returns (c, [(t, lanes)]) at that MMA."""
    ex, C = extremes(route, i), fp32(route.accs[i].start)
    prods = {(x, y): pr[i] for x, y, pr in table(route)}
    sg = "+" if s > 0 else "-"
    full, part, last = (prods[ex[k + sg][1]] for k in "FGU")
    carried, others = (route.words - 1) * route.slices, route.slices - 1
    c = C + LANES * carried * sum(full) + LANES * others * sum(p for p in part if s * p > 0)
    issued = [p for p in last if s * p > 0]
    rest = [p for p in last if s * p <= 0]
    k = max(rest, key=abs) if rest else max(issued, key=abs)
    if not rest:
        issued.remove(k)
    return c + LANES * sum(issued), [(k, LANES)]


def regime(c, terms: list, p: int) -> tuple[bool, int | None, Q]:
    """FpGridSmall at p for one step with accumulator c and terms [(t, lanes)], at the step's own largest grid:
    (inside, g, |c| + sum |t|)."""
    vals = [v2(c)] if c else []
    vals += [v2(t) for t, n in terms if t and n]
    total = abs(c) + sum(n * abs(t) for t, n in terms)
    if not vals:
        return True, None, total
    g = min(min(vals), 128 - p)
    return g >= -126 and total < 2 ** (g + p), g, total


def outside_witness(route: Route, p: int):
    """An input, in the kernel's issue order (each word's slices in turn), whose step falls outside the regime at its own
    largest grid. It tries each accumulator, MMA position and sign of c in the last word of a carry window, with the
    carried slices' lanes at the pair of largest sum, and the position's lanes all one pair (A), the same with one carried
    lane at the largest odd sum (B), or 31 lanes at the position's maximizing pair and one other (C). A position whose
    grid and largest |c| + sum |t| keep every input inside is skipped. None if no input is found."""
    tab = table(route)
    carried = (route.words - 1) * route.slices + (route.slices - 1)
    for i, acc in enumerate(route.accs):
        C = num(fp32(acc.start))
        full = [sum(pr[i]) for _, _, pr in tab]
        for j in range(len(acc.mmas)):
            pre = [sum(pr[i][:j]) for _, _, pr in tab]
            t = [pr[i][j] for _, _, pr in tab]
            g = min(grid(route, i, None if carried else j + 1), 128 - p)
            most = max(s * C + LANES * (carried * max(s * v for v in full) + max(s * a + abs(b) for a, b in zip(pre, t)))
                       for s in (1, -1))
            if most < 2 ** (g + p):
                continue
            for s in (1, -1):
                f = max(range(len(tab)), key=lambda q: s * full[q])
                u = max(range(len(tab)), key=lambda q: s * pre[q] + abs(t[q]))
                base = C + LANES * carried * full[f]
                odd = [q for q in range(len(tab)) if full[q] and v2(full[q]) == 0] if carried else []
                o = max(odd, key=lambda q: s * full[q]) if odd else None
                for q in range(len(tab)):
                    cands = [("A", base + LANES * pre[q], [(t[q], LANES)]),
                             ("C", base + (LANES - 1) * pre[u] + pre[q], [(t[u], LANES - 1), (t[q], 1)])]
                    if o is not None:
                        cands.append(("B", base - full[f] + full[o] + LANES * pre[q], [(t[q], LANES)]))
                    for fam, c, terms in cands:
                        inside, gq, total = regime(c, terms, p)
                        if not inside:
                            return {"acc": i, "mma": j, "family": fam, "pair": tab[q][:2], "g": gq, "total": total}
    return None


# ---- the word: recombination and its replay -----------------------------------------------------------------------------

def identity_bad(route: Route) -> int:
    """Operand pairs whose recombination is not the dot product (in units)."""
    bad = 0
    kind = route.read[0]
    ulps = [Q(2) ** (binade(fp32(a.start)) - 23) if a.start else None for a in route.accs]
    for x, y, prods in table(route):
        if kind == "bits":
            v = sum(w * sum(pr) / u for w, pr, u in zip(route.read[1], prods, ulps))
        else:
            v = sum(coef * sum(prods[i]) for i, coef in route.read[1])
        bad += v != x * y / route.unit
    return bad


def fold_ranges(route: Route) -> list[tuple[Q, Q, bool]]:
    """Per fold, the exact range of its result minus 1.5 2^23 over every input, and whether it is always an integer."""
    out, tab = [], table(route)
    for k in range(len(route.read[1])):
        vals = [sum(coef * sum(prods[i]) for i, coef in route.read[1][:k + 1]) for _, _, prods in tab]
        out.append((LANES * min(vals), LANES * max(vals), all(Q(v).denominator == 1 for v in vals)))
    return out


def replay_word(route: Route, words: list) -> bool:
    """The route's words over `words` (per word, per slice, 32 operand pairs), in this model: each accumulator the exact
    sum, read as its FP32 word (which must hold it), the folds rounded to nearest; each word against the dot product."""
    acc = [fp32(a.start) for a in route.accs]
    prev = [a.start for a in route.accs]
    for w, word in enumerate(words):
        if route.read[0] == "fold" or w % route.words == 0:
            acc, prev = [fp32(a.start) for a in route.accs], [a.start for a in route.accs]
        dot = 0
        for pairs in word:
            for x, y in pairs:
                dot += x * y / route.unit
                for i, a in enumerate(route.accs):
                    acc[i] += sum(route.planes(x, 0)[pa] * route.planes(y, 1)[pb] for pa, pb in a.mmas)
        bits = [fp32_rn(v) for v in acc]
        if any(fp32(b) != v for b, v in zip(bits, acc)):
            return False
        if route.read[0] == "bits":
            inc = sum(wt * (b - q) for wt, b, q in zip(route.read[1], bits, prev)) % 2 ** 32
            prev = bits
        else:
            v = fp32(MAGIC_BITS)
            for i, coef in route.read[1]:
                v = fp32(fp32_rn(coef * acc[i] + v))
            inc = (fp32_rn(v) - MAGIC_BITS) % 2 ** 32
        if Q(dot).denominator != 1 or inc != int(dot) % 2 ** 32:
            return False
    return True


def replay_inputs(route: Route, rng: random.Random) -> list[list]:
    """Two carry windows of words: random pairs, then every lane each extreme pair of the route's accumulators."""
    n = 2 * route.words
    dom = route.domain
    out = [[[(rng.choice(dom), rng.choice(dom)) for _ in range(LANES)] for _ in range(route.slices)] for _ in range(n)]
    for i in range(len(route.accs)):
        for _, q in extremes(route, i).values():
            out += [[[q] * LANES for _ in range(route.slices)] for _ in range(n)]
    return out


# ---- main ---------------------------------------------------------------------------------------------------------------

def inside(route: Route, p: int) -> bool:
    return all(w["g"] >= -126 and w["worst"] < w["bound"] for w in (worst(route, i, p) for i in range(len(route.accs))))


def constants_cancel(route: Route) -> bool:
    return sum(w * a.start for w, a in zip(route.read[1], route.accs)) % 2 ** 32 == route.read[2] % 2 ** 32


def check_route(checks: Checks, route: Route, dev: str, p: int) -> list[float]:
    """The checks of a route on a device that issues it; returns each accumulator's max |c| + sum |t| over 2^(g+p)."""
    r, ratios = route, []
    for i, acc in enumerate(r.accs):
        w = worst(r, i, p)
        ratios.append(float(w["worst"] / w["bound"]))
        tag = f"{dev} {r.name} acc {i} ({len(acc.mmas) * r.slices * r.words} MMAs a restart)"
        checks.check(f"{tag}: every MMA inside FpGridSmall {p}, over every input and issue order",
                     w["g"] >= -126 and w["worst"] < w["bound"],
                     f"max |c| + sum |t| = {w['worst']} < 2^({w['g']}+{p}) = {w['bound']} ({ratios[-1]:.4f} of it)")
        c, terms = uniform_witness(r, i, w["sign"])
        total = abs(c) + sum(n * abs(t) for t, n in terms)
        checks.check(f"{tag}: a uniform input attains that maximum", total == w["worst"], f"|c| + sum |t| = {total}")
        if r.read[0] == "bits":
            E = binade(fp32(acc.start))
            ulp_ok = grid(r, i) >= E - 23
            checks.check(f"{tag}: every partial sum in its start's binade [2^{E}, 2^{E + 1}) on its ulp 2^{E - 23}, "
                         f"so the word reads bits - bits(C)",
                         ulp_ok and Q(2) ** E <= w["lo"] and w["hi"] < Q(2) ** (E + 1),
                         f"partial sums in [{w['lo']}, {w['hi']}]")
    if r.read[0] == "fold":
        rng = fold_ranges(r)
        checks.check(f"{dev} {r.name}: each fold's exact result an integer in [2^23, 2^24), so fma.rn returns it",
                     all(ok and -2 ** 22 <= lo and hi < 2 ** 22 for lo, hi, ok in rng),
                     "; ".join(f"1.5 2^23 + [{lo}, {hi}]" for lo, hi, _ in rng))
    return ratios


def main() -> int:
    checks = Checks()
    rng = random.Random(20261019)
    e4 = e4m3_values()
    checks.check("E4M3: 254 finite codes, 253 values (two zeros), max 448", len(e4) == 253 and max(e4) == 448, len(e4))
    checks.check("FP32 round to nearest even: 2^24 + 1 -> 2^24, 2^24 + 3 -> 2^24 + 4, 1.5 2^23 is 0x4B400000",
                 fp32(fp32_rn(Q(2 ** 24 + 1))) == 2 ** 24 and fp32(fp32_rn(Q(2 ** 24 + 3))) == 2 ** 24 + 4
                 and fp32_rn(Q(3 * 2 ** 22)) == C0_BITS and fp32(C0_BITS) == 3 * 2 ** 22)

    for r in ROUTES.values():
        vals = {v for x in r.domain for side in (0, 1) for v in r.planes(x, side)}
        checks.check(f"{r.name} ({r.source}): every plane value is an E4M3 value", vals <= e4,
                     f"{len(vals)} values, |v| <= {max(map(abs, vals))}")
        checks.check(f"{r.name}: the recombination is the dot product on every operand pair", identity_bad(r) == 0,
                     f"{len(table(r))} pairs")
        if r.read[0] == "bits":
            checks.check(f"{r.name}: its starts' bits cancel mod 2^32 (sum w bits(C) = the offset it subtracts)",
                         constants_cancel(r), f"{sum(w * a.start for w, a in zip(r.read[1], r.accs)) % 2 ** 32:#x}")

    ratios = {}
    for dev, names in ISSUED.items():
        for name in names:
            ratios[dev, name] = max(check_route(checks, ROUTES[name], dev, P[dev]))
            checks.check(f"{dev} {name}: the word replayed in premise (a)'s model is the dot product",
                         replay_word(ROUTES[name], replay_inputs(ROUTES[name], rng)))
    order = sorted(ratios, key=lambda k: -ratios[k])
    checks.check("the tightest routes: k3 (its acc1 starts at C1 = 16493420, 283796 below 2^24), then B200's k4t "
                 "(its accumulator carries 7 words at L = 32)",
                 {n for _, n in order[:2]} == {"k3"} and order[2] == ("sm_100a", "k4t-L32"),
                 "; ".join(f"{d} {n} {ratios[d, n]:.4f}" for d, n in order[:5]))

    k3h = max(worst(ROUTES["k3h"], i, 14)["worst"] for i in range(3))
    e4r = worst(ROUTES["e4r-s1"], 0, 14)["worst"]
    checks.check("H200: k3h's largest k32 sum is 32 * 225 = 7200 (m m') < 2^13, e4r's 32 * 16^2 = 8192 units < 2^14",
                 k3h == 7200 < 2 ** 13 and e4r == 8192 < 2 ** 14, f"{k3h}, {e4r}")

    for name in H200_WRONG:
        wit = outside_witness(ROUTES[name], 14)
        checks.check(f"control: H200 (p = 14) {name}, measured wrong there, has a step outside the regime", wit is not None,
                     None if wit is None else f"acc {wit['acc']} MMA {wit['mma']} input {wit['family']} pair "
                     f"{wit['pair']}: |c| + sum |t| = {wit['total']} >= 2^({wit['g']}+14)")
    for name, rb in (("k4t-L32", 7), ("k4t-L64", 3)):
        r = ROUTES[name]
        wit = outside_witness(replace(r, words=rb + 1), 24)
        checks.check(f"control: B200 {name} at RB = {rb + 1} words, one past the kernel's, has a step outside the regime",
                     r.words == rb and wit is not None,
                     None if wit is None else f"MMA {wit['mma']} input {wit['family']}: |c| + sum |t| = {wit['total']} "
                     f">= 2^({wit['g']}+24)")
    if checks.mutations:
        k3, k3z = ROUTES["k3"], ROUTES["k3z"]
        checks.mutation("k4t rebasing every 8 words at L = 32 leaves the regime",
                        not inside(replace(ROUTES["k4t-L32"], words=8), 24))
        checks.mutation("k4t at L = 64 rebasing every 4 words leaves the regime",
                        not inside(replace(ROUTES["k4t-L64"], words=4), 24))
        checks.mutation("k3's acc1 started at 0x4B7FF000, near the top of its binade, leaves the regime",
                        not inside(replace(k3, accs=(Acc(k3.accs[0].mmas, 0x4B7FF000), k3.accs[1])), 24))
        checks.mutation("k3z issued on H200 leaves the regime: (-l)(l') terms land on sum (4h)(4h') up to 2^15",
                        not inside(k3z, 14))
        checks.mutation("k3h's m m' accumulator started at a binade constant, as k3b's, leaves H200's regime",
                        not inside(replace(ROUTES["k3h"], accs=(ROUTES["k3h"].accs[0], Acc(((1, 1),), C2_BITS),
                                                                ROUTES["k3h"].accs[2])), 14))
        checks.mutation("at p = 24 (a wide adder) k3z on H200 would have no step outside: the control needs p = 14",
                        outside_witness(k3z, 24) is None)
        checks.mutation("k3z folded with weight 16 on acc0 is not the dot product",
                        identity_bad(replace(k3z, read=("fold", ((0, 16), (1, 16))))) > 0)
        checks.mutation("k3 with acc2 started one ulp off: its constants no longer cancel mod 2^32",
                        not constants_cancel(replace(k3, accs=(k3.accs[0], Acc(k3.accs[1].mmas, C2_BITS + 1)))))
        checks.mutation("e4r on a +-32 grid on H200 leaves the regime: 32 * 32^2 = 2^15",
                        not inside(replace(ROUTES["e4r-s1"], domain=e4m3_grid(1, 32)), 14))
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
