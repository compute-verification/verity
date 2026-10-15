"""Which sm_120 MMA operand formats hold a small noise alphabet exactly, and the cheapest exact decomposition otherwise.

Exact (fractions), CPU only. For each alphabet and each operand format it asks whether some scale s puts every alphabet
value on the format's grid (alphabet ⊆ s·V), for three scale classes: a power of two (UE8M0, MXFP4/MXFP6/MXFP8), a UE4M3
value (NVFP4's block scale) and any positive real (a per-tensor alpha in the epilogue, on top of either). A block scale
can differ between blocks, but a random n×n noise matrix puts every alphabet value in almost every block of 16 or 32,
so one scale must hold the whole alphabet.

For an alphabet no FP4 scale holds, it checks the decomposition bounds by exhaustion over the alphabet: an entrywise
decomposition ab = Σ_i f_i(a) g_i(b) with r terms costs r FP4 products (r/2 of an int8 product in time). r = 1 is
representability; r = 2 is impossible (rank argument, `rank2_impossible`); r = 3 is the Karatsuba split built here.

    python benchmarks/d4_attacks/represent.py [--json OUT]
"""
from __future__ import annotations

import argparse
import itertools
import json
from fractions import Fraction as F


def minifloat(e_bits: int, m_bits: int, bias: int, finite_top: bool = True, nan_top_mantissa: bool = False) -> set[F]:
    """Non-negative finite values of a sign-magnitude minifloat (subnormals included)."""
    vals = set()
    for e in range(1 << e_bits):
        for m in range(1 << m_bits):
            if nan_top_mantissa and e == (1 << e_bits) - 1 and m == (1 << m_bits) - 1:
                continue  # E4M3's S.1111.111 is NaN
            if not finite_top and e == (1 << e_bits) - 1:
                continue  # E5M2's top exponent is inf/NaN
            if e == 0:
                v = F(m, 1 << m_bits) * F(2) ** (1 - bias)
            else:
                v = (1 + F(m, 1 << m_bits)) * F(2) ** (e - bias)
            vals.add(v)
    return vals


FORMATS = {
    "E2M1 (FP4)": minifloat(2, 1, 1),
    "E2M3 (FP6)": minifloat(2, 3, 1),
    "E3M2 (FP6)": minifloat(3, 2, 3),
    "E4M3 (FP8)": minifloat(4, 3, 7, nan_top_mantissa=True),
    "E5M2 (FP8)": minifloat(5, 2, 15, finite_top=False),
    "int8": {F(i) for i in range(128)},
}
UE4M3 = sorted(v for v in FORMATS["E4M3 (FP8)"] if v > 0)
UE8M0 = [F(2) ** e for e in range(-127, 128)]

ALPHABETS = {
    "{±1}": [-1, 1],
    "{±1,±3}": [-3, -1, 1, 3],
    "{±5,±7}": [-7, -5, 5, 7],
    "{-2..2}": list(range(-2, 3)),
    "{-4..4}": list(range(-4, 5)),
    "{-8..8}": list(range(-8, 9)),
}


def scales_holding(alphabet, grid: set[F]) -> list[F]:
    """Every s > 0 with |alphabet| ⊆ s·grid: s = max|a| / v for each nonzero grid value v, kept if it fits all."""
    mags = sorted({F(abs(a)) for a in alphabet if a != 0})
    top = mags[-1]
    out = []
    for v in grid:
        if v == 0:
            continue
        s = top / v
        if all(m / s in grid for m in mags):
            out.append(s)
    return sorted(out)


def rep_row(alphabet, grid):
    any_s = scales_holding(alphabet, grid)
    pow2 = [s for s in any_s if s.numerator & (s.numerator - 1) == 0 and s.denominator & (s.denominator - 1) == 0]
    ue4m3 = [s for s in any_s if s in set(UE4M3)]
    return {"any_real": [str(s) for s in any_s], "pow2_ue8m0": [str(s) for s in pow2], "ue4m3": [str(s) for s in ue4m3]}


def ratio_proof_57():
    """{±5,±7} on E2M1: one scale must map 5 and 7 to grid values v5, v7 with v7/v5 = 7/5; list every ratio."""
    g = sorted(v for v in FORMATS["E2M1 (FP4)"] if v > 0)
    ratios = sorted({b / a for a in g for b in g if b > a})
    return {"e2m1_positive": [str(v) for v in g], "ratios_gt_1": [str(r) for r in ratios], "has_7_5": F(7, 5) in ratios}


def rank2_impossible(alphabet, grid):
    """r = 2 needs F Gᵀ = a bᵀ with F, G two columns each: rank F + rank G <= 3 (Sylvester), so one has rank <= 1 and its
    nonzero columns are multiples of the alphabet itself, which no scale holds.  Returns the premise that makes it so."""
    return not scales_holding(alphabet, grid)


def karatsuba(alphabet, grid):
    """Find a = w·h + l with h, l, h + l all E2M1-representable (scale 1 or 1/2 grids), w > 0, then
    ab = (w² - w)·hh' + w·(h+l)(h'+l') + (1 - w)·ll', three FP4 products; verify over every (a, b) pair."""
    reps = {}
    for w in range(1, 9):
        ok = True
        choice = {}
        for a in alphabet:
            found = None
            for h in range(-8, 9):
                l = a - w * h
                if F(abs(h)) in grid and F(abs(l)) in grid and F(abs(h + l)) in grid:
                    found = (h, l)
                    break
            if found is None:
                ok = False
                break
            choice[a] = found
        if ok:
            reps[w] = choice
    for w, choice in sorted(reps.items()):
        for a, b in itertools.product(alphabet, repeat=2):
            h, l = choice[a]
            h2, l2 = choice[b]
            assert (w * w - w) * h * h2 + w * (h + l) * (h2 + l2) + (1 - w) * l * l2 == a * b
        return {"w": w, "split": {str(a): list(v) for a, v in choice.items()},
                "coefficients": [w * w - w, w, 1 - w]}
    return None


def max_dot(alphabet, k):
    m = max(abs(a) for a in alphabet)
    return k * m * m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    out = {"alphabets": {}}
    e2m1 = FORMATS["E2M1 (FP4)"]
    for name, alph in ALPHABETS.items():
        row = {"values": alph, "formats": {}}
        for fname, grid in FORMATS.items():
            row["formats"][fname] = rep_row(alph, grid)
        row["fp4_holds"] = bool(row["formats"]["E2M1 (FP4)"]["any_real"])
        row["max_abs_dot_k8192"] = max_dot(alph, 8192)
        row["fp32_exact_k8192"] = max_dot(alph, 8192) < 2 ** 24
        if not row["fp4_holds"]:
            row["rank2_impossible"] = rank2_impossible(alph, e2m1)
            row["karatsuba3"] = karatsuba(alph, e2m1)
        out["alphabets"][name] = row
    out["proof_57"] = ratio_proof_57()
    hdr = ["alphabet"] + list(FORMATS) + ["FP4 decomposition"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for name, row in out["alphabets"].items():
        cells = [name]
        for fname in FORMATS:
            r = row["formats"][fname]
            if not r["any_real"]:
                cells.append("no")
            else:
                tags = []
                if "1" in r["any_real"]:
                    tags.append("s=1")
                if r["pow2_ue8m0"]:
                    tags.append("UE8M0 " + r["pow2_ue8m0"][0])
                if r["ue4m3"]:
                    tags.append("UE4M3 " + r["ue4m3"][0])
                if not tags:
                    tags.append("alpha " + r["any_real"][0])
                cells.append(", ".join(tags))
        if row["fp4_holds"]:
            cells.append("1 product (direct)")
        else:
            k3 = row["karatsuba3"]
            cells.append(f"r=2 impossible; r=3 Karatsuba w={k3['w']} coeffs {k3['coefficients']}" if k3 else "none found")
        print("| " + " | ".join(cells) + " |")
    p = out["proof_57"]
    print(f"\nE2M1 positive values {p['e2m1_positive']}; ratios >1 {p['ratios_gt_1']}; 7/5 among them: {p['has_7_5']}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
