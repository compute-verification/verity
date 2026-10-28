"""FP8-g1's and W4A8-g1's committed values against the int8 records' operand ranges, and the served identity.

fpx_design credits g1 at the int8 record's ε because the one int8 GEMM
of R + b is the record's transcript. That needs every operand g1 commits or feeds the MMA to lie where the record's own
operands do, and the served output to be that GEMM less its rank-one terms. This checks both, on each record's ranges
as main's kernels state them:

- Dither-2S (B200, H200): both MMA operands value plus a fair bit, in s8.
- H1 = co32c-u16 (RTX PRO 6000; `edited_gemm/cgemm.cu` "per-token scales, +-110 weights, the narrow U",
  `cgemm_ref.py` `u_rows16`): activations X = V + b with V in [-126, 126]; weights W' = W + c + U, |W| <= 110,
  U in {-16..-1, 1..16}; W' in s8.
- co32c, Dither-C's 7-bit form (`cgemm_ref.py`'s non-h1 branch, `u_rows`): U = -64 + beta . (1, 2, 4, 8, 16, 32, 65),
  W' = W + c + U in s8. This is the form fpx_design's sm_120 arms read as H1.

For each, the largest E4M3 integer and the largest integer UE4M3 block scale whose g1 operands fit, and whether g1's
grids are "all even above 16" (fpx_design's card-scope remark).
Run: `python benchmarks/pouw/rt2_fp/g1_check.py [--json OUT]`.
"""
from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction

S8 = (-128, 127)


def e4m3_values() -> list[Fraction]:
    """Every finite non-negative E4M3 value (bias 7, no infinities, S.1111.111 NaN)."""
    out = set()
    for e in range(16):
        for m in range(8):
            if e == 15 and m == 7:
                continue
            out.add(Fraction(m, 8) * Fraction(2) ** -6 if e == 0 else (1 + Fraction(m, 8)) * Fraction(2) ** (e - 7))
    return sorted(out)


E4M3_INTS = [int(v) for v in e4m3_values() if v.denominator == 1]
E2M1X2 = [0, 1, 2, 3, 4, 6, 8, 12]
U16 = [u for u in range(-16, 17) if u != 0]
U64 = sorted({-64 + sum(w * ((beta >> p) & 1) for p, w in enumerate((1, 2, 4, 8, 16, 32, 65))) for beta in range(128)})


def grid_e4m3(top: int) -> list[int]:
    pos = [v for v in E4M3_INTS if v <= top]
    return sorted({s * v for v in pos for s in (1, -1)})


def grid_w4(scale_top: int) -> list[int]:
    return sorted({s * m * q for q in E2M1X2 for m in range(1, scale_top + 1) for s in (1, -1)})


def within(lo: int, hi: int, rng: tuple[int, int]) -> bool:
    return rng[0] <= lo and hi <= rng[1]


def operand_ranges(record: str, side: str, grid: list[int]) -> tuple[int, int]:
    """The MMA operand's range when the committed value runs over `grid` and the record adds its coins."""
    lo, hi = min(grid), max(grid) + 1
    if record in ("h1", "co32c") and side == "weights":
        u = U16 if record == "h1" else U64
        lo, hi = lo + min(u), hi + max(u)
    return lo, hi


def committed_ok(record: str, side: str, grid: list[int]) -> bool:
    if record == "h1" and side == "weights":
        return within(min(grid), max(grid), (-110, 110))
    if record in ("h1", "co32c") and side == "activations":
        return within(min(grid), max(grid), (-126, 126))
    return True


def fits(record: str, side: str, grid: list[int]) -> bool:
    return committed_ok(record, side, grid) and within(*operand_ranges(record, side, grid), S8)


def largest(record: str, side: str, family: str) -> int:
    tops = E4M3_INTS if family == "e4m3" else list(range(1, 128))
    make = grid_e4m3 if family == "e4m3" else grid_w4
    return max(t for t in tops if fits(record, side, make(t)))


def odd_above_16(grid: list[int]) -> list[int]:
    return sorted(v for v in grid if v > 16 and v % 2)


def served_identity(rng: random.Random, trials: int, k: int) -> dict:
    """sum_k (X - 1/2)(W - 1/2) = sum_k X W - (sum_k X + sum_k W) / 2 + k / 4, X = R_x + b, W = R_w + c, exactly; and the
    s32 accumulator's bound at the largest k it holds."""
    gx, gw = grid_e4m3(120), grid_e4m3(120)
    bad = 0
    for _ in range(trials):
        x = [rng.choice(gx) + rng.getrandbits(1) for _ in range(k)]
        w = [rng.choice(gw) + rng.getrandbits(1) for _ in range(k)]
        served = sum((Fraction(a) - Fraction(1, 2)) * (Fraction(b) - Fraction(1, 2)) for a, b in zip(x, w))
        gemm = sum(a * b for a, b in zip(x, w))
        bad += served != gemm - Fraction(sum(x) + sum(w), 2) + Fraction(k, 4)
    top = max(abs(v) for v in range(-120, 122)) ** 2
    return {"trials": trials, "k": k, "mismatches": bad, "word_max_per_k": top, "k_max_s32": (2 ** 31 - 1) // top}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args()
    out: dict = {"records": {}}
    claims = {"FP8-g1": ("e4m3", 120, 120), "FP8-g1 on sm_120 (FPX-DESIGN)": ("e4m3", 120, 60),
              "W4A8-g1": ("w4", None, 10), "W4A8-g1 on sm_120 (FPX-DESIGN)": ("w4", None, 5)}
    print("record       claim                              side         operand range   fits")
    for record in ("dither-2s", "h1", "co32c"):
        rec = out["records"][record] = {"largest": {}, "claims": {}}
        for side in ("activations", "weights"):
            for family in ("e4m3", "w4"):
                if family == "w4" and side == "activations":
                    continue
                rec["largest"][f"{side} {family}"] = largest(record, side, family)
        for name, (family, act_top, w_top) in claims.items():
            grids = {"weights": grid_e4m3(w_top) if family == "e4m3" else grid_w4(w_top)}
            if act_top is not None:
                grids["activations"] = grid_e4m3(act_top)
            for side, grid in grids.items():
                ok = fits(record, side, grid)
                rec["claims"][f"{name} {side}"] = ok
                print(f"{record:12} {name:34} {side:12} {str(operand_ranges(record, side, grid)):15} {ok}")
    for record, rec in out["records"].items():
        print(f"{record}: largest E4M3 integer (activations, weights) "
              f"{rec['largest']['activations e4m3']}, {rec['largest']['weights e4m3']}; "
              f"largest W4A8 block scale {rec['largest']['weights w4']}")
    odd = {"FP8-g1 (E4M3 integers to 120)": odd_above_16(grid_e4m3(120)),
           "W4A8-g1 scales to 10": odd_above_16(grid_w4(10)), "W4A8-g1 scales to 5": odd_above_16(grid_w4(5)),
           "W4A8-g1 scales to 9": odd_above_16(grid_w4(9))}
    out["odd_above_16"] = odd
    for name, vals in odd.items():
        print(f"odd committed values above 16, {name}: {vals or 'none'}")
    out["served_identity"] = served_identity(random.Random(20261019), 2000, 32)
    print("served identity:", out["served_identity"])
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
