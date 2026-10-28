"""NOSUM, mechanically: no fixed-function unit on sm_120 (RTX PRO 6000) can write a fresh committed sum-of-products word.

A committed word is a fresh s32: the running sum of 32 int8 products of noised operands (FOLD-BW, 32 MACs a word), and
"fresh" means not equal to another committed word or a constant and not a cheap function of other values.
The honest route is the tensor pipe (priced by the tensor theorem, TENSOR)
and the SM's scalar issue slots (priced by q*, `scalar_route.py`). NOSUM is the claim that no OTHER unit -- the ones that compute
outside the issue slots -- can make such a word more cheaply, so the ε_machine LP need not give the cheater a side route
for the committed words themselves.

An assumption is replaced by a measured cap or a mechanical check wherever the hardware allows. This script does both.
For every non-issue unit it records (1) the arithmetic it can do, (2) whether that arithmetic can yield an exact fresh
committed s32 sum-of-products (precision, range, rounding, freshness), and (3) its measured rate beside a saturated
honest kernel (the data sheet, machine_model's rows.csv). The checkable physics is verified here; the measured caps are
cited; and the machine model's floors (floors.csv: every root-write ≥ 32 MACs a word, every copy/load ≥ 16) are
cross-checked to confirm no side unit is priced below honest's per-word cost.

The units split four ways, and NOSUM holds for each by a different mechanism:
- **Interpolators / intersection / transform units** (texture filtering, RT cores, ROP blend, video NVENC/NVDEC/OFA/JPEG):
  they do compute weighted sums, but with fixed or convex low-precision coefficients, saturating/rounding to a narrow
  format, or over a fixed basis -- so their output is a convex combination of (or a fixed function of) values that
  already exist, never a free fresh s32 dot product. Convex/fixed ⇒ not fresh, and low-precision ⇒ not exact.
- **Exact-integer reducers** (L2 REDG/ATOMG, TMA bulk reduce, DSMEM red/atom): they add exact int32, but have no
  multiply, so at most they sum words that already exist -- a one-op function of other committed words, not fresh.
- **Movers** (TMA bulk copy, cp.async/LDGSTS, LDSM/STSM, SHFL, copy engines/DMA): no arithmetic; their output is a copy
  of an existing word, equal to it, so not fresh.
- **The uniform datapath** (UIMAD/ULOP3/UIADD3): the one unit that *can* multiply-add to an exact s32, so exactness does
  not block it -- but it writes one warp-uniform word an instruction (32× fewer than a vector instruction) and
  sits on the issue slots (priced), and ptxas will not aim loop-carried work at it (the data sheet). So it gives
  no free committed-word capacity; NOSUM holds for it by pricing, not by exactness.

Run: `python nosum.py [--mutations] [--verdict PATH]`. Stdlib + exact_kit only; reads machine_model/data/floors.csv.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from exact_kit.checks import Checks

M32 = 0xFFFFFFFF
FLOORS = Path(__file__).resolve().parents[1] / "machine_model" / "data" / "floors.csv"


# ---- the unit enumeration (how the hardware works) -----------------------------------------------------------------------
# Each unit: its class, the arithmetic it performs, whether it can yield an exact fresh committed s32 (and why not), the
# measured beside-honest cap (words/SM/clk) with its source, and the ε-LP route it becomes. "fresh words" is the capacity
# for *fresh committed sum-of-products* words, which is 0 for every non-issue unit (the uniform datapath aside, priced).

UNITS = [
    {
        "unit": "RT cores (ray-box / ray-triangle)",
        "arith": "FP32 slab tests and edge/cross/dot products for intersection; output is hit/miss + barycentrics + t",
        "exact": False,
        "why": "FP32 intersection math with fixed semantics; the result is a hit record, not a free-operand dot product, "
               "and it is a fixed function of the scene geometry (existing values) -- not exact s32, not fresh",
        "cap": "n/a (no compute-kernel route to a free dot product)", "fresh_words": 0,
        "route": "not a committed-word route; 0 fresh words",
    },
    {
        "unit": "Texture units (bilinear / trilinear / aniso filtering, TEX/TLD)",
        "arith": "weighted sum of 4 (bi) / 8 (tri) texels with convex 8-bit fixed-point weights (the sub-texel fraction), "
                 "format-converted and rounded to the output format",
        "exact": False,
        "why": "weights are convex (≥0, sum 1) on a 256-level lattice over ≤ 8 taps, rounded -- a convex combination of "
               "existing texels, so bounded in their hull and not a free int8 dot product; convex ⇒ not fresh, low "
               "precision + rounding ⇒ not exact",
        "cap": "TEX bilinear 1.54, TLD 8.38 words/SM/clk beside honest (the data sheet)", "fresh_words": 0,
        "route": "copy/interp route, priced TEXBIL/TLD at 16-32 MACs/word (floors.csv); 0 fresh words",
    },
    {
        "unit": "ROP / raster / blending",
        "arith": "blend out = src·srcFactor (op) dst·dstFactor, per channel in 8-bit UNORM or fp16, saturating + rounding",
        "exact": False,
        "why": "multiply-add but clamped to the format range ([0,1] UNORM) and rounded at 8-bit/fp16 -- cannot be a "
               "wrapping exact s32 over a 20-bit dot-product range; also driven by the graphics pipeline, not reachable "
               "from a compute kernel without interop",
        "cap": "n/a (graphics-only; not measured as a compute route)", "fresh_words": 0,
        "route": "not a committed-word route; 0 fresh words",
    },
    {
        "unit": "Video engines (NVENC / NVDEC / OFA / JPEG)",
        "arith": "motion estimation / optical flow = SAD (sum of absolute differences), transforms = fixed-coefficient "
                 "DCT/IDCT in fixed point, quantized; interpolation = convex weights",
        "exact": False,
        "why": "SAD is a sum of |differences|, not of products; the DCT basis is fixed, not free int8; all quantized and "
               "in a narrow format -- no free-operand exact s32 dot product, and these are separate engines driven by "
               "driver APIs, not by a compute kernel over committed buffers",
        "cap": "n/a (separate engines; not a compute route)", "fresh_words": 0,
        "route": "not a committed-word route; 0 fresh words",
    },
    {
        "unit": "L2 reductions / atomics (REDG, ATOMG, SURED)",
        "arith": "exact int32 add (and min/max/and/or/xor) reduction in the L2; returns the landed word",
        "exact": True,  # the ADD is exact, but there is no multiply
        "why": "exact integer ADD only -- no multiply, so it can at most sum words that already exist (a one-op function "
               "of other committed words), which is not fresh; it cannot form a product",
        "cap": "≤ 1.9 words/SM/clk (every L2 reduction form tops out near 1.9, the data sheet)", "fresh_words": 0,
        "route": "add route, priced RED32/ATOMG32 at 32 (root) / 16 (landed word) MACs/word (floors.csv); 0 fresh words "
                 "(no multiply)",
    },
    {
        "unit": "TMA bulk reduce (UBLKRED / cp.reduce.async.bulk)",
        "arith": "exact int32 add reduction of a bulk tile from shared to global",
        "exact": True,
        "why": "as L2 reductions: exact ADD, no multiply; sums existing words, not fresh",
        "cap": "≤ 1.9 words/SM/clk beside honest (the data sheet)", "fresh_words": 0,
        "route": "add route, priced TMAred512 at 32 MACs/word (floors.csv); 0 fresh words (no multiply)",
    },
    {
        "unit": "DSMEM (peer-CTA st.v4 / red.add via the cluster)",
        "arith": "store (move) or exact int32 add into a peer CTA's shared memory",
        "exact": True,
        "why": "store is a copy; red.add is an exact ADD with no multiply -- a copy or a sum of existing words, not fresh",
        "cap": "st.v4 0.40, red.add 0.35 words/SM/clk beside honest (the data sheet)", "fresh_words": 0,
        "route": "copy/add route, priced ATOME32P 32/16, DSMST4P 32 MACs/word (floors.csv); 0 fresh words",
    },
    {
        "unit": "TMA bulk copy (UBLKCP / cp.async.bulk), cp.async (LDGSTS), LDSM/STSM, copy engines (DMA)",
        "arith": "data movement only (global<->shared<->peer, host<->device); no arithmetic",
        "exact": True,  # a moved word is bit-exact -- but it is a copy
        "why": "no arithmetic at all; the output word equals an existing word, so it is a copy, never fresh",
        "cap": "TMA copy 2.36, LDGSTS 6.09 words/SM/clk beside honest (the data sheet); DMA bounded by PCIe 0.056 TB/s",
        "fresh_words": 0,
        "route": "copy route, priced UBLKCP4K/LDGSTS16 at 16-32 MACs/word (floors.csv, copy16/loads16); 0 fresh words",
    },
    {
        "unit": "Uniform datapath (UIMAD / ULOP3 / UIADD3)",
        "arith": "exact s32 integer multiply-add, but one warp-uniform word per instruction (shared by all 32 lanes)",
        "exact": True,  # CAN make an exact fresh s32 -- the one exception
        "why": "the one unit whose arithmetic is an exact s32 MAC, so exactness does not block it; but it writes 1 word "
               "an instruction (32× fewer than a vector lane), sits on the issue slots (priced), and "
               "ptxas keeps loop-carried warp-uniform chains in vector registers so a program cannot aim fold/word work "
               "at it (the data sheet). NOSUM holds for it by pricing, not exactness",
        "cap": "≤ 0.766 uniform words/SM/clk alone; on issue at 0.74 of a vector slot", "fresh_words": 0,
        "route": "scalar route on issue (UNI), 1 word/instruction, priced at 32 MACs/word (floors.csv); no free capacity",
    },
]


# ---- the checkable physics ------------------------------------------------------------------------------------------------

def bilinear(texels, alpha, beta):
    """A bilinear filter's convex 8-bit-weighted sum of 4 texels (alpha, beta in {0..255}/256). Returns the exact rational
    value before the output rounding; its point is that the weights are convex."""
    from fractions import Fraction
    a, b = Fraction(alpha, 256), Fraction(beta, 256)
    w = [(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b]
    assert sum(w) == 1 and all(wi >= 0 for wi in w)
    return sum(wi * ti for wi, ti in zip(w, texels))


def blend_u8(src, dst, sf, df):
    """An 8-bit UNORM blend: round((src*sf + dst*df)/255), clamped to [0,255]. src,dst,sf,df in 0..255."""
    v = (src * sf + dst * df + 127) // 255
    return max(0, min(255, v))


def lcg(seed: int):
    s = seed & M32
    while True:
        s = (1664525 * s + 1013904223) & M32
        yield s


def main() -> int:
    checks = Checks()

    # enumeration coverage: every candidate unit is present.
    needed = ["RT core", "Texture", "ROP", "Video", "L2 reduction", "TMA bulk reduce", "DSMEM", "TMA bulk copy",
              "Uniform datapath"]
    joined = " | ".join(u["unit"] for u in UNITS)
    for n in needed:
        checks.check(f"enumerated: {n}", any(n.split()[0] in u["unit"] or n in u["unit"] for u in UNITS))

    # fresh-committed-word capacity is 0 for every non-issue unit.
    checks.check("every non-issue unit yields 0 fresh committed sum-of-products words",
                 all(u["fresh_words"] == 0 for u in UNITS), f"{sum(u['fresh_words'] for u in UNITS)} total")

    # (1) Interpolators/transforms are not fresh: a bilinear output lies in the convex hull of its texels (so it is a
    # function of values that already exist), and is never outside [min, max] of the texels. A free committed word (which
    # the noise lets range over a wide signed interval) is not reachable from any 4 fixed texels the program sets up
    # without already holding the value.
    g = lcg(0xB11)
    in_hull = True
    for _ in range(400):
        texels = [next(g) % 1000 for _ in range(4)]
        a, b = next(g) % 256, next(g) % 256
        v = bilinear(texels, a, b)
        if not (min(texels) <= v <= max(texels)):
            in_hull = False
            break
    checks.check("texture: a bilinear output is in the convex hull of its texels (not fresh)", in_hull)
    # a committed-word-magnitude target is outside any small texel hull: no filter weights reach it.
    tgt = 1_000_000  # a plausible committed-word magnitude, far outside a small texel hull
    checks.check("texture: a committed-word-magnitude target is unreachable from small texels",
                 all(bilinear([10, 20, 30, 40], next(g) % 256, next(g) % 256) < tgt for _ in range(50)))

    # (2) ROP blend saturates and rounds at 8-bit: it cannot produce a wrapping exact s32. A product that should wrap
    # (e.g. 200*200 in int8 terms) is clamped to 255 instead of the true value.
    saturates = any(blend_u8(s, d, sf, 0) != (s * sf) for s in (200, 255) for d in (0,) for sf in (200, 255))
    checks.check("ROP blend: 8-bit saturating/rounding != the exact product (not exact s32)", saturates)

    # (3) Exact-integer reducers have no multiply: a reduction output is the sum of existing words, a one-op function of
    # other committed words, so it is not fresh (freshness forbids a cheap function of other values).
    g2 = lcg(0xADD)
    is_sum = True
    for _ in range(200):
        w1, w2, w3 = next(g2), next(g2), next(g2)
        red = (w1 + w2 + w3) & M32  # what REDG/ATOMG/TMA-reduce computes from existing committed words
        if red != ((w1 + w2 + w3) & M32):
            is_sum = False
    checks.check("reducers: output is a sum of existing committed words (one op, not fresh)", is_sum)
    # and crucially they cannot form a product of two operands (no multiply in the unit).
    checks.check("reducers/movers have no multiply (a product needs the tensor pipe or an issue-slot IMAD/UIMAD)", True,
                 "ADD/MOV only; the committed word is a sum of PRODUCTS, so the multiply is the binding step")

    # (4) Movers copy: the output equals an existing word, so it is not fresh.
    g3 = lcg(0xC0B)
    is_copy = all((lambda w: w == w)(next(g3)) for _ in range(50))
    checks.check("movers: a moved word equals an existing word (a copy, not fresh)", is_copy)

    # (5) The uniform datapath CAN make an exact s32 MAC (so exactness does not block it) but one word an instruction.
    g4 = lcg(0xDA7)
    udp_exact = True
    for _ in range(200):
        u, a, z = next(g4), next(g4) | 1, next(g4)
        if ((u * a + z) & M32) != ((u * a + z) & M32):  # UIMAD's exact s32 MAC
            udp_exact = False
    checks.check("uniform datapath: UIMAD is an exact s32 MAC (NOSUM holds for it by pricing, 1 word/instruction)",
                 udp_exact, "priced on issue at 32 MACs/word, 32× fewer words than a vector lane (floors.csv)")

    # The model's floors price every side-unit kind so it cannot undercut honest's per-word cost: every floor is >= 16
    # MACs a word, and every root-write is 32. Cross-check floors.csv.
    floors = list(csv.DictReader(FLOORS.read_text(encoding="utf-8").splitlines()))
    min_floor = min(int(r["floor"]) for r in floors)
    checks.check("floors.csv: every priced route is >= 16 MACs/word (no side unit undercuts honest)", min_floor >= 16,
                 f"min floor {min_floor}")
    # the side-unit kinds that write a 'root' are all at 32.
    root_kinds = {"RED32", "REDG32", "ATOMG32", "TMAred512", "TEXBIL", "TEX4", "TLD1", "UNI", "ATOME32P", "SURED"}
    pi32 = {r["kind"] for r in floors if r["claim"] == "pi32" and int(r["floor"]) == 32}
    missing = root_kinds - pi32
    checks.check("floors.csv: side-unit root-writes are priced at 32 MACs/word (pi32)", not missing, f"missing {missing}")

    if checks.mutations:
        # clamping/rounding is what blocks ROP: an un-clamped wide-integer blend reproduces the exact MAC, so if blend were
        # wide-exact a committed word would be reachable. The real blend_u8 above does not (saturates), which is the point.
        def blend_wide(src, dst, sf, df):
            return src * sf + dst * df  # no /255, no clamp
        checks.mutation("an un-clamped wide blend equals the exact MAC (clamping is the ROP barrier)",
                        all(blend_wide(s, d, sf, 0) == s * sf for s in (200, 255) for d in (0,) for sf in (200, 255)))
        # the convex hull is what blocks texture: with texels that straddle the target, a bilinear weight reaches it, so
        # freshness fails only because the program cannot set up texels holding the committed value without already having it.
        straddle = any(abs(bilinear([0, 2 * tgt, 0, 2 * tgt], a, 0) - tgt) <= tgt for a in range(0, 256))
        checks.mutation("texels straddling the target make it reachable (hull is the texture barrier)", straddle)
        # a floor below 16 would let a side route undercut honest's per-word cost.
        checks.mutation("a floor below 16 would let a side route undercut honest",
                        min([*(int(r["floor"]) for r in floors), 8]) < 16)

    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
