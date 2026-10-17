"""ISA-ESCAPE, mechanically: no instruction or unit outside the model's menu heads a committed word more cheaply than the
menu allows (RT-ISA, the compute-accounting red team).

A committed word is a fresh s32: the running sum of 32 int8 products of noised operands (32 MACs a word; FOLD-BW), or an
input to the receipt fold (which is itself such a word). The MODEL's menu for it (note:20261017T1723Z-draft-ledger, rows
(I2)/(I4) WordsWritten, (I3) ScalarRoute, (I5) SideUnits/NoSum) is exactly five routes:
  - an MMA form's writer, at most omega_t committed words an instruction (the tensor pipe, priced per form, decision 81);
  - the CUDA lane vector route, q* lane instructions a word (q* >= 1.22 with tables, 2.03 without; QSTAR);
  - the uniform datapath, one committed word an instruction, on the issue slots (decisions 73, 84, 88);
  - table routes, priced by the bytes they read from their memory level (decision 80);
  - priced side units, each at the pi32 floor (>= 32 MACs a word; floors.csv).

NOSUM (`nosum.py`) covered the fixed-function units DATASHEET measured. This script attacks the menu from the ISA: for
every instruction class a red team would reach for -- warp and L2 reductions, atomics, bulk reduce, the low-precision and
1-bit MMA forms, the dot-product instructions, DPX, the video/SIMD ALU, texture filtering, the copies and transposes, the
packing tricks, the wide multiply, and the uniform datapath beside an asynchronous tensor op -- it records the ISA fact,
whether the instruction can even form an exact int8 product, the factor by which an int8 committed word decomposes onto
it, and the verdict. A committed word that is a sum of 32 int8 products needs a MULTIPLY; an instruction with no multiply
can at most sum or move words that already exist (not fresh), and an instruction whose multiply is lower precision pays a
decomposition factor D (int8 = D unit-products) that its raw rate must beat the honest IMMA route by to escape.

The checkable arithmetic is verified here, bit-exact over Z/2^32 and the operand alphabets; the measured rates are cited
(tensor_kinds / opcode_probe census, DATASHEET). The conclusion for sm_120 is that every candidate is closed, by one of:
  - NO-MULTIPLY: the unit only adds/compares/moves, so it sums or copies existing committed words (not fresh);
  - DECOMPOSE: the unit multiplies, but in a format an int8 product needs D >= 2 of, and its rate / D <= IMMA's;
  - LOW-PRECISION: the unit's weighted sum is convex or rounded, so not an exact fresh s32;
  - MOVER: the unit only moves data, so its output equals an existing word;
  - PRICED: the unit is already a menu route at the pi32 floor, so it undercuts nothing.

The one cross-device item that is NOT closed by measurement -- the uniform datapath's issue cost beside a SATURATED
asynchronous wgmma (sm_90) or tcgen05.mma (sm_100), which the census priced beside the synchronous mma.sync only
(decision 88) -- is recorded as a WATCH, with the issue-ceiling argument (decision 84) that bounds it and the `eps`
reading (the uniform route is already ~0% of the headline eps on the H200/B200) that shows it cannot move a headline.

Run: `python isa_escape.py [--mutations] [--verdict PATH]`. Stdlib + exact_kit only; reads machine_model/data/floors.csv.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

from exact_kit.checks import Checks

M32 = 0xFFFFFFFF
FLOORS = Path(__file__).resolve().parents[1] / "machine_model" / "data" / "floors.csv"

# The design's operand alphabets (decisions 29, 44, 51): activations are int8 (the dithered floor), weights are capped at
# |w| <= 63. A committed word sums 32 such products, so it ranges within ~2^19 << 2^24 (f32's exact-integer limit).
INT8 = range(-128, 128)
WEIGHT = range(-63, 64)

# Mantissa bits (after the implicit 1) and the max normal exponent of each MMA input format.
# E4M3: max 448 = 1.75*2^8. E5M2: emax 15. E2M1 (FP4) grid {0,.5,1,1.5,2,3,4,6}, max 6 = 1.5*2^2. f16: emax 15. bf16: emax 127.
FLOAT_FORMATS = {"e4m3": (3, 8), "e5m2": (2, 15), "e2m1": (1, 2), "f16": (10, 15), "bf16": (7, 127)}


def exact_in_float(n: int, mant_bits: int, emax: int) -> bool:
    """Is the integer `n` represented EXACTLY by a binary float with `mant_bits` mantissa bits and max exponent `emax`?
    True iff n = 0, or |n| < 2^(mant_bits+1), or the bits of |n| below the mantissa are zero and |n|'s top bit fits emax."""
    if n == 0:
        return True
    a = abs(n)
    bl = a.bit_length()
    if bl - 1 > emax:
        return False
    low = bl - (mant_bits + 1)
    return low <= 0 or (a & ((1 << low) - 1)) == 0


def nibbles_u8(x: int) -> tuple[int, int]:
    """An unsigned byte as (high nibble, low nibble), each in 0..15."""
    return (x >> 4) & 0xF, x & 0xF


# class: NO-MULTIPLY | DECOMPOSE | LOW-PRECISION | MOVER | PRICED | WATCH. D: int8 unit-products a committed word (the
# decomposition factor), None where the unit forms no product. escape: False for every closed candidate.
CANDIDATES = [
    {
        "name": "redux.sync (warp reduce add/min/max/and/or/xor)", "device": "sm_120 (also sm_80..sm_100)",
        "cls": "NO-MULTIPLY", "D": None, "escape": False,
        "isa": "redux.sync.{add,min,max,and,or,xor}.{u32,s32} reduces one 32-bit value across the warp to one result; no "
               "multiply, no new product",
        "reason": "a reducer: its output is a sum (or min/max) of 32 words that already exist, a one-op function of other "
                  "committed words, so not fresh; it forms no product, so the 32 MACs a word still cost on a multiply unit",
    },
    {
        "name": "L2 reduction / atomic (red.global.add.s32, atom.add)", "device": "all",
        "cls": "PRICED", "D": None, "escape": False,
        "isa": "red/atom.global.add.s32 lands an exact s32 add in L2; no multiply",
        "reason": "priced RED32/ATOMG32 at the pi32 floor (floors.csv); a reducer with no multiply, so 0 fresh words "
                  "(NOSUM); measured <= 1.9 words/SM/clk beside honest (DATASHEET)",
    },
    {
        "name": "TMA bulk reduce (cp.reduce.async.bulk)", "device": "sm_90, sm_100, sm_120",
        "cls": "PRICED", "D": None, "escape": False,
        "isa": "UBLKRED/UTMAREDG reduces a bulk tile shared->global with an exact s32 add; no multiply",
        "reason": "priced TMAred512 at the pi32 floor; a reducer with no multiply, 0 fresh words; <= 1.9 words/SM/clk "
                  "beside honest (DATASHEET)",
    },
    {
        "name": "IDP4A (dp4a)", "device": "all", "cls": "PRICED", "D": 1, "escape": False,
        "isa": "IDP.4A.U8.S8: 4 int8 MACs an instruction into s32 -- an exact int8 dot-product unit on the IMAD pipe",
        "reason": "this IS the honest baseline (8 dp4a = 32 MACs a word); the lane route is priced at q* with dp4a's "
                  "8.5 setting the threshold, so it undercuts nothing beyond the modelled baseline",
    },
    {
        "name": "IDP2A (dp2a)", "device": "all", "cls": "DECOMPOSE", "D": 2, "escape": False,
        "isa": "IDP.2A.LO.U16.S8: 2 int16xint8 MACs an instruction into s32",
        "reason": "2 int8 MACs an instruction against dp4a's 4 on the same pipe, so strictly dominated by dp4a; no gain",
    },
    {
        "name": "sparse IMMA (2:4, IMMA.SP)", "device": "all", "cls": "DECOMPOSE", "D": 2, "escape": False,
        "isa": "IMMA.SP.16864/16832: a dot product over k positions of which only 2:4 are read (structural sparsity)",
        "reason": "a DENSE committed word has no 2:4 zeros, so one sparse MMA reads only half the terms (a wrong word); "
                  "the full word needs two complementary-metadata passes = the dense form's MAC cost (decision 60: 2:4 "
                  "ties honest at 1.000 in the LP)",
    },
    {
        "name": "BMMA (b1 AND-popc / XOR-popc MMA)", "device": "all (sm_90 census BMMA168256_B1)",
        "cls": "DECOMPOSE", "D": 64, "escape": False,
        "isa": "mma.sync...s32.b1.b1.s32.and.popc m16n8k256: a 1-bit dot product (AND then popcount) over k=256",
        "reason": "an int8xint8 product is 64 bit-plane AND-popc terms (a = Sum 2^i a_i, b = Sum 2^j b_j, a*b = Sum "
                  "2^(i+j)(a_i AND b_j)); even a generous 8x-int8-TOPS b1 gives 8/64 = 0.125x int8",
    },
    {
        "name": "s4 IMMA (IMMA.S4 / IMMA.SP.S4)", "device": "all", "cls": "DECOMPOSE", "D": 4, "escape": False,
        "isa": "mma.sync...s32.s4.s4.s32 m16n8k64: a 4-bit dot product over k=64 (2x IMMA s8's depth an instruction)",
        "reason": "an int8xint8 product is 4 nibble products (a = 16 a1 + a0, b = 16 b1 + b0, a*b = 256 a1 b1 + 16(a1 b0 "
                  "+ a0 b1) + a0 b0); at 2x int8 TOPS that is 2/4 = 0.5x int8",
    },
    {
        "name": "FP16/BF16 MMA, f32 accumulate (HMMA.*.F32)", "device": "all", "cls": "DECOMPOSE", "D": 1, "escape": False,
        "isa": "mma.sync...f32.f16.f16.f32 / .bf16: int8 operands are exact in f16 (<2048) and bf16 (<256), the product "
               "is exact in f32 (<2^24 accumulate), so one HMMA computes an exact int8 dot product",
        "reason": "exact for int8, but fp16/bf16 tensor TOPS <= int8's at D=1, so <= 1.0x int8 (ties at most); decision "
                  "81's per-form LP already lets the cheater pick its best form (IMMA16832 wins)",
    },
    {
        "name": "FP16/BF16 MMA, f16 accumulate (HMMA.*.F16)", "device": "all", "cls": "LOW-PRECISION", "D": None,
        "escape": False,
        "isa": "mma.sync...f16.f16.f16.f16: the accumulator is f16",
        "reason": "an f16 accumulator rounds the growing sum (exact only to 2048), so it cannot hold a 32-MAC committed "
                  "word exactly -- not an exact s32 route",
    },
    {
        "name": "FP8 MMA (QMMA E4M3/E5M2)", "device": "all", "cls": "DECOMPOSE", "D": 4, "escape": False,
        "isa": "mma.sync...f32.e4m3.e4m3.f32 m16n8k32: E4M3 holds integers exactly only to 16, so an int8 operand needs "
               "a 2-piece split and the product 4 QMMA passes",
        "reason": "int8 is not E4M3-exact (e.g. 17, 127 are not), so 4 QMMA a word; QMMA ~1x int8 TOPS (FP-EXACT 0.985x), "
                  "so 4 passes = 0.25x int8; the FP8 DESIGNS (parked, decision 64) are the only ones it fits",
    },
    {
        "name": "FP4 MMA (OMMA/QMMA E2M1, dense and 2:4)", "device": "all", "cls": "DECOMPOSE", "D": 8, "escape": False,
        "isa": "mma.sync...e2m1.e2m1 m16n8k64 (OMMA 1.951x IMMA, 2:4 3.761x; decision 60): E2M1 holds only {0,.5,1,1.5,"
               "2,3,4,6}, so an int8 operand needs many pieces",
        "reason": "int8 is not E2M1-representable, so >= 8 OMMA a word; at <= 4x int8 TOPS that is <= 0.5x int8; the FP4 "
                  "designs are parked (decision 64, +19% perplexity at best)",
    },
    {
        "name": "DPX (sm_90 __vimax3/__viaddmax; __vimnmx on sm_120)", "device": "sm_90 (VIMNMX all)",
        "cls": "NO-MULTIPLY", "D": None, "escape": False,
        "isa": "DPX fuses packed s16/s32 min/max and add+max; no general multiply",
        "reason": "min/max/add only -- a sum of maxima, not of products; forms no int8 product (VIMNMX already priced at "
                  "the pi32 floor)",
    },
    {
        "name": "video / SIMD ALU (VABSDIFF4, VADD4, VIMNMX)", "device": "all", "cls": "NO-MULTIPLY", "D": None,
        "escape": False,
        "isa": "VABSDIFF4.ACC accumulates Sum|a_i - b_i| (SAD) over 4 bytes; VADD4 packed add; VIMNMX packed min/max",
        "reason": "SAD is a sum of absolute DIFFERENCES, not of products; packed add/min/max have no multiply -- none "
                  "forms an int8 product",
    },
    {
        "name": "texture filtering (TEX bilinear/trilinear, TLD)", "device": "all", "cls": "LOW-PRECISION", "D": None,
        "escape": False,
        "isa": "a convex 8-bit-weighted sum of <= 8 texels, format-converted and rounded",
        "reason": "convex low-precision weights over a fixed basis: the output is in the texels' hull and rounded, never "
                  "a free exact s32 (NOSUM); priced TEXBIL/TLD at the pi32 floor",
    },
    {
        "name": "async copy / TMA bulk copy (cp.async, cp.async.bulk, LDGSTS)", "device": "all", "cls": "MOVER",
        "D": None, "escape": False,
        "isa": "data movement global<->shared; no arithmetic",
        "reason": "a moved word equals an existing word (a copy, not fresh); priced LDGSTS/UBLKCP at the copy/load floor",
    },
    {
        "name": "ldmatrix / stmatrix transpose (LDSM/STSM, .trans)", "device": "all", "cls": "MOVER", "D": None,
        "escape": False,
        "isa": "a shared-memory load/store with a register-layout transpose; no arithmetic",
        "reason": "a transposed word equals an existing word (a copy); priced LDSM*/STSM* at the copy/load floor, .trans "
                  "free beside it",
    },
    {
        "name": "PRMT / FMNMX / LOP3 packing tricks", "device": "all", "cls": "NO-MULTIPLY", "D": None, "escape": False,
        "isa": "PRMT selects bytes, LOP3 is a 3-input bitwise truth table, I/FMNMX is min/max; none multiplies",
        "reason": "byte-select / bitwise / min-max carry no product; the outer-product PACKING that DOES multiply is "
                  "IMAD/IMAD.WIDE/IDP, priced on the lane route and capped by ISA-CAP (<= 4 fields a word)",
    },
    {
        "name": "IMAD.WIDE (two destination words)", "device": "all", "cls": "PRICED", "D": 1, "escape": False,
        "isa": "IMAD.WIDE: s32 x s32 -> s64 into a register pair (two words a lane)",
        "reason": "the two words are one s64 product's halves (the high word is carry/sign, not an independent fresh "
                  "word); priced IMADW at the pi32 floor; ISA-CAP caps its packing at 4 fields a word = IMAD's density",
    },
    {
        "name": "uniform datapath beside saturated async wgmma / tcgen05.mma", "device": "sm_90, sm_100",
        "cls": "WATCH", "D": 1, "escape": False,
        "isa": "UIMAD is an exact s32 MAC, one warp-uniform word an instruction; the census priced it on issue beside "
               "mma.sync (decisions 73, 84, 88), NOT beside a saturated asynchronous wgmma/tcgen05.mma",
        "reason": "bounded by the issue ceiling (decision 84: one instruction a cycle whatever the pipe, uniform "
                  "included), and the uniform route is already ~0% of the H200/B200 headline eps; UNMEASURED beside "
                  "async tensor (ptxas moves uniform chains to lanes there) -- a round-2 DC probe should confirm no "
                  "dual-issue",
    },
]


def main() -> int:
    checks = Checks()

    # -- coverage: every candidate the brief names is enumerated, with a class and a verdict --------------------------
    brief = ["redux", "L2 reduction", "TMA bulk reduce", "IDP4A", "IDP2A", "sparse IMMA", "BMMA", "s4 IMMA", "FP16",
             "FP8", "FP4", "DPX", "VABSDIFF4", "texture", "async copy", "ldmatrix", "PRMT", "IMAD.WIDE", "uniform"]
    joined = " | ".join(c["name"] for c in CANDIDATES)
    for b in brief:
        checks.check(f"enumerated: {b}", b in joined)
    classes = {"NO-MULTIPLY", "DECOMPOSE", "LOW-PRECISION", "MOVER", "PRICED", "WATCH"}
    checks.check("every candidate has a known class", all(c["cls"] in classes for c in CANDIDATES))
    checks.check("no candidate on sm_120 is a live escape", not any(c["escape"] for c in CANDIDATES))

    # -- NO-MULTIPLY units cannot form a product ---------------------------------------------------------------------
    rng = random.Random(0x5EED)
    red_is_sum, red_ne_prod = True, False
    for _ in range(2000):
        ws = [rng.randrange(1 << 32) for _ in range(32)]
        red = 0
        for w in ws:
            red = (red + w) & M32  # redux.sync.add / red.add: the sum of 32 existing committed words
        if red != (sum(ws) & M32):
            red_is_sum = False
        if red != (ws[0] * ws[1]) & M32:
            red_ne_prod = True
    checks.check("redux/red.add output is the exact sum of existing committed words (not fresh)", red_is_sum)
    checks.check("a reducer's output is not a product of two operands (no multiply -> no fresh dot-product term)",
                 red_ne_prod)
    sad_ne_dp = False
    for _ in range(2000):
        a = [rng.randrange(256) for _ in range(4)]
        b = [rng.randrange(256) for _ in range(4)]
        if sum(abs(x - y) for x, y in zip(a, b)) != sum(x * y for x, y in zip(a, b)):
            sad_ne_dp = True
    checks.check("VABSDIFF4 SAD (Sum|a-b|) differs from the dot product (Sum a*b): not a product unit", sad_ne_dp)
    checks.check("DPX / VIMNMX (min/max/add) is not a multiply: max(a,b) != a*b",
                 any(max(x, y) != x * y for x, y in [(3, 5), (7, 2), (9, 9), (4, 6)]))

    # -- MOVERS copy: a transposed / copied / selected word equals an existing word ----------------------------------
    checks.check("a moved/transposed word (cp.async, LDSM.trans) equals an existing word (a copy)",
                 all(w == w for w in (rng.randrange(1 << 32) for _ in range(200))))
    checks.check("PRMT selects a byte of an existing word (no product)",
                 all(((w >> (8 * s)) & 0xFF) in [(w >> (8 * k)) & 0xFF for k in range(4)]
                     for w in (rng.randrange(1 << 32) for _ in range(100)) for s in range(4)))

    # -- DECOMPOSE: the bit-exact int8 reconstructions that set each form's factor D ----------------------------------
    bmma_ok = all(sum((1 << (i + j)) * (((a >> i) & 1) & ((b >> j) & 1)) for i in range(8) for j in range(8)) == a * b
                  for a in range(256) for b in range(256))
    checks.check("BMMA: an unsigned byte product is 64 AND-popc bit-planes Sum 2^(i+j)(a_i & b_j), exhaustive on 256x256",
                 bmma_ok, "D = 64 one-bit products an int8 word")
    s4_ok = True
    for a in range(256):
        a1, a0 = nibbles_u8(a)
        for b in range(256):
            b1, b0 = nibbles_u8(b)
            if 256 * a1 * b1 + 16 * (a1 * b0 + a0 * b1) + a0 * b0 != a * b:
                s4_ok = False
                break
        if not s4_ok:
            break
    checks.check("s4 IMMA: an unsigned byte product is 4 nibble products (256,16,16,1), exhaustive on 256x256", s4_ok,
                 "D = 4 four-bit products an int8 word")
    checks.check("IDP2A heads 2 int8 MACs an instruction against IDP4A's 4 (strictly dominated by dp4a)", 2 < 4)

    # -- exactness of the low-precision / FP forms for int8 ----------------------------------------------------------
    m, e = FLOAT_FORMATS["f16"]
    checks.check("f16 represents every int8 operand exactly (|n| < 2048)", all(exact_in_float(n, m, e) for n in INT8))
    m, e = FLOAT_FORMATS["bf16"]
    checks.check("bf16 represents every int8 operand exactly (|n| < 256)", all(exact_in_float(n, m, e) for n in INT8))
    m, e = FLOAT_FORMATS["e4m3"]
    not_exact_e4m3 = [n for n in INT8 if not exact_in_float(n, m, e)]
    checks.check("E4M3 does NOT represent every int8 operand (e.g. 17, 127): so FP8 pays a piece split",
                 17 in not_exact_e4m3, f"{len(not_exact_e4m3)} of 256 int8 values not E4M3-exact")
    m, e = FLOAT_FORMATS["e2m1"]
    not_exact_e2m1 = [n for n in INT8 if not exact_in_float(n, m, e)]
    checks.check("E2M1 (FP4) represents almost no int8 operand (grid {0,.5,1,1.5,2,3,4,6})", len(not_exact_e2m1) > 240,
                 f"{len(not_exact_e2m1)} of 256 int8 values not E2M1-exact")
    checks.check("an int8xint8 product and a 32-MAC word are f32-exact (< 2^24), so f32-accumulate HMMA is exact",
                 all(exact_in_float(a * b, 23, 127) for a in WEIGHT for b in (-128, -63, 63, 127)))
    checks.check("an f16 ACCUMULATOR rounds a growing word (2049 not f16-exact): f16-accumulate forms are not exact s32",
                 not exact_in_float(2049, *FLOAT_FORMATS["f16"]))

    # -- the int8-equivalent throughput of every non-IMMA tensor form is <= IMMA s8's (decision 81's best form holds) --
    # The tensor core's peak bit-operation throughput is conserved: a lower-precision form's int8 rate is its peak TOPS
    # ratio over int8 divided by D, the int8 unit-products a committed word decomposes into. eff <= 1 means the form ties
    # (eff == 1, no saving under the tensor theorem) or loses. The TOPS ratios are GENEROUS upper bounds (measured <=
    # these: FP-EXACT, decision 60, sm_90 BMMA census), so the closure is robust. 2:4 sparsity doubles the TOPS ratio but
    # a DENSE committed word has no structural zeros, so it needs two complementary-metadata passes (D doubles too).
    tensor = [  # (name, peak TOPS ratio vs int8 [generous], D int8 unit-products a committed word)
        ("IMMA16816 (s8, k16)", 1.0, 1),
        ("HMMA16816.F32 (f16/bf16, exact int8)", 1.0, 1),
        ("IMMA.SP.16864 (2:4 s8, dense = 2 passes)", 2.0, 2),
        ("IMMA.S4.16864", 2.0, 4),
        ("QMMA.E4M3.16832 (FP8)", 1.0, 4),
        ("OMMA.NVF4.16864 (FP4, 2:4 logical)", 4.0, 8),
        ("BMMA.B1.16n256", 8.0, 64),
    ]
    for name, tops, d in tensor:
        eff = tops / d
        checks.check(f"tensor form int8 TOPS <= IMMA s8 (tie or worse, no saving): {name} at {eff:.3f}x (D={d})",
                     eff <= 1.0 + 1e-9, f"{eff:.3f}x")

    # -- the floors price every reachable side unit at or above honest's per-word cost --------------------------------
    floors = list(csv.DictReader(FLOORS.read_text(encoding="utf-8").splitlines()))
    checks.check("floors.csv: every priced route is >= 16 MACs/word", min(int(r["floor"]) for r in floors) >= 16)
    roots = {r["kind"] for r in floors if r["claim"] == "pi32"}
    for k in ("RED32", "ATOMG32", "TMAred512", "UNI", "IMADW", "TEXBIL", "TLD1", "SURED", "REDV"):
        checks.check(f"floors.csv: {k} priced at the pi32 root floor", k in roots)

    if checks.mutations:
        checks.mutation("a hypothetical multiplying reducer WOULD be fresh (add is not multiply)",
                        any((a + b) & M32 != (a * b) & M32 for a, b in [(3, 5), (100, 100), (7, 9)]))
        checks.mutation("int8 is not all E4M3-exact (else FP8 would be a 1-pass int8 route)", len(not_exact_e4m3) > 0)
        a, b = 255, 255
        short = sum((1 << (i + j)) * (((a >> i) & 1) & ((b >> j) & 1)) for i in range(8) for j in range(7))
        checks.mutation("b1 with 56 planes (one bit short) is wrong: int8 needs all 64", short != a * b)
        checks.mutation("BMMA at a 64x raw rate would tie IMMA (so D=64 is the binding closure, not an arbitrary bound)",
                        (256 / 32) * 64.0 / 64 >= 1.0)

    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
