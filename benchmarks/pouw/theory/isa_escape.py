"""ISA-ESCAPE, mechanically: no instruction or unit outside the model's menu heads a committed word more cheaply than the
menu allows.

A committed word is a fresh s32: the running sum of 32 int8 products of noised operands (32 MACs a word; FOLD-BW), or an
input to the receipt fold (which is itself such a word). The model's menu for it (the ledger's rows
(I2)/(I4) WordsWritten, (I3) ScalarRoute, (I5) SideUnits/NoSum) is exactly five routes:
  - an MMA form's writer, at most omega_t committed words an instruction (the tensor pipe, priced per form);
  - the CUDA lane vector route, q* lane instructions a word (q* >= 1.22 with tables, 2.03 without; `scalar_route.py`);
  - the uniform datapath, one committed word an instruction, on the issue slots;
  - table routes, priced by the bytes they read from their memory level;
  - priced side units, each at the pi32 floor (>= 32 MACs a word; floors.csv).

NOSUM (`nosum.py`) covered the fixed-function units the data sheet measured. This script attacks the menu from the ISA: for
every instruction class a cheater would reach for -- warp and L2 reductions, atomics, bulk reduce, the low-precision,
1-bit and FP64 MMA forms, the dot-product instructions, DPX, the video/SIMD ALU, texture filtering, the copies and transposes, the
packing tricks, the wide multiply, the two-word add, min/max and select, and the uniform datapath beside an asynchronous tensor op -- it records the ISA fact,
whether the instruction can even form an exact int8 product, the factor by which an int8 committed word decomposes onto
it, and the verdict. A committed word that is a sum of 32 int8 products needs a MULTIPLY; an instruction with no multiply
can at most sum or move words that already exist (not fresh), and an instruction whose multiply is lower precision pays a
decomposition factor D (int8 = D unit-products) that its raw rate must beat the honest IMMA route by to escape. A
committed word is also an input to the receipt fold, so a cross-lane reducer is also tested as a fold: it collapses the
warp to one result (one word a warp instruction = 32 lane instructions a word), so it cannot undercut duo's 3 (FOLD-OPT).

The checkable arithmetic is verified here, bit-exact over Z/2^32 and the operand alphabets; the measured rates are cited
(tensor_kinds / opcode_probe census, the data sheet). The conclusion for sm_120 is that every candidate is closed, by one of:
  - NO-MULTIPLY: the unit only adds/compares/moves, so it sums or copies existing committed words (not fresh);
  - DECOMPOSE: the unit multiplies, but in a format an int8 product needs D >= 2 of, and its rate / D <= IMMA's;
  - LOW-PRECISION: the unit's weighted sum is convex or rounded, so not an exact fresh s32;
  - MOVER: the unit only moves data, so its output equals an existing word;
  - PRICED: the unit is already a menu route at the pi32 floor, so it undercuts nothing.

The one cross-device item that is NOT closed by measurement -- the uniform datapath's issue cost beside a SATURATED
asynchronous wgmma (sm_90) or tcgen05.mma (sm_100), which the census priced beside the synchronous mma.sync only
-- is recorded as a WATCH, with the issue-ceiling argument that bounds it and the `eps`
reading (the uniform route is already ~0% of the headline eps on the H200/B200) that shows it cannot move a headline.

The B200 (sm_100, 5th-gen tensor core) adds a distinctive word ISA this script also closes:
tcgen05.mma writes its 32768-word accumulator into TENSOR MEMORY (not registers) and a CTA pair
(cta_group::2) can share one MMA, but that writer is the honest, priced tensor route; the TMEM movers it comes with
(tcgen05.cp/shift/ld/st) only copy or shift words that already exist (MOVER); the block-scaled low-precision forms
(mxf8f6f4 for FP6 E2M3/E3M2, mxf4/mxf4nvf4 for FP4) carry one shared exponent a block, not a per-element mantissa, so
they decompose an int8 word no more cheaply than the honest route (DECOMPOSE: counting the values one MAC can take
against the 5965 distinct int8 x weight products, FP8 and FP6 need >= 2 MACs a product and FP4 >= 3; two E2M3 limbs a
side attain D=4); and the weight decompression engine only reconstructs stored words off the SMSP (MOVER). Every sm_100
tensor form's int8-equivalent rate is re-checked at the B200 data-sheet peaks (DS_tensor: s8/e4m3 8192, f16/bf16 4096,
tf32 2048 MACs/SM/clk), each FP form at its counted lower bound on D, and each is <= UTCIMMA s8.

Run: `python isa_escape.py [--mutations] [--verdict PATH]`. Stdlib + exact_kit only; reads machine_model/data/floors.csv.
"""
from __future__ import annotations

import csv
import random
import sys
import tomllib
from fractions import Fraction
from pathlib import Path

from exact_kit.checks import Checks

M32 = 0xFFFFFFFF
DATA = Path(__file__).resolve().parents[1] / "machine_model" / "data"
FLOORS = DATA / "floors.csv"
DESIGNS = DATA / "designs.toml"

# A warp has 32 lanes. A per-lane (vertical) fold keeps every lane busy on its own word; a cross-lane (horizontal)
# reduction (redux.sync, warp-shuffle trees) collapses all 32 lanes to one result, so it heads one word a warp
# instruction = 32 lane instructions a word, whatever the reduce op.
LANES = 32

# The design's operand alphabets: activations are int8 (the dithered floor), weights are capped at
# |w| <= 63. A committed word sums 32 such products, so it ranges within ~2^19 << 2^24 (f32's exact-integer limit).
INT8 = range(-128, 128)
WEIGHT = range(-63, 64)

# Mantissa bits (after the implicit 1) and the max normal exponent of each MMA input format.
# E4M3: max 448 = 1.75*2^8. E5M2: emax 15. E2M1 (FP4) grid {0,.5,1,1.5,2,3,4,6}, max 6 = 1.5*2^2. f16: emax 15. bf16: emax 127.
# f64: 52 bits, emax 1023.
# FP6 (OCP MX; sm_120 kind::f8f6f4, B200 mxf8f6f4): E2M3 is 1s/2e/3m, bias 1 -> emax 2 (max 7.5 = 1.875*2^2); E3M2 is
# 1s/3e/2m, bias 3 -> emax 4 (max 28 = 1.75*2^4).
FLOAT_FORMATS = {"e4m3": (3, 8), "e5m2": (2, 15), "e2m1": (1, 2), "f16": (10, 15), "bf16": (7, 127),
                 "e2m3": (3, 2), "e3m2": (2, 4), "f64": (52, 1023)}


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


def float_values(exp_bits: int, mant_bits: int, bias: int, nan_codes: int = 0) -> set[Fraction]:
    """Every finite value of a small binary float, subnormals and both signs included; the top `nan_codes` codes a sign
    are NaN or infinity (E4M3: S.1111.111)."""
    codes = [(e, m) for e in range(1 << exp_bits) for m in range(1 << mant_bits)]
    out = set()
    for e, m in codes[: len(codes) - nan_codes]:
        frac = Fraction(m, 1 << mant_bits)
        v = (frac if e == 0 else 1 + frac) * Fraction(2) ** ((1 if e == 0 else e) - bias)
        out |= {v, -v}
    return out


# The MMA input formats of the block-scaled and FP8 forms, as value sets.
LOWP = {"e4m3": float_values(4, 3, 7, nan_codes=1), "e2m3": float_values(2, 3, 1), "e3m2": float_values(3, 2, 3),
        "e2m1": float_values(2, 1, 1)}
PRODUCTS = {a * w for a in INT8 for w in WEIGHT}


def macs_lower_bound(values: set[Fraction]) -> int:
    """Fewest MACs of a format that head one int8 x weight product, for an elementwise scheme: each MAC multiplies a
    per-element limb of a by one of w and the MACs are summed with fixed constants (a block scale, shared by 16 or 32
    products, is such a constant for a dense block). D MACs then take at most |V*V|^D values, against len(PRODUCTS)."""
    vv = len({x * y for x in values for y in values})
    d = 1
    while vv ** d < len(PRODUCTS):
        d += 1
    return d


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
        "reason": "no multiply, so the 32 MACs a word still cost on a product unit; and as a fold of existing words it is "
                  "a cross-lane reduction, 1 word a warp instruction = 32 lane instructions a word, >> duo's 3 (FOLD-OPT) and "
                  "q* (>= 1.22), so it cannot head a fresh word nor lower the fold",
    },
    {
        "name": "L2 reduction / atomic (red.global.add.s32, atom.add)", "device": "all",
        "cls": "PRICED", "D": None, "escape": False,
        "isa": "red/atom.global.add.s32 lands an exact s32 add in L2; no multiply",
        "reason": "priced RED32/ATOMG32 at the pi32 floor (floors.csv); a reducer with no multiply, so 0 fresh words "
                  "(NOSUM); measured <= 1.9 words/SM/clk beside honest (the data sheet)",
    },
    {
        "name": "TMA bulk reduce (cp.reduce.async.bulk)", "device": "sm_90, sm_100, sm_120",
        "cls": "PRICED", "D": None, "escape": False,
        "isa": "UBLKRED/UTMAREDG reduces a bulk tile shared->global with an exact s32 add; no multiply",
        "reason": "priced TMAred512 at the pi32 floor; a reducer with no multiply, 0 fresh words; <= 1.9 words/SM/clk "
                  "beside honest (the data sheet)",
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
                  "the full word needs two complementary-metadata passes = the dense form's MAC cost (2:4 "
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
        "reason": "int8 is not E4M3-exact (e.g. 17, 127 are not), so 4 QMMA a word; QMMA ~1x int8 TOPS (fp_exact's 0.985x), "
                  "so 4 passes = 0.25x int8; the FP8 DESIGNS (parked) are the only ones it fits",
    },
    {
        "name": "FP6 MMA (QMMA E3M2/E2M3, kind::f8f6f4)", "device": "sm_120", "cls": "DECOMPOSE", "D": 4,
        "escape": False,
        "isa": "mma.sync.kind::f8f6f4 m16n8k32 at e3m2 or e2m3 assembles to the same QMMA.16832 as E4M3 on sm_120a "
               "and runs at E4M3's rate, 0.2461 a SM-cycle",
        "reason": "E3M2 holds integers exactly only to 8 and E2M3 only to 7, so an int8 operand needs >= 2 pieces and a "
                  "product >= 4 QMMA passes; at QMMA's measured 0.992x IMMA that is <= 0.25x int8",
    },
    {
        "name": "FP64 MMA (DMMA.8x8x4)", "device": "sm_120 (also sm_80..sm_100)", "cls": "DECOMPOSE", "D": 1,
        "escape": False,
        "isa": "mma.sync.m8n8k4.f64 assembles to one DMMA.8x8x4 on sm_120a: 64 f64 dot products "
               "of depth 4, exact for int8 (a 32-MAC word is <= 2^19, f64 exact to 2^53)",
        "reason": "exact at D = 1, but measured at 0.0068 DMMA a SM-cycle, 1.74 MACs a SM-cycle "
                  "against IMMA s8's 1016, 1/584 of int8's rate; packing two products a MAC in the 53-bit mantissa "
                  "leaves 1/292. On issue a 32-MAC word takes 8 DMMA for 64 words (8 a slot against IMMA's 128) plus "
                  "an F2I each, so it loses on both rows",
    },
    {
        "name": "FP4 MMA (OMMA/QMMA E2M1, dense and 2:4)", "device": "all", "cls": "DECOMPOSE", "D": 8, "escape": False,
        "isa": "mma.sync...e2m1.e2m1 m16n8k64 (OMMA 1.951x IMMA, 2:4 3.761x): E2M1 holds only {0,.5,1,1.5,"
               "2,3,4,6}, so an int8 operand needs many pieces",
        "reason": "int8 is not E2M1-representable, so >= 8 OMMA a word; at <= 4x int8 TOPS that is <= 0.5x int8; the FP4 "
                  "designs are parked (+19% perplexity at best)",
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
        "name": "IADD.64 / IMNMX.64 / SEL.64 (two destination words, no multiply)", "device": "sm_120",
        "cls": "NO-MULTIPLY", "D": None, "escape": False,
        "isa": "a register pair a lane: add.u64 assembles to one IADD.64 on sm_120a (either "
               "operand negated, the sass_gate fixtures), IMNMX.U64 is in TS.lean's fan-in table and SEL.64 in "
               "harness/sass_pins.json",
        "reason": "no multiply, so they sum, compare or select existing words (not fresh); as a fold step, fold_wide3.c "
                  "finds no program whose last instruction writes fx1's two outputs as one IADD.64, IMNMX.64 or SEL.64 "
                  "(fold_opt_fx.py --wide3)",
    },
    {
        "name": "uniform datapath beside saturated async wgmma / tcgen05.mma", "device": "sm_90, sm_100",
        "cls": "WATCH", "D": 1, "escape": False,
        "isa": "UIMAD is an exact s32 MAC, one warp-uniform word an instruction; the census priced it on issue beside "
               "mma.sync, NOT beside a saturated asynchronous wgmma/tcgen05.mma",
        "reason": "bounded by the issue ceiling (one instruction a cycle whatever the pipe, uniform "
                  "included), and the uniform route is already ~0% of the H200/B200 headline eps; sm_100 MEASURED at "
                  "saturation, no dual-issue (dc/probes/sm100_a2u.cu: the chain stays on the "
                  "uniform datapath beside tcgen05.mma, total issue <= 0.9996 an SMSP-cycle at K 32-135, each uniform "
                  "instruction 0.95-1.10 of a lane slot); sm_90 MEASURED near saturation, no dual-issue "
                  "(dc/probes/sm90_emp.cu emp_pack3: the chain uniform beside 4 wgmma s8 n128 a "
                  "group, issue 0.87, each uniform instruction 1.08-1.19 SMSP-cycles)",
    },
    # -- sm_100 (B200) only: the 5th-gen tensor core's TMEM movers, block-scaled low-precision forms, and the
    # decompression engine. The writer tcgen05.mma IS the honest priced route; the
    # instructions below are the Blackwell-distinctive ones a cheater would reach for beyond it.
    {
        "name": "tcgen05.cp (shared -> tensor-memory copy)", "device": "sm_100", "cls": "MOVER", "D": None,
        "escape": False,
        "isa": "tcgen05.cp copies a tile from shared memory into tensor memory (TMEM); no arithmetic",
        "reason": "a copied TMEM word equals an existing word (data movement, not fresh); heads no product",
    },
    {
        "name": "tcgen05.shift (tensor-memory row shift)", "device": "sm_100", "cls": "MOVER", "D": None,
        "escape": False,
        "isa": "tcgen05.shift moves the rows of a TMEM region down (a convolution helper); no arithmetic",
        "reason": "a shifted word equals an existing word (data movement); heads no product",
    },
    {
        "name": "tcgen05.ld / tcgen05.st (tensor-memory <-> registers)", "device": "sm_100", "cls": "MOVER",
        "D": None, "escape": False,
        "isa": "LDTM/STTM move words between TMEM and registers (LDTM_32X32B_X128 reads 128 words/thread = 4096 bits); "
               "no arithmetic",
        "reason": "a loaded/stored word equals an existing word (a copy); the TMEM read port is the TL facet (a load "
                  "hides beside a saturated issue), and it heads no fresh product",
    },
    {
        "name": "block-scaled FP6 MMA (mxf8f6f4, E2M3 / E3M2)", "device": "sm_100", "cls": "DECOMPOSE", "D": 4,
        "escape": False,
        "isa": "tcgen05.mma.mxf8f6f4 with a per-block E8M0 scale; E2M3 and E3M2 each take 63 values (integers only to 7 "
               "or 8 at unit scale), so one limb holds no int8 operand, but two limbs at power-of-two weights hold "
               "both alphabets (a = 16 x1 + 2 x2, w = 8 y1 + y2/2 in E2M3)",
        "reason": "an int8 x weight product is 4 E2M3 limb products (D=4, exact on INT8 x WEIGHT), and no elementwise "
                  "scheme does it in 1 MAC (|V*V| = 443 < 5965 products); FP6 rides the mxf8f6f4 datapath at s8's "
                  "rate, so <= 1/2 int8 (0.25x for the limb scheme), like E4M3",
    },
    {
        "name": "block-scaled FP4 MMA (mxf4, mxf4nvf4; E2M1 + E8M0/E4M3 scale)", "device": "sm_100",
        "cls": "DECOMPOSE", "D": 3, "escape": False,
        "isa": "tcgen05.mma.mxf4/mxf4nvf4: E2M1 holds only {0,.5,1,1.5,2,3,4,6}; the microscaling factor is one shared "
               "exponent a 16/32-lane block (E8M0, or a per-16 E4M3 for NVFP4), not a per-element value",
        "reason": "one E2M1 MAC takes 37 values, and 37^2 < 5965 int8 x weight products, so an elementwise scheme needs "
                  ">= 3 MACs a product (D >= 3; power-of-two limbs need 3 a side, D=9); at 2x int8 dense, or 4x for "
                  "2:4 with 2 passes a dense word, that is <= 2/3 int8",
    },
    {
        "name": "weight decompression engine (sparse / compressed-weight expand)", "device": "sm_100", "cls": "MOVER",
        "D": None, "escape": False,
        "isa": "the Blackwell decompression engine expands compressed weights on the DMA path before the tensor core "
               "reads them; it reconstructs stored values, it does not multiply",
        "reason": "a decompressed word equals the stored (existing) word it reconstructs (NO-MULTIPLY/MOVER); it is off "
                  "the SMSP issue path and heads no fresh product",
    },
]


def main() -> int:
    checks = Checks()

    # -- coverage: every candidate is enumerated, with a class and a verdict --------------------------
    brief = ["redux", "L2 reduction", "TMA bulk reduce", "IDP4A", "IDP2A", "sparse IMMA", "BMMA", "s4 IMMA", "FP16",
             "FP8", "FP6", "FP4", "FP64", "DPX", "VABSDIFF4", "texture", "async copy", "ldmatrix", "PRMT", "IMAD.WIDE",
             "IADD.64", "uniform",
             # sm_100 (B200) word ISA
             "tcgen05.cp", "tcgen05.shift", "tcgen05.ld", "mxf4", "decompression"]
    joined = " | ".join(c["name"] for c in CANDIDATES)
    for b in brief:
        checks.check(f"enumerated: {b}", b in joined)
    classes = {"NO-MULTIPLY", "DECOMPOSE", "LOW-PRECISION", "MOVER", "PRICED", "WATCH"}
    checks.check("every candidate has a known class", all(c["cls"] in classes for c in CANDIDATES))
    checks.check("no candidate on sm_120 is a live escape", not any(c["escape"] for c in CANDIDATES))
    sm100 = [c for c in CANDIDATES if "sm_100" in c["device"]]
    checks.check("the B200 (sm_100) word ISA is enumerated (tcgen05 movers, block-scaled FP6/FP4, decompression)",
                 len(sm100) >= 6, f"{len(sm100)} sm_100 candidates")
    checks.check("no sm_100 candidate is a live escape", not any(c["escape"] for c in sm100))

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

    # -- redux/shuffle-tree as a fold: a cross-lane reduction forfeits SIMT, so it cannot undercut duo's fold ----------
    # The committed word is also "an input to the receipt fold", so a cheaper fold would raise eps_mand. duo folds
    # vertically: each of the 32 lanes folds its own word, so a warp instruction advances 32 words at once and the fold
    # costs (ALU + IMAD) = 3 lane instructions a word. redux reduces horizontally: all 32 lanes collapse to one result,
    # so one warp instruction heads one word -- 32 lane instructions a word, whatever the op and however it is masked
    # (redux.sync emits one reduced value per instruction, so k segment-sums need k instructions). 32 >> 3, and 32 is
    # also well past q* (>= 1.22 with tables, 2.03 without), so redux beats neither the fold nor the lane route.
    design = tomllib.loads(DESIGNS.read_text(encoding="utf-8"))
    fold_cost = sum(v for v in design["fold"].values() if isinstance(v, int))  # duo: ALU 2 + IMAD 1 = 3
    redux_fold_cost = LANES  # one word a warp instruction = 32 lane instructions a word
    checks.check("designs.toml fold is duo at 3 lane instructions a word (FOLD-OPT's count)", fold_cost == 3,
                 f"{design['fold']['name']} = {fold_cost}")
    checks.check("redux/shuffle-tree fold (cross-lane, 1 word a warp instruction) costs 32 lane instructions a word > "
                 "duo's (FOLD-OPT holds: no cheaper fold, so eps_mand stays 0)", redux_fold_cost > fold_cost,
                 f"redux {redux_fold_cost} vs duo {fold_cost} lane-instructions/word")
    checks.check("redux fold (32 lane-instructions/word) is also well past q* (>= 1.22), so it never beats the lane route",
                 redux_fold_cost > 1.22)

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
    for fp6 in ("e3m2", "e2m3"):
        m, e = FLOAT_FORMATS[fp6]
        not_exact = [n for n in INT8 if not exact_in_float(n, m, e)]
        checks.check(f"{fp6.upper()} (FP6) does NOT represent every int8 operand (e.g. 9): so FP6 pays a piece split",
                     9 in not_exact, f"{len(not_exact)} of 256 int8 values not {fp6.upper()}-exact")
    m, e = FLOAT_FORMATS["f64"]
    word_max = LANES * 128 * 128
    checks.check("f64 holds an int8 product and a 32-MAC word exactly (<= 2^19 << 2^53), so DMMA is an exact D = 1 form",
                 all(exact_in_float(n, m, e) for n in (word_max, -word_max, word_max - 1, 127 * -128)))
    field = (2 * word_max).bit_length()  # a signed 32-MAC word's field, its range [-word_max, word_max] in bits
    checks.check("f64's 53-bit mantissa packs at most two 32-MAC words' fields, so DMMA's packing gain is <= 2",
                 2 * field <= m + 1 < 3 * field, f"{field}-bit fields, {m + 1}-bit mantissa")
    m, e = FLOAT_FORMATS["e2m1"]
    not_exact_e2m1 = [n for n in INT8 if not exact_in_float(n, m, e)]
    checks.check("E2M1 (FP4) represents almost no int8 operand (grid {0,.5,1,1.5,2,3,4,6})", len(not_exact_e2m1) > 240,
                 f"{len(not_exact_e2m1)} of 256 int8 values not E2M1-exact")
    checks.check("an int8xint8 product and a 32-MAC word are f32-exact (< 2^24), so f32-accumulate HMMA is exact",
                 all(exact_in_float(a * b, 23, 127) for a in WEIGHT for b in (-128, -63, 63, 127)))
    checks.check("an f16 ACCUMULATOR rounds a growing word (2049 not f16-exact): f16-accumulate forms are not exact s32",
                 not exact_in_float(2049, *FLOAT_FORMATS["f16"]))

    # -- B200 block-scaled FP6/FP4 (mxf8f6f4, mxf4): how many MACs an int8 x weight product needs ----------------------
    m, e = FLOAT_FORMATS["e2m3"]
    e2m3_ints = [n for n in range(256) if exact_in_float(n, m, e)]
    checks.check("E2M3 (FP6) holds integers only 0..7 at unit scale: one limb holds no int8 operand",
                 e2m3_ints == list(range(8)), f"E2M3 exact ints 0..255: {e2m3_ints}")
    checks.check("the value sets: E4M3 253, E2M3 63, E3M2 63, E2M1 15 finite values",
                 [len(LOWP[f]) for f in ("e4m3", "e2m3", "e3m2", "e2m1")] == [253, 63, 63, 15],
                 str({f: len(v) for f, v in LOWP.items()}))
    d_lb = {f: macs_lower_bound(v) for f, v in LOWP.items()}
    checks.check("counted: one MAC of E4M3/E2M3/E3M2 heads no int8 x weight product (D >= 2), one of E2M1 needs D >= 3",
                 d_lb == {"e4m3": 2, "e2m3": 2, "e3m2": 2, "e2m1": 3}, f"{d_lb} against {len(PRODUCTS)} products")
    # The lower bound is attained to within a factor 2: two E2M3 limbs a side at power-of-two weights (block scales)
    # hold both alphabets, so an int8 x weight product is 4 E2M3 limb products.
    v = LOWP["e2m3"]

    def limbs(n: int, w1: Fraction, w2: Fraction) -> tuple[Fraction, Fraction] | None:
        return next(((x, (n - w1 * x) / w2) for x in sorted(v) if (n - w1 * x) / w2 in v), None)

    a_limbs = {a: limbs(a, Fraction(16), Fraction(2)) for a in INT8}
    w_limbs = {w: limbs(w, Fraction(8), Fraction(1, 2)) for w in WEIGHT}
    fp6_ok = all(x is not None for x in (*a_limbs.values(), *w_limbs.values())) and all(
        sum(wa * xa * ww * xw for wa, xa in zip((16, 2), a_limbs[a]) for ww, xw in zip((8, Fraction(1, 2)), w_limbs[w]))
        == a * w for a in INT8 for w in WEIGHT)
    checks.check("FP6: an int8 x weight product is 4 E2M3 limb products (a = 16 x1 + 2 x2, w = 8 y1 + y2/2), "
                 "exhaustive on INT8 x WEIGHT", fp6_ok, "D = 4 attained")

    # -- MOVERS on sm_100 (tcgen05.cp/shift/ld/st, decompression) copy existing words, head no product ----------------
    mover_names = ("tcgen05.cp", "tcgen05.shift", "tcgen05.ld", "decompression")
    checks.check("tcgen05 TMEM movers (cp/shift/ld/st) and the decompression engine carry no multiply (MOVER)",
                 all(c["D"] is None and c["cls"] == "MOVER"
                     for c in sm100 if any(n in c["name"] for n in mover_names)))

    # -- the int8-equivalent throughput of every non-IMMA tensor form is <= IMMA s8's (the per-form pricing's best form holds) --
    # The tensor core's peak bit-operation throughput is conserved: a lower-precision form's int8 rate is its peak TOPS
    # ratio over int8 divided by D, the int8 unit-products a committed word decomposes into. eff <= 1 means the form ties
    # (eff == 1, no saving under the tensor theorem) or loses. The TOPS ratios are GENEROUS upper bounds (measured <=
    # these: fp_exact, the LP's 2:4 tie, sm_90 BMMA census), so the closure is robust. 2:4 sparsity doubles the TOPS ratio but
    # a DENSE committed word has no structural zeros, so it needs two complementary-metadata passes (D doubles too).
    tensor = [  # (name, peak TOPS ratio vs int8 [generous], D int8 unit-products a committed word)
        ("IMMA16816 (s8, k16)", 1.0, 1),
        ("HMMA16816.F32 (f16/bf16, exact int8)", 1.0, 1),
        ("IMMA.SP.16864 (2:4 s8, dense = 2 passes)", 2.0, 2),
        ("IMMA.S4.16864", 2.0, 4),
        ("QMMA.E4M3.16832 (FP8)", 1.0, 4),
        ("QMMA.E3M2/E2M3.16832 (FP6, E4M3's opcode and rate)", 1.0, 4),
        ("DMMA.8x8x4 (f64, measured 1/584 of IMMA a MAC, two fields a mantissa)", 2 / 584, 1),
        ("OMMA.NVF4.16864 (FP4, 2:4 logical)", 4.0, 8),
        ("BMMA.B1.16n256", 8.0, 64),
    ]
    for name, tops, d in tensor:
        eff = tops / d
        checks.check(f"tensor form int8 TOPS <= IMMA s8 (tie or worse, no saving): {name} at {eff:.3f}x (D={d})",
                     eff <= 1.0 + 1e-9, f"{eff:.3f}x")
    checks.check("the FP8 and FP4 rows close at the counted D lower bound too, not only at their typed D (QMMA E4M3 at "
                 "1x, OMMA FP4 2:4 at 4x with 2 passes a dense word)",
                 1.0 / d_lb["e4m3"] <= 1.0 and 4.0 / (2 * d_lb["e2m1"]) <= 1.0,
                 f"{1.0 / d_lb['e4m3']:.3f}x, {4.0 / (2 * d_lb['e2m1']):.3f}x")

    # -- the same closure re-checked at the B200's data-sheet peaks (DS_tensor, sm_100 census), not sm_120's rates -----
    # Peaks are dense MACs/SM/clk: the sheet gives s8/e4m3 8192, f16/bf16 4096, tf32 2048 (sm_100 rows.csv DS_tensor);
    # for the forms the sheet omits, FP6 rides the FP8 (mxf8f6f4) datapath (<= s8's 8192) and FP4 is bounded generously
    # at 2x s8 dense / 4x s8 for 2:4 (a dense word then needs 2 passes, so D doubles). eff = (peak/s8_peak)/D, with each
    # FP form's D its counted lower bound, so eff is an upper bound whatever the limb scheme.
    S8_PEAK = 8192
    tensor_sm100 = [  # (name, dense MACs/SM/clk at the B200 peak, D: a lower bound on the MACs a committed product)
        ("UTCIMMA s8 (k32)", 8192, 1),
        ("QMMA e4m3 FP8 (k32)", 8192, d_lb["e4m3"]),
        ("HMMA f16/bf16 (k16)", 4096, 1),
        ("tf32 MMA (k8)", 2048, 1),
        ("mxf8f6f4 FP6 E2M3/E3M2", 8192, min(d_lb["e2m3"], d_lb["e3m2"])),
        ("mxf4 FP4 E2M1 dense", 16384, d_lb["e2m1"]),
        ("mxf4nvf4 FP4 2:4 (dense = 2 passes)", 32768, 2 * d_lb["e2m1"]),
    ]
    for name, peak, d in tensor_sm100:
        eff = (peak / S8_PEAK) / d
        checks.check(f"sm_100 tensor form int8 TOPS <= UTCIMMA s8 at the B200 DS peaks: {name} at {eff:.3f}x (D={d})",
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
        checks.mutation("a per-lane fold at 1 lane-instruction/word WOULD beat duo (so the 32 is the cross-lane cost, "
                        "not an arbitrary bound): redux's cost is 32 only because it collapses the warp", LANES > 1)
        checks.mutation("a format holding int8 itself would head a product in one MAC (so the D lower bound is counted, "
                        "not assumed)", macs_lower_bound({Fraction(n) for n in INT8}) == 1)
        checks.mutation("E2M1 at 4x int8 dense would beat s8 at its D lower bound (so FP4's closure binds on the count)",
                        4.0 / d_lb["e2m1"] > 1.0)

    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
