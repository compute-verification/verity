"""The stacking property (`verity/stacking-property` v1): P(c, R_in, L) on a Program, read with its census (`census`,
verity/census/v1: its steps, its primitives' instances, and each root value's producer, bits and readers).

A carrier is an audited block whose contents were chosen after the seed and hold another block's encoding. Such an encoding
reaches the circuit's values only as input bytes or through an encode inside the circuit, and an encode needs carries: a
floor, a comparison, a selection, a wrap or a clamp. P bounds both, and how long a value is held.

The free set is closed. A gate is free only when it is

- a constant (``const``: `CONST_ID`);
- an s8-family core step (``family``, `FAMILY`: `BlackwellS8ImmaDot32_v1`, its u8 x s8 sibling `BlackwellU8S8ImmaDot32_v1`,
  and PoUW's wrapping dots `PouwSegDot_v1`, `PouwSegDot0_v1` (C = 0) and `PouwDot16U8S8_v1`, ``D = C + sum_k a_k * b_k``)
  whose window the interval pass proves: D's bound lies in s32, from its operands' ranges as their producers give them, so
  it never clamps or wraps. Their Definitions on bits (`_v2`) are listed in the family for the census, as Lean's `classOf`
  lists them; the pass reads their bodies down to the 1-bit gates, never as a step whose window it proves; or
- an integer affine map with constant coefficients (``affine``: `I32Add_v1`, `I32Sub_v1`, `I32Mul_v1` with a constant
  operand, and the BF16-to-FP32 widening `Bf16ToF32_v1`, its operand's encoding times 2^16 read as signed) that the interval
  pass proves never wraps (the widening never does) over operand ranges within their width's signed reading: an operand's
  range past it is never clipped to it, and the map is budgeted.

Every other primitive the table lists is budgeted, one unit per output value (a primitive returns one value), in one of four
classes: ``roundings`` (FP arithmetic, casts and dots, special functions, integer floors, mods and narrowing, integer products
reduced mod 2^w (`I32Mul_v1` of two runtime values, the integer dots outside the family), bit operations on words and field
extractions, limb steps, and an add or subtract that may wrap), ``compare_select`` (comparisons, selections, min, max and clamps,
`F32Sat_v1` and `S32ClampS8` among them, the 1-bit logic ops, the top-p keep mask, the lifting's tag tests), ``gathers`` (a
data-dependent read of one of V words, counted in fresh index bits) and ``folds`` (the receipt folds' steps, `PouwDuoAbsorb_v1`
and fx1's `PouwFx1Absorb`). Only gates with an operand that depends on a run input are budgeted: a gate of constants and
registered words computes the same function before the seed. A primitive the table doesn't list is refused. The table (`CLASSES`, and
the families `class_of` reads) is Lean's `Accounting.Gates.classOf` id for id where Lean lists the id (`Bf16ToF32_v1` affine
there too), and a closed table for the families built on demand that Lean doesn't list yet. The exact-int8 profile's
primitives (vLLM's `int8_exact`) aren't in Lean's registry yet: its casts, f64 products and `LutIndex_v1` (min(floor(max(d, 0)
a / 2^32), 255)) are roundings, and `Exp2Lut127_v1` is a gather.

A gather's index bits are those that tell its outcomes apart (`GATHERS`): ceil(log2(V + 1)) for an entry of V or the family's
out-of-range constant, n for the entry at the index's low n bits (8 for `Exp2Lut127_v1`, an 8-bit index into a constant table of
256 entries), and ceil(log2(V + 3)) for a lifted row of V (or ⊥, an invalid index, or out of range). Each distinct index a body
reads counts once per kind, so one routing index read by many gathers counts
once; a constant index is wiring. A gather whose index and column both depend on a run input pays one ``gathers`` unit more, as
Lean's `Gates.units` does: it is a V-way select among run-time values, and without that unit a tree of gathers sharing indices
would compute any function of b index bits for b units. A field extraction (`BitAtx{W}`, `PouwWindowWord_v1`) is a rounding, not a gather: at a
constant position it still computes. A lift (`Lifted[P]_v1`, `Lifted[P]_v2`, ``P{ORD=k}``) is P's class with a tag test, so a
free P's lift is budgeted (a constant's a select on the enable, the rest roundings). A lifted gather also scans every operand's
tag (any invalid row gives INV), so it pays one ``compare_select`` unit beside its index bits, and it isn't exempt from B.

Restricted B (Lean's `Gates.Restricted`, its `OperandBits B`). A budgeted gate other than an unlifted gather reads at most B
operand bits: the widths of its operands' leaves, summed (at least the bits of their declared intervals, which Lean counts). A
primitive past it (an FP or integer dot, PoUW's 576-bit squares, a lifted gather's rows) is refused: its decomposition into gates
of B operand bits isn't charged. B is 128 by default (`OPERAND_BITS`), the red team's.

Ranges. A constant is its value, a proven core step its D, a proven affine map its exact result, a gate `RANGES` lists what its
rule derives from its operands' ranges (`fmaxf` and `fminf` of a non-negative float, and the low byte of an f32's truncation
below 128.0, over f32 bit patterns read as s32, which order +0..+inf as the floats; `Exp2Lut127_v1`'s table's hull with 0, as
Lean's `hull0`, [0, 127] over the table its evaluator holds; `S32LowByte`'s code of a word in s8, the word itself, so the LUT
softmax's P keeps [0, 127] and its P V' is proven to 132,096 keys, Lean's `LutPvWindow`), and every other value any word of its
width (an s8 operand -128..127, an accumulator all of s32). Port types (`ports`, PROTOCOL.md §12) narrow a range, read through
`ports.reading` as signed: a primitive's or Definition's value lies in the interval it declares, or a primitive's in what its
transfer gives from its operands' ranges. A range comes from its producer and the producer's declaration, never from a reader's
bound, and never from an input's declared bound (its census fact): evaluation doesn't enforce it, so no declared ``In(bound)`` is
taken on trust and a run input spans its width. So an accumulator from a run input, a split-K combine that may wrap, or a chain
long enough to clamp is never proven. A `scan`'s carry is iterated to a fixpoint, exactly up to 2^16
iterations and then widened to any word.

A word is read at its own width. A reference to a leaf of another width than the reading position's type declares (an s32
word read back as s8 lanes: a limb split no gate computes) is a ``reinterpretation``, refused wherever it occurs, an argument
or a body's return.

P(c, R_in, L) holds, at B operand bits, when under the audit's windows (`Window`: a range of the census's steps, the bytes the
audit certified as read in it and its duration):

1. ``unlisted``: no gate is refused;
2. ``reinterpretation``: no word is read at another width (the sites are listed);
3. ``operand-bits``: the Program is Restricted B (the primitives that aren't are listed with their operand bits);
4. ``s8-window``: every core step is proven in its window (the sites that aren't are listed);
5. ``uncovered``: every step lies in a window;
6. ``roundings``, ``compare_select``, ``gathers``, ``folds``: in each window, each class's units are at most c[class] per
   certified byte read;
7. ``inputs``: in each window, the run-input bytes the Program first reads there are at most R_in (bytes per second, 100 KB/s
   by default) times its duration;
8. ``lifetime``: no dynamic value (computed, or a run input) is held more than L steps. A computed value is born in its
   producer's step, a run input in the step before the first that reads it (or has it among a selection's candidates), and
   it is held until the last such step. L None (the default) leaves lifetimes to the audit's re-keying, and the verdict
   reports the longest.

The sink, coin and limb-split types of the s8 family's companions (the receipt fold, the noise derivation, `LimbSplit`) are
circuits' Build types; their primitives are budgeted gate by gate, never free (`LimbSplit`'s byte, carry and clamp, and the
H1 + fx1 tile's noise, digit, one-stream and fold primitives, v4's dead-tile mask and J10-R's call seeds and block counter,
each with its Definition on bits, are listed). Bits become bytes exactly (bits / 8), and every comparison is exact (`fractions.Fraction`): c, R_in and a window's
duration are given as an int, a Fraction or a str, and a float, which would be read at its binary value, is refused.
"""
from __future__ import annotations

import hashlib
import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral, Rational
from typing import Any, Callable, Iterator, Mapping, Sequence

from . import ports
from .census import SCHEMA as CENSUS, step_of
from .codec import canonical_json
from .defs import Body, CompositeDefinition, PrimitiveDefinition, bind
from .intervals import norm, ref_intervals
from .program import RUN, input_id, is_input
from .refs import Affine, Concat, Explicit, Refs, Strided
from .types import Array, Tuple, Type

TAG = "verity/stacking-property"
VERSION = 1
SCHEMA = f"{TAG}/v{VERSION}"

BUDGETED = ("roundings", "compare_select", "gathers", "folds")
REFUSED = ("unlisted", "reinterpretation")

S8 = (-(1 << 7), (1 << 7) - 1)
U8 = (0, (1 << 8) - 1)
S16 = (-(1 << 15), (1 << 15) - 1)
S32 = (-(1 << 31), (1 << 31) - 1)
#: R_in's default: bytes per second
R_IN = 100_000
#: Restricted B's default: the operand bits a budgeted gate other than an unlifted gather reads
OPERAND_BITS = 128

#: a constant primitive's id (`PROTOCOL.md` §9)
CONST_ID = re.compile(r"^Const([1-9][0-9]*)\[0x([0-9a-f]+)\]_v1$")
#: the gathers by id: the index operand and the kind of its bits, of n (the id's, or the fixed n given; ``range``:
#: ceil(log2(n + 1)), an entry of n or the out-of-range constant; ``low``: n, the index's low n bits; ``lifted``:
#: ceil(log2(n + 3)), a row of n, ⊥, an invalid index or out of range)
GATHERS: tuple[tuple[str, int, str, int | None], ...] = (
    (r"^Gather(?:Bf16|F32|I32|E4m3|E2m1|E8m0)x([1-9][0-9]*)_v1$", 0, "range", None),  # (index: i32, column: Array<V, word>)
    (r"^TableRead([1-9][0-9]*)x[1-9][0-9]*_v1$", 0, "low", None),  # (index, table)
    (r"^LGatherRow[1-9][0-9]*x([1-9][0-9]*)(?:o[0-9a-f]+)?_v1$", 0, "lifted", None),  # (idx: Value<33>, rows: Array<N, Value<w+1>>)
    (r"^Exp2Lut127_v1$", 0, "low", 8),  # (t: Value<8>): rint(127 * 2^(-t/16)), a constant table of 256 entries
)
_BITS: dict[str, Callable[[int], int]] = {"range": lambda n: n.bit_length(), "low": lambda n: n,
                                          "lifted": lambda n: (n + 2).bit_length()}
#: the other families built on demand, by id, and their class
FAMILIES: tuple[tuple[str, str], ...] = (
    (r"^BitAtx[1-9][0-9]*_v1$", "roundings"),  # (word, i): bit i of one runtime word, a field extraction
    (r"^(?:And|Xor)(?:[2-9]|[1-9][0-9]+)_v1$", "compare_select"),  # Boolean logic on n bits, as And2_v1 and Xor2_v1
    (r"^TopPMaskWordx[1-9][0-9]*_v1$", "compare_select"),  # the top-p keep mask of V lanes
    (r"^L(?:Select[1-9][0-9]*|FixedSelect[1-9][0-9]*x[1-9][0-9]*|IsActive[1-9][0-9]*|IsEOS[1-9][0-9]*|StepLive[1-9][0-9]*"
     r"|Enable|EmitNext\[H=[1-9][0-9]*\])_v1$", "compare_select"),  # the lifting's tag tests and selects
    (r"^L?SplitsForSMS[1-9][0-9]*_v1$", "roundings"),  # an integer division of a live count
    (r"^LiveCount[1-9][0-9]*_v1$", "roundings"),  # the sum of R bits
    (r"^LBot[1-9][0-9]*_v1$", "const"),  # the ⊥ encoding
)
#: the lifts of a primitive P, by id: P's class with a tag test
LIFTS = (r"^Lifted\[(.+)\]_v[12]$", r"^(.+)\{ORD=[0-9]+\}$")
_GATHERS = tuple((re.compile(p), i, k, n) for p, i, k, n in GATHERS)
_FAMILIES = tuple((re.compile(p), c) for p, c in FAMILIES)
_LIFTS = tuple(re.compile(p) for p in LIFTS)
_LIFTED = {"const": "compare_select", "affine": "roundings", "family": "roundings"}


@dataclass(frozen=True)
class Step:
    """An s8-family core step's shape: k products a step, parameters (acc, a, b), a and b in ``a`` and ``b``, D in s32; with
    no ``acc`` parameter (``acc`` False) it is the step from 0."""

    k: int
    a: tuple[int, int]
    b: tuple[int, int]
    acc: bool = True


FAMILY: dict[str, Step] = {
    **dict.fromkeys(("BlackwellS8ImmaDot32_v1", "BlackwellS8ImmaDot32_v2", "PouwSegDot_v1", "PouwSegDot_v2"),
                    Step(32, S8, S8)),
    **dict.fromkeys(("PouwSegDot0_v1", "PouwSegDot0_v2"), Step(32, S8, S8, acc=False)),
    **dict.fromkeys(("BlackwellU8S8ImmaDot32_v1", "BlackwellU8S8ImmaDot32_v2"), Step(32, U8, S8)),
    "PouwDot16U8S8_v1": Step(16, U8, S8),
}
#: the primitives the pass tries as affine maps; `I32Mul_v1` is one only with a constant operand, and `Bf16ToF32_v1` is
#: its operand times 2^16
AFFINE = ("I32Add_v1", "I32Sub_v1", "I32Mul_v1", "Bf16ToF32_v1")


def _ids(names: str) -> list[str]:
    return names.split()


CLASSES: dict[str, str] = {
    **dict.fromkeys(FAMILY, "family"),
    **dict.fromkeys(("I32Add_v1", "I32Sub_v1", "Bf16ToF32_v1"), "affine"),
    **dict.fromkeys(_ids(
        "AmpereBF16TcDot16_v2 Bf16AddF2fp_v1 Bf16Add_v1 Bf16MulBf16_v1 Bf16MulF32_v1 "
        "BlackwellE4m3QmmaDot32_v1 BlackwellE5m2QmmaDot32_v1 BlackwellMxf4OmmaDot64_v1 "
        "BlackwellNvf4OmmaDot64_v1 DivFullRcp_v1 DivFullScaleA_v1 DivFullScaleA_v2 E2m1ToBf16_v1 E4m3ToF32_v1 "
        "E8m0DequantBf16_v1 E8m0ShlBf16_v1 E8m0ToBf16_v1 F2fpBf16_v1 F32AddFtz_v1 F32AddFtz_v2 F32Add_v1 F32Add_v2 "
        "F32BitsShl23_v1 F32CeilFtz_v1 F32Div_v1 F32Div_v2 F32ErfRn_v1 F32ExpRn_v1 F32Fabs_v1 F32FmaFtz_v1 F32FmaFtz_v2 "
        "F32FmaRm_v1 F32FmaRm_v2 F32FmaSubFtz_v1 F32FmaSubFtz_v2 F32Fma_v1 F32Fma_v2 F32MulFtz_v1 F32MulFtz_v2 F32Mul_v1 "
        "F32MulWide_v1 F32Mul_v2 F32Neg_v1 F32Rint_v1 F32Sqrt_v1 F32SubFtz_v1 F32SubFtz_v2 F32Sub_v1 F32Sub_v2 "
        "F32TanhRn_v1 F32ToBf16Rn_v1 F32ToE2m1Sat_v1 F32ToE4m3Sat_v1 F32ToS8Rzi_v1 F32ToU32RzByte_v1 F64Mul_v1 "
        "F64ToU32RniSat_v1 Fa2InvSum_v1 Fa3InvSum_v1 GeluErfBf16_v1 "
        "GeluTanhMulBf16_v1 GumbelNoiseLane_v1 GumbelNoiseLane_v2 GumbelStreamKey_v1 HopperBF16WgmmaDot16_v1 "
        "HopperE4m3MmaV2Dot32_v1 HopperE4m3QgmmaDot32_v1 HopperTf32WgmmaDot8_v1 I32Div_v1 I32Mul_v1 I32Rem_v1 I32ToF32_v1 "
        "I64ToI32_v1 LutIndex_v1 MufuEx2Ftz_v1 MufuLg2Mantissa_v1 MufuRcpFtz_v1 MufuSqrtFtz_v1 MufuTanh_v1 NcpDot16I7_v1 "
        "NcpRneClampI7_v1 "
        "NvLogf_v1 NvLogf_v2 PouwAdd64Hi_v1 PouwAdd64Lo_v1 PouwAdd8_v1 PouwAddSub8_v1 PouwBf16Neg_v1 "
        "PouwBlackwellE4m3Window_v1 PouwCh32_v1 PouwChi32_v1 PouwETilde_v1 PouwF32SqrtRn_v1 PouwFormX3_v1 PouwFormX_v1 "
        "PouwHopperE4m3Window_v1 PouwI32ToF32Rn_v1 PouwMaj32_v1 PouwNegBetaSum16_v1 PouwNoiseByte_v1 PouwPackBE4_v1 "
        "PouwRotl64Hi_v1 PouwRotl64Lo_v1 PouwSha512BigSigma0Hi_v1 PouwSha512BigSigma0Lo_v1 PouwSha512BigSigma1Hi_v1 "
        "PouwSha512BigSigma1Lo_v1 PouwSha512SmallSigma0Hi_v1 PouwSha512SmallSigma0Lo_v1 PouwSha512SmallSigma1Hi_v1 "
        "PouwSha512SmallSigma1Lo_v1 PouwSqAcc_v1 PouwSqCeil_v1 PouwSub8_v1 PouwWindowWord_v1 PouwXor32_v1 RopeOutAdd_v1 "
        "RopeOut_v1 RsqrtApprox_v1 S32ByteCarry_v1 S32ByteCarry_v2 S32ClampRest_v1 S32ClampRest_v2 "
        "S32LowByte_v1 S32LowByte_v2 S32ToF32Rn_v1 S8ToF32_v1 SiluMulBf16_v1 SiluMulBf16_v2 TanhF32Rn_v1 "
        "TopPNumkeep_v1 U32And_v1 U32Xor_v1 Ue8m0CeilByte_v1 Ue8m0InvScale_v1 "
        "PouwDigitHi_v1 PouwDigitHi_v2 PouwDigitLo_v1 PouwDigitLo_v2 PouwI8AddBit_v1 PouwI8AddBit_v2 PouwI8Add_v1 "
        "PouwI8Add_v2 PouwI8Neg_v1 PouwI8Neg_v2 PouwLow8_v1 PouwLow8_v2 PouwNoiseBlock_v1 PouwNoiseBlock_v2 "
        "PouwOneStream2_v1 PouwOneStream2_v2 PouwOneStream3_v1 PouwOneStream3_v2 PouwPeelSeg_v1 PouwPeelSeg_v2 "
        "PouwResAdd_v1 PouwResAdd_v2 PouwSx8_v1 PouwSx8_v2 PouwU129_v1 PouwU129_v2 PouwUnitSeeds_v1 PouwUnitSeeds_v2 "
        "PouwWrapSeg_v1 PouwWrapSeg_v2 PouwXinFinal_v1 PouwXinFinal_v2 PouwXor32_v2 "
        "PouwCallSeeds_v1 PouwCallSeeds_v2 PouwAdd64_v1 PouwAdd64_v2"
    ), "roundings"),
    **dict.fromkeys(_ids(
        "Bf16GtStrict_v1 F32Eq_v1 F32Fmaxf_v1 F32Fminf_v1 F32Geu_v1 F32GtStrict_v1 F32IsFinite_v1 F32MaxRef_v1 F32Max_v1 "
        "GuardNegInfZero_v1 I32Eq_v1 I32Le_v1 NcpAbsMax_v1 PouwBlackwellE4m3SaltDead_v1 PouwHopperE4m3SaltDead_v1 "
        "PouwSel32_v1 PouwSqGe_v1 PouwWinOk_v1 SelectBf16_v1 SelectBit_v1 SelectF32_v1 SelectI32_v1 U32Ge_v1 "
        "F32Sat_v1 S32ClampS8_v1 S32ClampS8_v2 And2_v1 Xor2_v1 BitAnd_v1 BitOr_v1 BitXor_v1 BitNot_v1 Not_v1 "
        "PouwDigitZ_v1 PouwDigitZ_v2 PouwWrapBits_v1 PouwWrapBits_v2 PouwXWrap_v1 PouwXWrap_v2 "
        "PouwOrBit_v1 PouwOrBit_v2 PouwMask32_v1 PouwMask32_v2"
    ), "compare_select"),
    **dict.fromkeys(("PouwDuoAbsorb_v1", "PouwFx1Absorb_v1", "PouwFx1Absorb_v2"), "folds"),
}

def _unlift(fid: str) -> tuple[str, bool]:
    """The primitive a lift (`LIFTS`) wraps, innermost, and whether `fid` is one."""
    lifted = False
    while True:
        m = next((m for p in _LIFTS if (m := p.fullmatch(fid))), None)
        if m is None:
            return fid, lifted
        fid, lifted = m[1], True


def gather(fid: str) -> tuple[int, int, str, bool] | None:
    """A gather's index operand, its index bits and their kind (`GATHERS`), and whether it is lifted (`LGatherRow`, or a lift
    of a gather: its tag scan reads every operand); None when `fid` isn't a gather."""
    inner, lifted = _unlift(fid)
    for pat, i, kind, n in _GATHERS:
        m = pat.fullmatch(inner)
        if m:
            return i, _BITS[kind](n if n is not None else int(m[1])), kind, lifted or kind == "lifted"
    return None


def _column(fid: str) -> bool:
    """Whether `fid` is an unlifted gather, whose operand bits are the column its index bits pay for."""
    g = gather(fid)
    return g is not None and not g[3]


def class_of(fid: str) -> str | None:
    """A primitive's class by its id: `CLASSES`, else ``const``, ``gathers`` or a `FAMILIES` class by family, else a lift's
    (P's class, a free P's budgeted), else None (unlisted, refused)."""
    c = CLASSES.get(fid)
    if c is not None:
        return c
    inner, lifted = _unlift(fid)
    if lifted:
        c = class_of(inner)
        return _LIFTED.get(c, c) if c is not None else None
    if CONST_ID.fullmatch(fid):
        return "const"
    if gather(fid):
        return "gathers"
    return next((c for p, c in _FAMILIES if p.fullmatch(fid)), None)


def spec() -> dict:
    """The table as canonical data: what `digest` hashes."""
    return {"format": TAG, "version": VERSION, "budgeted": list(BUDGETED), "refused": list(REFUSED),
            "classes": dict(sorted(CLASSES.items())), "const": CONST_ID.pattern, "affine": list(AFFINE),
            "gathers": [list(g) for g in GATHERS], "families": [list(f) for f in FAMILIES], "lifts": list(LIFTS),
            "lifted": dict(sorted(_LIFTED.items())), "ranges": sorted(RANGES),
            "family": {f: {"k": s.k, "a": list(s.a), "b": list(s.b), "acc": list(S32) if s.acc else [0, 0]}
                       for f, s in sorted(FAMILY.items())}}


def digest() -> str:
    """``SHA-512(TAG || "/v" || version || 0 || canon(spec))``: the table a verdict names and a statement can pin."""
    return hashlib.sha512(f"{TAG}/v{VERSION}".encode() + b"\0" + canonical_json(spec())).hexdigest()


# ---- the pass: run dependence, ranges and classes ------------------------------------------------------------------------

Range = tuple[int, int] | None  # a value's signed range at its width; None: any word of its width
Segs = tuple[tuple[int, int, Range], ...]  # a collection's leaves [lo, hi) and their range, covering it
_EMPTY = (1, 0)  # the union's identity: an empty range
_EXACT_ITERATIONS = 1 << 16
_MAX_SEGS = 256  # a return with more segments is one segment, their union


@dataclass
class Tally:
    """What gates computed: budgeted units by class, refused gates by (code, primitive), proven core steps, their products
    and proven affine maps, and the core steps not proven by primitive. ``index``: the gather index bits read from the
    Definition's parameters, by index leaves (refs over its parameters), which its caller counts (`add` leaves them out)."""

    units: Counter = field(default_factory=Counter)
    refused: Counter = field(default_factory=Counter)
    family: int = 0
    macs: int = 0
    affine: int = 0
    unproven: Counter = field(default_factory=Counter)
    index: dict = field(default_factory=dict)

    def add(self, o: "Tally", m: int = 1) -> None:
        for mine, theirs in ((self.units, o.units), (self.refused, o.refused), (self.unproven, o.unproven)):
            for key, n in theirs.items():
                mine[key] += m * n
        self.family += m * o.family
        self.macs += m * o.macs
        self.affine += m * o.affine


def _union(a: Range, b: Range) -> Range:
    if a is None or b is None:
        return None
    if a[0] > a[1]:
        return b
    if b[0] > b[1]:
        return a
    return (min(a[0], b[0]), max(a[1], b[1]))


def _meet(a: Range, b: Range) -> Range:
    """Both ranges at once; any word when they share no value (one of them is wrong, and neither is taken)."""
    if a is None or b is None:
        return b if a is None else a
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if lo <= hi else None


def _port_range(iv: Any, w: int | None) -> Range:
    """A port interval of a `Value<w>` as a range: its encodings read as signed (`ports.as_reading`, through `ports.reading`);
    any word when it is none, holds nothing, or is no one signed interval."""
    return None if iv is None or w is None else ports.as_reading(tuple(iv), w)


def _declared(fn: PrimitiveDefinition, rng: Sequence[Range]) -> Range:
    """What a primitive's ports say of its value: what its transfer gives from its operands' ranges, else its declared interval."""
    pt = ports.port_types(fn)
    if pt is None:
        return None
    if pt.transfer is None:
        return next((_port_range(o.interval, w) for _lo, _hi, o, w in pt.ret if o.interval is not None), None)
    ivs = [None if r is None or r[0] > r[1] else ports.reading(r, _width(t, 0, t.leaves)) for r, (_n, t) in zip(rng, fn.params)]
    return _port_range(pt.transfer(*ivs), fn.ret.w)


def _declared_segs(segs: Segs, runs: Sequence[tuple]) -> Segs:
    """A Definition's return segments, each narrowed by the interval its ports declare there (`runs`: `PortTypes.ret`)."""
    cuts = sorted({x for lo, hi, o, _w in runs if o.interval is not None for x in (lo, hi)})
    out: list[tuple[int, int, Range]] = []
    for a, b, r in segs:
        for lo, hi in zip([a] + [x for x in cuts if a < x < b], [x for x in cuts if a < x < b] + [b]):
            got = next((_port_range(o.interval, w) for p, q, o, w in runs if p <= lo and hi <= q and o.interval is not None), None)
            out.append((lo, hi, _meet(r, got)))
    return _merge(out)


def _merge(segs: Sequence[tuple[int, int, Range]]) -> Segs:
    """Adjacent segments of one range as one."""
    merged: list[tuple[int, int, Range]] = []
    for a, b, r in segs:
        if merged and merged[-1][2] == r and merged[-1][1] == a:
            merged[-1] = (merged[-1][0], b, r)
        else:
            merged.append((a, b, r))
    return tuple(merged)


def _signed(w: int, v: int) -> int:
    return v - (1 << w) if v >> (w - 1) else v


def _clip(r: Range, bound: tuple[int, int]) -> tuple[int, int]:
    if r is None or r[0] > r[1]:
        return bound
    return (max(r[0], bound[0]), min(r[1], bound[1]))


def _within(r: tuple[int, int], bound: tuple[int, int]) -> bool:
    return bound[0] <= r[0] and r[1] <= bound[1]


def _operand(r: Range, fmt: tuple[int, int]) -> tuple[int, int]:
    """An operand's range in its format; a signed range that reaches below 0, read as unsigned, is the whole format."""
    return fmt if fmt[0] >= 0 and r is not None and r[0] < 0 else _clip(r, fmt)


def _point(r: Range) -> int | None:
    return r[0] if r is not None and r[0] == r[1] else None


def _in(r: Range, bound: tuple[int, int]) -> tuple[int, int] | None:
    """`r` within `bound`: `bound` when `r` is any word (or empty), None when it leaves `bound` (never clipped to it)."""
    if r is None or r[0] > r[1]:
        return bound
    return r if _within(r, bound) else None


def _affine(fid: str, rng: Sequence[Range]) -> tuple[int, int] | None:
    """An affine map's exact result range over s32 operands (the widening's over an s16 encoding), or None when it isn't
    affine (a product of two non-constants) or an operand's range leaves its width's signed reading."""
    if fid == "Bf16ToF32_v1":
        x = _in(rng[0], S16)
        return None if x is None else (x[0] << 16, x[1] << 16)
    a, b = rng[0], rng[1]
    x, y = _in(a, S32), _in(b, S32)
    if x is None or y is None:
        return None
    if fid == "I32Add_v1":
        return (x[0] + y[0], x[1] + y[1])
    if fid == "I32Sub_v1":
        return (x[0] - y[1], x[1] - y[0])
    k = _point(a) if _point(a) is not None else _point(b)
    if k is None:
        return None
    o = y if _point(a) is not None else x
    return (min(k * o[0], k * o[1]), max(k * o[0], k * o[1]))


F32_POS = (0, 0x7F800000)  # +0 to +inf, f32 bit patterns read as s32: ordered as the floats


def _f32_pos(r: Range) -> bool:
    return r is not None and r[0] <= r[1] and _within(r, F32_POS)


def _fmaxf(a: Range, b: Range) -> Range:
    """`fmaxf`: at least a non-negative operand, so non-negative itself (a NaN or -0 operand gives the other or +0)."""
    pa, pb = _f32_pos(a), _f32_pos(b)
    if pa and pb:
        return (max(a[0], b[0]), max(a[1], b[1]))  # type: ignore[index]
    if pa or pb:
        return ((a if pa else b)[0], F32_POS[1])  # type: ignore[index]
    return None


def _fminf(a: Range, b: Range) -> Range:
    """`fminf`: at most a non-negative operand (a NaN operand gives the other; a negative float or -0 reads below 0)."""
    pa, pb = _f32_pos(a), _f32_pos(b)
    if pa and pb:
        return (min(a[0], b[0]), min(a[1], b[1]))  # type: ignore[index]
    if pa or pb:
        return (S32[0], (a if pa else b)[1])  # type: ignore[index]
    return None


def _trunc(bits: int) -> int:
    return int(struct.unpack("<f", struct.pack("<I", bits))[0])


def _rz_byte(x: Range) -> Range:
    """`F32ToU32RzByte`'s low byte of `cvt.rzi.u32.f32` below 128.0: the truncation; a NaN, -0 or negative float (read below 0)
    gives 0."""
    if x is None or x[0] > x[1] or x[1] >= 0x43000000:
        return None
    return (_trunc(x[0]) if x[0] >= 0 else 0, _trunc(x[1]) if x[1] >= 0 else 0)


#: `Exp2Lut127_v1`'s table, rint(127 * 2^(-t/16)) for t < 256, as its evaluator (vLLM's `engine.int8_attention.LUT`, which
#: `verity` doesn't import) and Lean's `exp2Lut127` hold it: `round` is rint's half to even (t = 16, 63.5, is 64)
EXP2_LUT127_TABLE = tuple(round(127 * 2 ** (-t / 16)) for t in range(256))
#: `Exp2Lut127_v1`'s value: the hull of its table's entries and 0 (Lean's `hull0`)
EXP2_LUT127 = (min(0, *EXP2_LUT127_TABLE), max(0, *EXP2_LUT127_TABLE))


def _low_byte(y: Range) -> Range:
    """`S32LowByte`'s s8 code, sext8(y): y itself when y lies in s8, so P's codes keep the LUT's hull (Lean's
    `S32LowByteRange`)."""
    return y if y is not None and y[0] <= y[1] and _within(y, S8) else None


#: the budgeted gates whose value's range a rule derives from their operands' ranges (`F32_POS`)
RANGES: dict[str, Callable[..., Range]] = {"F32Fmaxf_v1": _fmaxf, "F32Fminf_v1": _fminf, "F32ToU32RzByte_v1": _rz_byte,
                                           "Exp2Lut127_v1": lambda _t: EXP2_LUT127, "S32LowByte_v1": _low_byte,
                                           "S32LowByte_v2": _low_byte}


def _colls(refs: Refs, cache: dict | None = None) -> frozenset[tuple[str, int]]:
    """The collections `refs` reads, as (space, index). `cache` keeps each refs it answers for beside its answer, so an id
    it holds is never reused."""
    if cache is not None and (hit := cache.get(id(refs))) is not None:
        return hit[1]
    if isinstance(refs, (Affine, Strided)):
        out = frozenset({(refs.space, refs.idx)}) if refs.count else frozenset()
    elif isinstance(refs, Concat):
        out = frozenset().union(*(_colls(p) for p in refs.parts))
    elif isinstance(refs, Explicit):
        out = frozenset((s, i) for s, i, _leaf in refs.items)
    else:
        out = frozenset((s, i) for s, i, _leaf in refs)
    if cache is not None:
        cache[id(refs)] = (refs, out)
    return out


def _dyn(refs: Refs, params: Sequence[bool], nodes: Sequence[bool], cache: dict | None = None) -> bool:
    return any(params[i] if s == "p" else nodes[i] for s, i in _colls(refs, cache))


def _seg_union(segs: Segs, lo: int = 0, hi: int | None = None) -> Range:
    out: Range = _EMPTY
    for a, b, r in segs:
        if b > lo and (hi is None or a < hi):
            out = _union(out, r)
    return out


def _intervals(refs: Refs, cache: dict | None = None) -> tuple[tuple[str, int, tuple[tuple[int, int], ...]], ...]:
    """The leaf intervals `refs` reads, by collection and normalized (none for a parameter, which is read whole); `cache`
    as `_colls`'s."""
    if cache is not None and (hit := cache.get(id(refs))) is not None:
        return hit[1]
    ivs: dict[tuple[str, int], list] = {}
    ref_intervals(refs, ivs)
    out = tuple((s, i, () if s == "p" else tuple(norm(iv))) for (s, i), iv in ivs.items())
    if cache is not None:
        cache[id(refs)] = (refs, out)
    return out


def _read(refs: Refs, params: Sequence[Range], segs: Sequence[Segs], cache: dict | None = None) -> Range:
    out: Range = _EMPTY
    for s, i, ivs in _intervals(refs, cache):
        if s == "p":
            out = _union(out, params[i])
            continue
        for lo, hi in ivs:
            out = _union(out, _seg_union(segs[i], lo, hi))
    return out


def _read_segs(refs: Refs, params: Sequence[Range], segs: Sequence[Segs]) -> Segs:
    """The segments of a body's return, in leaf order: an affine run of a node's leaves keeps that node's segments."""
    out: list[tuple[int, int, Range]] = []
    pos = 0
    for part in (refs.parts if isinstance(refs, Concat) else [refs]):
        n = part.count
        if n and isinstance(part, Affine) and part.space == "n":
            for a, b, r in segs[part.idx]:
                lo, hi = max(a, part.base), min(b, part.base + n)
                if lo < hi:
                    out.append((pos + lo - part.base, pos + hi - part.base, r))
        elif n:
            out.append((pos, pos + n, _read(part, params, segs)))
        pos += n
    merged = _merge(out)
    if len(merged) > _MAX_SEGS:
        return ((0, pos, _seg_union(merged)),)
    return merged


def _key(refs: Refs) -> Any:
    return tuple(refs) if refs.count <= 64 else canonical_json(refs.to_json()).decode()


def _bits(t: Type) -> int:
    """The widths of `t`'s leaves, summed."""
    w = t.homogeneous_width()
    return t.leaves * w if w is not None else sum(t.leaf_widths())


def _width(t: Type, lo: int, hi: int) -> int | None:
    """The one width of `t`'s leaves [lo, hi), or None when they have two (structurally, never leaf by leaf)."""
    w = t.homogeneous_width()
    if w is not None:
        return w
    if isinstance(t, Array):
        L = t.elem.leaves
        a, b = lo // L, (hi - 1) // L
        if a == b:
            return _width(t.elem, lo - a * L, hi - a * L)
        ws = {_width(t.elem, lo - a * L, L), _width(t.elem, 0, hi - b * L)} | ({None} if b - a > 1 else set())
    else:
        ws, pos = set(), 0
        for f in (t.fields if isinstance(t, Tuple) else tuple(f for _name, f in t.fields)):  # type: ignore[attr-defined]
            n = f.leaves
            if pos < hi and pos + n > lo:
                ws.add(_width(f, max(lo, pos) - pos, min(hi, pos + n) - pos))
            pos += n
    return ws.pop() if len(ws) == 1 else None


def _span(t: Type, pos: int, ct: Type, base: int, n: int) -> bool:
    a, b = _width(t, pos, pos + n), _width(ct, base, base + n)
    if a is not None or b is not None:
        return a == b
    h = n // 2
    return _span(t, pos, ct, base, h) and _span(t, pos + h, ct, base + h, n - h)


def _parts(refs: Refs) -> Iterator[Refs]:
    if isinstance(refs, Concat):
        for p in refs.parts:
            yield from _parts(p)
    elif refs.count:
        yield refs


def _at_width(t: Type, refs: Refs, types: Callable[[str, int], Type]) -> bool:
    """Whether each of `refs` names a leaf of the width `t` declares at its position (`types`: a collection's type)."""
    pos = 0
    for part in _parts(refs):
        n = part.count
        if isinstance(part, Affine):
            ok = _span(t, pos, types(part.space, part.idx), part.base, n)
        else:
            w, ivs = _width(t, pos, pos + n), {}  # type: ignore[var-annotated]
            ref_intervals(part, ivs)
            ok = w is not None and all(_width(types(s, i), lo, hi) == w for (s, i), iv in ivs.items() for lo, hi in norm(iv))
            ok = ok or all(_width(t, pos + k, pos + k + 1) == _width(types(s, i), j, j + 1) for k, (s, i, j) in enumerate(part))
        if not ok:
            return False
        pos += n
    return True


def _reinterpretations(body: Body) -> list[int | None]:
    """The nodes of `body` with an argument read at another width, and None when its return is."""
    def types(s: str, i: int) -> Type:
        return body.params[i][1] if s == "p" else body.nodes[i].out

    out: list[int | None] = [k for k, nd in enumerate(body.nodes) if not all(_at_width(a.type, a.refs, types) for a in nd.args)]
    return out + ([] if _at_width(body.ret_type, body.ret, types) else [None])


class _Pass:
    def __init__(self) -> None:
        self.memo: dict[tuple, tuple[Segs, bool, Tally]] = {}
        self.keep: list[Any] = []
        self.sites: dict[tuple[str, int], tuple[Range, Range]] = {}
        self.widths: dict[int, frozenset[int | None]] = {}
        self.shapes: dict[str, tuple[str | None, int]] = {}  # each primitive's class and operand bits
        self.colls: dict[int, tuple[Refs, frozenset[tuple[str, int]]]] = {}
        self.ivs: dict[int, tuple[Refs, tuple]] = {}

    def prim(self, fn: PrimitiveDefinition, dyn: Sequence[bool], rng: Sequence[Range], where: tuple[str, int]) -> tuple[Range, Tally]:
        t = Tally()
        cls = class_of(fn.id)
        if fn.id not in self.shapes:
            self.shapes[fn.id] = (cls, sum(_bits(ty) for _n, ty in fn.params))
        if cls is None:
            t.refused[("unlisted", fn.id)] += 1
            return None, t
        if cls == "const":
            m = CONST_ID.fullmatch(fn.id)
            v = _signed(int(m[1]), int(m[2], 16)) if m else None
            return ((v, v) if v is not None else None), t
        if cls == "family":
            st = FAMILY[fn.id]
            ab = rng[1:] if st.acc else rng
            c, a, b = _clip(rng[0], S32) if st.acc else (0, 0), _operand(ab[0], st.a), _operand(ab[1], st.b)
            corners = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
            s = (c[0] + st.k * min(corners), c[1] + st.k * max(corners))
            if _within(s, S32):
                t.family, t.macs = 1, st.k
                return s, t
            prev = self.sites.get(where)
            self.sites[where] = (c, s) if prev is None else (_union(prev[0], c), _union(prev[1], s))
            t.unproven[fn.id] += 1
            return (max(s[0], S32[0]), min(s[1], S32[1])), t
        if fn.id in AFFINE:
            r = _affine(fn.id, rng)
            if r is not None and _within(r, S32):
                t.affine = 1
                return r, t
            if cls == "affine":
                cls = "roundings"
        g = gather(fn.id) if cls == "gathers" else None
        if any(dyn) and (g is None or g[3]):
            t.units["compare_select" if g else cls] += 1
        if g and dyn[g[0]] and any(d for j, d in enumerate(dyn) if j != g[0]):
            t.units["gathers"] += 1
        rule = RANGES.get(fn.id)
        return (rule(*rng) if rule else None), t

    def call(self, fn: Any, dyn: tuple[bool, ...], rng: tuple[Range, ...], where: tuple[str, int]) -> tuple[Segs, bool, Tally]:
        if isinstance(fn, PrimitiveDefinition):
            r, t = self.prim(fn, dyn, rng, where)
            return ((0, 1, _meet(r, _declared(fn, rng))),), any(dyn), t
        key = (id(fn), dyn, rng)
        hit = self.memo.get(key)
        if hit is None:
            self.keep.append(fn)
            segs: list[Segs] = []
            dyns: list[bool] = []
            out = Tally()
            up: dict = {}
            for _k, t in self.walk(fn.body, dyn, rng, fn.id, segs, dyns, up):
                out.add(t)
            out.index = up
            ret = _read_segs(fn.body.ret, rng, segs)
            if (pt := ports.port_types(fn)) is not None and pt.ret:
                ret = _declared_segs(ret, pt.ret)
            hit = self.memo[key] = (ret, _dyn(fn.body.ret, dyn, dyns, self.colls), out)
        return hit

    def walk(self, body: Body, pdyn: Sequence[bool], prng: Sequence[Range], def_id: str, segs: list[Segs],
             dyns: list[bool], up: dict) -> Iterator[tuple[int, Tally]]:
        """Each node of `body` with the tally of its gates (a batch's and a scan's whole), filling its output's segments and
        run dependence, and a reinterpreting return as its last node's. Gather index bits, by index and kind: an index
        computed in this body counts once, in the tally of the first node that reads it; one read from the parameters goes in
        `up` for the caller; a batch member's own index (a mapped one) counts once a member."""
        bad = self.widths.get(id(body))
        if bad is None:
            self.keep.append(body)
            bad = self.widths[id(body)] = frozenset(_reinterpretations(body))
        fresh: dict[Any, int] = {}

        def note(refs: Refs, bits: int, kind: str, t: Tally) -> None:
            key = (kind, _key(refs))
            if any(s == "p" for s, _i in _colls(refs, self.colls)):
                up[key] = (refs, max(bits, up.get(key, (refs, 0))[1]), kind)
            elif bits > fresh.get(key, 0):
                t.units["gathers"] += bits - fresh.get(key, 0)
                fresh[key] = bits

        for k, nd in enumerate(body.nodes):
            if is_input(nd.fn):
                when, _w = input_id(nd.fn.id)  # type: ignore[misc]
                self.shapes.setdefault(nd.fn.id, ("input", 0))
                segs.append(((0, nd.out.leaves, None),))
                dyns.append(when == RUN)
                continue
            ad = [_dyn(a.refs, pdyn, dyns, self.colls) for a in nd.args]
            ar = [_read(a.refs, prng, segs, self.ivs) for a in nd.args]
            if nd.form == "scan":
                s, od, t = self.scan(nd, ad, ar, def_id, k)
            else:
                s, od, t0 = self.call(nd.fn, tuple(ad), tuple(ar), (def_id, k))
                batch = nd.form == "batch"
                if batch:
                    s = ((0, nd.out.leaves, _seg_union(s)),)
                t = Tally()
                t.add(t0, nd.n if batch else 1)
                for refs, bits, kind in t0.index.values():
                    leaves = list(refs)
                    if batch and any(nd.axes[i] is not None for _s, i, _j in leaves):
                        t.units["gathers"] += nd.n * bits
                    else:
                        note(Explicit(tuple(nd.args[i].refs[j] for _s, i, j in leaves)), bits, kind, t)
                g = gather(nd.fn.id) if isinstance(nd.fn, PrimitiveDefinition) else None
                if g and ad[g[0]]:
                    i, bits, kind, _lifted = g
                    if batch and nd.axes[i] is not None:
                        t.units["gathers"] += nd.n * bits
                    else:
                        note(nd.args[i].refs, bits, kind, t)
            if k in bad:
                t.refused[("reinterpretation", def_id)] += 1
            segs.append(s)
            dyns.append(od)
            yield k, t
        if None in bad:
            t = Tally()
            t.refused[("reinterpretation", def_id)] += 1
            yield len(body.nodes) - 1, t

    def scan(self, nd: Any, ad: list[bool], ar: list[Range], def_id: str, k: int) -> tuple[Segs, bool, Tally]:
        cl = nd.out.fields[0].leaves
        d = tuple([any(ad)] + ad[1:])
        carry, rest = ar[0], tuple(ar[1:])
        ys: Range = _EMPTY
        out = Tally()
        for i in range(nd.n):
            if i == _EXACT_ITERATIONS:
                carry = None
            s, _od, t = self.call(nd.fn, d, (carry,) + rest, (def_id, k))
            nxt = _union(carry, _seg_union(s, 0, cl))
            ys = _union(ys, _seg_union(s, cl, None))
            reps = nd.n - i if nxt == carry else 1
            out.add(t, reps)
            if t.index:
                out.units["gathers"] += reps * sum(bits for _refs, bits, _kind in t.index.values())  # an iteration's index counts anew
            if nxt == carry:
                break
            carry = nxt
        return ((0, cl, carry), (cl, nd.out.leaves, ys)), d[0], out


# ---- the analysis and the check -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Analysis:
    """What P reads of a Program: per step (the census's) its budgeted units by class and the run-input bits it first reads;
    over the Program its primitives and inputs (``gates``: each one's class, instances and operand bits), its
    refusals (gates by primitive, reinterpretations by the Definition whose body reads at another width), proven core steps
    with their products and proven affine maps, the core steps not proven with their sites, and the longest a dynamic value is
    held."""

    steps: tuple[dict, ...]
    gates: dict
    refused: dict
    family: int
    macs: int
    affine: int
    unproven: dict
    sites: tuple[dict, ...]
    max_lifetime: int

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "classes": digest(), "steps": list(self.steps), "gates": self.gates, "refused": self.refused,
                "family": self.family, "macs": self.macs, "affine": self.affine, "unproven": self.unproven,
                "sites": list(self.sites), "max_lifetime": self.max_lifetime}


def count(fn: Any, dyn: Sequence[bool], ranges: Sequence[Range] | None = None) -> Tally:
    """The tally of one call of `fn` (a Definition or a primitive) whose parameters depend on a run input as `dyn` says, with
    value ranges `ranges` (None: any word)."""
    if isinstance(fn, CompositeDefinition):
        fn = bind(fn)
    rng = tuple(ranges) if ranges is not None else (None,) * len(dyn)
    p = _Pass()
    if isinstance(fn, PrimitiveDefinition):
        return p.call(fn, tuple(dyn), rng, (fn.id, 0))[2]
    out = Tally()
    up: dict = {}
    for _k, t in p.walk(fn.body, tuple(dyn), rng, fn.id, [], [], up):
        out.add(t)
    if up:
        out.units["gathers"] += sum(bits for _refs, bits, _kind in up.values())
    return out


def analyze(program: Any, census: Mapping) -> Analysis:
    """P's reading of `program` with its census (verity/census/v1 of this Program: its steps, and its root values' producers,
    bits, readers, selections and facts). A census of another schema or of another root is refused (ValueError)."""
    nodes = program.root.body.nodes
    if census.get("schema") != CENSUS or len(census["nodes"]["callee"]) != len(nodes) or census["inputs"] != sum(
            1 for nd in nodes if is_input(nd.fn)):
        raise ValueError(f"a {CENSUS} of this Program's root ({len(nodes)} nodes), not {census.get('schema')!r} with "
                         f"{len(census.get('nodes', {}).get('callee', []))}")
    S = len(census["steps"]["ends"])
    step = step_of(census)
    units = [Counter() for _ in range(S)]
    whole = Tally()
    vals = census["values"]
    p = _Pass()
    for k, t in p.walk(program.root.body, (), (), program.root.id, [], [], {}):
        whole.add(t)
        for cls, n in t.units.items():
            units[step(k)][cls] += n
    I = census["inputs"]
    binding = [census["classes"][census["callees"][i]]["binding"] for i in census["nodes"]["callee"][:I]]
    pools: dict[int, set[int]] = {}
    for k, vs in zip(census["selects"]["call"], census["selects"]["values"]):
        for v in vs:
            pools.setdefault(v, set()).add(step(k))
    first = [0] * S
    longest = 0
    for v, (j, bits, reads) in enumerate(zip(vals["producer"], vals["bits"], vals["reads"])):
        touched = {step(k) for k in reads} | pools.get(v, set())
        if j >= I:
            birth = step(j)
        elif binding[j] != "run" or not touched:
            continue
        else:
            birth = min(touched) - 1
            first[birth + 1] += bits
        longest = max(longest, max(touched, default=birth) - birth)
    steps = tuple({"units": {c: units[s][c] for c in BUDGETED}, "input_bits": first[s]} for s in range(S))
    refused: dict[str, dict[str, int]] = {}
    for (code, fid), n in sorted(whole.refused.items()):
        refused.setdefault(code, {})[fid] = n
    sites = tuple({"definition": d, "node": n, "acc": list(c), "sum": list(r)} for (d, n), (c, r) in sorted(p.sites.items()))
    gates = {g: {"class": cls, "instances": census["classes"][g]["instances"], "operand_bits": b}
             for g, (cls, b) in sorted(p.shapes.items())}
    return Analysis(steps, gates, refused, whole.family, whole.macs, whole.affine, dict(sorted(whole.unproven.items())), sites,
                    longest)


@dataclass(frozen=True)
class Window:
    """A window: a range of the census's steps, the bytes the audit certified as read in it, and its duration in seconds."""

    steps: range
    reads: int
    seconds: Any


@dataclass(frozen=True)
class Verdict:
    """Whether P holds, why not (``reasons``: each a ``code`` with its numbers) and the numbers it read."""

    ok: bool
    reasons: tuple[dict, ...]
    numbers: dict

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "classes": digest(), "ok": self.ok, "reasons": list(self.reasons), "numbers": self.numbers}


def check(program: Any, census: Mapping, *, c: Mapping[str, Any], windows: Sequence[Window], R_in: Any = R_IN,
          L: int | None = None, B: int = OPERAND_BITS) -> Verdict:
    """P(c, R_in, L) at B operand bits on `program` and its census under the audit's `windows`. c maps each budgeted class to
    its units per certified byte read, R_in is bytes per second, L is steps (None: left to re-keying). c, R_in and a window's
    seconds are exact: an int, a Fraction or a str `Fraction` reads. A window outside the census's steps, a c without every
    budgeted class, a float among the exact numbers, or a B below 1 is refused (ValueError)."""
    a = analyze(program, census)
    return judge(a, c=c, windows=windows, R_in=R_in, L=L, B=B)


def _exact(x: Any, what: str) -> Fraction:
    """`x` as the rational it names. A float is refused: `Fraction` reads it at its binary value (0.1 a little past 1/10)."""
    if isinstance(x, float):
        raise ValueError(f"{what} is the float {x!r}, read at its binary value: give it exactly, as Fraction({str(x)!r})")
    if not isinstance(x, (Rational, str)):
        raise ValueError(f"{what} is an int, a Fraction or a str such as '1/10', not {x!r}")
    return Fraction(x)


def judge(a: Analysis, *, c: Mapping[str, Any], windows: Sequence[Window], R_in: Any = R_IN, L: int | None = None,
          B: int = OPERAND_BITS) -> Verdict:
    """P(c, R_in, L) at B operand bits on an `Analysis` (as `check`)."""
    if set(c) != set(BUDGETED):
        raise ValueError(f"c gives units per certified byte for exactly {list(BUDGETED)}, not {sorted(c)}")
    if not (isinstance(B, int) and not isinstance(B, bool) and B >= 1):
        raise ValueError(f"B is an integer >= 1, not {B!r}")
    cq = {k: _exact(c[k], f"c[{k!r}]") for k in BUDGETED}
    R = _exact(R_in, "R_in")
    S = len(a.steps)
    for win in windows:
        if (not isinstance(win.steps, range) or win.steps.step != 1 or not 0 <= win.steps.start < win.steps.stop <= S
                or not isinstance(win.reads, Integral) or win.reads < 0
                or _exact(win.seconds, "a window's seconds") <= 0):
            raise ValueError(f"a window is a non-empty range of the census's {S} steps with an integer reads >= 0 and seconds "
                             f"> 0, not {win}")
    reasons: list[dict] = [{"code": code, "definitions" if code == "reinterpretation" else "primitives": a.refused[code]}
                           for code in REFUSED if code in a.refused]
    wide = {g: s["operand_bits"] for g, s in a.gates.items()
            if s["class"] in BUDGETED and s["operand_bits"] > B and not _column(g)}
    if wide:
        reasons.append({"code": "operand-bits", "B": B, "primitives": wide})
    if a.unproven:
        reasons.append({"code": "s8-window", "steps": a.unproven, "sites": list(a.sites)})
    covered = set().union(*(set(win.steps) for win in windows)) if windows else set()
    if len(covered) < S:
        reasons.append({"code": "uncovered", "steps": sorted(set(range(S)) - covered)})
    rows = []
    for win in windows:
        u = {k: sum(a.steps[s]["units"][k] for s in win.steps) for k in BUDGETED}
        inb, sec = Fraction(sum(a.steps[s]["input_bits"] for s in win.steps), 8), Fraction(win.seconds)
        row = {"steps": [win.steps.start, win.steps.stop], "reads": win.reads, "seconds": str(sec), "units": u,
               "input_bytes": str(inb)}
        rows.append(row)
        for k in BUDGETED:
            if u[k] > cq[k] * win.reads:
                reasons.append({"code": k, **row, "c": str(cq[k])})
        if inb > R * sec:
            reasons.append({"code": "inputs", **row, "R_in": str(R)})
    if L is not None and a.max_lifetime > L:
        reasons.append({"code": "lifetime", "max_lifetime": a.max_lifetime, "L": L})
    total = {k: sum(s["units"][k] for s in a.steps) for k in BUDGETED}
    numbers = {"c": {k: str(v) for k, v in cq.items()}, "R_in": str(R), "L": L, "B": B, "windows": rows, "units": total,
               "family_steps": a.family, "macs": a.macs, "affine": a.affine, "max_lifetime": a.max_lifetime}
    return Verdict(not reasons, tuple(reasons), numbers)


__all__ = ["AFFINE", "Analysis", "BUDGETED", "CLASSES", "CONST_ID", "EXP2_LUT127", "EXP2_LUT127_TABLE", "F32_POS", "FAMILIES",
           "FAMILY", "GATHERS", "LIFTS", "OPERAND_BITS", "RANGES", "REFUSED", "R_IN", "S16", "S32", "S8", "SCHEMA", "Step",
           "TAG", "Tally", "U8", "VERSION", "Verdict", "Window", "analyze", "check", "class_of", "count", "digest", "gather",
           "judge", "spec"]
