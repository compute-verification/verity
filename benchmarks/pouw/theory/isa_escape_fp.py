"""ISA-ESCAPE-FP: ISA-ESCAPE's menu for FP words, per device, mechanically.

Three committed words, each credited at its device's row of the data sheet (A3):
  (i)   an FP8 word: a sum of products of FP8-grid operands (an integer grid such as int5 or the E4M3 integers, or every
        finite E4M3 value for an FP-native design), credited at the E4M3 row;
  (ii)  an FP4 word: a sum of products of E2M1-grid operands with block scales, credited at the block-scaled FP4 row
        (the E4M3 row on H200, which has no FP4 kind);
  (iii) an int8 word computed on FP pipes (C1's k3/k4/k9), credited at the int8 row (IMMA, wgmma s8, kind::i8).
on the RTX PRO 6000 (sm_120), H200 (sm_90a) and B200 (sm_100a).

For every tensor form that could head such a word the script takes its rate against the credited row (rho: the data
sheet's dense MACs, twice that for a 2:4 form's logical MACs) and a lower bound D on the passes a word needs on it, and
closes the form when rho / D <= 1. D is a COUNTING bound that reads no FP semantics: with A = e_l the word is y_l (the
alphabets hold 0 and 1), so B's codes and scales must determine every element. For an ELEMENTWISE route (each element's
product formed in J slots of its own, the form of every construction known: k3, k4, k9, isa_escape.py's D = 4 and 8),
J b-bit codes with an s-bit scale per k elements carry J (b + s/k) >= log2 |S| bits; a 2:4 pass with metadata fixed in
advance carries half the elements (2J passes), and with data-dependent metadata b/2 + s/k + log2(6)/4 bits a position.
It needs only A7 (an MMA's output is a function of its operand registers) and the formats' code widths (PTX ISA 9.4,
section 5.2.3). The value-level bound (the word depends on codes only through their values: PTX's `D = A*B + C`) is
reported beside it as strictness; no verdict reads it.

Outcomes: CLOSED (rho / D < 1), TIE (= 1: a menu form when every alphabet value has a code, which the menu's LP must
list; else a counting-only tie, with no construction known and no proof that it closes), WATCH (closed with
data-independent 2:4 metadata, open by counting with data-dependent metadata: its gain is the data's zero patterns, an
epsilon_skip), OPEN. A JOINT route (elements spread across shared slots) is bounded by counting alone only at
log2 |S| / (bits a position), without rounding up; `joint_gap` is rho over that bound, and every form where it exceeds 1
is listed. The rank floor (`rank_lower`: exact products and sums on every input) closes those at rho <= 1 on the integer
kinds, where mma and wgmma state it; on FP kinds nothing does (premise (a) covers the small grid only), so its closures
there stay open, as do the pairs at rho >= 2 everywhere, until a lemma bounds a joint route's passes. The units with no tensor form (vector, uniform, tables, reductions, movers, side units) keep
isa_escape.py's classes, re-read for FP words.

It also replays premise (a), the small-grid exactness `Accounting.Assumptions.TS.Fp8GridExact` and `Fp4GridExact`
(Security/Definitions/Accounting/Assumptions/FpMma.lean): on the bit-exact models of record transcribed from Lean
(`Pouw.Fp8Atom.groupSum` at sm_120's 26-bit and Hopper's 14-bit adder, `Pouw.PearlC.Fp4.Fp4Atom.step`), a step whose
accumulator and products are multiples of 2^g (g >= -126) with magnitudes summing below 2^(g+p) writes the exact sum, at
p = 24 (26-bit adder, FP4 align-add) and p = 14 (Hopper), and controls just outside it don't. The models are not the
devices: that the devices compute them is the premise, measured only.

Run: `python isa_escape_fp.py [--mutations] [--verdict PATH]`. Stdlib + exact_kit only; reads nothing.
"""
from __future__ import annotations

import functools
import itertools
import math
import random
import sys
from fractions import Fraction as Q

from exact_kit.checks import Checks

DEVICES = ("sm_120", "sm_90a", "sm_100a")

# ---- formats, from their encodings (PTX ISA 9.4 section 5.2.3; the Lean codecs Pouw.Fp8Atom.decode, Pouw.PearlC.Fp4) ----


def minifloat(c: int, ebits: int, mbits: int, bias: int, nan, inf=()) -> Q | str | None:
    """A sign-exponent-mantissa code's exact value; "nan" / "inf" for the specials the format has."""
    if c in nan:
        return "nan"
    if c in inf:
        return "inf"
    s = (c >> (ebits + mbits)) & 1
    e = (c >> mbits) & ((1 << ebits) - 1)
    m = c & ((1 << mbits) - 1)
    sig, ex = (m, 1 - bias - mbits) if e == 0 else ((1 << mbits) + m, e - bias - mbits)
    v = Q(sig) * Q(2) ** ex
    return -v if s else v


FORMATS = {  # name: (code bits, decoder)
    "e4m3": (8, lambda c: minifloat(c, 4, 3, 7, nan={0x7F, 0xFF})),
    "e5m2": (8, lambda c: minifloat(c, 5, 2, 15, nan={0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF}, inf={0x7C, 0xFC})),
    "e3m2": (6, lambda c: minifloat(c, 3, 2, 3, nan=())),
    "e2m3": (6, lambda c: minifloat(c, 2, 3, 1, nan=())),
    "e2m1": (4, lambda c: minifloat(c, 2, 1, 1, nan=())),
    "s8": (8, lambda c: Q(c - 256 if c >= 128 else c)),
    "f16": (16, None),    # the wide kinds are read through exact_in: (fraction bits, emax, emin) in WIDE_GRID
    "bf16": (16, None),
    "tf32": (19, None),
}
WIDE_GRID = {"f16": (10, 15, -14), "bf16": (7, 127, -126), "tf32": (10, 127, -126)}
SCALES = {"ue4m3": (7, 16), "ue8m0": (8, 32)}   # (code bits, elements a scale: NVFP4 per 16, MXFP4 per 32)


@functools.lru_cache(maxsize=None)
def values(fmt: str) -> frozenset:
    bits, dec = FORMATS[fmt]
    return frozenset(v for v in (dec(c) for c in range(1 << bits)) if not isinstance(v, str))


def exact_in(n: Q, mant: int, emax: int, emin: int) -> bool:
    """Is `n` a binary float with `mant` fraction bits and exponents in [emin, emax] (subnormals at emin)?"""
    if n == 0:
        return True
    a = abs(n)
    e = math.floor(math.log2(a)) if a.denominator == 1 and a.numerator.bit_length() < 60 else None
    if e is None:
        e = a.numerator.bit_length() - a.denominator.bit_length()
        while Q(2) ** e > a:
            e -= 1
        while Q(2) ** (e + 1) <= a:
            e += 1
    if e > emax:
        return False
    q = a / Q(2) ** (max(e, emin) - mant)
    return q.denominator == 1


INT8 = [Q(i) for i in range(-128, 128)]
E4M3 = values("e4m3")
ALPHABETS = {  # the design's operand values (S); each holds 0 and 1, which the counting bound reads
    "int8": set(INT8),
    "int5": {Q(i) for i in range(-16, 16)},
    "pm16": {Q(i) for i in range(-16, 17)},
    "e4m3i": {v for v in E4M3 if v.denominator == 1},
    "g1s": {v + b for v in E4M3 if v.denominator == 1 and abs(v) <= 120 for b in (0, 1)},  # FPX-DESIGN's R + b
    "e4m3": E4M3,
    "e2m1": values("e2m1"),
    "e2m1x2": {2 * v for v in values("e2m1")},
}

# ---- the data sheets' tensor rates (A3), dense MACs per SM per clock; a 2:4 form does twice the logical MACs ----
# sm_120: RTX PRO 6000 Blackwell sheet (kind::f8f6f4 at the FP8 rate, block-scaled FP4 at twice it; FP16/BF16 with FP32
# accumulate at half; TF32 a quarter). sm_90a: H200 sheet (= H100 SXM: INT8 = FP8, FP16 half, TF32 a quarter, FP64
# tensor 67 TF). sm_100a: B200 sheet (INT8 = FP8 = FP6 4.5 P dense, FP4 9 P, FP16 2.25 P, TF32 1.1 P, FP64 37 T). DMMA
# rows are generous upper bounds (no FP64 tensor on sm_120). Only these ratios are read.
RATES = {
    "sm_120": {"s8": 1024, "e4m3": 1024, "e5m2": 1024, "e3m2": 1024, "e2m3": 1024, "e2m1": 1024, "nvf4": 2048,
               "mxf4": 2048, "f16": 512, "bf16": 512, "tf32": 256, "f64": 0},
    "sm_90a": {"s8": 4096, "e4m3": 4096, "e5m2": 4096, "f16": 2048, "bf16": 2048, "tf32": 1024, "f64": 256},
    "sm_100a": {"s8": 8192, "e4m3": 8192, "e5m2": 8192, "e3m2": 8192, "e2m3": 8192, "e2m1": 8192, "mxf8": 8192,
                "nvf4": 16384, "mxf4": 16384, "f16": 4096, "bf16": 4096, "tf32": 2048, "f64": 256},
}
SPARSE = {"sm_120": {"s8", "e4m3", "e5m2", "e3m2", "e2m3", "e2m1", "nvf4", "mxf4", "f16", "bf16", "tf32"},
          "sm_90a": {"s8", "e4m3", "e5m2", "f16", "bf16", "tf32"},
          "sm_100a": {"s8", "e4m3", "e5m2", "e3m2", "e2m3", "e2m1", "mxf8", "nvf4", "mxf4", "f16", "bf16", "tf32"}}
# each kind's element format and block scale (None: unscaled)
KIND = {"s8": ("s8", None), "e4m3": ("e4m3", None), "e5m2": ("e5m2", None), "e3m2": ("e3m2", None),
        "e2m3": ("e2m3", None), "e2m1": ("e2m1", None), "mxf8": ("e4m3", "ue8m0"), "nvf4": ("e2m1", "ue4m3"),
        "mxf4": ("e2m1", "ue8m0"), "f16": ("f16", None), "bf16": ("bf16", None), "tf32": ("tf32", None),
        "f64": ("f64", None)}
WIDE = {"f16": 16, "bf16": 16, "tf32": 19, "f64": 64}   # code bits of the wide kinds (each holds every alphabet here)
CREDIT = {"i": "e4m3", "ii": {"sm_120": "nvf4", "sm_90a": "e4m3", "sm_100a": "nvf4"}, "iii": "s8"}
WORDS = {"i": ("int5", "pm16", "g1s", "e4m3i", "e4m3"), "ii": ("e2m1", "e2m1x2"), "iii": ("int8",)}


def credit_row(word: str, dev: str) -> str:
    c = CREDIT[word]
    return c[dev] if isinstance(c, dict) else c


def code_bits(kind: str) -> float:
    fmt, _ = KIND[kind]
    return WIDE.get(fmt) or FORMATS[fmt][0]


def value_bits(kind: str) -> float:
    """log2 of the symbols a value-semantics word can tell apart: the finite values, and one NaN class."""
    fmt, _ = KIND[kind]
    if fmt in WIDE:
        return WIDE[fmt]
    vals = len(values(fmt))
    has_nan = any(FORMATS[fmt][1](c) == "nan" for c in range(1 << FORMATS[fmt][0]))
    return math.log2(vals + (1 if has_nan else 0))


META = math.log2(6) / 4   # bits a 2:4 group's metadata adds a position when it depends on the data


def scale_bits(kind: str) -> float:
    _, scale = KIND[kind]
    return SCALES[scale][0] / SCALES[scale][1] if scale else 0.0


def planes_lower(alpha: int, kind: str, sparse: bool, meta_data_dependent: bool, value_level: bool = False) -> int:
    """The counting bound on an elementwise route's passes over the vector on `kind`.

    Elementwise: each element's product is formed in slots of its own (J of them), as in every construction on the menu.
    Its operand's J codes and its blocks' scales must tell |S| values apart: J (b + s/k) >= log2 |S| (the scale's bits
    counted as the element's, which only helps the route). Dense, that is J passes. A 2:4 pass with metadata fixed in
    advance carries half the elements, so 2J. With data-dependent metadata a pass carries b/2 + s/k + log2(6)/4 bits a
    position (its zero patterns are the data's), the bound the WATCH reads."""
    b = value_bits(kind) if value_level else code_bits(kind)
    s = scale_bits(kind)
    lg = math.log2(alpha)
    j = max(1, math.ceil(lg / (b + (2 * s if sparse else s)) - 1e-9))
    if not sparse:
        return j
    if not meta_data_dependent:
        return 2 * j
    return min(2 * j, max(1, math.ceil(lg / (b / 2 + s + META) - 1e-9)))


def info_lower(alpha: int, kind: str, sparse: bool) -> float:
    """The bound on any route, joint encodings included: the slots' codes, scales and metadata must determine the
    operand vector, so a pass over the vector carries at most b (b/2 + metadata, 2:4) + s/k bits a position. No
    rounding up: a joint route can spread elements across shared slots."""
    b, s = code_bits(kind), scale_bits(kind)
    return math.log2(alpha) / ((b / 2 + META if sparse else b) + s)


def rank_lower(alpha: int, kind: str, sparse: bool) -> float:
    """`info_lower` with the rank floor: an exact bilinear unit forms a k-term dot product in no fewer than k products,
    so one pass (two with fixed 2:4 metadata, one with data-dependent). It reads the word's value semantics (exact
    products and sums), vendor-stated for the integer kinds' s32 and measured only for the FP kinds."""
    return max(info_lower(alpha, kind, sparse), 1.0)


def fits(alpha_set, kind: str) -> bool:
    """Does every alphabet value have a code of `kind` (so one plane carries it, a block-scaled kind at one fixed
    power-of-two scale)?"""
    fmt, scale = KIND[kind]
    if fmt == "f64":
        return True
    if fmt in WIDE_GRID:
        return all(exact_in(v, *WIDE_GRID[fmt]) for v in alpha_set)
    shifts = range(-8, 9) if scale else (0,)
    return any({v / Q(2) ** e for v in alpha_set} <= values(fmt) for e in shifts)


def verdicts() -> list[dict]:
    rows = []
    for word, alphas in WORDS.items():
        for dev in DEVICES:
            credit = credit_row(word, dev)
            base = RATES[dev][credit]
            for a in alphas:
                S = ALPHABETS[a]
                for kind, rate in RATES[dev].items():
                    for sparse in (False, True):
                        if sparse and kind not in SPARSE[dev]:
                            continue
                        if rate == 0:
                            continue
                        rho = Q(rate * (2 if sparse else 1), base)
                        d_fixed = planes_lower(len(S), kind, sparse, False)
                        d_data = planes_lower(len(S), kind, sparse, True)
                        d_value = planes_lower(len(S), kind, sparse, False, value_level=True)
                        eff_fixed, eff_data = rho / d_fixed, rho / d_data
                        if eff_data < 1:
                            out = "CLOSED"
                        elif eff_data == 1:
                            out = "TIE"
                        elif eff_fixed <= 1:
                            out = "WATCH"
                        else:
                            out = "OPEN"
                        rows.append({"word": word, "device": dev, "alphabet": a, "credit": credit,
                                     "route": kind + (".sp" if sparse else ""), "rho": rho, "D": d_data,
                                     "D_fixed_meta": d_fixed, "D_value": d_value, "eff": eff_data,
                                     "eff_fixed_meta": eff_fixed, "outcome": out,
                                     "joint_gap": float(rho) / info_lower(len(S), kind, sparse),
                                     "rank_gap": float(rho) / rank_lower(len(S), kind, sparse),
                                     "one_plane": (not sparse) and fits(S, kind), "fits": fits(S, kind)})
    return rows


# ---- the units with no tensor form, re-read for FP words (isa_escape.py's classes) ----
OTHER = [
    ("FFMA / FFMA2 / HFMA2 / IMAD / IDP4A lane routes", "PRICED",
     "a lane instruction heads at most two words (A6), so a vector-headed FP or int word is ScalarRoute's, at the 1/2 floor; "
     "the multiply's format is irrelevant to that count"),
    ("uniform datapath (UFFMA, UFMUL, UFADD, UIMAD)", "PRICED",
     "one word an instruction on an issue slot (A2u, measured); UFFMA/UFMUL/UFADD appear in the sm_120 census"),
    ("product tables (LDS/LDG/LDSM/TEX lookups of x*y)", "PRICED",
     "a table-headed word is priced by the bytes it reads and at most two words a lane (A6); a 4-word TEX "
     "or an LDS.128 writes more than 64 bits a lane, which A6's docstring must bound"),
    ("texture filtering (bilinear/trilinear)", "PRICED",
     "multiplies by 9-bit fixed-point weights (CUDA Programming Guide, texture fetching): a side unit at TEXBIL's pi32 "
     "floor (SideUnits); not an exact FP32 sum of FP8 products"),
    ("MUFU ex2/lg2 (multiply as exp(log a + log b))", "PRICED",
     "lane instructions on the 16-lane XU pipe, several a product: ScalarRoute's 1/2 floor, and XU's floor"),
    ("DPX (VIMNMX3, VIADDMNMX), VABSDIFF4, PRMT, LOP3", "NO-MULTIPLY",
     "add/min/max/select/logic only: no fresh product"),
    ("red.add.f32 / atom.add.f32, red.async, cp.reduce.async.bulk .add, tcgen05.ld.red", "NO-MULTIPLY",
     "reducers add or compare existing words; RED32/REDV/TMAred512 priced at the pi32 floor"),
    ("multimem.ld_reduce / NVLink SHARP", "NO-MULTIPLY", "an in-switch add of existing words, off the SM"),
    ("cvt to e4m3x2/e5m2x2/e2m1x2 (F2FP), spdecompress/spcompress, ldmatrix, cp.async, TMA", "MOVER",
     "conversion or movement: the output is an existing value, rounded or copied"),
    ("tcgen05.mma scale-input-d (D = A*B + D * 2^-n)", "MOVER",
     "a power-of-two scaling of the accumulator inside the MMA: it recombines digit planes for free but multiplies no "
     "operand, so it changes no plane count"),
    ("FP16-accumulate MMAs (HMMA.F16, wgmma f16 acc, kind::f16 with f16 D)", "LOW-PRECISION",
     "an f16 accumulator rounds a word past 2^11: not an exact FP32 or s32 word"),
    ("kind::ti16 (s1z4m11, K = 16, s32)", "WATCH",
     "PTX 9.4, sm_107f only: not on B200 (sm_100a); a 12-bit integer tensor kind would join the int8 menu on sm_107"),
]
REQUIRED_OTHER = ["FFMA", "uniform", "tables", "texture", "MUFU", "DPX", "red.add", "multimem", "cvt", "scale-input-d",
                  "FP16-accumulate", "ti16"]

# ---- constructions (upper bounds), checked exhaustively over every operand pair ----


def k3_planes(x: int):
    """emu8's k3 limbs: x = 16h + l, l in [-8, 7], m = h + l; A planes [4h, m, -l], B planes [4h', m', l']."""
    h = (x + 8) >> 4
    l = x - 16 * h
    return h, l, h + l


def k3_ok() -> tuple[bool, str]:
    e4 = ALPHABETS["e4m3i"]
    bad = 0
    for x in range(-128, 128):
        h, l, m = k3_planes(x)
        if not {Q(4 * h), Q(m), Q(-l), Q(l)} <= e4:
            return False, f"plane value of {x} not an E4M3 integer"
        for y in range(-128, 128):
            h2, l2, m2 = k3_planes(y)
            if 15 * (4 * h * 4 * h2 + (-l) * l2) + 16 * m * m2 != x * y:
                bad += 1
    return bad == 0, f"{bad} wrong of 65536"


def k3h_ok() -> tuple[bool, str]:
    """H200's k3h: unscaled limbs h, m, l, word = 240 sum(h h') + 16 sum(m m') - 15 sum(l l'); a k32 step's three sums stay
    below 2^14 (p = 14), so each wgmma E4M3 step from +0 is inside the small grid."""
    worst = 0
    for x in range(-128, 128):
        h, l, m = k3_planes(x)
        for y in range(-128, 128):
            h2, l2, m2 = k3_planes(y)
            if 240 * h * h2 + 16 * m * m2 - 15 * l * l2 != x * y:
                return False, f"{x}*{y}"
            worst = max(worst, abs(h * h2), abs(m * m2), abs(l * l2))
    return 32 * worst < 2 ** 14, f"largest limb product {worst}, a k32 sum <= {32 * worst} < 2^14"


def k4_ok() -> tuple[bool, str]:
    e4 = ALPHABETS["e4m3i"]
    for x in range(-128, 128):
        h, l, _ = k3_planes(x)
        if not {Q(16 * h), Q(l)} <= e4:
            return False, f"plane of {x}"
        for y in range(-128, 128):
            h2, l2, _ = k3_planes(y)
            if (16 * h + l) * (16 * h2 + l2) != x * y:
                return False, f"{x}*{y}"
    return True, "4 products (16h)(16h'), (16h)l', l(16h'), ll' into one accumulator"


def k9_digits(x: int):
    """Three balanced base-8 digits in [-4, 3]: x = 64 d2 + 8 d1 + d0 (every int8)."""
    d = []
    for _ in range(3):
        r = ((x + 4) % 8) - 4
        d.append(r)
        x = (x - r) // 8
    return d, x


def k9_ok() -> tuple[bool, str]:
    e2 = ALPHABETS["e2m1"]
    for x in range(-128, 128):
        d, rest = k9_digits(x)
        if rest != 0 or not {Q(v) for v in d} <= e2:
            return False, f"digits of {x}: {d} rest {rest}"
        if 64 * d[2] + 8 * d[1] + d[0] != x:
            return False, f"recombination of {x}"
    return True, "every int8 is three E2M1 digits in [-4, 3] at weights 64, 8, 1 (UE8M0 powers of two): 9 products"


def e2m1_three_plane_cover() -> tuple | None:
    """A weight triple (powers of two) with {w1 c1 + w2 c2 + w3 c3 : c in E2M1} covering every int8: r_A = 3 (2 planes
    give at most 15^2 = 225 < 256 values)."""
    vals = sorted(ALPHABETS["e2m1"])
    target = set(INT8)
    ws = [Q(2) ** k for k in range(-1, 7)]
    for w in itertools.combinations(ws, 3):
        sums = {w[0] * a + w[1] * b + w[2] * c for a in vals for b in vals for c in vals}
        if target <= sums:
            return w
    return None


# ---- the bit-exact models of record, transcribed from Lean (Pouw.Fp8Atom.Atom, Pouw.PearlC.Fp4) ----

ZERO = (False, 0, 0)


def fp32_term(w: int):
    e, m, s = (w >> 23) % 256, w % (1 << 23), (w >> 31) % 2 == 1
    assert e != 255, "finite accumulators only"
    if e == 0:
        return ZERO if m == 0 else (s, -126, m)
    return (s, e - 127, (1 << 23) | m)


def term_value(t) -> Q:
    neg, e, mag = t
    v = Q(mag) * Q(2) ** (e - 23)
    return -v if neg else v


def to_adder(width: int, m: int) -> int:
    return m << (width - 24) if width >= 24 else m >> (24 - width)


def from_adder(width: int, m: int) -> int:
    return m >> (width - 24) if width >= 24 else m << (24 - width)


def group_sum(width: int, zero_exp: int, ts):
    E = zero_exp
    for neg, e, mag in ts:
        if mag != 0 and E < e:
            E = e
    total = 0
    for neg, e, mag in ts:
        if mag:
            a = to_adder(width, mag) >> max(0, E - e)
            total += -a if neg else a
    if total == 0:
        return ZERO
    a = abs(total)
    bl = a.bit_length()
    e = E + bl - width
    mag = a >> (bl - width) if width < bl else a << (width - bl)
    if e < -126:
        mag >>= -126 - e
        e = -126
    mag = from_adder(width, mag)
    return ZERO if mag == 0 else (total < 0, e, mag)


def e4m3_fields(c: int):
    e, m = (c >> 3) % 16, c % 8
    return c >> 7 == 1, (m if e == 0 else 8 + m), max(e, 1)


def e4m3_product(a: int, b: int):
    sa, ga, ea = e4m3_fields(a)
    sb, gb, eb = e4m3_fields(b)
    mag = (ga * gb) << 17
    return ZERO if mag == 0 else (sa != sb, ea + eb - 14, mag)


def fp8_step(width: int, zero_exp: int, acc: int, a, b) -> Q:
    """`Pipeline.step` on finite words and codes: the word's value (one group of 32; overflow is out of these cases)."""
    t = group_sum(width, zero_exp, [fp32_term(acc)] + [e4m3_product(x, y) for x, y in zip(a, b)])
    assert t[2] == 0 or t[1] <= 127
    return term_value(t)


PIPELINES = {"sm_120": (26, -133, 24), "sm_90a": (14, -139, 14)}   # width, floor, the premise's p


def fp32_word(v: Q) -> int:
    """The FP32 word of a value that is one (normal or zero)."""
    if v == 0:
        return 0
    s, a = (1 if v < 0 else 0), abs(v)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Q(2) ** e > a:
        e -= 1
    while Q(2) ** (e + 1) <= a:
        e += 1
    m = a / Q(2) ** (e - 23)
    assert m.denominator == 1 and 1 << 23 <= m < 1 << 24 and -126 <= e <= 127, v
    return (s << 31) | ((e + 127) << 23) | (int(m) & 0x7FFFFF)


def nib_fields(c: int):
    e, m = (c >> 1) % 4, c % 2
    return c >> 3 == 1, (m if e == 0 else 2 + m), max(e, 1)


def nib_value(c: int) -> Q:
    s, g, e = nib_fields(c)
    v = Q(g) * Q(2) ** (e - 2)
    return -v if s else v


def numerator(c: int) -> int:
    s, g, e = nib_fields(c)
    n = g << (e - 1)
    return -n if s else n


def scale_dyadic(fmt: str, s: int):
    if fmt == "ue8m0":
        return None if s == 255 else (8, s - 130)
    if s % 128 == 127:
        return None
    c = s % 128
    e, m = (c >> 3) % 16, c % 8
    return (m if e == 0 else 8 + m), max(e, 1) - 10


def to_f32_rz(n: int, lsb: int):
    mag = abs(n)
    e_val = mag.bit_length() - 1 + lsb
    grid = e_val - 23 if -126 <= e_val else -149
    shift = grid - lsb
    q = mag >> shift if shift > 0 else mag << -shift
    if q == 0:
        return 0
    e = q.bit_length() - 1 + grid
    if e > 127:
        return None
    bits = ((e + 127) << 23) | (q & 0x7FFFFF) if e >= -126 else q
    return bits | (0x80000000 if n < 0 else 0)


def trunc_rz(m: int, e: int, lsb: int) -> int:
    if lsb <= e:
        return m * (1 << (e - lsb))
    q = abs(m) >> (lsb - e)
    return -q if m < 0 else q


def fp4_step(fmt: str, acc: int, a, b, sa, sb, k=64, group=16, frac=27, window=36, floor=-174):
    """`Fp4Atom.step` (sm120Nvf4 / sm120Mxf4): the word written, or None outside the model."""
    block = SCALES[fmt][1]
    ts = []
    for j in range(k // group):
        ps = [numerator(x) * numerator(y) for x, y in zip(a[j * group:(j + 1) * group], b[j * group:(j + 1) * group])]
        da, db = scale_dyadic(fmt, sa[j * group // block]), scale_dyadic(fmt, sb[j * group // block])
        if da is None or db is None:
            return None
        ts.append((sum(ps) * da[0] * db[0], -2 + da[1] + db[1], any(ps) and da[0] != 0 and db[0] != 0))
    cneg, cexp, cmag = fp32_term(acc)
    anchors = [u - frac for _, u, live in ts if live] + ([cexp - (window - 1)] if cmag else [])
    if not anchors:
        return 0
    lsb = max(max(anchors), floor)
    total = sum(trunc_rz(m, u, lsb) for m, u, _ in ts) + trunc_rz(-cmag if cneg else cmag, cexp - 23, lsb)
    return 0 if total == 0 else to_f32_rz(total, lsb)


def word_value(w: int) -> Q:
    return term_value(fp32_term(w))


def v2(x: Q) -> int:
    n, d = x.numerator, x.denominator
    return ((n & -n).bit_length() - 1) - ((d & -d).bit_length() - 1)


def small_grid(p: int, c: Q, ts: list) -> bool:
    """The premise's hypothesis (`FpGridSmall p c ts`): some g >= -126 with g + p <= 128, every term a multiple of 2^g,
    and their magnitudes summing below 2^(g + p). The best g is the least 2-adic valuation (capped at 128 - p)."""
    nz = [t for t in [c] + ts if t != 0]
    if not nz:
        return True
    g = min(min(v2(t) for t in nz), 128 - p)
    return g >= -126 and abs(c) + sum(abs(t) for t in ts) < Q(2) ** (g + p)


def e4m3_codes_of(vals: set) -> dict:
    out = {}
    for c in range(256):
        v = FORMATS["e4m3"][1](c)
        if not isinstance(v, str) and v in vals and v not in out:
            out[v] = c
    return out


def fp8_cases(rng: random.Random, p: int, n: int):
    """Steps inside the small grid: integer-grid codes at a power-of-two grid 2^g and accumulators on it."""
    grids = [(0, [Q(i) for i in range(-16, 17)]), (-3, [Q(i, 8) for i in range(-15, 16)]),
             (2, [Q(4 * i) for i in range(-16, 17) if Q(4 * i) in E4M3])]
    codes = e4m3_codes_of(E4M3)
    for _ in range(n):
        g, vals = rng.choice(grids)
        sub = [v for v in vals if v in codes]
        a = [codes[rng.choice(sub)] for _ in range(32)]
        b = [codes[rng.choice([Q(1), Q(-1), Q(2), Q(0), Q(1, 2)] if g == -3 else [Q(1), Q(-1), Q(2), Q(0)])]
             for _ in range(32)]
        prods = [FORMATS["e4m3"][1](x) * FORMATS["e4m3"][1](y) for x, y in zip(a, b)]
        room = Q(2) ** (g + p) - sum(abs(t) for t in prods) - 1
        c = Q(rng.randrange(-int(room / Q(2) ** g), int(room / Q(2) ** g) + 1)) * Q(2) ** g if room > 0 else Q(0)
        if rng.random() < 0.3 and room > 0:   # an accumulator at the top of the room (k4s's binade restarts)
            c = (room // Q(2) ** g) * Q(2) ** g
        yield fp32_word(c), a, b, c, prods


def fp4_cases(rng: random.Random, fmt: str, n: int):
    scale_codes = ([0x38, 0x40, 0x30, 0x48] if fmt == "ue4m3" else [127, 128, 126, 129])   # 1, 2, 1/2, 4
    for _ in range(n):
        a = [rng.randrange(16) for _ in range(64)]
        b = [rng.randrange(16) for _ in range(64)]
        ns = 64 // SCALES[fmt][1]
        sa = [rng.choice(scale_codes) for _ in range(ns)]
        sb = [rng.choice(scale_codes) for _ in range(ns)]
        blk = SCALES[fmt][1]

        def sv(s):
            m, e = scale_dyadic(fmt, s)
            return Q(m) * Q(2) ** e
        prods = [nib_value(a[l]) * nib_value(b[l]) * sv(sa[l // blk]) * sv(sb[l // blk]) for l in range(64)]
        c = Q(rng.randrange(-2 ** 12, 2 ** 12)) * Q(1, 8)
        yield fp32_word(c), a, b, sa, sb, c, prods


def main() -> int:
    checks = Checks()
    rng = random.Random(20261019)

    # -- the formats from their encodings (PTX ISA 9.4 section 5.2.3) --------------------------------------------------
    counts = {f: len(values(f)) for f in ("e4m3", "e5m2", "e3m2", "e2m3", "e2m1")}
    checks.check("E4M3: 254 finite codes, 253 values (two zeros), NaN only 0x7F/0xFF, max 448",
                 sum(not isinstance(FORMATS["e4m3"][1](c), str) for c in range(256)) == 254 and counts["e4m3"] == 253
                 and max(E4M3) == 448 and FORMATS["e4m3"][1](0x7E) == 448, counts)
    checks.check("E5M2: 248 finite codes (2 infinities, 6 NaNs), max 57344",
                 sum(not isinstance(FORMATS["e5m2"][1](c), str) for c in range(256)) == 248
                 and max(values("e5m2")) == 57344)
    checks.check("E3M2 and E2M3: 64 codes, 63 values each, max 28 and 7.5",
                 counts["e3m2"] == counts["e2m3"] == 63 and max(values("e3m2")) == 28 and max(values("e2m3")) == Q(15, 2))
    checks.check("E2M1: 16 codes, all finite, 15 values: +-{0, 1/2, 1, 3/2, 2, 3, 4, 6}",
                 values("e2m1") == {s * Q(v, 2) for v in (0, 1, 2, 3, 4, 6, 8, 12) for s in (1, -1)})
    checks.check("E2M1 decode equals Pouw.PearlC.Fp4.decode on every nibble",
                 all(FORMATS["e2m1"][1](c) == nib_value(c) for c in range(16)))
    checks.check("UE4M3 per 16 (7 bits, NaN 0x7F) and UE8M0 per 32 (NaN 0xFF): 127 and 255 valid scale codes",
                 sum(scale_dyadic("ue4m3", s) is not None for s in range(128)) == 127
                 and sum(scale_dyadic("ue8m0", s) is not None for s in range(256)) == 255)
    e4i = ALPHABETS["e4m3i"]
    checks.check("the E4M3 integers are 109 values (0..16, then 8 a binade to 448)", len(e4i) == 109, len(e4i))
    checks.check("every alphabet holds 0 and 1 (the counting bound recovers an element with A = e_l)",
                 all({Q(0), Q(1)} <= S for S in ALPHABETS.values()))
    checks.check("int8 is not inside E4M3 (17, 127 have no code): one FP8 plane does not carry int8",
                 not set(INT8) <= E4M3 and Q(17) not in E4M3 and Q(127) not in E4M3)
    checks.check("int5, the +-16 grid and the E4M3 integers below 128 fit int8 (one IMMA plane each)",
                 all(ALPHABETS[a] <= set(INT8) for a in ("int5", "pm16")) and not e4i <= set(INT8))
    checks.check("f16, bf16 and tf32 hold every int8 and every E4M3 value exactly",
                 all(fits(set(INT8) | E4M3, k) for k in ("f16", "bf16", "tf32")))

    # -- the counting bound (no FP semantics) ----------------------------------------------------------------------
    checks.check("int8 on E2M1: 2 planes by codes (16^2 = 256), 3 by values (15^2 = 225 < 256)",
                 planes_lower(256, "e2m1", False, False) == 2 and planes_lower(256, "e2m1", False, False, True) == 3)
    checks.check("int8 on E4M3: 1 plane by codes (256 codes), 2 by values (253 + NaN < 256)",
                 planes_lower(256, "e4m3", False, False) == 1 and planes_lower(256, "e4m3", False, False, True) == 2)
    checks.check("int8 on block-scaled FP4 with data-dependent scales: 2 planes (NVF4 4 + 7/16, MXF4 4 + 8/32 bits an element)",
                 planes_lower(256, "nvf4", False, False) == planes_lower(256, "mxf4", False, False) == 2)
    checks.check("int8 on 2:4 FP4, fixed metadata: 4 passes (2 planes, each carried in 2 passes)",
                 planes_lower(256, "nvf4", True, False) == planes_lower(256, "mxf4", True, False) == 4)
    checks.check("int8 on 2:4 FP4, data-dependent metadata: 3 passes (+ log2(6)/4 bits): the WATCH",
                 planes_lower(256, "nvf4", True, True) == 3)
    checks.check("int8 on 2:4 int8 (IMMA.SP): 2 passes either way (2:4 ties the dense form)",
                 planes_lower(256, "s8", True, False) == planes_lower(256, "s8", True, True) == 2)
    checks.check("E4M3 values (FP-native word) on E2M1: 2 planes by codes, 3 by values (225 < 253)",
                 planes_lower(253, "e2m1", False, False) == 2 and planes_lower(253, "e2m1", False, False, True) == 3)

    # -- the verdicts, per device ----------------------------------------------------------------------------------
    rows = verdicts()
    for dev in DEVICES:
        for word in WORDS:
            mine = [r for r in rows if r["device"] == dev and r["word"] == word]
            bad = [f"{r['alphabet']}/{r['route']} eff {float(r['eff']):.3f}" for r in mine if r["outcome"] == "OPEN"]
            checks.check(f"{dev} word ({word}): no tensor form is OPEN to an elementwise route (rho / D <= 1 with "
                         "fixed metadata)", not bad and mine, bad)
            watch = sorted({r["route"] for r in mine if r["outcome"] == "WATCH"})
            checks.check(f"{dev} word ({word}): every WATCH is a 2:4 form (data-dependent metadata only)",
                         all(w.endswith(".sp") for w in watch), watch)
            ties = [r for r in mine if r["outcome"] == "TIE"]
            menu = sorted({f"{r['alphabet']}/{r['route']}" for r in ties if r["fits"]})
            counting = sorted({f"{r['alphabet']}/{r['route']}" for r in ties if not r["fits"]})
            checks.check(f"{dev} word ({word}): ties, {len(menu)} menu forms (every value has a code: the LP lists "
                         f"them) and {len(counting)} by counting only (no such code and no construction known; that "
                         f"they close needs value semantics and a rank lemma, open)",
                         True, {"menu": menu, "counting_only": counting})
    for dev in DEVICES:
        dense = [r for r in rows if r["device"] == dev and not r["route"].endswith(".sp")]
        checks.check(f"{dev}: every dense tensor form closes or ties to elementwise routes by codes alone (no FP premise)",
                     all(r["eff"] <= 1 for r in dense))
        mine = [r for r in rows if r["device"] == dev]
        joint = sorted({(r["route"], r["word"]) for r in mine if r["joint_gap"] > 1})
        ranked = sorted({(r["route"], r["word"]) for r in mine if r["rank_gap"] > 1})
        worst = max(mine, key=lambda r: r["joint_gap"])
        worst_r = max(mine, key=lambda r: r["rank_gap"])
        fp_rank = sorted({(r["route"], r["word"]) for r in mine
                          if r["joint_gap"] > 1 and r["rank_gap"] <= 1 and r["route"].split(".")[0] != "s8"})
        checks.check(f"{dev}: joint encodings: counting alone leaves {len(joint)} (form, word) pairs open; the rank "
                     f"floor (exact products and sums on every input) leaves {len(ranked)}, all at rho = 2 or more "
                     "(block-scaled FP4 against FP8/int8 credits, 2:4 forms); closing those is open. Of the pairs it "
                     f"closes, {len(fp_rank)} are on FP kinds, where nothing states that exactness (premise (a) covers "
                     "the small grid only), so they stay open too",
                     all(r["rho"] >= 2 for r in mine if r["rank_gap"] > 1),
                     {"worst_counting": f"{worst['alphabet']}/{worst['route']} {worst['joint_gap']:.3f}",
                      "worst_rank": f"{worst_r['alphabet']}/{worst_r['route']} {worst_r['rank_gap']:.3f}",
                      "open_with_rank": [f"{f}:{w}" for f, w in ranked],
                      "open_on_fp_kinds": [f"{f}:{w}" for f, w in fp_rank]})
    imma_ties = [r for r in rows if r["word"] == "i" and r["route"] == "s8" and r["alphabet"] in ("int5", "pm16", "g1s")]
    checks.check("an FP8 word on an int8-fitting grid (int5, +-16, g1s's R + b) ties on IMMA / wgmma s8 / kind::i8 in "
                 "one plane: the LP must list the int8 form (and IMMA writes s32, saving the FP word's extraction)",
                 len(imma_ties) == 9 and all(r["outcome"] == "TIE" and r["one_plane"] for r in imma_ties))
    h200_fp4 = [r for r in rows if r["word"] == "ii" and r["device"] == "sm_90a" and r["route"] in ("s8", "e4m3")]
    checks.check("H200 has no FP4 kind: an FP4 word there is credited at E4M3 and ties on s8 (2x E2M1 is int8) and E4M3",
                 all(r["outcome"] == "TIE" for r in h200_fp4) and len(h200_fp4) == 4)
    checks.check("B200: kind::f8f6f4 E2M1 (unscaled) runs at the FP8 rate, half the block-scaled kinds'",
                 RATES["sm_100a"]["e2m1"] * 2 == RATES["sm_100a"]["nvf4"])

    # -- the constructions (upper bounds) --------------------------------------------------------------------------
    ok, det = k3_ok()
    checks.check("k3: every int8 product is 15(4h 4h' - l l') + 16 m m' with E4M3-integer planes (3 slots)", ok, det)
    ok, det = k3h_ok()
    checks.check("k3h: 240 hh' + 16 mm' - 15 ll' on unscaled limbs, each k32 sum inside H200's p = 14", ok, det)
    ok, det = k4_ok()
    checks.check("k4: (16h + l)(16h' + l') in 4 E4M3 products (4 slots)", ok, det)
    ok, det = k9_ok()
    checks.check("k9: int8 as 3 E2M1 digits, 9 products", ok, det)
    w = e2m1_three_plane_cover()
    checks.check("int8 is 3 E2M1 planes at power-of-two weights (r_A = 3): a product needs 4..9 by the rank bound",
                 w is not None, [str(x) for x in w] if w else None)

    # -- premise (a) replayed on the models of record --------------------------------------------------------------
    for dev, (width, floor, p) in PIPELINES.items():
        n, wrong = 0, []
        for acc, a, b, c, prods in fp8_cases(rng, p, 600):
            if not small_grid(p, c, prods):
                continue
            n += 1
            got = fp8_step(width, floor, acc, a, b)
            if got != c + sum(prods):
                wrong.append((hex(acc), str(got), str(c + sum(prods))))
        checks.check(f"premise (a) on {dev}'s model (width {width}): every small-grid step at p = {p} is exact",
                     n >= 300 and not wrong, {"cases": n, "wrong": wrong[:3]})
    codes = e4m3_codes_of(E4M3)
    q4 = [codes[Q(1, 4)]] * 16 + [0] * 16
    got = fp8_step(26, -133, fp32_word(Q(2) ** 23), q4, q4)
    checks.check("control, sm_120: sixteen 2^-4 products sum to 1, but each sits 27 bits below a 2^23 accumulator: the "
                 "model drops them all (outside p = 24)",
                 got == Q(2) ** 23 and not small_grid(24, Q(2) ** 23, [Q(1, 16)] * 16), str(got))
    h2 = [codes[Q(1, 2)]] * 3 + [0] * 29
    hb = [codes[Q(1, 2)]] * 2 + [codes[Q(1)]] + [0] * 29
    got = fp8_step(26, -133, fp32_word(Q(2) ** 23), h2, hb)
    checks.check("the window, sm_120: 2^-2 + 2^-2 + 2^-1 25 bits below a 2^23 accumulator is kept (2^23 + 1): the 26-bit "
                 "rule is stronger than premise (a), which does not cover it",
                 got == Q(2) ** 23 + 1 and not small_grid(24, Q(2) ** 23, [Q(1, 4), Q(1, 4), Q(1, 2)]), str(got))
    o1 = [codes[Q(1)]] + [0] * 31
    got = fp8_step(14, -139, fp32_word(Q(2) ** 14), o1, o1)
    checks.check("control, Hopper: 2^14 + 1 needs 15 bits; the 14-bit adder truncates it, outside p = 14",
                 got == Q(2) ** 14 and not small_grid(14, Q(2) ** 14, [Q(1)]), str(got))
    for fmt in ("ue4m3", "ue8m0"):
        n, wrong = 0, []
        for acc, a, b, sa, sb, c, prods in fp4_cases(rng, fmt, 300):
            if not small_grid(24, c, prods):
                continue
            n += 1
            wd = fp4_step(fmt, acc, a, b, sa, sb)
            if wd is None or word_value(wd) != c + sum(prods):
                wrong.append(str(c + sum(prods)))
        checks.check(f"premise (a) on sm_120's FP4 model ({fmt}): every small-grid step at p = 24 is exact",
                     n >= 250 and not wrong, {"cases": n, "wrong": wrong[:3]})
    one_code = 0b0010   # E2M1 1.0
    a = [one_code] + [0] * 63
    wd = fp4_step("ue8m0", fp32_word(Q(2) ** 40), a, a, [127, 127], [127, 127])
    checks.check("control, FP4: a unit product 40 bits below the accumulator is dropped (outside p = 24)",
                 word_value(wd) == Q(2) ** 40 and not small_grid(24, Q(2) ** 40, [Q(1)]))

    # -- the 2:4 WATCH: how often a digit plane offers a free zero pattern -------------------------------------------
    planes = {"k3 4h": lambda x: 4 * k3_planes(x)[0], "k3 m": lambda x: k3_planes(x)[2],
              "k3 l": lambda x: k3_planes(x)[1], "k9 d2": lambda x: k9_digits(x)[0][2]}
    zero = {name: sum(f(x) == 0 for x in range(-128, 128)) / 256 for name, f in planes.items()}
    group = {name: 1 - (1 - z) ** 4 - 4 * z * (1 - z) ** 3 for name, z in zero.items()}
    tile = {name: g ** 256 for name, g in group.items()}
    checks.check("uniform int8, k3/k9 digit planes: a 4-group has two zeros w.p. <= 0.6, a 16x64 FP4 tile's 256 groups "
                 "all at once w.p. < 1e-50 (the WATCH's gain is data-dependent: epsilon_skip)",
                 max(group.values()) < 0.6 and max(tile.values()) < 1e-50,
                 {k: [round(zero[k], 4), round(group[k], 4), f"{tile[k]:.1e}"] for k in planes})

    # -- the units with no tensor form -----------------------------------------------------------------------------
    joined = " | ".join(n for n, _, _ in OTHER)
    for r in REQUIRED_OTHER:
        checks.check(f"enumerated: {r}", r in joined)
    checks.check("every non-tensor unit has a known class",
                 all(c in {"NO-MULTIPLY", "MOVER", "PRICED", "LOW-PRECISION", "WATCH"} for _, c, _ in OTHER))

    if checks.mutations:
        with checks.mutant("block-scaled FP4 at 4x the FP8 rate (not 2x) opens int8 and int5 on nvf4"):
            saved = dict(RATES["sm_120"])
            RATES["sm_120"]["nvf4"] = RATES["sm_120"]["mxf4"] = 4 * RATES["sm_120"]["s8"]
            try:
                bad = [r for r in verdicts() if r["device"] == "sm_120" and r["outcome"] == "OPEN"]
                checks.check("sm_120: no tensor form is OPEN to an elementwise route", not bad, len(bad))
            finally:
                RATES["sm_120"] = saved
        with checks.mutant("2:4 with fixed metadata counted as one plane a pass (not two)"):
            orig = planes_lower

            def one_pass(alpha, kind, sparse, meta, value_level=False):
                d = orig(alpha, kind, sparse, meta, value_level)
                return max(1, d // 2) if sparse and not meta else d
            globals()["planes_lower"] = one_pass
            try:
                bad = [r for r in verdicts() if r["outcome"] == "OPEN"]
                checks.check("no tensor form is OPEN to an elementwise route", not bad, len(bad))
            finally:
                globals()["planes_lower"] = orig
        checks.mutation("a single 2:4 pass for a dense int8 plane would escape (rho 2 / D 1)",
                        Q(2 * RATES["sm_120"]["s8"], RATES["sm_120"]["s8"]) / 1 > 1)
        checks.mutation("2 E2M1 planes cannot carry int8 by value (15^2 < 256): a claimed 2-plane value route is false",
                        15 ** 2 < 256)
        acc = fp32_word(Q(2) ** 24)
        got = fp8_step(26, -133, acc, o1, o1)
        checks.mutation("p = 25 on sm_120 is false: 2^24 + 1 inside a p = 25 grid, the model writes 2^24",
                        small_grid(25, Q(2) ** 24, [Q(1)]) and got != Q(2) ** 24 + 1)
        got = fp8_step(14, -139, fp32_word(Q(2) ** 20), o1, o1)
        checks.mutation("p = 24 on Hopper's 14-bit adder is false: 2^20 + 1 is inside p = 24, the model drops the 1",
                        small_grid(24, Q(2) ** 20, [Q(1)]) and got != Q(2) ** 20 + 1)
        checks.mutation("int8 into E2M1 at 2 planes and power-of-two weights does not cover (r_A > 2)",
                        not any(set(INT8) <= {w1 * x + w2 * y for x in ALPHABETS["e2m1"] for y in ALPHABETS["e2m1"]}
                                for w1, w2 in itertools.combinations([Q(2) ** k for k in range(-1, 7)], 2)))

    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
