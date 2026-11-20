"""Word steps as gates, shared by the Boolean versions of the word modules built on `dsa` (`boolean_io_shell`, `boolean_stop_rule`):
core's gate builder for each word primitive, with the GPU's NaN word for the FP32 ones; no table or lookup gate.  Beside them, five
word primitives and their Boolean versions, read by modules of more than one model:

    F32ToU32RzByte_v1 / _v2      (x: f32) -> byte: the low byte of `cvt.rzi.u32.f32`
    Ue8m0CeilByte_v1 / _v2       (s: f32) -> e8m0: the UE8M0 byte of a scale word, its exponent rounded up
    I32Mul_v1 / _v2, I32Div_v1 / _v2, I32Rem_v1 / _v2
                                 (a, b: i32) -> i32: mul.lo.s32, div.s32, rem.s32 (C `int` * / %)
"""
from __future__ import annotations

from functools import cache

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import (
    NOT,
    ONE,
    ZERO,
    add,
    and_reduce,
    compress_columns,
    const_bits,
    lzc,
    multiply,
    mux,
    or_reduce,
    shl_window,
    sub,
)
from verity.circuits.boolean.trace import definition
from verity.circuits.defs import primitive
from verity.circuits.ops.boolean import mufu as BM
from verity.circuits.ops.boolean import scalar as SC
from verity.circuits.ops.boolean.softmax import f32_gt, ftz
from verity.circuits.silicon import fp as FP
from verity.circuits.types import Array, Value

from verity_vllm.definitions.evidence import EVIDENCE
from verity_vllm.definitions.registry import boolean_dsa as BD
from verity_vllm.definitions.registry.prims import F32, I32

PTX = FP.F32_NAN_PTX
M32 = 0xFFFFFFFF
E8M0 = Value(8)
F32_BITS, E8_BITS = Array(32, BL.BIT), Array(8, BL.BIT)
I32_BITS = F32_BITS

_PTX = "model-untested: as the PTX ISA defines it; tests/properties/dsv4_qnorm_insert_capture_gpu.py is its hardware check"
_CONF = ("model-untested: read from the CuTe DSL kernels and their PTX and SASS at the commits the module docstring pins; "
         "tests/properties/dsv4_compressor_capture_gpu.py is its hardware check")


def _nan(u: int) -> bool:
    return (u & 0x7F800000) == 0x7F800000 and (u & 0x7FFFFF) != 0


def f32_to_u32_rz_byte(x: int) -> int:
    """The low byte of `cvt.rzi.u32.f32` (SASS `F2I.U32.TRUNC.NTZ`): x truncated toward zero and saturated to [0, 2^32 - 1] (a
    NaN, every negative and -0 to 0; +inf and every x >= 2^32 to 0xFFFFFFFF)."""
    x &= M32
    e = (x >> 23) & 0xFF
    if _nan(x) or x >> 31 or e < 127:
        return 0
    if e >= 127 + 32:
        return 0xFF
    m, k = (x & 0x7FFFFF) | 0x800000, e - 127
    return ((m << (k - 23)) if k >= 23 else (m >> (23 - k))) & 0xFF


@primitive("F32ToU32RzByte", 1, [("x", F32)], E8M0, conformance=_PTX)
def F32ToU32RzByte(x: int) -> int:
    """`static_cast<uint8_t>(float)` as nvcc compiles it: `cvt.rzi.u32.f32` and `st.global.u8` of its low byte
    (`f32_to_u32_rz_byte`)."""
    return f32_to_u32_rz_byte(x)


@primitive("Ue8m0CeilByte", 1, [("s", F32)], E8M0, conformance=_CONF)
def Ue8m0CeilByte(s: int) -> int:
    """The UE8M0 byte of a positive scale word, its exponent rounded up: `((bits(s) + 0x7FFFFF) >> 23) & 0xFF` (add.s32,
    shr.u32, and.b32; cutedsl:369, 1415)."""
    return (((s & M32) + 0x7FFFFF) & M32) >> 23 & 0xFF


def _signed(a: int) -> int:
    a &= M32
    return a - (1 << 32) if a & 0x80000000 else a


def i32_divrem(a: int, b: int) -> tuple[int, int]:
    """C `int` a / b and a % b (truncating toward zero, the remainder a's sign), modulo 2^32; INT_MIN / -1 wraps to INT_MIN with
    remainder 0, and b = 0 gives (-1, a)."""
    sa, sb = _signed(a), _signed(b)
    if sb == 0:
        return M32, a & M32
    q = abs(sa) // abs(sb)
    if (sa < 0) != (sb < 0):
        q = -q
    return q & M32, (sa - q * sb) & M32


@primitive("I32Mul", 1, [("a", I32), ("b", I32)], I32, conformance="exact-model-tested")
def I32Mul(a: int, b: int) -> int:
    """`mul.lo.s32`: a * b modulo 2^32."""
    return (a * b) & M32


@primitive("I32Div", 1, [("a", I32), ("b", I32)], I32, conformance="exact-model-tested")
def I32Div(a: int, b: int) -> int:
    """`div.s32` (C `int` /): `i32_divrem`'s quotient."""
    return i32_divrem(a, b)[0]


@primitive("I32Rem", 1, [("a", I32), ("b", I32)], I32, conformance="exact-model-tested")
def I32Rem(a: int, b: int) -> int:
    """`rem.s32` (C `int` %): `i32_divrem`'s remainder."""
    return i32_divrem(a, b)[1]


#: the stages of the right shift by 31 - k, largest first, each with the low bits its later stages read
_STAGES = ((16, 23), (8, 15), (4, 11), (2, 9), (1, 8))


def f32_to_u32_rz_byte_bits(C, x):
    """`f32_to_u32_rz_byte`: with k = e - 127 (no borrow: x >= 1 in magnitude), x < 2^32 iff k < 32, and then the
    truncation is (1.m << 8) >> (31 - k), 31 - k the NOT of k's five low bits; 0xFF from k >= 32 (+inf too), 0 for a NaN, a
    negative or k < 0."""
    m, e, s = x[:23], x[23:31], x[31]
    k, below = sub(C, e, const_bits(127, 8), 8)
    big = C.AND(NOT(below), or_reduce(C, k[5:]))
    nan = C.AND(_all(C, e), or_reduce(C, m))
    w = [ZERO] * 8 + list(m) + [ONE]
    for (sh, keep), r in zip(_STAGES, (NOT(k[4]), NOT(k[3]), NOT(k[2]), NOT(k[1]), NOT(k[0]))):
        w = _stage(C, r, w, sh, keep)
    ok = C.AND(NOT(s), NOT(nan))
    sat, fits = C.AND(ok, big), C.AND(ok, C.AND(NOT(below), NOT(big)))
    return [sat ^ C.AND(fits, b) for b in w[:8]]


def _stage(C, r, w, sh, keep):
    """`mux(r, w >> sh, w)` to ``keep`` bits, each r AND v made once: in the first stage a bit read shifted at i and unshifted at
    i + sh (its other operand zero) would be ANDed with r twice in a lowering that does not share terms."""
    ands = {}
    out = []
    for i in range(keep):
        v = (w[i + sh] if i + sh < len(w) else ZERO) ^ w[i]
        if v.key() not in ands:
            ands[v.key()] = C.AND(r, v)
        out.append(w[i] ^ ands[v.key()])
    return out


def _all(C, bits):
    out = bits[0]
    for b in bits[1:]:
        out = C.AND(out, b)
    return out


def ue8m0_byte(C_, s):
    """`Ue8m0CeilByte`: bits 23..30 of the 32-bit sum bits(s) + 0x7FFFFF."""
    return add(C_, list(s), const_bits(0x7FFFFF, 32), width=32)[23:31]


F32ToU32RzByteV2 = definition("F32ToU32RzByte", 2, f32_to_u32_rz_byte_bits, (("x", F32_BITS),), E8_BITS, word=F32ToU32RzByte,
                              doc="the low byte of cvt.rzi.u32.f32 on bits: F32ToU32RzByte_v1")
Ue8m0CeilByteV2 = definition("Ue8m0CeilByte", 2, ue8m0_byte, (("s", F32_BITS),), E8_BITS, word=Ue8m0CeilByte,
                             doc="The UE8M0 byte of a scale word on bits: Ue8m0CeilByte_v1")


class GateOps:
    """Word steps as gates on ``C``: each value a list of forms (LSB first), each method the gates of the word primitive of the
    same name in a word module's ``WORD`` (`dsa`'s and the modules built on it)."""

    def __init__(self, C):
        self.C = C

    def word(self, bits: int):
        return const_bits(bits, 32)

    def neg(self, x):
        return FP.f32_neg(x)

    def add(self, a, b):
        return FP.f32_add(self.C, a, b, nan_word=PTX)

    def sub(self, a, b):
        return FP.f32_sub(self.C, a, b, nan_word=PTX)

    def mul(self, a, b):
        return FP.f32_mul(self.C, a, b, nan_word=PTX)

    def fma(self, a, b, c):
        return FP.f32_fma(self.C, a, b, c, nan_word=PTX)

    def fma_ftz(self, a, b, c):
        C = self.C
        return ftz(C, FP.f32_fma(C, ftz(C, a), ftz(C, b), ftz(C, c), nan_word=PTX))

    def rcp(self, b):
        return BM.div_full_rcp(self.C, b)

    def scale_a(self, a, b):
        return BM.div_full_scale_a(self.C, a, b)

    def rsqrt(self, x):
        return BM.rsqrt_approx(self.C, x)

    def ex2(self, x):
        return BM.mufu_ex2(self.C, x)

    def fabs(self, x):
        return list(x[:31]) + [ZERO]

    def fmaxf(self, a, b):
        return SC.fmaxmin(self.C, a, b, True)

    def fminf(self, a, b):
        return SC.fmaxmin(self.C, a, b, False)

    def e4m3(self, x):
        return SC.f32_to_e4m3_sat(self.C, x)

    def iadd(self, a, b):
        return add(self.C, a, b, width=32)

    def iand(self, a, b):
        return [self.C.AND(x, y) for x, y in zip(a, b)]

    def isub(self, a, b):
        return sub(self.C, a, b, 32)[0]

    def i2f(self, a):
        return BD.i32_to_f32(self.C, a)

    def ceil(self, x):
        return BD.f32_ceil_ftz(self.C, x)

    def cvt(self, h):
        return FP.bf16_to_f32(h)

    def bf16(self, u):
        return FP.f32_to_bf16(self.C, u, 0x7FFF)

    def fsel(self, c, a, b):
        return mux(self.C, c, a, b)

    def gt(self, a, b):
        return f32_gt(self.C, a, b)[0]

    def ile(self, a, b):
        """Signed a <= b: not b < a, the borrow of (b ^ 2^31) - (a ^ 2^31)."""
        flip = lambda x: list(x[:31]) + [NOT(x[31])]  # noqa: E731
        return NOT(sub(self.C, flip(b), flip(a), 32)[1])



class SelectGateOps(GateOps):
    """`GateOps` with a word module's ``isel``, ``band`` and ``bnot``, and ``bit(x, i)``."""

    def isel(self, c, a, b):
        return mux(self.C, c, a, b)

    def band(self, a, b):
        return self.C.AND(a, b)

    def bnot(self, a):
        return NOT(a)

    def bit(self, x, i: int):
        return x[i]


def _neg_if(C, c, w):
    """c ? -w : w on 32 bits."""
    return add(C, [b ^ c for b in w], [c], width=32)


def divrem(C, a, b):
    """`i32_divrem` on bits: |a| / |b| by restoring division (|b| <= 2^31, so the partial remainder fits 32 bits), the
    quotient negated when the signs differ and the remainder when a is negative, and a zero divisor's quotient -1 (its remainder is
    then a, the division's own)."""
    sa, sb = a[31], b[31]
    ua, ub = _neg_if(C, sa, a), _neg_if(C, sb, b)
    r, q = [ZERO] * 32, [ZERO] * 32
    for i in reversed(range(32)):
        r = [ua[i]] + r[:31]
        d, borrow = sub(C, r, ub, 32)
        q[i] = NOT(borrow)
        r = mux(C, q[i], d, r)
    zero = NOT(or_reduce(C, b))
    q = _neg_if(C, sa ^ sb, q)
    return [x ^ C.AND(zero, NOT(x)) for x in q], _neg_if(C, sa, r)


I32DivV2 = definition("I32Div", 2, lambda C, a, b: divrem(C, a, b)[0], (("a", I32_BITS), ("b", I32_BITS)), I32_BITS, word=I32Div,
                      doc="div.s32 on bits: I32Div_v1, `divrem`'s quotient")
I32RemV2 = definition("I32Rem", 2, lambda C, a, b: divrem(C, a, b)[1], (("a", I32_BITS), ("b", I32_BITS)), I32_BITS, word=I32Rem,
                      doc="rem.s32 on bits: I32Rem_v1, `divrem`'s remainder")
I32MulV2 = definition("I32Mul", 2, lambda C, a, b: multiply(C, a, b)[:32], (("a", I32_BITS), ("b", I32_BITS)), I32_BITS, word=I32Mul,
                      doc="mul.lo.s32 on bits: I32Mul_v1, the product's low 32 bits")


# --- MUFU.LG2 on [1, 2) ---------------------------------------------------------------------------------------------------------

LG2_OFFSET_BITS, LG2_SQUARE_FROM, LG2_SHIFT, LG2_SCALE = 17, 19, 2, -36
#: (C0, C1, C2) per segment g = m >> 17
LG2_COEFFS = (
    (29509, 47272, -1454), (6148432709, 46545, -1410), (12203000645, 45840, -1370), (18166489925, 45156, -1330),
    (24041640773, 44492, -1292), (29831013189, 43847, -1254), (35537077061, 43221, -1220), (41162191685, 42612, -1184),
    (46708667205, 42020, -1152), (52178604869, 41444, -1118), (57574130501, 40884, -1088), (62897230661, 40339, -1060),
    (68149822277, 39809, -1036), (73333732165, 39292, -1008), (78450783045, 38788, -982), (83502646085, 38297, -958),
    (88490939205, 37818, -932), (93417288517, 37351, -910), (98283180869, 36896, -890), (103090078533, 36451, -866),
    (107839419205, 36018, -850), (112532534085, 35594, -828), (117170770757, 35180, -808), (121755394885, 34776, -792),
    (126287618885, 34380, -770), (130768622405, 33994, -754), (135199564613, 33616, -736), (139581543237, 33247, -722),
    (143915623237, 32886, -708), (148202869573, 32532, -692), (152444248901, 32186, -678), (156640731973, 31847, -662),
    (160793285445, 31516, -652), (164902798149, 31191, -638), (168970179397, 30872, -622), (172996260677, 30561, -614),
    (176981869381, 30255, -600), (180927816517, 29955, -586), (184834904901, 29662, -578), (188703847237, 29374, -566),
    (192535413573, 29091, -552), (196330328901, 28814, -542), (200089273157, 28542, -532), (203812897605, 28276, -524),
    (207501906757, 28014, -514), (211156931397, 27757, -506), (214778549061, 27505, -498), (218367394629, 27257, -488),
    (221924049733, 27013, -476), (225449096005, 26774, -468), (228943086405, 26540, -464), (232406541125, 26309, -454),
    (235840037701, 26082, -446), (239244047173, 25859, -438), (242619102021, 25640, -432), (245965656901, 25424, -422),
    (249284195141, 25213, -418), (252575208261, 25004, -408), (255839150917, 24799, -402), (259076424517, 24598, -398),
    (262287512389, 24399, -390), (265472783173, 24204, -384), (268632675141, 24012, -378), (271767585605, 23823, -372),
)
#: the measurement the coefficients reproduce: ``lg2.approx.ftz.f32`` of ``0x3F800000 | m`` for every m (`registry.mla_capture`'s
#: probe), the SHA-256 of the 2^23 little-endian u32 words
_DIFFTEST, _CAPTURE = EVIDENCE["cc12_difftest"], EVIDENCE["dsv4_moe_capture"]
LG2_MEASURED = {"device": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "cc": "12.0",
                "run_record": _DIFFTEST["run_record"], "words": 1 << 23,
                "sha256": "7fd9be9a924e40a0ee18ba20e7f5158135112947597c8dd86131d3d4d8f9dbfa"}


def truncated_square(u):
    """Q(u) for a 17-bit offset (an int, or an array of integers wide enough for 2^16): the folded partial products of u^2 (u_a
    at column 2a, u_a u_b at a + b + 1 for a < b) in columns >= 19, shifted right by 19."""
    nb, p = LG2_OFFSET_BITS, LG2_SQUARE_FROM
    bit = [(u >> a) & 1 for a in range(nb)]
    q = u * 0
    for a in range(nb):
        if 2 * a >= p:
            q = q + (bit[a] << (2 * a - p))
        for b in range(max(a + 1, p - a - 1), nb):
            q = q + ((bit[a] & bit[b]) << (a + b + 1 - p))
    return q


def lg2_interpolant(m: int) -> int:
    """G at the mantissa m: ``(C0[g] + C1[g] u + C2[g] Q(u)) >> 2``, g = m >> 17, u = m mod 2^17."""
    c0, c1, c2 = LG2_COEFFS[m >> LG2_OFFSET_BITS]
    u = m & ((1 << LG2_OFFSET_BITS) - 1)
    return (c0 + c1 * u + c2 * truncated_square(u)) >> LG2_SHIFT


def mufu_lg2_mantissa_bits(x: int) -> int:
    """MUFU.LG2 of 1.m, m = x & 0x7FFFFF: G 2^-36 rounded toward zero, and +0 at m = 0."""
    m = x & 0x7FFFFF
    if m == 0:
        return 0
    G = lg2_interpolant(m)
    e = G.bit_length() - 1
    frac = G << (23 - e) if e <= 23 else G >> (e - 23)
    return (e + 127 + LG2_SCALE) << 23 | frac & 0x7FFFFF


@primitive("MufuLg2Mantissa", 1, [("x", F32)], F32,
           conformance="exact-model-tested: lg2.approx.ftz.f32 on every 1.m of [1, 2), 8,388,608 words measured on a cc 12.0 GPU "
                       f"({_DIFFTEST['run_record']}) equal to the interpolant; captured on a cc 9.0 GPU (132 SMs) too: "
                       "lg2.approx.ftz.f32 and lg2.approx.f32 on every 1.m are those words, lg2(2) = 1 and a NaN 0x7FFFFFFF "
                       f"({_CAPTURE['run_record']}, the harness at {_CAPTURE['harness']}, tests/properties/dsv4_moe_capture_gpu.py)",
           tables=(("lg2", "library"),))
def MufuLg2Mantissa(x: int) -> int:
    """MUFU.LG2 (``lg2.approx.f32``) of 1.m, m = x's 23 mantissa bits: the unit's quadratic interpolator rounded toward zero on
    the 2^-36 grid, +0 at m = 0."""
    return mufu_lg2_mantissa_bits(x)


# --- MUFU.LG2 on [1, 2) --------------------------------------------------------------------------------------------------------

#: the columns of 2v (2v < 2^39 on every mantissa), and the low bits of 2v below G
LG2_COLUMNS, LG2_DROP = 39, 1 + LG2_SHIFT
LG2_SEGMENT_BITS = 23 - LG2_OFFSET_BITS
LG2_SQUARE_BITS = 2 * LG2_OFFSET_BITS - LG2_SQUARE_FROM


@cache
def _lg2_plan() -> tuple[tuple[int, int], tuple[tuple[int, ...], ...]]:
    """The segment functions of `table_word`'s encoding for `LG2_COEFFS`: the digit signs n_k of C1 (K1 of them), of
    C2 (K2), then the 39 bits of 2 C0 with the digits' constants folded in, each as the monomials of the segment bits in its
    algebraic normal form; and (K1, K2)."""
    c0 = [2 * c for c, _, _ in LG2_COEFFS]
    rows, Ks = [], []
    for n, cs in ((LG2_OFFSET_BITS, [c for _, c, _ in LG2_COEFFS]), (LG2_SQUARE_BITS, [c for _, _, c in LG2_COEFFS])):
        K, B = BM._digits(cs)
        Ks.append(K)
        rows += [[1 - ((b >> k) & 1) for b in B] for k in range(K)]
        c0 = [c - ((1 << n) - 1) * ((((1 << K) - 1) ^ b) + 1) for c, b in zip(c0, B)]
    rows += [[(c >> j) & 1 for c in c0] for j in range(LG2_COLUMNS)]
    for r in rows:
        for i in range(LG2_SEGMENT_BITS):
            for g in range(len(r)):
                if g >> i & 1:
                    r[g] ^= r[g ^ (1 << i)]
    return (Ks[0], Ks[1]), tuple(tuple(g for g, a in enumerate(r) if a) for r in rows)


def mufu_lg2_mantissa(C, x):
    """`MufuLg2Mantissa_v1` on bits: the interpolant G as one carry-save sum, then G 2^-36 rounded toward zero, +0 at m = 0."""
    (K1, K2), anf = _lg2_plan()
    m = list(x[:23])
    u = m[:LG2_OFFSET_BITS]
    terms = BM._monomials(C, m[LG2_OFFSET_BITS:])
    fns = [BM._xor_all(terms[i] for i in r) for r in anf]
    cols: list[list] = [[] for _ in range(LG2_COLUMNS)]
    for v, signs in ((u, fns[:K1]), (BM.truncated_square(C, u, LG2_SQUARE_FROM), fns[K1:K1 + K2])):
        for k, n_k in enumerate(signs):
            for j, b in enumerate(v):
                if j + k < LG2_COLUMNS:
                    cols[j + k].append(b ^ n_k)
        for j, b in enumerate(v):
            cols[j].append(NOT(b))
    for j, b in enumerate(fns[K1 + K2:]):
        cols[j].append(b)
    G = compress_columns(C, cols)[LG2_DROP:LG2_COLUMNS]
    top = [ZERO] * (64 - len(G)) + list(G)
    _, z = lzc(C, top)
    mant = shl_window(C, top, z, 40, 63)
    exp, _ = sub(C, const_bits(len(G) - 1 + 127 + LG2_SCALE, 8), z, 8)
    # m = 0 from the NOTs of u the sum reads and the segment bits' zero test, the XOR of all their monomials (no AND), so that
    # no AND repeats a monomial or a product of the square
    nz = NOT(and_reduce(C, [NOT(b) for b in u] + [BM._xor_all(terms)]))
    return [C.AND(nz, b) for b in mant + exp] + [ZERO]


MufuLg2MantissaV2 = definition("MufuLg2Mantissa", 2, mufu_lg2_mantissa, (("x", F32_BITS),), F32_BITS, word=MufuLg2Mantissa,
                               doc="MUFU.LG2 of 1.m on bits: MufuLg2Mantissa_v1's interpolator as one carry-save sum, the word "
                                   "rounded toward zero by a leading-zero count and a shift")

__all__ = ["F32ToU32RzByte", "F32ToU32RzByteV2", "I32Div", "I32DivV2", "I32Mul", "I32MulV2", "I32Rem", "I32RemV2",
           "MufuLg2Mantissa", "MufuLg2MantissaV2", "Ue8m0CeilByte", "Ue8m0CeilByteV2"]
