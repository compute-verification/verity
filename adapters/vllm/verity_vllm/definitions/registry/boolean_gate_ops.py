"""Word steps as gates, shared by the Boolean versions of the word modules built on `dsa` (`boolean_io_shell`, `boolean_stop_rule`):
core's gate builder for each word primitive, with the GPU's NaN word for the FP32 ones; no table or lookup gate.  Beside them, five
word primitives and their Boolean versions, read by modules of more than one model:

    F32ToU32RzByte_v1 / _v2      (x: f32) -> byte: the low byte of `cvt.rzi.u32.f32`
    Ue8m0CeilByte_v1 / _v2       (s: f32) -> e8m0: the UE8M0 byte of a scale word, its exponent rounded up
    I32Mul_v1 / _v2, I32Div_v1 / _v2, I32Rem_v1 / _v2
                                 (a, b: i32) -> i32: mul.lo.s32, div.s32, rem.s32 (C `int` * / %)
"""
from __future__ import annotations

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import NOT, ONE, ZERO, add, const_bits, multiply, mux, or_reduce, sub
from verity.circuits.boolean.trace import definition
from verity.circuits.defs import primitive
from verity.circuits.ops.boolean import mufu as BM
from verity.circuits.ops.boolean import scalar as SC
from verity.circuits.ops.boolean.softmax import f32_gt, ftz
from verity.circuits.silicon import fp as FP
from verity.circuits.types import Array, Value

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
