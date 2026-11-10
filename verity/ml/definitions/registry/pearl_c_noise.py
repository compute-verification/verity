"""Pearl-C -h3's noise line as Definitions: a line's 32 E4M3 codes from the 32 bytes of its hash.

-h3 samples a noise line from 32 bytes ``raw`` of SHA-512 (`verity.ml.protocols.pouw.schemes.pearl_c.sample_line` after its
hash; in Lean `Pouw.PearlC.NoiseSha512`'s `lineInts`, `lineScale` and `lineList`):

    x_p     = (raw_p mod 128) + 1, negated when raw_p >= 128                         PearlCLineInt
    root    = ⌊√(1024 · Σ_p x_p²)⌋                                                    PearlCLineNormRoot
    scale   = bf16(f32(norm32) / f32(bf16(root)))                                    PearlCLineScale
    code_p  = e4m3(f32(bf16(f32(bf16(x_p)) · f32(scale))))                           PearlCLineCode, PearlCLineCodes

``raw`` is the line's hash, `pearl_c.s512` of ``"line"`` over the line's key, side, factor and index, and ``norm32`` the
BF16 word of 32 times the line's norm: `NORM_E` for an E line (norm 256), `NORM_F` for an F line (norm 16). bf16 is
`F2fpBf16_v1`, the product `F32Mul_v2`, the quotient `F32Div_v2` and e4m3 `F32ToE4m3Sat_v1`, so a line is total on every
encoding. `pearl_c.sample_line` rejects a non-finite step; at the two norms none is (a root is 181 to 23,170, a product at
most 128 times the scale), and there the tests hold a line to it.

A byte string is an ``Array<n, Value<8>>`` in byte order. Each Definition has a word form (``_v1``) and a Boolean version
(``_v2``) whose word view it is; on bits a byte is 8 bits, least significant first.
"""
from __future__ import annotations

import math

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, compress_columns, const_bits, multiply
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import bind, composite, primitive
from verity.ml.circuits.ops.boolean.scalar import f32_to_e4m3_sat
from verity.ml.circuits.ops.boolean.sqrt import isqrt
from verity.ml.circuits.ops.fp32 import F32DivV2, F32MulV2
from verity.ml.circuits.ops.prims import Bf16ToF32, F2fpBf16, F32ToE4m3Sat, S32ToF32Rn
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.types import Array, Value

__all__ = ["BYTES32", "NORM_E", "NORM_F", "PearlCLineCode", "PearlCLineCodeV2", "PearlCLineCodes", "PearlCLineCodesV2",
           "PearlCLineInt", "PearlCLineIntV2", "PearlCLineNormRoot", "PearlCLineNormRootV2", "PearlCLineScale",
           "PearlCLineScaleV2"]

U8, BF16, U32 = Value(8), Value(16), Value(32)
#: a line's hash, and its 32 codes
BYTES32 = Array(32, U8)
#: the BF16 words of 32 times a line's norm: an E line's 256 and an F line's 16
NORM_E, NORM_F = 0x4600, 0x4400
#: `F2fpBf16_v1`'s word for a NaN
_NAN_BF16 = 0x7FFF

_LINE = "protocol-defined; equal to `pearl_c.sample_line` on its vectors (verity/ml/definitions/tests/test_pearl_c_noise.py)"


@primitive("PearlCLineInt", 1, [("b", U8)], U32, conformance=_LINE)
def PearlCLineInt(b: int) -> int:
    """A line's integer from its byte, as an s32 word: (b mod 128) + 1, negated when b >= 128; so 1 to 128 in magnitude."""
    x = (int(b) & 0x7F) + 1
    return (-x if int(b) >> 7 else x) & 0xFFFFFFFF


@primitive("PearlCLineNormRoot", 1, [("raw", BYTES32)], U32, conformance=_LINE)
def PearlCLineNormRoot(*raw: int) -> int:
    """⌊√(1024 · Σ x²)⌋ over a line's 32 integers (`PearlCLineInt_v1` of each byte): 32 times the line's norm before scaling,
    rounded down; 181 to 23,170."""
    return math.isqrt(1024 * sum(((int(b) & 0x7F) + 1) ** 2 for b in raw))


def _int_bits(C, b: list) -> list:
    """`PearlCLineInt_v1` on bits: with v = b mod 128 and s = b's top bit, v + 1 when s is clear and NOT v (−(v + 1)) when
    set, which is (v XOR s) + NOT s on 8 bits with s as the operand's top bit; the 24 bits above are s."""
    s = b[7]
    return add(C, [x ^ s for x in b[:7]] + [s], [], cin=NOT(s), width=8) + [s] * 24


def _root_bits(C, *raw: list) -> list:
    """`PearlCLineNormRoot_v1` on bits: with v = b mod 128, Σ (v + 1)² = Σ v² + 2 Σ v + 32, each v² its diagonal bits and
    one AND a pair of bits, all in one carry-save sum of 20 bits; then the digit-by-digit root of the sum times 1024."""
    cols: list[list] = [[] for _ in range(14)]
    cols[5].append(ONE)
    for b in raw:
        v = b[:7]
        for i in range(7):
            cols[2 * i].append(v[i])
            cols[i + 1].append(v[i])
            for j in range(i + 1, 7):
                cols[i + j + 1].append(C.AND(v[i], v[j]))
    total = compress_columns(C, cols)[:20]
    return isqrt(C, [ZERO] * 10 + total) + [ZERO] * 17


PearlCLineIntV2 = definition("PearlCLineInt", 2, _int_bits, (("b", BL.bit_type(U8)),), BL.bit_type(U32), word=PearlCLineInt,
                             conformance=_LINE, doc="A line's integer from its byte, on bits: PearlCLineInt_v1")
PearlCLineNormRootV2 = definition("PearlCLineNormRoot", 2, _root_bits, (("raw", BL.bit_type(BYTES32)),), BL.bit_type(U32),
                                  word=PearlCLineNormRoot, conformance=_LINE,
                                  doc="⌊√(1024 · Σ x²)⌋ over a line's 32 integers, on bits: PearlCLineNormRoot_v1")


@composite("PearlCLineScale", 1, [], lambda S: ((("raw", BYTES32), ("norm32", BF16)), BF16), conformance=_LINE)
def PearlCLineScale(B, S, raw, norm32):
    """A line's scale, a BF16 word: bf16(f32(norm32) / f32(bf16(root))) for root = `PearlCLineNormRoot_v1` of the line's
    bytes and ``norm32`` the BF16 word of 32 times the line's norm, by `S32ToF32Rn_v1`, `F2fpBf16_v1` and `F32Div_v2`
    (`pearl_kw.bf16_div`)."""
    root = B.call(F2fpBf16, B.call(S32ToF32Rn, B.call(PearlCLineNormRoot, raw)))
    return B.call(F2fpBf16, B.call(F32DivV2, B.call(Bf16ToF32, norm32), B.call(Bf16ToF32, root)))


@composite("PearlCLineCode", 1, [], lambda S: ((("b", U8), ("scale", BF16)), U8), conformance=_LINE)
def PearlCLineCode(B, S, b, scale):
    """One code of a line: e4m3(f32(bf16(f32(bf16(x)) · f32(scale)))) for x = `PearlCLineInt_v1` of its byte, by
    `S32ToF32Rn_v1`, `F2fpBf16_v1`, `F32Mul_v2` (`pearl_kw.bf16_mul`) and `F32ToE4m3Sat_v1`."""
    x = B.call(F2fpBf16, B.call(S32ToF32Rn, B.call(PearlCLineInt, b)))
    y = B.call(F2fpBf16, B.call(F32MulV2, B.call(Bf16ToF32, x), B.call(Bf16ToF32, scale)))
    return B.call(F32ToE4m3Sat, B.call(Bf16ToF32, y))


def _bf16_bits(C, x: list) -> list:
    return fp.f32_to_bf16(C, x, _NAN_BF16)


def _scale_bits(C, *el: list) -> list:
    root = _bf16_bits(C, fp.s32_to_f32(C, _root_bits(C, *el[:32])))
    return _bf16_bits(C, fp.f32_div(C, fp.bf16_to_f32(el[32]), fp.bf16_to_f32(root), nan_word=fp.F32_NAN_PTX))


def _code_bits(C, b: list, scale: list) -> list:
    """`PearlCLineCode_v1` on bits, its product rounded once. The integer x = ±(v + 1) is exact as a BF16 word, and its FP32
    product with a BF16 word is exact short of overflow (16 significant bits, a multiple of FP32's least subnormal), so the
    product's BF16 word is the one nearest (v + 1) · sig · 2^(e − 134), for the scale's significand sig and exponent e: one
    `round_fp`, which never shifts right since x is at least 1. An infinite scale overflows there to its infinity, and a NaN
    is `F2fpBf16_v1`'s."""
    S = fp.unpack_fp(C, scale, 8, 7)
    s = b[7] ^ S.s
    product = multiply(C, add(C, b[:7], [], cin=ONE, width=8), S.sig)
    y = fp.round_fp(C, s, add(C, S.e_eff, const_bits(8, 4), width=10), 0, product, s, 8, 7, underflow=False)
    y = [OR(C, S.nan, f) for f in y[:7]] + y[7:15] + [C.AND(NOT(S.nan), y[15])]
    return f32_to_e4m3_sat(C, fp.bf16_to_f32(y))


def _codes_bits(C, *el: list) -> list[list]:
    scale = _scale_bits(C, *el)
    return [_code_bits(C, b, scale) for b in el[:32]]


PearlCLineScaleV2 = definition(
    "PearlCLineScale", 2, _scale_bits, (("raw", BL.bit_type(BYTES32)), ("norm32", BL.bit_type(BF16))), BL.bit_type(BF16),
    word=bind(PearlCLineScale), conformance=_LINE,
    doc="A line's scale on bits, its steps' builders in one circuit: PearlCLineScale_v1")
PearlCLineCodeV2 = definition(
    "PearlCLineCode", 2, _code_bits, (("b", BL.bit_type(U8)), ("scale", BL.bit_type(BF16))), BL.bit_type(U8),
    word=bind(PearlCLineCode), conformance=_LINE,
    doc="One code of a line on bits, its product of 8-bit significands rounded once to BF16: PearlCLineCode_v1")


@composite("PearlCLineCodes", 1, [], lambda S: ((("raw", BYTES32), ("norm32", BF16)), BYTES32), conformance=_LINE)
def PearlCLineCodes(B, S, raw, norm32):
    """A line's 32 E4M3 codes from its hash: `PearlCLineCode_v1` of each byte at the line's `PearlCLineScale_v1`
    (`pearl_c.sample_line` after its hash)."""
    return B.batch(PearlCLineCode, raw, B.call(PearlCLineScale, raw, norm32), axes=(0, None))


PearlCLineCodesV2 = definition(
    "PearlCLineCodes", 2, _codes_bits, (("raw", BL.bit_type(BYTES32)), ("norm32", BL.bit_type(BF16))), BL.bit_type(BYTES32),
    word=bind(PearlCLineCodes), conformance=_LINE,
    doc="A line's 32 codes on bits, in one circuit, so what a code reads of the scale alone is computed once: "
        "PearlCLineCodes_v1")
