"""`registry.dense`'s activations as Boolean Definitions: the next version of each word Definition, the same function on bits.

    GeluTanhMulBf16_v2   (g, u: Array<16, Value<1>>) -> Array<16, Value<1>>                      word view: GeluTanhMulBf16_v1
    GeluTanhMul_v2{I}    (gu: Array<2I, Array<16, Value<1>>>) -> Array<I, Array<16, Value<1>>>   word view: GeluTanhMul_v1{I}

`GeluTanhMulBf16_v2` is traced from `gelu_tanh_mul` (`verity.ml.circuits.boolean.trace.definition`), so its And gates are that builder's
live ANDs: ``a = gelu_tanh_bf16(g)``, then ``bf16_rn(f32(a) * (f32(u) + 0.0f))`` through `verity.ml.circuits.silicon.fp` with F2FP's NaN
0x7FFF. ``a = (T)gelu_tanh_kernel((float)g)`` is a function of the 16-bit g alone (`dense.gelu_tanh_bf16_bits`), and its word
Definition reads no MUFU (``tanhf`` is `TanhF32Rn_v1`, RN32 of the real tanh), so neither does this. Off a band of exponent
fields e it is exact arithmetic on g's fields; on the band it reads a constant table of the word function's own values
(`elementwise.bf16_band`):

    g NaN, or -inf                 0x7FFF  ((0.5f * -inf) * 0 is NaN)
    e < BAND_LO                    bf16_rn(g / 2)  (1 + tanh(inner) rounds to 1)
    g > 0, e > BAND_HI[0]          g (+inf included)
    g < 0, e > BAND_HI[1]          -0  (1 + tanh(inner) is 0)
    BAND_LO <= e <= BAND_HI[sign]  g's sign, and the 15 magnitude bits of the table

The kernel's ``y + 0.0f`` turns ``u = -0`` into +0 and is the identity on every other word that reaches the output (a NaN ends
in the cast's 0x7FFF). `GeluTanhMul_v2{I}` is `GeluTanhMul_v1`'s body on bits: a batch of I `GeluTanhMulBf16_v2` Calls over
(gu[i], gu[I + i]), so each member's 16 output bits are one unit under `Q_word_v1`.

Gemma-2's RMSNorm chain and logit softcap read `registry.dense`'s element-wise rows, here on bits (BF16 = Array<16, Value<1>>,
F32 = Array<32, Value<1>>; each word view is its _v1 at the same statics, `Bf16MulScalarTensor_v1` being `registry.b1`'s):

    SquareBf16_v2{N}                              (x: Array<N, BF16>) -> Array<N, F32>        RN32(f32(x_j) * f32(x_j))
    SquareF32_v2{N}                               (x: Array<N, F32>) -> Array<N, F32>         RN32(x_j * x_j)
    AddScalarF32_v2{N,C}                          (x: Array<N, F32>) -> Array<N, F32>         RN32(x_j + f32(C))
    AddScalarBf16_v2{N,C}                         (w: Array<N, BF16>) -> Array<N, F32>        RN32(f32(w_j) + f32(C))
    AddWidenedBf16_v2{N}                          (x, r: Array<N, BF16>) -> Array<N, F32>     RN32(f32(x_j) + f32(r_j))
    ScaleRowBf16_v2{N}                            (x: Array<N, BF16>, s: Array<1, F32>) -> Array<N, F32>
    ScaleRowF32_v2{N}                             (x: Array<N, F32>, s: Array<1, F32>) -> Array<N, F32>
    MulVecF32_v2{N}                               (x, w: Array<N, F32>) -> Array<N, F32>      F32Mul_v3(x_j, w_j)
    Bf16DivScalar_v2{N,C}, Bf16MulScalar_v2{N,C}  (x: Array<N, BF16>) -> Array<N, BF16>       Bf16MulF32_v1(x_j, K)
    Bf16MulScalarTensor_v2{N}                     (x: Array<N, BF16>, s: Array<1, BF16>) -> Array<N, BF16>

Each is a batch of one element Definition per member, traced from `verity.ml.circuits.silicon.fp`'s builders with the IR's NaN words,
and computes no value twice (`Q_word`'s no-recompute rule): ``x * x`` decodes x once (`SquareF32Element_v1`, not
``F32Mul_v3(x, x)``); a constant static is folded into the gates (`elementwise.specialized`; `Bf16DivScalar`'s K is ATen's
RN32(1 / C), `dense.reciprocal_bits`, `Bf16MulScalar`'s f32(C)); a scalar every member shares has its own gates once per row
(`elementwise.shared_operand`). None reads a MUFU, and no sub-Call is a word Definition.
"""
from __future__ import annotations

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.ops.boolean.elementwise import bf16_band
from verity.ml.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, and_reduce, const_bits, or_reduce, sub
from verity.ml.circuits.boolean.trace import definition
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import dense as D
from verity_vllm.program.registry import prims as P

__all__ = ["AddScalarBf16", "AddScalarBf16Element", "AddScalarF32", "AddWidenedBf16", "AddWidenedBf16Element", "Bf16DivScalar",
           "Bf16MulScalar", "Bf16MulScalarTensor", "GELU_BAND_HI", "GELU_BAND_LO", "GeluTanhMul", "GeluTanhMulBf16", "MulVecF32",
           "ScaleRowBf16", "ScaleRowBf16Element", "ScaleRowF32", "SquareBf16", "SquareBf16Element", "SquareF32", "gelu_tanh_bf16",
           "gelu_tanh_mul"]

BF16 = Array(16, BL.BIT)
F32 = Array(32, BL.BIT)
NAN_F2FP = P.BF16_NAN_F2FP

#: the exponent fields `gelu_tanh_bf16` reads from its table, per sign
GELU_BAND_LO = 118
GELU_BAND_HI = (128, 129)


def _half(C, g):
    """``bf16_rn(g / 2)``'s 15 magnitude bits: (for e <= 1, the 8-bit significand halved, ties to even; for e >= 2, the
    exponent field less one)."""
    m, e = list(g[:7]), list(g[7:15])
    return add(C, m[1:] + [e[0]], [C.AND(m[0], m[1])], width=8) + [ZERO] * 7, m + sub(C, e, [ONE], 8)[0]


def gelu_tanh_bf16(C, g):
    """``(T)gelu_tanh_kernel((float)g)``, the activation half of `GeluTanhMulBf16_v1`, for a 16-bit g (LSB first)."""
    m, e, s = list(g[:7]), list(g[7:15]), g[15]
    nan = C.AND(and_reduce(C, e), OR(C, or_reduce(C, m), s))
    _, below = sub(C, e, const_bits(GELU_BAND_LO, 8), 8)
    _, under = sub(C, e, const_bits(GELU_BAND_HI[0] + 1, 8), 8)
    keep = C.AND(C.AND(NOT(s), NOT(under)), NOT(nan))
    low = C.AND(below, NOT(or_reduce(C, e[1:])))
    half_low, half_high = _half(C, g)
    band = bf16_band(C, g, D.gelu_tanh_bf16_bits, GELU_BAND_LO, GELU_BAND_HI)
    mag = [x ^ C.AND(low, y) ^ C.AND(below ^ low, z) ^ C.AND(keep, w) ^ nan
           for x, y, z, w in zip(band, half_low, half_high, g[:15])]
    return mag + [C.AND(s, NOT(nan))]


def gelu_tanh_mul(C, g, u):
    """`GeluTanhMulBf16_v1`: ``bf16_rn(f32(gelu_tanh_bf16(g)) * (f32(u) + 0.0f))``, every NaN 0x7FFF."""
    a = gelu_tanh_bf16(C, g)
    u0 = list(u[:15]) + [C.AND(u[15], or_reduce(C, u[:15]))]
    return fp.f32_to_bf16(C, fp.f32_mul(C, fp.bf16_to_f32(a), fp.bf16_to_f32(u0)), NAN_F2FP)


GeluTanhMulBf16 = definition("GeluTanhMulBf16", 2, gelu_tanh_mul, (("g", BF16), ("u", BF16)), BF16, word=D.GeluTanhMulBf16,
                             doc="vLLM act_and_mul_kernel<gelu_tanh> on bits: GeluTanhMulBf16_v1, "
                                 "bf16(f32(bf16(gelu_tanh(g))) * (f32(u) + 0.0f)), NaN 0x7FFF")


def _gelu_tanh_mul(B, S, gu):
    """vLLM `gelu_tanh_and_mul` on bits: `out[i] = GeluTanhMulBf16_v2(gu[i], gu[I + i])`."""
    return B.batch(GeluTanhMulBf16, gu[0:S.I], gu[S.I:2 * S.I], axes=(0, 0))


GeluTanhMul = EW.on_bits("GeluTanhMul", 2, ("I",), lambda S: ((("gu", Array(2 * S.I, BF16)),), Array(S.I, BF16)),
                         _gelu_tanh_mul, D.GeluTanhMul)


def _f32_bits(c: float) -> int:
    """``f32(c)``, the constant `dense.f32c` makes of a static."""
    return P.bits_of(np.float32(float(c)))


SquareBf16Element = definition("SquareBf16Element", 1, lambda C, x: fp.f32_mul(C, fp.bf16_to_f32(x), fp.bf16_to_f32(x), ir_nan=True),
                               (("x", BF16),), F32, doc="RN32(f32(x) * f32(x)), F32Mul_v1's NaN words")
AddWidenedBf16Element = definition(
    "AddWidenedBf16Element", 1, lambda C, x, r: fp.f32_add(C, fp.bf16_to_f32(x), fp.bf16_to_f32(r), ir_nan=True),
    (("x", BF16), ("r", BF16)), F32, doc="RN32(f32(x) + f32(r)), F32Add_v1's NaN words")
AddScalarF32Element = EW.specialized("AddScalarF32Element", 1, ["K"], lambda S: lambda C, x: fp.f32_add(C, x, const_bits(S.K, 32), ir_nan=True),
                                     lambda S: ((("x", F32),), F32), doc="RN32(x + K) for the f32 word K, folded; F32Add_v1's NaN words")
AddScalarBf16Element = EW.specialized(
    "AddScalarBf16Element", 1, ["K"], lambda S: lambda C, w: fp.f32_add(C, fp.bf16_to_f32(w), const_bits(S.K, 32), ir_nan=True),
    lambda S: ((("w", BF16),), F32), doc="RN32(f32(w) + K) for the f32 word K, folded; F32Add_v1's NaN words")
Bf16MulScalarElement = EW.specialized("Bf16MulScalarElement", 1, ["K"], lambda S: lambda C, x: EW.bf16_mul_f32(C, x, const_bits(S.K, 32)),
                                      lambda S: ((("x", BF16),), BF16), doc="Bf16MulF32_v1(x, K) for the f32 word K, folded")
SquareF32Element = definition("SquareF32Element", 1, lambda C, x: fp.f32_mul(C, x, x, ir_nan=True), (("x", F32),), F32,
                              doc="RN32(x * x), F32Mul_v1's NaN words: x decoded once")
ScaleRowF32Scalar, ScaleRowF32Element = EW.shared_operand(
    ("ScaleRowF32Scalar", "ScaleRowF32Element"), 1, lambda C, x, s: fp.f32_mul(C, x, s, ir_nan=True), (("x", F32), ("s", F32)), F32, 1,
    ("the gates of RN32(x * s) that read only s, once per row", "RN32(x * s), F32Mul_v1's NaN words, from s's own gates"))
ScaleRowBf16Scalar, ScaleRowBf16Element = EW.shared_operand(
    ("ScaleRowBf16Scalar", "ScaleRowBf16Element"), 1, lambda C, x, s: fp.f32_mul(C, fp.bf16_to_f32(x), s, ir_nan=True),
    (("x", BF16), ("s", F32)), F32, 1,
    ("the gates of RN32(f32(x) * s) that read only s, once per row", "RN32(f32(x) * s), F32Mul_v1's NaN words, from s's own gates"))
Bf16MulScalarTensorScalar, Bf16MulScalarTensorElement = EW.shared_operand(
    ("Bf16MulScalarTensorScalar", "Bf16MulScalarTensorElement"), 1, EW.Bf16MulBf16.build, (("x", BF16), ("s", BF16)), BF16, 1,
    ("the gates of Bf16MulBf16_v1(x, s) that read only s, once per row", "Bf16MulBf16_v1(x, s) from s's own gates"))


def _row(x: Array, out: Array, *more: tuple[str, tuple[int | None, Array]]):
    """The signature of a row over ``x: Array<N, x>`` (and ``more``) to ``Array<N, out>``; a ``None`` length is N's."""
    return lambda S: ((("x", Array(S.N, x)),) + tuple((n, Array(S.N if k is None else k, t)) for n, (k, t) in more),
                      Array(S.N, out))


def _square_bf16(B, S, x):
    """`out_j = RN32(f32(x_j) * f32(x_j))` on bits."""
    return B.batch(SquareBf16Element, x, axes=(0,))


def _square_f32(B, S, x):
    """`out_j = RN32(x_j * x_j)` on bits."""
    return B.batch(SquareF32Element, x, axes=(0,))


def _add_scalar_f32(B, S, x):
    """`out_j = RN32(x_j + f32(C))` on bits, the constant folded."""
    return B.batch(bind(AddScalarF32Element, K=_f32_bits(S.C)), x, axes=(0,))


def _add_scalar_bf16(B, S, w):
    """`out_j = RN32(f32(w_j) + f32(C))` on bits, the constant folded."""
    return B.batch(bind(AddScalarBf16Element, K=_f32_bits(S.C)), w, axes=(0,))


def _add_widened_bf16(B, S, x, r):
    """`out_j = RN32(f32(x_j) + f32(r_j))` on bits."""
    return B.batch(AddWidenedBf16Element, x, r, axes=(0, 0))


def _scale_row_bf16(B, S, x, s):
    """`out_j = RN32(f32(x_j) * s)` on bits, s's own gates once."""
    return B.batch(ScaleRowBf16Element, x, s[0], B.call(ScaleRowBf16Scalar, s[0]), axes=(0, None, None))


def _scale_row_f32(B, S, x, s):
    """`out_j = RN32(x_j * s)` on bits, s's own gates once."""
    return B.batch(ScaleRowF32Element, x, s[0], B.call(ScaleRowF32Scalar, s[0]), axes=(0, None, None))


def _mul_vec_f32(B, S, x, w):
    """`out_j = F32Mul_v3(x_j, w_j)`."""
    return B.batch(EW.F32Mul, x, w, axes=(0, 0))


def _bf16_div_scalar(B, S, x):
    """`out_j = Bf16MulF32_v1(x_j, RN32(1 / C))` on bits (ATen's CPU-scalar reciprocal), the constant folded."""
    return B.batch(bind(Bf16MulScalarElement, K=D.reciprocal_bits(float(S.C))), x, axes=(0,))


def _bf16_mul_scalar(B, S, x):
    """`out_j = Bf16MulF32_v1(x_j, f32(C))` on bits, the constant folded."""
    return B.batch(bind(Bf16MulScalarElement, K=_f32_bits(S.C)), x, axes=(0,))


def _bf16_mul_scalar_tensor(B, S, x, s):
    """`out_j = Bf16MulBf16_v1(x_j, s)` on bits, the scalar a graph value shared by the row and its own gates once."""
    return B.batch(Bf16MulScalarTensorElement, x, s[0], B.call(Bf16MulScalarTensorScalar, s[0]), axes=(0, None, None))


SquareBf16 = EW.on_bits("SquareBf16", 2, ("N",), _row(BF16, F32), _square_bf16, D.SquareBf16)
SquareF32 = EW.on_bits("SquareF32", 2, ("N",), _row(F32, F32), _square_f32, D.SquareF32)
AddScalarF32 = EW.on_bits("AddScalarF32", 2, ("N", "C"), _row(F32, F32), _add_scalar_f32, D.AddScalarF32)
AddScalarBf16 = EW.on_bits("AddScalarBf16", 2, ("N", "C"), lambda S: ((("w", Array(S.N, BF16)),), Array(S.N, F32)),
                           _add_scalar_bf16, D.AddScalarBf16)
AddWidenedBf16 = EW.on_bits("AddWidenedBf16", 2, ("N",), _row(BF16, F32, ("r", (None, BF16))), _add_widened_bf16,
                            D.AddWidenedBf16)
ScaleRowBf16 = EW.on_bits("ScaleRowBf16", 2, ("N",), _row(BF16, F32, ("s", (1, F32))), _scale_row_bf16, D.ScaleRowBf16)
ScaleRowF32 = EW.on_bits("ScaleRowF32", 2, ("N",), _row(F32, F32, ("s", (1, F32))), _scale_row_f32, D.ScaleRowF32)
MulVecF32 = EW.on_bits("MulVecF32", 2, ("N",), _row(F32, F32, ("w", (None, F32))), _mul_vec_f32, D.MulVecF32)
Bf16DivScalar = EW.on_bits("Bf16DivScalar", 2, ("N", "C"), _row(BF16, BF16), _bf16_div_scalar, D.Bf16DivScalar)
Bf16MulScalar = EW.on_bits("Bf16MulScalar", 2, ("N", "C"), _row(BF16, BF16), _bf16_mul_scalar, D.Bf16MulScalar)
Bf16MulScalarTensor = EW.on_bits("Bf16MulScalarTensor", 2, ("N",), _row(BF16, BF16, ("s", (1, BF16))),
                                 _bf16_mul_scalar_tensor, b1.Bf16MulScalarTensor)
