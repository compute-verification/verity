"""`registry.dense`'s element-wise rows as Boolean Definitions: the next version of each word Definition, the same function on
bits. Gemma-2's GeGLU on bits is `registry.boolean_tanh`'s `GeluTanhMul_v4{I}`.

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
    AddScalarF32_v4{N}, AddScalarBf16_v4{N}, Bf16DivScalar_v4{N}, Bf16MulScalar_v4{N}
                                                  the same rows with the f32 scalar an input (``c``; the division's
                                                  ``inv = RN32(1 / C)``); word views `value_inputs`' _v3

Each is a batch of one element Definition per member, traced from `verity.ml.circuits.silicon.fp`'s builders with the evaluator's NaN words,
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
from verity.ml.circuits.boolean.forms import const_bits
from verity.ml.circuits.boolean.trace import definition
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import dense as D
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry import value_inputs as VI

__all__ = ["AddScalarBf16", "AddScalarBf16Element", "AddScalarBf16In", "AddScalarF32", "AddScalarF32In", "AddWidenedBf16",
           "AddWidenedBf16Element", "Bf16DivScalar", "Bf16DivScalarIn", "Bf16MulScalar", "Bf16MulScalarIn", "Bf16MulScalarTensor",
           "MulVecF32", "ScaleRowBf16", "ScaleRowBf16Element", "ScaleRowF32", "SquareBf16", "SquareBf16Element", "SquareF32"]

BF16 = Array(16, BL.BIT)
F32 = Array(32, BL.BIT)


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


# The scalar rows with their f32 scalar an input (`value_inputs`' _v3 words, which a `registered_values` program registers): the
# scalar's own gates once per row, then one element per member.

AddScalarF32InScalar, AddScalarF32InElement = EW.shared_operand(
    ("AddScalarF32InScalar", "AddScalarF32InElement"), 1, lambda C, x, c: fp.f32_add(C, x, c, ir_nan=True), (("x", F32), ("c", F32)),
    F32, 1, ("the gates of RN32(x + c) that read only c, once per row", "RN32(x + c), F32Add_v1's NaN words, from c's own gates"))
AddScalarBf16InScalar, AddScalarBf16InElement = EW.shared_operand(
    ("AddScalarBf16InScalar", "AddScalarBf16InElement"), 1, lambda C, w, c: fp.f32_add(C, fp.bf16_to_f32(w), c, ir_nan=True),
    (("w", BF16), ("c", F32)), F32, 1,
    ("the gates of RN32(f32(w) + c) that read only c, once per row", "RN32(f32(w) + c), F32Add_v1's NaN words, from c's own gates"))
Bf16MulF32InScalar, Bf16MulF32InElement = EW.shared_operand(
    ("Bf16MulF32InScalar", "Bf16MulF32InElement"), 1, EW.bf16_mul_f32, (("x", BF16), ("s", F32)), BF16, 1,
    ("the gates of Bf16MulF32_v1(x, s) that read only s, once per row", "Bf16MulF32_v1(x, s) from s's own gates"))


def _add_scalar_f32_in(B, S, x, c):
    """`out_j = RN32(x_j + c)` on bits, c's own gates once."""
    return B.batch(AddScalarF32InElement, x, c, B.call(AddScalarF32InScalar, c), axes=(0, None, None))


def _add_scalar_bf16_in(B, S, w, c):
    """`out_j = RN32(f32(w_j) + c)` on bits, c's own gates once."""
    return B.batch(AddScalarBf16InElement, w, c, B.call(AddScalarBf16InScalar, c), axes=(0, None, None))


def _bf16_div_scalar_in(B, S, x, inv):
    """`out_j = Bf16MulF32_v1(x_j, inv)` on bits, `inv = RN32(1 / C)` an input, its own gates once."""
    return B.batch(Bf16MulF32InElement, x, inv, B.call(Bf16MulF32InScalar, inv), axes=(0, None, None))


def _bf16_mul_scalar_in(B, S, x, c):
    """`out_j = Bf16MulF32_v1(x_j, c)` on bits, `c = f32(C)` an input, its own gates once."""
    return B.batch(Bf16MulF32InElement, x, c, B.call(Bf16MulF32InScalar, c), axes=(0, None, None))


def _row_in(x: Array, out: Array, name: str, xname: str = "x"):
    return lambda S: (((xname, Array(S.N, x)), (name, F32)), Array(S.N, out))


AddScalarF32In = EW.on_bits("AddScalarF32", 4, ("N",), _row_in(F32, F32, "c"), _add_scalar_f32_in, VI.AddScalarF32V3)
AddScalarBf16In = EW.on_bits("AddScalarBf16", 4, ("N",), _row_in(BF16, F32, "c", "w"), _add_scalar_bf16_in, VI.AddScalarBf16V3)
Bf16DivScalarIn = EW.on_bits("Bf16DivScalar", 4, ("N",), _row_in(BF16, BF16, "inv"), _bf16_div_scalar_in, VI.Bf16DivScalarV3)
Bf16MulScalarIn = EW.on_bits("Bf16MulScalar", 4, ("N",), _row_in(BF16, BF16, "c"), _bf16_mul_scalar_in, VI.Bf16MulScalarV3)
