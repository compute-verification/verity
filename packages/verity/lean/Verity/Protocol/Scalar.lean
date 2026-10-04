/-!
# Scalar primitives: `verity.ml.scalar`

A transcription of the primitives of `verity.ml.scalar` that PoUW's tile-check Programs call, function for function and
branch for branch, with `scalar._order` and `cast.f32_is_nan` that they read. Words are natural numbers in
`Verity.TC.Spec`'s encoding: an f32 or an i32 (two's complement, bit 31 the sign) is a word below `2^32`, a bf16 a word below
`2^16`, a bit `0` or `1`, and a bit result is the `Nat` `0` or `1`, as the IR has it. Each function is total on `Nat`, takes
its operands unmasked as the Python does, and is the Python's function on every non-negative integer, so on every word of
its operands' widths. The bit operations are Python's (`&&&` is `&`, `^^^` is `^`), and `Int` holds the Python's signed
integers.
-/

namespace Verity.Scalar

/-- `cast.f32_is_nan`: the exponent field all ones and a nonzero fraction. -/
def f32IsNan (u : Nat) : Bool := (u &&& 0x7F800000 == 0x7F800000) && (u &&& 0x7FFFFF != 0)

/-- `scalar._order(u, width)`: an integer whose order is the value order of the non-NaN float encodings of `width` bits,
the sign bit `2^(width - 1)` and the magnitude below it; `+0` and `-0` share `0`. -/
def order (u width : Nat) : Int :=
  let sign := 1 <<< (width - 1)
  let u := u &&& ((sign <<< 1) - 1)
  let m := u &&& (sign - 1)
  if u &&& sign ≠ 0 then -(m : Int) else m

/-- `scalar.F32_NAN_PTX`: the GPU's canonical FP32 NaN, which `max.f32` writes for two NaN operands. -/
def F32_NAN_PTX : Nat := 0x7FFFFFFF

/-- `F32Fabs_v1` (`fabsf`): the sign bit cleared, a NaN's payload kept. The NaN rule is the Python's, documented but
unmeasured: no GPU capture runs `fabsf` on a NaN. -/
def f32Fabs (x : Nat) : Nat := x &&& 0x7FFFFFFF

/-- `F32Neg_v1` (`neg.f32`): the sign bit flipped, so `+0` becomes `-0` and a NaN keeps its payload. The NaN rule is the
Python's, documented but unmeasured: no GPU capture runs `neg.f32` on a NaN. -/
def f32Neg (a : Nat) : Nat := (a ^^^ 0x80000000) &&& 0xFFFFFFFF

/-- `F32Fmaxf_v1` (`fmaxf`, PTX `max.f32`), `scalar._fmaxmin(a, b, True)`: IEEE maxNum. Two NaN operands give
`F32_NAN_PTX`, one NaN operand gives the other operand's word unchanged; otherwise the operand larger in `order`, and of
two operands equal in value the first, except that of two zeros `+0` wins: `max(+0, -0) = max(-0, +0) = +0`. -/
def f32Fmaxf (a b : Nat) : Nat :=
  let na := f32IsNan a
  let nb := f32IsNan b
  if na && nb then F32_NAN_PTX
  else if na then b
  else if nb then a
  else
    let ka := order a 32
    let kb := order b 32
    if ka = kb then
      if a &&& 0x7FFFFFFF = 0 ∧ b &&& 0x7FFFFFFF = 0 then (if a = 0 ∨ b = 0 then 0 else 0x80000000)
      else a
    else if ka > kb then a else b

/-- `I32Le_v1`: signed 32-bit `a ≤ b`, each operand read as two's complement with bit 31 its sign; `1` or `0`. -/
def i32Le (a b : Nat) : Nat :=
  let sa : Int := if a &&& 0x80000000 ≠ 0 then (a : Int) - 2 ^ 32 else a
  let sb : Int := if b &&& 0x80000000 ≠ 0 then (b : Int) - 2 ^ 32 else b
  if sa ≤ sb then 1 else 0

/-- `I32Add_v1`: the 32-bit add, wrapping modulo `2^32`. -/
def i32Add (a b : Nat) : Nat := (a + b) &&& 0xFFFFFFFF

/-- `BitAnd_v1`: `(a & b) & 1`, so `bitAnd 0 x = 0` for every `x`. -/
def bitAnd (a b : Nat) : Nat := a &&& b &&& 1

/-- `SelectI32_v1`: `c ? a : b` on the low bit of `c`. -/
def selectI32 (c a b : Nat) : Nat := if c &&& 1 ≠ 0 then a else b

/-- `SelectBf16_v1`: `c ? a : b` on the low bit of `c`. -/
def selectBf16 (c a b : Nat) : Nat := if c &&& 1 ≠ 0 then a else b

end Verity.Scalar
