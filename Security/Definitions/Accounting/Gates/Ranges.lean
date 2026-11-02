import Mathlib.Data.Rat.Floor
import Definitions.Accounting.Gates.Model

/-!
# Primitives' output intervals: the evaluators, and property/v1's rules for them (trusted)

The evaluators of the budgeted primitives whose value's interval property/v1 derives from its operands' (`RANGES` in
`verity/ml/circuits/stacking.py`: P's producers), with those rules, of the quantizers and the LUT softmax, whose
interval is their own (`benchmarks/ppl/steps.py`), and of the exact-int8 profile's primitives (vLLM's served int8
linear and LUT attention, `verity_vllm/program/registry/int8_exact.py`). An f32 value is its encoding read as s32, as
property/v1 reads it: `+0 … +inf` is `[0, 0x7F800000]`, ordered as the floats, and a negative float or `−0` reads below
0.
`Guarantees.Gates` states that each interval holds its evaluator's value.
-/

namespace Accounting.Gates

/-- `z ∈ [a]`. -/
def Iv.Mem (a : Iv) (z : ℤ) : Prop := a.1 ≤ z ∧ z ≤ a.2

/-- A 32-bit word read as two's complement. -/
def S32 (z : ℤ) : Prop := s32Min ≤ z ∧ z ≤ s32Max

/-- An operand's range as property/v1 carries it: an interval, or `none`, its type's whole range. -/
abbrev Rng := Option Iv

/-- `z` lies in the range (anywhere, for `none`). -/
def Rng.Holds : Rng → ℤ → Prop
  | none, _ => True
  | some r, z => r.Mem z

/-! ## f32 on s32 readings -/

/-- The encoding's sign-cleared bits. -/
def f32Mag (a : ℤ) : ℤ := if a < 0 then a + 2 ^ 31 else a

/-- A NaN: exponent all ones and a nonzero significand. -/
def f32NaN (a : ℤ) : Bool := decide (0x7F800000 < f32Mag a)

/-- An integer ordered as the non-NaN floats, `+0` and `−0` both 0 (`scalar._order`). -/
def f32Key (a : ℤ) : ℤ := if a < 0 then -(a + 2 ^ 31) else a

/-- The GPU's canonical NaN, `max.f32` of two NaNs (`scalar.F32_NAN_PTX`). -/
def f32NanPtx : ℤ := 0x7FFFFFFF

/-- CUDA `fmaxf` (`want`) or `fminf` (`scalar._fmaxmin`): a NaN operand gives the other, two NaNs the canonical NaN,
and `+0` is above `−0`. -/
def f32MaxMin (want : Bool) (a b : ℤ) : ℤ :=
  if f32NaN a && f32NaN b then f32NanPtx
  else if f32NaN a then b
  else if f32NaN b then a
  else if f32Key a = f32Key b then (if want then max a b else min a b)
  else if want then (if f32Key b < f32Key a then a else b)
  else if f32Key a < f32Key b then a else b

/-- `F32Fmaxf` (PTX `max.f32`). -/
def f32Fmaxf (a b : ℤ) : ℤ := f32MaxMin true a b

/-- `F32Fminf` (PTX `min.f32`). -/
def f32Fminf (a b : ℤ) : ℤ := f32MaxMin false a b

/-- The low byte of `cvt.rzi.u32.f32` on an encoding `u ≤ +inf` of a non-negative float
(`dsv4_qnorm_insert.f32_to_u32_rz_byte`): below 1 it truncates to 0, from `2^32` (`+inf` too) it saturates to
`0xFFFFFFFF`, and otherwise it is the truncation's low byte. -/
def rzByteNat (u : ℕ) : ℕ :=
  let e := u / 2 ^ 23
  let m := u % 2 ^ 23 + 2 ^ 23
  if e < 127 then 0 else if 159 ≤ e then 255
  else if 150 ≤ e then m * 2 ^ (e - 150) % 256 else m / 2 ^ (150 - e) % 256

/-- `F32ToU32RzByte`: a NaN, a negative float or `−0` gives 0. -/
def f32RzByte (a : ℤ) : ℤ := if a < 0 ∨ 0x7F800000 < a then 0 else rzByteNat a.toNat

/-! ## property/v1's rules for P's producers -/

/-- `+inf` (`stacking.F32_POS`'s top). -/
def f32Inf : ℤ := 0x7F800000

/-- A range within `+0 … +inf` (`stacking._f32_pos`). -/
def Iv.f32Pos (r : Iv) : Bool := decide (r.1 ≤ r.2 ∧ 0 ≤ r.1 ∧ r.2 ≤ f32Inf)

/-- `stacking._fmaxf`: two operands in `+0 … +inf` give the hull of their maxima, one gives at least its low end. -/
def fmaxfRule : Rng → Rng → Rng
  | some a, some b =>
      if a.f32Pos && b.f32Pos then some (max a.1 b.1, max a.2 b.2)
      else if a.f32Pos then some (a.1, f32Inf) else if b.f32Pos then some (b.1, f32Inf) else none
  | some a, none => if a.f32Pos then some (a.1, f32Inf) else none
  | none, some b => if b.f32Pos then some (b.1, f32Inf) else none
  | none, none => none

/-- `stacking._fminf`: two operands in `+0 … +inf` give the hull of their minima, one gives at most its high end. -/
def fminfRule : Rng → Rng → Rng
  | some a, some b =>
      if a.f32Pos && b.f32Pos then some (min a.1 b.1, min a.2 b.2)
      else if a.f32Pos then some (s32Min, a.2) else if b.f32Pos then some (s32Min, b.2) else none
  | some a, none => if a.f32Pos then some (s32Min, a.2) else none
  | none, some b => if b.f32Pos then some (s32Min, b.2) else none
  | none, none => none

/-- `stacking._rz_byte`: an operand below `128.0` (`0x43000000`) gives its ends' truncations, 0 for an end below 0. -/
def rzByteRule : Rng → Rng
  | some r =>
      if r.1 ≤ r.2 ∧ r.2 < 0x43000000 then
        some (if 0 ≤ r.1 then f32RzByte r.1 else 0, if 0 ≤ r.2 then f32RzByte r.2 else 0)
      else none
  | none => none

/-! ## Quantizers and the LUT softmax (`benchmarks/ppl/steps.py`) -/

/-- `r` clamped to `[lo, lo + w]`. -/
def clampTo (lo : ℤ) (w : ℕ) (r : ℤ) : ℤ := max lo (min (lo + w) r)

/-- `clamp(round(x/s), lo, lo + w)`, for `v = x/s` and any rounding `rnd` (`torch.round`'s half to even among them):
`quant`, `quant_rows` and the weights' grids at `[−top, top]`, P's at `[0, p_top]`. -/
def quantize (lo : ℤ) (w : ℕ) (rnd : ℚ → ℤ) (v : ℚ) : ℤ := clampTo lo w (rnd v)

/-- A `b`-bit dither, `m = 2^(b−1) − 1`: `⌊v⌋` clamped to `[lo, lo + w]`, less `m`, plus coins `R` on
`{0, …, 2^b − 1}`. `act_quant` clamps `v` to `[−t, t]`, `t = top − 1 − m` (so `lo = −t`, `w = 2t`); `floor_quant`
picks the scale so that `⌊v⌋` lies in `[m − top − 1, top − m − 1]`, where the clamp changes nothing. -/
def dither (lo : ℤ) (w m : ℕ) (v : ℚ) (R : Fin (2 * m + 2)) : ℤ := clampTo lo w ⌊v⌋ - m + R

/-- The hull of a column's words and 0, the value outside it of a gather whose fill is 0. -/
def hull0 (T : List ℤ) : Iv := (T.foldr min 0, T.foldr max 0)

/-- The LUT softmax's element: a floor, a min and an 8-bit gather (the model's `IntOp.gather`, fill 0) into the table
`T`. -/
def lutSoftmax (T : List ℤ) (d : ℚ) : ℤ := (IntOp.gather none 0 : IntOp Unit).raw (fun _ => 0) (min ⌊d⌋ 255 :: T)

/-! ## The exact-int8 profile's primitives (`verity_vllm/program/registry/int8_exact.py`) -/

/-- The truncation toward zero of the non-negative float whose encoding is `u` (`+inf` past every bound). -/
def truncNat (u : ℕ) : ℕ :=
  let e := u / 2 ^ 23
  let m := u % 2 ^ 23 + 2 ^ 23
  if e < 127 then 0 else if 150 ≤ e then m * 2 ^ (e - 150) else m / 2 ^ (150 - e)

/-- `F32ToS8Rzi` (`cvt.rzi.sat.s8.f32`), its code read as s8: the float's truncation toward zero, saturated to
`[−128, 127]`; a NaN gives 0. -/
def f32ToS8Rzi (a : ℤ) : ℤ :=
  if f32NaN a then 0
  else if a < 0 then -min 128 ((truncNat (f32Mag a).toNat : ℕ) : ℤ) else min 127 ((truncNat a.toNat : ℕ) : ℤ)

/-- `127.0` (`C_TOP`). -/
def f32Top : ℤ := 0x42FE0000

/-- `−127.0` (`C_NEG_TOP`), read as s32. -/
def f32NegTop : ℤ := 0xC2FE0000 - 2 ^ 32

/-- `Int8QuantElement`'s clamp and cast of the rounded quotient `r = rint(RN32(x/s))`: `F32Fmaxf` against `−127.0`,
`F32Fminf` against `127.0`, then `F32ToS8Rzi`. -/
def int8Clamp (r : ℤ) : ℤ := f32ToS8Rzi (f32Fminf (f32Fmaxf r f32NegTop) f32Top)

/-- `LutIndex`: `min(⌊max(d, 0) · a / 2^32⌋, 255)`, `d` read as s32 and `a` unsigned (fattn's `umulhi(m − S, A)`). -/
def lutIndex (d a : ℤ) : ℤ := min (max d 0 * (a % 2 ^ 32) / 2 ^ 32) 255

/-- `Exp2Lut127`'s table: `LUT[t] = rint(127 · 2^(−t/16))`, `t < 256`. -/
def exp2Lut127 : List ℤ :=
  [127, 122, 116, 112, 107, 102, 98, 94, 90, 86, 82, 79, 76, 72, 69, 66, 64, 61, 58, 56, 53, 51, 49, 47, 45, 43,
    41, 39, 38, 36, 35, 33, 32, 30, 29, 28, 27, 26, 24, 23, 22, 21, 21, 20, 19, 18, 17, 17, 16, 15, 15, 14, 13,
    13, 12, 12, 11, 11, 10, 10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3,
    3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1] ++ List.replicate 128 0

/-- `S32LowByte`: a word's low byte read as s8 (`sext8`). -/
def s32LowByte (z : ℤ) : ℤ := (z + 128) % 256 - 128

end Accounting.Gates
