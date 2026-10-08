import Evaluate.Fp32

/-!
# The word casts

Executable transcriptions of the casts of `verity.ml.circuits.ops.prims` and of the functions of
`verity.ml.circuits.silicon.cast` they evaluate, copies of `Security/Definitions/Core/Prims.lean` where it defines them.
Words as in `Evaluate.Scalar`: an E4M3 code is below `2^8` and an E2M1 code below `2^4`.

`cast.f32_to_e4m3_sat_word` computes with exact rationals, every one of which is a dyadic `m · 2^e`, carried here as its
numerator and exponent (`dyadicLt`, `roundEvenDyadic`).
-/

namespace Evaluate.Casts

open Evaluate.Fp32 (f32ToBf16RnBits BF16_NAN_TORCH BF16_NAN_F2FP)

/-- Python's `int.bit_length`. Must equal `Verity.TC.bitLength`. -/
def bitLength (n : Nat) : Nat := if n = 0 then 0 else n.log2 + 1

/-- `Bf16ToF32_v1`, `cast.bf16_to_f32_word`: the exact widening. Must equal `Verity.Prims.bf16ToF32`. -/
def bf16ToF32 (a : Nat) : Nat := (a &&& 0xFFFF) <<< 16

/-- `F32ToBf16Rn_v1`, `cast.f32_to_bf16_word`: round to nearest even, every NaN `0x7FC0` (the software word). Security:
none yet; it is `Verity.Fp32.f32ToBf16RnBits a BF16_NAN_TORCH`. -/
def f32ToBf16Rn (a : Nat) : Nat := f32ToBf16RnBits a BF16_NAN_TORCH

/-- `F2fpBf16_v1` (`cvt.rn.bf16.f32`), `cast.f32_to_bf16_hw_word`: round to nearest even, every NaN `0x7FFF`. Must equal
`Verity.Prims.f2fpBf16`, which is `Verity.TC.Spec.cvtRnBf16F32`. -/
def f2fpBf16 (a : Nat) : Nat := f32ToBf16RnBits a BF16_NAN_F2FP

/-- `cast.E4M3_NAN`. Must equal `Verity.Prims.E4M3_NAN`. -/
def E4M3_NAN : Nat := 0x7F

/-- `E4m3ToF32_v1`, `cast.e4m3_to_f32_word`: an E4M3 code widened exactly; both NaN codes to `0x7FFFE000`. Must equal
`Verity.Prims.e4m3ToF32`. -/
def e4m3ToF32 (x : Nat) : Nat :=
  let b := x &&& 0xFF
  if b &&& 0x7F = E4M3_NAN then 0x7FFFE000
  else
    let sign := (b &&& 0x80) <<< 24
    let e := (b >>> 3) &&& 0xF
    let m := b &&& 0x7
    if e = 0 then
      if m = 0 then sign
      else
        let p := bitLength m - 1
        sign ||| ((p + 127 - 9) <<< 23) ||| ((m - (1 <<< p)) <<< (23 - p))
    else sign ||| ((e + 127 - 7) <<< 23) ||| (m <<< 20)

/-- `m · 2^e < c · 2^f`, exactly. Must equal `Verity.Prims.dyadicLt`. -/
def dyadicLt (m : Nat) (e : Int) (c : Nat) (f : Int) : Bool :=
  let k := min e f
  decide ((m <<< (e - k).toNat) < (c <<< (f - k).toNat))

/-- `m · 2^e` rounded to an integer, to nearest with ties to even. Must equal `Verity.Prims.roundEvenDyadic`. -/
def roundEvenDyadic (m : Nat) (e : Int) : Nat :=
  if 0 ≤ e then m <<< e.toNat
  else
    let s := (-e).toNat
    let n := m >>> s
    let r := m % 2 ^ s
    if 2 * r > 2 ^ s ∨ (2 * r = 2 ^ s ∧ n &&& 1 = 1) then n + 1 else n

/-- `F32ToE4m3Sat_v1` (`cvt.rn.satfinite.e4m3x2.f32`, one lane), `cast.f32_to_e4m3_sat_word`: NaN to `E4M3_NAN`; `±inf`
and every `|x| ≥ 464` to `±0x7E`; otherwise round to nearest even onto the E4M3 grid. Must equal
`Verity.Prims.f32ToE4m3Sat`. -/
def f32ToE4m3Sat (x : Nat) : Nat :=
  let u := x &&& 0xFFFFFFFF
  let sign := if u >>> 31 ≠ 0 then 0x80 else 0
  let exp := (u >>> 23) &&& 0xFF
  let mant := u &&& 0x7FFFFF
  if exp = 0xFF then (if mant ≠ 0 then E4M3_NAN else sign ||| 0x7E)
  else if exp = 0 ∧ mant = 0 then sign
  else
    let sig := if exp = 0 then mant else (1 <<< 23) ||| mant
    let ex : Int := if exp = 0 then -149 else (exp : Int) - 150
    if !dyadicLt sig ex 464 0 then sign ||| 0x7E
    else
      let e : Int := (bitLength sig : Int) - 1 + ex
      let stepExp : Int := if e ≥ -6 then e - 3 else -9
      let n := roundEvenDyadic sig (ex - stepExp)
      if n = 0 then sign
      else if !dyadicLt n stepExp 464 0 then sign ||| 0x7E
      else if dyadicLt n stepExp 1 (-6) then sign ||| (n <<< (stepExp + 9).toNat)
      else
        let e2 : Int := (bitLength n : Int) - 1 + stepExp
        let m3 := ((n <<< 3) >>> (bitLength n - 1)) - 8
        sign ||| ((e2 + 7).toNat <<< 3) ||| m3

/-- `prims.E2M1_NAN_CODE`. Must equal `Verity.Prims.E2M1_NAN_CODE`. -/
def E2M1_NAN_CODE : Nat := 0x7

/-- `cast.E2M1_UP`. Must equal `Verity.Prims.E2M1_UP`. -/
def E2M1_UP : List (Nat × Bool) :=
  [(0x3E800000, false), (0x3F400000, true), (0x3FA00000, false), (0x3FE00000, true), (0x40200000, false),
   (0x40600000, true), (0x40A00000, false)]

/-- `F32ToE2m1Sat_v1` (`cvt.rn.satfinite.e2m1x2.f32`, one lane), `cast.f32_to_e2m1_sat_word(x, E2M1_NAN_CODE)`. Must equal
`Verity.Prims.f32ToE2m1Sat`. -/
def f32ToE2m1Sat (x : Nat) : Nat :=
  let u := x &&& 0xFFFFFFFF
  let sign := if u >>> 31 ≠ 0 then 0x8 else 0
  let mag := u &&& 0x7FFFFFFF
  if mag > 0x7F800000 then E2M1_NAN_CODE
  else sign ||| E2M1_UP.countP (fun (bound, up) => mag > bound || (up && mag == bound))

end Evaluate.Casts
