import Definitions.Core.TC.Spec

/-!
# Casts: the word casts of `verity.ml.prims`

A transcription of the casts of `verity.ml.prims` that PoUW's tile-check Programs call, and of the functions of
`verity.ml.tc.cast` they evaluate, branch for branch. Words are natural numbers in `Verity.TC.Spec`'s encoding: an f32 is a
word below `2^32`, a bf16 a word below `2^16`, an E4M3 code a word below `2^8` and an E2M1 code a word below `2^4`. Each
function is total on `Nat`, masks its operand as the Python does, and is the Python's function on every non-negative
integer, so on every word of its operand's width.

`cast.f32_to_e4m3_sat_word` computes with exact rationals (`Fraction`), every one of which is a dyadic `m · 2^e`; here each
is carried as its numerator `m` and exponent `e`, and each comparison and rounding of the Python is the same comparison
and rounding of those integers (`dyadicLt`, `roundEvenDyadic`). Subtractions of naturals are written so that they never
truncate (`p + 127 - 9` for the Python's `p - 9 + 127`).
-/

namespace Verity.Prims

open Verity.TC

/-- `Bf16ToF32_v1`, `cast.bf16_to_f32_word`: the exact widening, the BF16 word as the high half of the FP32 word. -/
def bf16ToF32 (a : Nat) : Nat := (a &&& 0xFFFF) <<< 16

/-- `F2fpBf16_v1` (SASS `F2FP.BF16.F32.PACK_AB`, `cvt.rn.bf16.f32`), `cast.f32_to_bf16_hw_word`: round to nearest even,
a finite word that rounds past the largest finite BF16 to `±inf`, `±inf` kept, every NaN to `0x7FFF`. It is
`Verity.TC.Spec.cvtRnBf16F32` (`total.cvt_rn_bf16_f32`), the same function: the Python's mask of the operand to 32 bits
changes neither the NaN test, which reads bits 0 to 30, nor the 16 bits the rounding keeps. -/
def f2fpBf16 (a : Nat) : Nat := Spec.cvtRnBf16F32 a

/-- `cast.E4M3_NAN`: OCP E4M3's NaN, the code `0x7F` (and `0xFF`); E4M3 has no infinities, and its largest finite code
`0x7E` is 448. -/
def E4M3_NAN : Nat := 0x7F

/-- `E4m3ToF32_v1` (`float(__nv_fp8_e4m3)`), `cast.e4m3_to_f32_word`: an E4M3 code widened to f32 exactly; both NaN codes
to `0x7FFFE000` (the f16 canonical NaN `0x7FFF` widened, the sign dropped). That NaN word is the Python's, from the
documented conversion chain and unmeasured: no GPU capture widens `0x7F` or `0xFF`. A subnormal code `m · 2^-9` is
`(1 + (m - 2^p) / 2^p) · 2^(p - 9)` with `p = ⌊log₂ m⌋`. -/
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

/-- `m · 2^e < c · 2^f`, exactly: both sides scaled by `2^-(min e f)` to naturals. -/
def dyadicLt (m : Nat) (e : Int) (c : Nat) (f : Int) : Bool :=
  let k := min e f
  decide ((m <<< (e - k).toNat) < (c <<< (f - k).toNat))

/-- `m · 2^e` rounded to an integer, to nearest with ties to even: `cast.f32_to_e4m3_sat_word`'s `n` for `q = m · 2^e`,
from `n = ⌊q⌋` and `r = q - n` (rounded up when `r > 1/2`, or `r = 1/2` and `n` is odd). For `e ≥ 0`, `q` is an integer;
for `e < 0`, `r · 2^(-e)` is `m`'s low `-e` bits. -/
def roundEvenDyadic (m : Nat) (e : Int) : Nat :=
  if 0 ≤ e then m <<< e.toNat
  else
    let s := (-e).toNat
    let n := m >>> s
    let r := m % 2 ^ s
    if 2 * r > 2 ^ s ∨ (2 * r = 2 ^ s ∧ n &&& 1 = 1) then n + 1 else n

/-- `F32ToE4m3Sat_v1` (`cvt.rn.satfinite.e4m3x2.f32`, one lane), `cast.f32_to_e4m3_sat_word`: NaN to `E4M3_NAN`, its sign
dropped; `±inf`, and every `|x|` at or past the 448|480 tie 464, to `±0x7E` (448, the sign kept); otherwise round to
nearest even onto the E4M3 grid, whose step is `2^(e - 3)` in the binade `2^e ≤ |x| < 2^(e + 1)` for `e ≥ -6` and `2^-9`
below `2^-6`; `|x| < 2^-10` and the tie `2^-10` round to a signed zero.

The finite magnitude `mag` is `sig · 2^ex` (`mant / 2^149` for a subnormal word, `(2^23 | mant) · 2^(exp - 150)`
otherwise), `e = ⌊log₂ mag⌋` is `bitLength sig - 1 + ex` (the Python finds it from the fraction's bit lengths), `q =
mag / 2^step_exp` is `sig · 2^(ex - step_exp)`, and the rounded value `val` is `n · 2^step_exp`, with `⌊log₂ val⌋ =
bitLength n - 1 + step_exp` and `(val / 2^e2 - 1) · 8 = 8n / 2^(bitLength n - 1) - 8`. -/
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

/-- `prims.E2M1_NAN_CODE`: the code `cvt.rn.satfinite.e2m1x2.f32` writes for a NaN (E2M1 has none), `+6`, as measured on
the RTX PRO 6000. -/
def E2M1_NAN_CODE : Nat := 0x7

/-- `cast.E2M1_UP`: the seven code boundaries of round to nearest even onto the E2M1 magnitudes
`{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, each `|x|` as an f32 word and `true` when a tie on it rounds up to the even code. -/
def E2M1_UP : List (Nat × Bool) :=
  [(0x3E800000, false), (0x3F400000, true), (0x3FA00000, false), (0x3FE00000, true), (0x40200000, false),
   (0x40600000, true), (0x40A00000, false)]

/-- `F32ToE2m1Sat_v1` (`cvt.rn.satfinite.e2m1x2.f32`, one lane), `cast.f32_to_e2m1_sat_word(x, E2M1_NAN_CODE)`: the sign bit
kept (a signed zero included), the magnitude rounded to nearest even onto the E2M1 grid, the code being how many of the
`E2M1_UP` boundaries `|x|` passes, so that `|x| > 5` and `±inf` saturate to `±6` (code 7); a NaN gives `E2M1_NAN_CODE`,
its sign dropped. -/
def f32ToE2m1Sat (x : Nat) : Nat :=
  let u := x &&& 0xFFFFFFFF
  let sign := if u >>> 31 ≠ 0 then 0x8 else 0
  let mag := u &&& 0x7FFFFFFF
  if mag > 0x7F800000 then E2M1_NAN_CODE
  else sign ||| E2M1_UP.countP (fun (bound, up) => mag > bound || (up && mag == bound))

end Verity.Prims
