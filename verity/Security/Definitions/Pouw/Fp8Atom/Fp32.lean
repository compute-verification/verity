import Mathlib.Data.Rat.Floor
import Definitions.Pouw.Fp8Atom.Atom

/-!
# The FP32 add of the running words (trusted definitions)

`rnAdd x y` is `RN_f32(x + y)`: the exact sum rounded to a 24-bit significand, ties to even, with no exponent range.
It is IEEE-754 binary32 addition (CUDA `FADD`, `__fadd_rn`) wherever that result is finite and normal or zero, which
is every running word of H-1's chains (no overflow and no subnormal in range). It is the `fpSem` of
`Pouw.Dimension.H100.runWords` for FADD.

* `binade x` is `⌊log₂ x⌋` for `0 < x`, computed from `Nat.log2` of the numerator and denominator.
* `ulp x` is `2^(binade |x| − 23)`, the spacing of 24-bit significands in the binade of `x`.
* `roundEven m` rounds a rational to the nearest integer, ties to even.
* `rn x` is `± roundEven (|x| / ulp x) · ulp x`; `Representable x` says `x = m · 2^e` with `|m| < 2^24`.
-/

namespace Pouw.Fp8Atom

/-- `⌊log₂ x⌋` for `0 < x`: `2 ^ binade x ≤ x < 2 ^ (binade x + 1)`. -/
def binade (x : ℚ) : ℤ :=
  let k : ℤ := (x.num.natAbs.log2 : ℤ) - x.den.log2
  if x < 2 ^ k then k - 1 else k

/-- The spacing of 24-bit significands in the binade of `x ≠ 0`. -/
def ulp (x : ℚ) : ℚ := 2 ^ (binade |x| - 23)

/-- Round to the nearest integer, ties to even. -/
def roundEven (m : ℚ) : ℤ :=
  let q := ⌊m⌋
  if m - q < 1 / 2 then q else if 1 / 2 < m - q then q + 1 else if q % 2 = 0 then q else q + 1

/-- Round to a 24-bit significand, ties to even, with no exponent range: the sign, times `|x| / ulp x` rounded to an
integer, times `ulp x`. -/
def rn (x : ℚ) : ℚ := (if x < 0 then -1 else 1) * (roundEven (|x| / ulp x) : ℚ) * ulp x

/-- **The FP32 add** `RN_f32(x + y)`. -/
def rnAdd (x y : ℚ) : ℚ := rn (x + y)

/-- A value with a 24-bit significand at some exponent. -/
def Representable (x : ℚ) : Prop := ∃ m e : ℤ, |m| < 2 ^ 24 ∧ x = m * 2 ^ e

/-- A finite FP32 word's value; `none` for NaN and the infinities. -/
def wordQ (w : Nat) : Option ℚ := (fp32Value w).finite?.map Term.value

/-- An FP32 add conformance vector: `c` is the word IEEE binary32 addition writes for the words `a` and `b`. -/
structure AddVec where
  a : Nat
  b : Nat
  c : Nat

/-- An add vector holds: all three words are finite and `rnAdd` maps the values of `a` and `b` to the value of `c`. -/
def AddVec.holds (v : AddVec) : Bool :=
  match wordQ v.a, wordQ v.b, wordQ v.c with
  | some x, some y, some z => decide (rnAdd x y = z)
  | _, _, _ => false

end Pouw.Fp8Atom
