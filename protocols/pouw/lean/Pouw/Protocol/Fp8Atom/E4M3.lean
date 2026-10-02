import Mathlib.Algebra.Order.Field.Rat

/-!
# The OCP E4M3 codec (trusted definitions)

An E4M3 code is a byte `c : Fin 256`: bit 7 is the sign, bits 6..3 the exponent field `e` (bias 7), bits 2..0 the
mantissa field `m`. This is `verity.ml.tc.term.E4M3` (`Format("e4m3", 4, 3, "e4m3")`) and `total_fp8.e4m3_class`:

* `0x7F` and `0xFF` (`e = 15`, `m = 7`) are NaN. E4M3 has **no infinities**; `0x7E` is 448.
* `0x00` and `0x80` are the two zeros.
* The other 252 codes are nonzero finite values; subnormals (`e = 0`, `m ≠ 0`) keep their unnormalised significand at
  exponent `−9`. So 254 codes are finite, the zeros included.

A finite code has the value `(−1)^sign · sig · 2^lsbExp`, with `sig = m` (subnormal) or `8 + m` (normal) and
`lsbExp = max e 1 − 10`: `decode_e4m3`'s `(sign, exp, sig)`, whose `exp` is `1 − 7 − 3 = −9` for `e = 0` and
`e − 7 − 3` otherwise.
-/

namespace Pouw.Fp8Atom

/-- An E4M3 code: one byte. -/
abbrev Code := Fin 256

/-- The sign bit (bit 7). -/
def Code.sign (c : Code) : Bool := c.val >>> 7 == 1

/-- The exponent field `e` (bits 6..3). -/
def Code.expField (c : Code) : Nat := (c.val >>> 3) % 16

/-- The mantissa field `m` (bits 2..0). -/
def Code.manField (c : Code) : Nat := c.val % 8

/-- NaN: `0x7F` or `0xFF` (exponent and mantissa fields all ones). -/
def Code.isNaN (c : Code) : Bool := c.val % 128 == 127

/-- A zero: `0x00` or `0x80`. -/
def Code.isZero (c : Code) : Bool := c.val % 128 == 0

/-- The significand, hidden bit included for normals: `m` if `e = 0`, else `8 + m`. -/
def Code.sig (c : Code) : Nat := if c.expField = 0 then c.manField else 8 + c.manField

/-- `(e or 1)`, the exponent field with subnormals read at `1`. -/
def Code.expOr1 (c : Code) : Nat := max c.expField 1

/-- The weight of the significand's least significant bit: `2^(max e 1 − 10)`. -/
def Code.lsbExp (c : Code) : Int := (c.expOr1 : Int) - 10

/-- The exact value of a finite code; `none` for the two NaNs. -/
def decode (c : Code) : Option ℚ :=
  if c.isNaN then none else some ((if c.sign then -1 else 1) * (c.sig : ℚ) * (2 : ℚ) ^ c.lsbExp)

end Pouw.Fp8Atom
