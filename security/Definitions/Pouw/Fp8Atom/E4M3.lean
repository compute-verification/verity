module

public import Mathlib.Data.Rat.Init
public import Mathlib.Tactic.ToAdditive
import Mathlib.Tactic.Bound.Init
import Mathlib.Tactic.Finiteness.Attr
import Mathlib.Tactic.SetLike

/-! The OCP E4M3 codec, `verity.ml.tc.term.E4M3`, which has no infinities and is NaN at `0x7F` and `0xFF`. -/

@[expose] public section

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
