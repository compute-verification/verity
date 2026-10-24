import Definitions.Pouw.Fp8Atom.Fp32

/-! The proof-only part of `Definitions.Pouw.Fp8Atom.Fp32`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.Fp8Atom

/-- A value with a 24-bit significand at some exponent. -/
def Representable (x : ℚ) : Prop := ∃ m e : ℤ, |m| < 2 ^ 24 ∧ x = m * 2 ^ e

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
