import Definitions.Pouw.Fp8Atom.Atom

/-! The proof-only part of `Definitions.Pouw.Fp8Atom.Atom`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.Fp8Atom

/-- `ADA_E4M3_M16N8K32`: the same adder, two groups of 16. -/
def adaE4m3K32 : Pipeline := ⟨[16, 16], 14, -139⟩

/-- One `ADA_E4M3_M16N8K32` step. -/
def adaStep (acc : Nat) (a b : Fin 32 → Code) : Nat := adaE4m3K32.step acc (List.ofFn a) (List.ofFn b)

/-- A conformance vector: the accumulator word, the packed codes `a` and `b`, and the word `verity.ml.tc` computes. -/
structure Vec where
  acc : Nat
  a : Nat
  b : Nat
  d : Nat

/-- A vector holds for `step`. -/
def Vec.holds (step : Nat → (Fin 32 → Code) → (Fin 32 → Code) → Nat) (v : Vec) : Bool :=
  step v.acc (codes v.a) (codes v.b) == v.d

end Pouw.Fp8Atom
