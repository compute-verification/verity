module

public import Definitions.Core
public import Mathlib.Analysis.Real.Sqrt

/-! The inference-only policy (`Complies`) and the fresh input an inference block reads (`freshInput`). -/

@[expose] public section

namespace InferenceOnly

open Verity
open scoped ENNReal

/-- The fresh input that the gates `R` of `d`'s circuit read, in bits: the width of `in(R) \ θ`. -/
def freshInput (d : RegisteredCircuit) (R : Finset (Fin d.D.N)) : ℕ := d.D.width (d.D.into R \ d.θ)

/-- A policy: a condition on declarations. -/
abbrev Policy : Type := RegisteredCircuit → Prop

def Complies (c k : ℕ) : Policy := fun d =>
  ∃ (B : ℕ) (P : Partition d.D B), P.IsCK d c k

/-- **The policy at a rate**, as the verifier's check reads it on a circuit of any widths: `d`'s circuit divides into
inference blocks of at most `c` gates, each reading at most `α` fresh input bits per gate (`freshInput R ≤ α · size R`,
multiplied out so that a block with no gates needs no division). -/
def CompliesRate (c : ℕ) (α : ℝ) : Policy := fun d =>
  ∃ (n : ℕ) (P : Partition d.D n),
    ∀ u, d.D.size (P.gates u) ≤ c ∧ (freshInput d (P.gates u) : ℝ) ≤ α * d.D.size (P.gates u)

/-- **At most `n` `(c, k)`-inference blocks**: `d`'s operation gates divide into at most `n` proof blocks of a
`(c, k)`-partition (`Verity.Partition.IsCK`). -/
def CompliesAtMost (c k n : ℕ) : Policy := fun d =>
  ∃ B ≤ n, ∃ P : Partition d.D B, P.IsCK d c k

noncomputable def Reads (c : ℕ) (C : RegisteredCircuit) : ℕ :=
  sInf {r | ∃ (n : ℕ) (P : Partition C.D n), (∀ u, C.D.size (P.gates u) ≤ c) ∧ r = ∑ u, freshInput C (P.gates u)}

end InferenceOnly
