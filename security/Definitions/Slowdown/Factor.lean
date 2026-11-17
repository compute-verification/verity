module

public import Definitions.InferenceOnly

/-! The slowdown factor of a policy on a training circuit. -/

@[expose] public section

open InferenceOnly

namespace Slowdown

open Verity
open scoped ENNReal

/-- `⊤` when no declaration implements `t` and complies with `policy`. -/
noncomputable def slowdown (policy : Policy) (t : RegisteredCircuit) : ℝ≥0∞ :=
  ⨅ (d : RegisteredCircuit) (_ : t.Implements d ∧ policy d), (d.D.size d.D.computed : ℝ≥0∞) / t.D.size t.D.computed

end Slowdown
