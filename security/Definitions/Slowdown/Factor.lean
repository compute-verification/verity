module

public import Definitions.InferenceOnly

/-!
# The slowdown factor

`slowdown policy t` is the factor by which every declaration that complies with `policy` and implements `t` is larger
than `t`.
-/

@[expose] public section

open InferenceOnly

namespace Slowdown

open Verity
open scoped ENNReal

/-- **The slowdown factor** `ρ_policy(t)`: the least ratio of `size(d)` to `size(t)` over the declarations `d` that
implement `t` and comply with `policy`, and `⊤` when none does. -/
noncomputable def slowdown (policy : Policy) (t : RegisteredCircuit) : ℝ≥0∞ :=
  ⨅ (d : RegisteredCircuit) (_ : t.Implements d ∧ policy d), (d.D.size d.D.computed : ℝ≥0∞) / t.D.size t.D.computed

end Slowdown
