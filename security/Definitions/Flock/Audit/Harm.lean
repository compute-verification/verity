module

public import Definitions.Flock.Audit.Law
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-!
# Harm bounds

A consumer weights an incorrect proof block by what it costs it: its output bits for exfiltration, its work for compute. `IsHarmBound
L h δ H` is the specification of core's `verity.core.protocols.profile.Stratified.harm_bound`, "an upper bound on
`sum_{u in B} harm(u)` over the incorrect sets `B` with `escape >= delta`". The Python function computes one by a greedy
fill in floating point; this file states what it must return and does not prove that implementation. `Audit/HarmBound.lean`
proves it for the stratified sampling design at closed-form bounds: the greedy fill's value, over the risk `k_s c / n_s` or the exact
risk, at the price where the fill stops.
-/

@[expose] public section

namespace FlockSoundness.Audit

open Finset
open scoped ENNReal
variable {n : ℕ}

/-- **`harm_bound`'s specification**: every incorrect set that escapes with probability at least `δ` has harm at most
`H`. -/
def IsHarmBound (L : Law n) (h : Fin n → ℝ) (δ : ℝ≥0∞) (H : ℝ) : Prop :=
  ∀ B : Finset (Fin n), δ ≤ L.escape B → ∑ u ∈ B, h u ≤ H

end FlockSoundness.Audit
