import Definitions.Flock.Audit.Harm
import Proofs.Flock.Soundness.Audit.Law

/-!
# Harm bounds

A consumer weights a wrong unit by what it costs it: its output bits for exfiltration, its work for compute. `IsHarmBound
L h δ H` is the specification of core's `verity.core.protocols.profile.Stratified.harm_bound`, "an upper bound on
`sum_{u in B} harm(u)` over the wrong-unit sets `B` with `escape >= delta`". The Python function computes one by a greedy
fill in floating point; this file states what it must return and does not prove that implementation. `Audit/HarmBound.lean`
proves it for the stratified law at closed-form bounds: the greedy fill's value, over the risk `k_s c / n_s` or the exact
risk, at the price where the fill stops.
-/

namespace FlockSoundness.Audit

open Finset
open scoped ENNReal
variable {n : ℕ}

/-- A set whose harm exceeds a harm bound escapes with probability below `δ`. -/
theorem escape_lt_of_harm_gt {L : Law n} {h : Fin n → ℝ} {δ : ℝ≥0∞} {H : ℝ} (hH : IsHarmBound L h δ H)
    {B : Finset (Fin n)} (hB : H < ∑ u ∈ B, h u) : L.escape B < δ :=
  lt_of_not_ge fun h' => absurd (hH B h') (not_le.2 hB)

end FlockSoundness.Audit
