import Proofs.Flock.Soundness.Audit.Law

/-!
# Harm bounds

A consumer weights a wrong unit by what it costs it: its output bits for exfiltration, its work for compute. `IsHarmBound
L h δ H` is the specification of core's `verity.proofs.profile.Stratified.harm_bound`, "an upper bound on
`sum_{u in B} harm(u)` over the wrong-unit sets `B` with `escape >= delta`". The Python function computes one by a greedy
fill in floating point; this file states what it must return and does not prove that implementation.
-/

namespace FlockSoundness.Audit

open Finset
open scoped ENNReal

variable {n : ℕ}

/-- **`harm_bound`'s specification**: every wrong-unit set that escapes with probability at least `δ` has harm at most
`H`. -/
def IsHarmBound (L : Law n) (h : Fin n → ℝ) (δ : ℝ≥0∞) (H : ℝ) : Prop :=
  ∀ B : Finset (Fin n), δ ≤ L.escape B → ∑ u ∈ B, h u ≤ H

/-- A set whose harm exceeds a harm bound escapes with probability below `δ`. -/
theorem escape_lt_of_harm_gt {L : Law n} {h : Fin n → ℝ} {δ : ℝ≥0∞} {H : ℝ} (hH : IsHarmBound L h δ H)
    {B : Finset (Fin n)} (hB : H < ∑ u ∈ B, h u) : L.escape B < δ :=
  lt_of_not_ge fun h' => absurd (hH B h') (not_le.2 hB)

end FlockSoundness.Audit
