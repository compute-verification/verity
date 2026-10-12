import Definitions.Accounting.Physical

/-!
# The physical assumptions' first consequences

Causality at any latency makes wires run forward in time inside every unit, the form the split needs.
-/

namespace Accounting.SecurityProofs.Actual

open Verity

variable {L : Type*} (A : Placed L)

theorem forward_of_causal {ℓ : L → L → ℕ} (h : A.Causal ℓ) (X : Finset L) : A.Forward X :=
  fun g a ha _ _ _ => (Nat.le_add_right _ _).trans (h g a ha)

theorem causal_of_feasible {H : Hardware L} {ann : Annotation A} (h : Feasible H A ann) : A.Causal H.ℓ := h.1

end Accounting.SecurityProofs.Actual
