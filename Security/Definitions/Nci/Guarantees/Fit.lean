import Definitions.Nci.Assumptions.Fit

/-!
# NCI over windows: the fit bound

A window that reads at most `S` bits uses at most `S / b_w` weights under F1, so when `S < N* b_w` no window holds a
model that reaches the target loss. Each thread's work then spans several windows, and each window it enters other than
its start reads `w_e` bits of it under F4. Counting those entries against what the windows read bounds the counted
multiplications with no `(d, e)`-threading at window scale. At `S ≥ N* b_w` it says nothing. `FitBoundSlots` is the same
bound with weights counted in parameter slots, so that it applies to windows that span many optimizer steps; `FitBound`
is its case of one slot per weight gate.
-/

namespace Nci.Guarantees

open Verity

/-- **The fit bound**: for a division `P` of the reference circuit `c` into `k` windows that each read at most
`S < N* b_w` bits of novel information, under F3 (`Assumptions.Reach`) and F1 with F4 (`Assumptions.Fits`), the counted
multiplications number at most `k · reuse · N* S² / (4 w_e (N* b_w − S))`, written multiplied out. At `reuse = 2`
(forward and backward) it reads `#windows · 2 N* S² / (4 w_e (N* b_w − S))`. -/
def FitBound : Prop :=
  ∀ (c : Registered) (T : Threading c) (k : ℕ) (P : Partition c.D k) (start : Fin T.m → Fin k)
    (Nstar : ℕ) (S b_w w_e : ℝ),
    S < Nstar * b_w → (∀ j, (novelReads c (P.gates j) : ℝ) ≤ S) →
    Assumptions.Reach c T Nstar → Assumptions.Fits c T P start b_w w_e →
    4 * w_e * (Nstar * b_w - S) * T.M.card ≤ k * T.reuse * Nstar * S ^ 2

/-- **The fit bound over parameter slots**: `FitBound` with its premises counted in the slots of a slot map `σ`, F3 as
`Assumptions.ReachSlots` and F1 with F4 as `Assumptions.FitsSlots`, so that a window spanning several optimizer steps is
charged once for each parameter it carries. The bound is `FitBound`'s, `k · reuse · N* S² / (4 w_e (N* b_w − S))`,
written multiplied out. With one slot per weight gate it is `FitBound`. -/
def FitBoundSlots : Prop :=
  ∀ (c : Registered) (T : Threading c) (σ : T.Slots) (k : ℕ) (P : Partition c.D k) (start : Fin T.m → Fin k)
    (Nstar : ℕ) (S b_w w_e : ℝ),
    S < Nstar * b_w → (∀ j, (novelReads c (P.gates j) : ℝ) ≤ S) →
    Assumptions.ReachSlots c T σ Nstar → Assumptions.FitsSlots c T σ P start b_w w_e →
    4 * w_e * (Nstar * b_w - S) * T.M.card ≤ k * T.reuse * Nstar * S ^ 2

end Nci.Guarantees
