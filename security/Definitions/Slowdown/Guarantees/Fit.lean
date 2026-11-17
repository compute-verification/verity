module

public import Definitions.Slowdown.Assumptions.Fit

/-! The fit bound over windows, which says nothing at `S ≥ N* b_w`. -/

@[expose] public section

open InferenceOnly

namespace Slowdown.Guarantees

open Verity

/-- **The fit bound**: for a division `P` of the reference training algorithm `c` into `k` windows that each read at most
`S < N* b_w` fresh input bits, under F3 (`Assumptions.Reach`) and F1 with F4 (`Assumptions.Fits`), the counted
multiplications number at most `k · reuse · N* S² / (4 w_e (N* b_w − S))`, written multiplied out. At `reuse = 2`
(forward and backward) it reads `#windows · 2 N* S² / (4 w_e (N* b_w − S))`. -/
def FitBound : Prop :=
  ∀ (c : RegisteredCircuit) (T : Threading c) (k : ℕ) (P : Partition c.D k) (start : Fin T.m → Fin k)
    (Nstar : ℕ) (S b_w w_e : ℝ),
    S < Nstar * b_w → (∀ j, (freshInput c (P.gates j) : ℝ) ≤ S) →
    Assumptions.Reach c T Nstar → Assumptions.Fits c T P start b_w w_e →
    4 * w_e * (Nstar * b_w - S) * T.M.card ≤ k * T.reuse * Nstar * S ^ 2

/-- **The fit bound over parameter slots**: `FitBound` with its premises counted in the slots of a slot map `σ`, F3 as
`Assumptions.ReachSlots` and F1 with F4 as `Assumptions.FitsSlots`, so that a window spanning several optimizer steps is
charged once for each parameter it carries. The bound is `FitBound`'s, `k · reuse · N* S² / (4 w_e (N* b_w − S))`,
written multiplied out. With one slot per weight gate it is `FitBound`. -/
def FitBoundSlots : Prop :=
  ∀ (c : RegisteredCircuit) (T : Threading c) (σ : T.Slots) (k : ℕ) (P : Partition c.D k) (start : Fin T.m → Fin k)
    (Nstar : ℕ) (S b_w w_e : ℝ),
    S < Nstar * b_w → (∀ j, (freshInput c (P.gates j) : ℝ) ≤ S) →
    Assumptions.ReachSlots c T σ Nstar → Assumptions.FitsSlots c T σ P start b_w w_e →
    4 * w_e * (Nstar * b_w - S) * T.M.card ≤ k * T.reuse * Nstar * S ^ 2

end Slowdown.Guarantees
