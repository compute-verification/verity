module

public import Definitions.Slowdown.Assumptions.Fit

/-!
# The inference-only policy over windows: the fit bound

A window that reads at most `S` bits uses at most `S / b_w` weights under F1, so when `S < N* b_w` no window holds a
model that reaches the target loss. Each thread's work then spans several windows, and each window it enters other than
its start reads `w_e` bits of it under F4. Counting those entries against what the windows read bounds the counted
multiplications with no `(d, e)`-threading at window scale. At `S ≥ N* b_w` it says nothing. `FitBoundSlots` states it
with weights counted in the parameter slots of a slot map, so that it applies to windows that span many optimizer
steps; at one slot per weight gate its premises are `Assumptions.Reach` and `Assumptions.Fits`.
-/

@[expose] public section

open InferenceOnly

namespace Slowdown.Guarantees

open Verity

/-- **The fit bound over parameter slots**: for a division `P` of the reference training algorithm `c` into `k` windows
that each read at most `S < N* b_w` fresh input bits, with the premises counted in the slots of a slot map `σ`, F3 as
`Assumptions.ReachSlots` and F1 with F4 as `Assumptions.FitsSlots`, so that a window spanning several optimizer steps is
charged once for each parameter it carries, the counted multiplications number at most
`k · reuse · N* S² / (4 w_e (N* b_w − S))`, written multiplied out. At `reuse = 2` (forward and backward) it reads
`#windows · 2 N* S² / (4 w_e (N* b_w − S))`. With one slot per weight gate the premises are `Assumptions.Reach` and
`Assumptions.Fits`. -/
def FitBoundSlots : Prop :=
  ∀ (c : RegisteredCircuit) (T : Threading c) (σ : T.Slots) (k : ℕ) (P : Partition c.D k) (start : Fin T.m → Fin k)
    (Nstar : ℕ) (S b_w w_e : ℝ),
    S < Nstar * b_w → (∀ j, (freshInput c (P.gates j) : ℝ) ≤ S) →
    Assumptions.ReachSlots c T σ Nstar → Assumptions.FitsSlots c T σ P start b_w w_e →
    4 * w_e * (Nstar * b_w - S) * T.M.card ≤ k * T.reuse * Nstar * S ^ 2

end Slowdown.Guarantees
