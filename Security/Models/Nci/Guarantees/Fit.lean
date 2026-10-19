import Models.Nci.Assumptions.Fit

/-!
# NCI over windows: the fit bound

A window that reads at most `S` bits uses at most `S / b_w` weights under F1, so when `S < N* b_w` no window holds a
model that reaches the target loss. Each thread's work then spans several windows, and each window it enters other than
its start reads `w_e` bits of it under F4. Counting those entries against what the windows read bounds the counted
multiplications with no `(d, e)`-threading at window scale ("The fit bound", Notion, section 2). At `S ≥ N* b_w` it says
nothing.
-/

namespace Nci.Guarantees

open Verity

/-- **The fit bound** ("The fit bound", Notion, sections 1 and 9): for a division `P` of the reference circuit `c` into
`k` windows that each read at most `S < N* b_w` bits of novel information, under F3 (`Assumptions.Reach`) and F1 with F4
(`Assumptions.Fits`), the counted multiplications number at most `k · reuse · N* S² / (4 w_e (N* b_w − S))`, written
multiplied out. At `reuse = 2` (forward and backward) it is the page's `#windows · 2 N* S² / (4 w_e (N* b_w − S))`. -/
def FitBound : Prop :=
  ∀ (c : Declaration) (T : Threading c) (k : ℕ) (P : Partition c.D k) (start : Fin T.m → Fin k)
    (Nstar : ℕ) (S b_w w_e : ℝ),
    S < Nstar * b_w → (∀ j, (novelReads c (P.gates j) : ℝ) ≤ S) →
    Assumptions.Reach c T Nstar → Assumptions.Fits c T P start b_w w_e →
    4 * w_e * (Nstar * b_w - S) * T.M.card ≤ k * T.reuse * Nstar * S ^ 2

end Nci.Guarantees
