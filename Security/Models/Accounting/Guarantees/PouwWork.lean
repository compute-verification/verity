import Models.Accounting.Assumptions.PouwWork

/-!
# PoUW's work lower bound on an isolated node (trusted)

The converged model's `pouw_local_off` at help 0 (CMP-12 v3.1), on the actual circuit: an accepted run's node did the
work itself, in credits, except with `δ_det + δ_luck + δ_skip`. It rests on A2 (`PouwDetection`), A2b (`PouwLuck`), A3
(`PouwFloorOff`) and A5 at `H = 0` (`PouwHelpBound`). It reads no capacity, clock or resource: all of PoUW's GPU
content is in A3.

Proved by `Accounting.SecurityProofs.PouwLocalOff : Accounting.Guarantees.PouwLocalOff` (`Proofs.Accounting.PouwWork`).
-/

namespace Accounting.Guarantees

open Accounting.PouwWork

/-- **PoUW's work lower bound on an isolated node.** For any strategy `S` on the verifier's coins, any node `X`, credit
pricing `π` and marking `m`: if the verifier accepts with fewer than `Nmin` right blocks with probability at most
`δ_det` (A2), with `Nmin` right at the coin but not at every coin with probability at most `δ_luck` (A2b), the work
floor holds at `m` off an event of probability at most `δ_skip` (A3), and the node takes in no help on any accepted
coin (A5 at 0), then the verifier accepts while the node's own gates outside `m` made fewer than `X = Wb · Nmin`
credits in the window with probability at most `δ_det + δ_luck + δ_skip`. It says the node computed; not which of its
GPUs did. -/
def PouwLocalOff : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (L : Type) (S : Ω → Actual.Run L) (P : Verdict Ω L) (X : Finset L)
    (π : ∀ r, Fin (S r).A.N → ℕ) (m : ∀ r, Finset (Fin (S r).A.N)) (Wb Nmin ρ : ℕ) (δdet δluck δskip : ℝ),
    Assumptions.PouwDetection S P Nmin δdet → Assumptions.PouwLuck S P Nmin δluck →
    Assumptions.PouwFloorOff S P X π m Wb Nmin ρ δskip → Assumptions.PouwHelpBound S P X 0 →
    Pouw.pr (fun r => Accepts S P r ∧ credits S P X π m r < Wb * Nmin) ≤ δdet + δluck + δskip

end Accounting.Guarantees
