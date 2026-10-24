import Definitions.Pouw.Game.Defs

/-!
# NCP's named assumption (trusted)

A `Prop` taken as a hypothesis by the statements that use it, never an `axiom`.
-/

namespace Pouw.NCP.Assumptions

open Finset

variable {Q R S : Type}

/-- **Assumption TT_NCP(γ₀), tight stacked transcript** (red-team N2), for a cost model `CM`, a protocol `P` (the
named assumption is its instance at `CM = M_4090` and `P` an NCP instance with `IsLead`), a domain `D`, a target `γ₀`
and an error `ε(q, N)`: TT (`Pouw.Assumptions.TT`) with every checked-correct unit credited `3·m·k·n`, the work of the
stacked product.

Side condition: `γ₀ ≥ d/(3k)` on every unit. The final checked value is `A·B`, free at a degenerate input, and the
one-segment adversary makes the event sure below it (`Pouw.NCP.GameProofs.oneSegment`). -/
def TTNCP [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ) : Prop :=
  CM.Queries ∧
  ∀ U : Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T / (1 - γ₀) < ∑ u ∈ checkedSet P U x.1 x.2 (A.transcript U x.1 x.2), 3 * ((U.layout.shape u).Wmm : ℝ))
      ≤ ε q U.layout.N

end Pouw.NCP.Assumptions
