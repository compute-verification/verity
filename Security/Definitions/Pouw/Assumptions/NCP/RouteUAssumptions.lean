import Definitions.Pouw.NCP.RouteU

/-!
# NCP's named assumption on route U (trusted)

A `Prop` taken as a hypothesis by the statements that use it, never an `axiom`. It is `TT_NCP`
(`Pouw.NCP.Assumptions.TTNCP`) restated for NCP's checked values on route U, `Params.toProtocolU`, whose checked
values are route U's int32 accumulator at bias `β` (`mmWU β`, from `c₀ = −β·Σ Y`) instead of the unbiased one; the
named assumption is its instance at ncp-v1's `β = 128`. Neither variant
reduces to the other in W1 cost: converting one checked word into the other costs one priced add.
-/

namespace Pouw.NCP.Assumptions

open Finset

variable {Q R S : Type}

/-- **Assumption TT_NCP_U(γ₀), tight stacked transcript on route U**, for a bias `β`, a cost model `CM`, NCP's
parameters `P` (the named assumption is its instance at `β = 128`, `CM = M_4090` and `P` with `IsLead`), a domain
`D`, a target `γ₀` and an error `ε(q, N)`: `TTNCP` of route U's protocol `P.toProtocolU β` at `P.d`, with every
checked-correct unit credited `3·m·k·n`, the work of the stacked product. Like `TTNCP`, it is claimed only on a domain
where `γ₀ ≥ P.d/(3k)` on every unit. -/
def TTNCP_U (β : ℤ) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Params Q R S)
    (D : UnitLayout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ) : Prop :=
  (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, (P.d : ℝ) ≤ γ₀ * (3 * (U.layout.shape u).k)) →
  CM.Queries ∧
  ∀ U : Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T / (1 - γ₀) < ∑ u ∈ checkedSet (P.toProtocolU β) U x.1 x.2 (A.transcript U x.1 x.2),
        3 * ((U.layout.shape u).Wmm : ℝ))
      ≤ ε q U.layout.N

end Pouw.NCP.Assumptions
