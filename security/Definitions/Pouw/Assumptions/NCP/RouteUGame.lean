module

public import Definitions.Pouw.Assumptions.NCP.Assumptions
public import Definitions.Pouw.Assumptions.NCP.RouteUAssumptions
public import Definitions.Pouw.Game.Params
public import Definitions.Pouw.NCP.Witness

/-! NCP's game instance on route U, with the lead's satisfiability witness `LeadUWitness`. -/

@[expose] public section

namespace Pouw.NCP.RouteU

open Finset

/-- **The hypotheses of `GapFromTTNCP_U` hold jointly at the lead** (`GamePinned.LeadWitness` on route U). -/
def LeadUWitness : Prop :=
  ∀ (β : ℤ) (f : ℕ), ∃ (P : Params Unit Unit Salt) (Lay : Layout), P.IsLead ∧ P.RSigned ∧
    (∀ (sh : Shape) (b : List Unit), Shape.InDefaultDomain sh → Unsigned sh.k (P.expWt sh b).2) ∧ P.QueryBound 0 ∧
    (∀ (L : Layout) (u : ℕ), P.Wref L u = wrefU f L u) ∧
    (P.toProtocolU β).Complete leadDomain ∧
    AdmitsRef (outputPriced3 Unit Unit Salt) (P.toProtocolU β) leadDomain (fun _ => 0) ∧
    (∀ g₀ : ℝ, 0 ≤ g₀ → g₀ < 1 →
      Assumptions.TTNCP_U β (outputPriced3 Unit Unit Salt) P leadDomain g₀ (fun _ _ => 0)) ∧
    (∀ U : Workload, U.InDomain leadDomain → ∀ u < U.layout.N, 2 ^ 10 ≤ (U.layout.shape u).n) ∧
    (⟨Lay, fun _ => 0⟩ : Workload).InDomain leadDomain ∧ 0 < Lay.N ∧ ∀ u < Lay.N, P.d ∣ (Lay.shape u).k

end Pouw.NCP.RouteU
