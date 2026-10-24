import Definitions.Pouw.Assumptions.NCP.RouteUAssumptions
import Definitions.Pouw.NCP.Witness
import Definitions.Pouw.Assumptions.NCP.Assumptions
import Definitions.Pouw.Game.Params

/-!
# NCP's game instance on route U: pinned statements (trusted)

`G_γ` from `TT_NCP_U` for route U's protocol at bias `β` and its honest reference `wrefU f = 3·m·k·n + f·m·k`,
whose ratio to the credited `3·m·k·n` is `1 + f/(3n)`, its two instances (ncp-v1: `β = 128`, `f = 24`, `1 + 8/n`;
C2: `β = 129`, `f = 20`, `1 + 20/(3n)`), and the satisfiability witnesses for its hypotheses. All but the lead's
witness (`LeadUWitness`) are in `Pouw.Guarantees.NCP`.
-/

namespace Pouw.NCP.RouteU

open Finset

/-- **The hypotheses of `GammaFromTTNCP_U` hold jointly at the lead** (`GamePinned.LeadWitness` on route U): for
every bias `β` and count `f`, an NCP instance with `IsLead`, permutation-matrix `R` on domain shapes, no oracle
queries and `W_ref = wrefU f`; route U's protocol at `β` is complete on `leadDomain`, the output-priced model runs
its `H_ref` with no oracle calls, TT_NCP_U(γ₀) holds there with error 0 at every `γ₀ ∈ [0, 1)`, every in-domain
unit has `n ≥ 2^10` (so `n₀ = 2^10`, `Ω_max = 1 + f/3072`), and a layout with at least one unit and `d ∣ k` on
every unit is in the domain. -/
def LeadUWitness : Prop :=
  ∀ (β : ℤ) (f : ℕ), ∃ (P : Params Unit Unit Salt) (Lay : Layout), P.IsLead ∧ P.RSigned ∧
    (∀ (sh : Shape) (b : List Unit), Shape.InDefaultDomain sh → Unsigned sh.k (P.expWt sh b).2) ∧ P.QueryBound 0 ∧
    (∀ (L : Layout) (u : ℕ), P.Wref L u = wrefU f L u) ∧
    (P.toProtocolU β).Complete leadDomain ∧
    AdmitsRef (outputPriced3 Unit Unit Salt) (P.toProtocolU β) leadDomain (fun _ => 0) ∧
    (∀ γ₀ : ℝ, 0 ≤ γ₀ → γ₀ < 1 →
      Assumptions.TTNCP_U β (outputPriced3 Unit Unit Salt) P leadDomain γ₀ (fun _ _ => 0)) ∧
    (∀ U : Workload, U.InDomain leadDomain → ∀ u < U.layout.N, 2 ^ 10 ≤ (U.layout.shape u).n) ∧
    (⟨Lay, fun _ => 0⟩ : Workload).InDomain leadDomain ∧ 0 < Lay.N ∧ ∀ u < Lay.N, P.d ∣ (Lay.shape u).k

end Pouw.NCP.RouteU
