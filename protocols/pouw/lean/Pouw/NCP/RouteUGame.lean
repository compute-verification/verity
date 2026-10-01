import Pouw.NCP.RouteUAssumptions
import Pouw.NCP.Witness
import Pouw.NCP.Assumptions
import Pouw.Game.Params

/-!
# NCP's game instance on route U: pinned statements (trusted)

`G_γ` from `TT_NCP_U` for route U's protocol at bias `β` and its honest reference `wrefU f = 3·m·k·n + f·m·k`,
whose ratio to the credited `3·m·k·n` is `1 + f/(3n)`, its two instances (ncp-v1: `β = 128`, `f = 24`, `1 + 8/n`;
C2: `β = 129`, `f = 20`, `1 + 20/(3n)`), and the satisfiability witnesses for its hypotheses.
-/

namespace Pouw.NCP.RouteU

open Finset

/-- **`TT_NCP_U` is `TT_NCP` of route U's protocol**, by definition. -/
def TTNCPUIff : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Params Q R S) (β : ℤ) (D : Layout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ),
    Assumptions.TTNCP_U β CM P D γ₀ ε ↔ Assumptions.TTNCP CM (P.toProtocolU β) D γ₀ ε

/-- **`G_γ` from TT_NCP_U** (`GamePinned.GammaFromTTNCP` on route U): for every bias `β`, count `f` of forming
units per position, cost model, NCP instance whose `W_ref` is route U's, `wrefU f`, and domain on which every unit
has at least `n₀ > 0` columns, TT_NCP_U(γ₀) at `β` with error `ε` gives `G_γ` for route U's protocol at `β` with
`Ω_max = 1 + f/(3n₀)`, so `γ = 1 − (1 − γ₀)/(1 + f/(3n₀))`, and `η = ε`. -/
def GammaFromTTNCP_U : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Params Q R S) (β : ℤ) (f : ℕ) (D : Layout → Prop) (γ₀ : ℝ) (n₀ : ℕ)
    (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < n₀ → (∀ (L : Layout) (u : ℕ), P.Wref L u = wrefU f L u) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, n₀ ≤ (U.layout.shape u).n) →
    Assumptions.TTNCP_U β CM P D γ₀ ε →
    Gγ CM (P.toProtocolU β) D (reductionGamma γ₀ (1 + f / (3 * n₀))) ε

/-- **`G_γ` from TT_NCP_U at ncp-v1** (`GammaFromTTNCP_U` at `β = 128`, `f = 24`): `Ω_max = 1 + 8/n₀`, so
`γ = 1 − (1 − γ₀)/(1 + 8/n₀)`. -/
def GammaFromTTNCP_U_v1 : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Params Q R S) (D : Layout → Prop) (γ₀ : ℝ) (n₀ : ℕ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < n₀ → (∀ (L : Layout) (u : ℕ), P.Wref L u = wrefU 24 L u) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, n₀ ≤ (U.layout.shape u).n) →
    Assumptions.TTNCP_U 128 CM P D γ₀ ε →
    Gγ CM (P.toProtocolU 128) D (reductionGamma γ₀ (1 + 8 / n₀)) ε

/-- **`G_γ` from TT_NCP_U at C2** (`GammaFromTTNCP_U` at `β = 129`, `f = 20`): `Ω_max = 1 + 20/(3n₀)`, so
`γ = 1 − (1 − γ₀)/(1 + 20/(3n₀))`. -/
def GammaFromTTNCP_U_C2 : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Params Q R S) (D : Layout → Prop) (γ₀ : ℝ) (n₀ : ℕ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < n₀ → (∀ (L : Layout) (u : ℕ), P.Wref L u = wrefU 20 L u) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, n₀ ≤ (U.layout.shape u).n) →
    Assumptions.TTNCP_U 129 CM P D γ₀ ε →
    Gγ CM (P.toProtocolU 129) D (reductionGamma γ₀ (1 + 20 / (3 * n₀))) ε

/-- **Satisfiability witness for TT_NCP_U** at every `γ₀ ∈ [0, 1)` (`GamePinned.TTNCPWitness` on route U): for every
bias `β` and NCP instance with at most `c` queries per unit and `W_ref ≥ 3·W_mm`, in the output-priced model at
`3·m·k·n` TT_NCP_U(γ₀) holds with error 0, and one program runs route U's `H_ref` at cost `≤ W_ref` with `N·c`
oracle calls. -/
def TTNCPUWitness : Prop :=
  ∀ (Q R : Type) [Fintype Q] [DecidableEq Q] [Fintype R] (P : Params Q R Salt) (β : ℤ) (c : ℕ) (γ₀ : ℝ),
    0 ≤ γ₀ → γ₀ < 1 → P.QueryBound c →
    (∀ U : Workload, U.InDomain leadDomain → ∀ u < U.layout.N, 3 * (U.layout.shape u).Wmm ≤ P.Wref U.layout u) →
    Assumptions.TTNCP_U β (outputPriced3 Q R Salt) P leadDomain γ₀ (fun _ _ => 0) ∧
    AdmitsRef (outputPriced3 Q R Salt) (P.toProtocolU β) leadDomain (fun U => U.layout.N * c)

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
