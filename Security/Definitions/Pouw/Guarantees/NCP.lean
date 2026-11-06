module

public import Definitions.Pouw.Assumptions.NCP.Assumptions
public import Definitions.Pouw.Assumptions.NCP.RouteUAssumptions
public import Definitions.Pouw.Assumptions.NCP.RouteUGame
public import Definitions.Pouw.Assumptions.NCP.RouteUWord
public import Definitions.Pouw.Game.Params
public import Definitions.Pouw.NCP.Pinned
public import Definitions.Pouw.NCP.Protocol
public import Definitions.Pouw.NCP.Relation
public import Definitions.Pouw.NCP.RouteU
public import Definitions.Pouw.NCP.Witness

/-!
# NCP's guarantees (trusted)

Each guarantee is a `def G : Prop`, stated over `Definitions.Pouw`'s definitions with the named assumptions it rests on as
hypotheses (`Pouw.Assumptions`). Its proof is the theorem `Pouw.SecurityProofs.NCP.G : Pouw.Guarantees.NCP.G`. The
lead's algebra (`Definitions.Pouw.NCP.Pinned`), the shift by 24 under the rejection rule (`Definitions.Pouw.NCP.Protocol`),
the rejection rule's witness (`Definitions.Pouw.NCP.Relation`), route U's checked values (`Definitions.Pouw.NCP.RouteU`), its
game instance (`Pouw.Assumptions.NCP.RouteUGame`) and its word layer (`Pouw.Assumptions.NCP.RouteUWord`). Each keeps
the text, the docstring and the `open`s of the statement it was in its source module. `ShiftBy24Admissible` is the type
of the theorem `Pouw.NCP.GameProofs.shiftBy24_admissible`, its binders written as a `∀`, in that theorem's `open`s but
`Pouw.Proofs`, which only proof modules declare.

`Pouw.NCP.RouteU.GammaFromTTNCP_U`, which `GammaFromTTNCP_U_v1` and `GammaFromTTNCP_U_C2` instantiate, keeps its name
and namespace and comes just before them, as in its source: `GammaFromTTNCP_U_C2` reuses the auxiliary proof Lean makes
for its `3 : ℝ` (`Pouw.NCP.RouteU.GammaFromTTNCP_U._proof_1`), and the audit reads that proof by name.
-/

@[expose] public section

namespace Pouw.Guarantees.NCP

/-! ## The lead's algebra (`Definitions.Pouw.NCP.Pinned`) -/

section

open Pouw.NCP Pouw.NCP.Pinned

end

/-! ## The shift by 24 under the rejection rule (`Definitions.Pouw.NCP.Protocol`) -/

section

open Pouw.NCP Pouw.NCP.GameProofs Finset QTree

end

/-! ## The rejection rule's witness (`Definitions.Pouw.NCP.Relation`) -/

section

open Pouw.NCP Pouw.NCP.Relation Finset

end

/-! ## Route U's checked values (`Definitions.Pouw.NCP.RouteU`) -/

section

open Pouw.NCP Pouw.NCP.RouteU Finset

end

/-! ## Route U's game instance (`Pouw.Assumptions.NCP.RouteUGame`) -/

section

open Pouw.NCP Pouw.NCP.RouteU Finset

end

end Pouw.Guarantees.NCP

namespace Pouw.NCP.RouteU

open Finset

/-- **`G_γ` from TT_NCP_U** (`GamePinned.GammaFromTTNCP` on route U): for every bias `β`, count `f` of forming
units per position, cost model, NCP instance whose `W_ref` is route U's, `wrefU f`, and domain on which every unit
has at least `n₀ > 0` columns and `γ₀ ≥ P.d/(3k)`, TT_NCP_U(γ₀) at `β` with error `ε` gives `G_γ` for route U's protocol at `β` with
`Ω_max = 1 + f/(3n₀)`, so `γ = 1 − (1 − γ₀)/(1 + f/(3n₀))`, and `η = ε`. -/
def GammaFromTTNCP_U : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Params Q R S) (β : ℤ) (f : ℕ) (D : Layout → Prop) (γ₀ : ℝ) (n₀ : ℕ)
    (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < n₀ → (∀ (L : Layout) (u : ℕ), P.Wref L u = wrefU f L u) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, n₀ ≤ (U.layout.shape u).n) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, (P.d : ℝ) ≤ γ₀ * (3 * (U.layout.shape u).k)) →
    Assumptions.TTNCP_U β CM P D γ₀ ε →
    Gγ CM (P.toProtocolU β) D (reductionGamma γ₀ (1 + f / (3 * n₀))) ε

end Pouw.NCP.RouteU

namespace Pouw.Guarantees.NCP

section

open Pouw.NCP Pouw.NCP.RouteU Finset

/-- **`G_γ` from TT_NCP_U at ncp-v1** (`GammaFromTTNCP_U` at `β = 128`, `f = 24`): `Ω_max = 1 + 8/n₀`, so
`γ = 1 − (1 − γ₀)/(1 + 8/n₀)`. -/
def GammaFromTTNCP_U_v1 : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Params Q R S) (D : Layout → Prop) (γ₀ : ℝ) (n₀ : ℕ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < n₀ → (∀ (L : Layout) (u : ℕ), P.Wref L u = wrefU 24 L u) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, n₀ ≤ (U.layout.shape u).n) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, (P.d : ℝ) ≤ γ₀ * (3 * (U.layout.shape u).k)) →
    Assumptions.TTNCP_U 128 CM P D γ₀ ε →
    Gγ CM (P.toProtocolU 128) D (reductionGamma γ₀ (1 + 8 / n₀)) ε

/-- **`G_γ` from TT_NCP_U at C2** (`GammaFromTTNCP_U` at `β = 129`, `f = 20`): `Ω_max = 1 + 20/(3n₀)`, so
`γ = 1 − (1 − γ₀)/(1 + 20/(3n₀))`. -/
def GammaFromTTNCP_U_C2 : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Params Q R S) (D : Layout → Prop) (γ₀ : ℝ) (n₀ : ℕ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < n₀ → (∀ (L : Layout) (u : ℕ), P.Wref L u = wrefU 20 L u) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, n₀ ≤ (U.layout.shape u).n) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, (P.d : ℝ) ≤ γ₀ * (3 * (U.layout.shape u).k)) →
    Assumptions.TTNCP_U 129 CM P D γ₀ ε →
    Gγ CM (P.toProtocolU 129) D (reductionGamma γ₀ (1 + 20 / (3 * n₀))) ε

end

/-! ## Route U's word layer (`Pouw.Assumptions.NCP.RouteUWord`) -/

section

open Pouw.NCP Pouw.NCP.RouteU Finset Pouw.NCP.Relation

end

end Pouw.Guarantees.NCP
