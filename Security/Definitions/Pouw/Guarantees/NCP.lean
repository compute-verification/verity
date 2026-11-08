module

public import Definitions.Pouw.NCP.Pinned
public import Definitions.Pouw.NCP.Protocol
public import Definitions.Pouw.NCP.Relation
public import Definitions.Pouw.NCP.RouteU
public import Definitions.Pouw.Assumptions.NCP.RouteUGame
public import Definitions.Pouw.Assumptions.NCP.RouteUWord

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

`Pouw.NCP.RouteU.GapFromTTNCP_U`, which `GapFromTTNCP_U_v1` and `GapFromTTNCP_U_C2` instantiate, keeps its name
and namespace and comes just before them, as in its source: `GapFromTTNCP_U_C2` reuses the auxiliary proof Lean makes
for its `3 : ℝ` (`Pouw.NCP.RouteU.GapFromTTNCP_U._proof_1`), and the audit reads that proof by name.
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

/-- **`G_g` from TT_NCP_U** (`GamePinned.GapFromTTNCP` on route U): for every bias `β`, count `f` of forming
units per position, cost model, NCP instance whose `W_ref` is route U's, `wrefU f`, and domain on which every unit
has at least `n₀ > 0` columns and `g₀ ≥ P.d/(3k)`, TT_NCP_U(g₀) at `β` with error `δ` gives `G_g` for route U's protocol at `β` with
`Ω_max = 1 + f/(3n₀)`, so `g = 1 − (1 − g₀)/(1 + f/(3n₀))`, and the same failure probability `δ`. -/
def GapFromTTNCP_U : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Params Q R S) (β : ℤ) (f : ℕ) (D : Layout → Prop) (g₀ : ℝ) (n₀ : ℕ)
    (δ : ℕ → ℕ → ℝ),
    g₀ < 1 → 0 < n₀ → (∀ (L : Layout) (u : ℕ), P.Wref L u = wrefU f L u) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, n₀ ≤ (U.layout.shape u).n) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, (P.d : ℝ) ≤ g₀ * (3 * (U.layout.shape u).k)) →
    Assumptions.TTNCP_U β CM P D g₀ δ →
    GGap CM (P.toProtocolU β) D (reductionGap g₀ (1 + f / (3 * n₀))) δ

end Pouw.NCP.RouteU

namespace Pouw.Guarantees.NCP

section

open Pouw.NCP Pouw.NCP.RouteU Finset

/-- **`G_g` from TT_NCP_U at ncp-v1** (`GapFromTTNCP_U` at `β = 128`, `f = 24`): `Ω_max = 1 + 8/n₀`, so
`g = 1 − (1 − g₀)/(1 + 8/n₀)`. -/
def GapFromTTNCP_U_v1 : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Params Q R S) (D : Layout → Prop) (g₀ : ℝ) (n₀ : ℕ) (δ : ℕ → ℕ → ℝ),
    g₀ < 1 → 0 < n₀ → (∀ (L : Layout) (u : ℕ), P.Wref L u = wrefU 24 L u) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, n₀ ≤ (U.layout.shape u).n) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, (P.d : ℝ) ≤ g₀ * (3 * (U.layout.shape u).k)) →
    Assumptions.TTNCP_U 128 CM P D g₀ δ →
    GGap CM (P.toProtocolU 128) D (reductionGap g₀ (1 + 8 / n₀)) δ

/-- **`G_g` from TT_NCP_U at C2** (`GapFromTTNCP_U` at `β = 129`, `f = 20`): `Ω_max = 1 + 20/(3n₀)`, so
`g = 1 − (1 − g₀)/(1 + 20/(3n₀))`. -/
def GapFromTTNCP_U_C2 : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Params Q R S) (D : Layout → Prop) (g₀ : ℝ) (n₀ : ℕ) (δ : ℕ → ℕ → ℝ),
    g₀ < 1 → 0 < n₀ → (∀ (L : Layout) (u : ℕ), P.Wref L u = wrefU 20 L u) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, n₀ ≤ (U.layout.shape u).n) →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N, (P.d : ℝ) ≤ g₀ * (3 * (U.layout.shape u).k)) →
    Assumptions.TTNCP_U 129 CM P D g₀ δ →
    GGap CM (P.toProtocolU 129) D (reductionGap g₀ (1 + 20 / (3 * n₀))) δ

end

/-! ## Route U's word layer (`Pouw.Assumptions.NCP.RouteUWord`) -/

section

open Pouw.NCP Pouw.NCP.RouteU Finset Pouw.NCP.Relation

end

end Pouw.Guarantees.NCP
