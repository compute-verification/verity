module

public import Definitions.Pouw.Assumptions.GameStatements
public import Definitions.Pouw.Basic.Prob
public import Definitions.Pouw.Game.Defs
public import Definitions.Pouw.Game.Params
public import Definitions.Pouw.Assumptions.Game

/-!
# The game's guarantees (trusted)

Each guarantee is a `def G : Prop`, stated over `Definitions.Pouw`'s definitions with the named assumptions it rests on as
hypotheses (`Pouw.Assumptions`). Its proof is the theorem `Pouw.SecurityProofs.G : Pouw.Guarantees.G`. (b), (c) and (d)
of `Definitions.Pouw.Assumptions.GameStatements`: Theorem 1, work-weighted sampling, and TT's resource exhaustion end to end. Each keeps the
text, the docstring and the `open`s of the statement it was in its source module.
-/

@[expose] public section

namespace Pouw.Guarantees

open Pouw.GameStatements Finset

/-- **(d) End to end: TT gives resource exhaustion.** Under TT(g₀) and the `Ω_max` bound, for every in-domain
workload, budget, oracle bound and adversary, the audit with `t` work-weighted samples satisfies
`Pr[accept ∧ T < (1 − g)(1 − ε_s)·W ∧ the run doesn't break SHA-512] ≤ (1 − ε_s)^t + δ(q, N)` with
`g = 1 − (1 − g₀)/Ω_max`. -/
def EndToEnd : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (g₀ Ωmax : ℝ) (δ : ℕ → ℕ → ℝ),
    g₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D g₀ δ →
    ∀ U : Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary CM, A.Bounded U T q → ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
          AuditAccepts (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - reductionGap g₀ Ωmax) * (1 - εs) * totalWref P U.layout ∧ ¬ A.Breaks U x.1 x.2.1)
        ≤ (1 - εs) ^ t + δ q U.layout.N

end Pouw.Guarantees

namespace Pouw.Proofs
open Finset

/-- **(a)** `G_g` from TT, with `g = 1 − (1 − g₀)/Ω_max` and `δ = ε_TT(q)`. -/
def _root_.Pouw.Guarantees.GapFromTT : Prop :=
  GameStatements.GapFromTT

end Pouw.Proofs
