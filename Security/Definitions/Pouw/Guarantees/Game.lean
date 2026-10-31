import Definitions.Pouw.Assumptions.GameStatements
import Definitions.Pouw.Basic.Prob
import Definitions.Pouw.Game.Defs
import Definitions.Pouw.Game.Params
import Definitions.Pouw.Assumptions.Game

/-!
# The game's guarantees (trusted)

Each guarantee is a `def G : Prop`, stated over `Definitions.Pouw`'s definitions with the named assumptions it rests on as
hypotheses (`Pouw.Assumptions`). Its proof is the theorem `Pouw.SecurityProofs.G : Pouw.Guarantees.G`. (b), (c) and (d)
of `Definitions.Pouw.Assumptions.GameStatements`: Theorem 1, work-weighted sampling, and TT's resource exhaustion end to end. Each keeps the
text, the docstring and the `open`s of the statement it was in its source module.
-/

namespace Pouw.Guarantees

open Pouw.GameStatements Finset

/-- **(b) Theorem 1** (§3.5). If `G_γ` holds with error `η` and, for an adversary within `(T, q)`, the audit (any
coins `C` drawn independently of `H` and `s`, any acceptance predicate) satisfies
`Pr[accept ∧ ∑_{u ∉ K} W_ref(u) > ε_s·W] ≤ δ_s`, then `Pr[accept ∧ T < (1 − γ)(1 − ε_s)·W] ≤ δ_s + η`. -/
def Theorem1 : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ),
    γ < 1 → Gγ CM P D γ η →
    ∀ U : Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary CM, A.Bounded U T q →
    ∀ (C : Type) [Fintype C] (accept : (Q → R) → S → C → Prop) (εs δs : ℝ),
      pr (fun x : (Q → R) × S × C => accept x.1 x.2.1 x.2.2 ∧
          εs * totalWref P U.layout <
            ∑ u ∈ range U.layout.N \ correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1), (P.Wref U.layout u : ℝ))
        ≤ δs →
      pr (fun x : (Q → R) × S × C => accept x.1 x.2.1 x.2.2 ∧ T < (1 - γ) * (1 - εs) * totalWref P U.layout)
        ≤ δs + η q U.layout.N

/-- **(c) Work-weighted sampling.** `t` independent samples, each unit drawn with probability `∝ w(u)`; the audit
accepts iff every sample is in `K` (which may depend on the rest of the randomness `α`). Then
`Pr[accept ∧ ∑_{u ∉ K} w(u) > ε_s·∑_u w(u)] ≤ (1 − ε_s)^t` for `ε_s ≤ 1`. -/
def WorkWeightedSampling : Prop :=
  ∀ (α : Type) [Fintype α] (w : ℕ → ℕ) (N t : ℕ) (K : α → Finset ℕ) (εs : ℝ), εs ≤ 1 →
    pr (fun x : α × (Fin t → Ticket w N) => AuditAccepts (K x.1) x.2 ∧
        εs * ∑ u ∈ range N, (w u : ℝ) < ∑ u ∈ range N \ K x.1, (w u : ℝ)) ≤ (1 - εs) ^ t

/-- **(d) End to end: TT gives resource exhaustion.** Under TT(γ₀) and the `Ω_max` bound, for every in-domain
workload, budget, oracle bound and adversary, the audit with `t` work-weighted samples satisfies
`Pr[accept ∧ T < (1 − γ)(1 − ε_s)·W] ≤ (1 − ε_s)^t + ε(q, N)` with `γ = 1 − (1 − γ₀)/Ω_max`. -/
def EndToEnd : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D γ₀ ε →
    ∀ U : Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary CM, A.Bounded U T q → ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
          AuditAccepts (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - reductionGamma γ₀ Ωmax) * (1 - εs) * totalWref P U.layout)
        ≤ (1 - εs) ^ t + ε q U.layout.N

end Pouw.Guarantees

namespace Pouw.Proofs
open Finset

/-- **(a)** `G_γ` from TT, with `γ = 1 − (1 − γ₀)/Ω_max` and `η = ε_TT(q)`. -/
def _root_.Pouw.Guarantees.GammaFromTT : Prop :=
  GameStatements.GammaFromTT

end Pouw.Proofs
