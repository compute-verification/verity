import Definitions.Pouw.Assumptions.PerCoin
import Definitions.Pouw.Guarantees.Game

/-!
# The game's guarantees per coin (trusted)

`Pouw.Guarantees.GammaFromTT` and `Pouw.Guarantees.EndToEnd` with TT in its per-coin form
(`Pouw.Assumptions.TTPerCoin`). Every adversary is quantified: there is no budget `T` and no oracle cap `q`. The work
an adversary is held to is its own cost on the coin it ran on, and the error is read at its own query count
(`Adversary.queries`). Each event carries `¬ Collides` of the finder built from the adversary (`finderQueries`), so a
collision of the hash the adversary found is charged to that finder alone, outside every theorem here.

Each guarantee takes the cost model's query trees (`hQ : CM.Queries`) as a parameter, since the finder is read from
them. `TTPerCoin` holds at some such `hQ`, and any two are equal.
-/

namespace Pouw.Guarantees

open Pouw.Pinned Finset

/-- **(a) per coin**: `G_γ` from `TTPerCoin` at `γ = 1 − (1 − γ₀)/Ω_max`. For every in-domain workload and every
adversary, the probability that it wins at its own cost (its correct units' reference work exceeds its cost over
`1 − γ`) while its finder finds no collision is at most `ε(A.queries U, N)`. -/
def GammaFromTTPerCoin : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ)
    (hQ : CM.Queries),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TTPerCoin CM P D γ₀ ε →
    ∀ U : Workload, U.InDomain D → ∀ A : Adversary CM,
      pr (fun x : (Q → R) × S =>
          Wins P U (reductionGamma γ₀ Ωmax) (CM.cost A.online x.1 U A.act (A.pre.eval x.1) x.2)
            (A.transcript U x.1 x.2) x.1 x.2 ∧
          ¬ Collides x.1 (finderQueries hQ A U x.1 x.2))
        ≤ ε (A.queries U) U.layout.N

/-- **(d) per coin: TT gives resource exhaustion.** Under `TTPerCoin` at `γ₀` and the `Ω_max` bound, for every
in-domain workload and every adversary, the audit with `t` work-weighted samples satisfies
`Pr[accept ∧ cost < (1 − γ)(1 − ε_s)·W ∧ ¬ the finder collides] ≤ (1 − ε_s)^t + ε(A.queries U, N)` with
`γ = 1 − (1 − γ₀)/Ω_max`, the cost the adversary's own on the run's coin. -/
def EndToEndPerCoin : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ)
    (hQ : CM.Queries),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TTPerCoin CM P D γ₀ ε →
    ∀ U : Workload, U.InDomain D → ∀ A : Adversary CM, ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
          AuditAccepts (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
          CM.cost A.online x.1 U A.act (A.pre.eval x.1) x.2.1 <
            (1 - reductionGamma γ₀ Ωmax) * (1 - εs) * totalWref P U.layout ∧
          ¬ Collides x.1 (finderQueries hQ A U x.1 x.2.1))
        ≤ (1 - εs) ^ t + ε (A.queries U) U.layout.N

end Pouw.Guarantees
