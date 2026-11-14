module

public import Definitions.Pouw.Assumptions.GameStatements
public import Definitions.Pouw.Assumptions.Deadline
public import Definitions.Pouw.Assumptions.Game
public import Definitions.Pouw.PearlC.Deadline
public import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
public import Definitions.Pouw.Assumptions.PearlC.TTOutW1
public import Definitions.Pouw.Game.Params
public import Definitions.Pouw.PearlC.Deadline
public import Definitions.Pouw.PearlC.DeviceKernelWref
public import Definitions.Pouw.PearlC.Tile
public import Definitions.Pouw.Game.Deadline
public import Definitions.Pouw.PearlC.Game
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Work between the salt and the deadline (trusted)

A proof of compute utilization: acceptance means at least `W` of work after the salt and before a deadline. `EndToEnd`
gives the work and its freshness (preprocessing never reads the salt); the deadline is `Definitions.Pouw.Game.Deadline`'s
timed run and verifier's rule, and time meets the cost model through one named assumption, the honest clock
(`Pouw.Assumptions.HonestClock`).

* `WorkBetweenSaltAndDeadline`: `EndToEnd` with the budget on the run's spending after the salt and by the deadline, under
  the honest clock. Its hypotheses are `EndToEnd`'s, `Adversary.Bounded`'s online cost read in the time window
  (`Adversary.BoundedBy`), and the clock.
* `SpareCapacityAtDeadline`: remote's exhaustion. A declared fleet that can spend `C·(td − ts)` in the window has at most
  `C·(td − ts) − (1 − g)(1 − ε_s)·W` of it to spare for undeclared work, except with the same probability.
* `PearlCSampledByDeadline`: the same in Pearl-C's game, from `GGapSampled`, so at every device record from
  `PearlC.SampledDevRev1K`. Its instances at the served record's numbers, on the four layouts the sampled g is pinned
  at, are lemmas (`Proofs.Pouw.Lemmas.Deadline`).
-/

@[expose] public section

namespace Pouw.Guarantees

open Pouw.GameStatements Finset

/-- **Work between the salt and the deadline.** Under TT(g₀), the `Ω_max` bound and an honest clock, for every in-domain
workload, adversary, timed run, deadline `td` and budget: if the run spends at most `T` after the salt's release `ts` and
by `td`, and makes at most `q` oracle calls, then the audit with `t` work-weighted samples accepts a transcript received
by `td` while `T < (1 − g)(1 − ε_s)·W`, `g = 1 − (1 − g₀)/Ω_max`, and the run doesn't break SHA-512, with probability
at most `(1 − ε_s)^t + δ(q, N)`. That is, except with that probability, an accepted transcript means at least
`(1 − g)(1 − ε_s)·W` of cost spent between the salt and the deadline, or a collision among the run's own SHA-512
inputs. -/
def WorkBetweenSaltAndDeadline : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (g₀ Ωmax : ℝ) (δ : ℕ → ℕ → ℝ),
    g₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D g₀ δ →
    ∀ U : Workload, U.InDomain D → ∀ (A : Adversary CM) (run : TimedRun Q R S) (td : ℝ),
    Assumptions.HonestClock (CM.adversaries U) A run td →
    ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → A.BoundedBy U run td T q → ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
          AcceptsBy run td x.1 x.2.1 (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - reductionGap g₀ Ωmax) * (1 - εs) * totalWref P U.layout ∧ ¬ A.Breaks U x.1 x.2.1)
        ≤ (1 - εs) ^ t + δ q U.layout.N

/-- **Remote's exhaustion: the spare capacity at the deadline.** A declared fleet of capacity `C` per unit of time can
spend `C·(td − ts)` between the salt's release and the deadline. If the run's spending in that window and undeclared
work `Y` together fit in it on every oracle and salt, then, except with probability `(1 − ε_s)^t + δ(q, N)`, an
accepted transcript received by `td` from a run that doesn't break SHA-512 leaves `Y ≤ C·(td − ts) − (1 − g)(1 − ε_s)·W`: the undeclared spare capacity is at
most the fleet's capacity less the work the audit certifies. -/
def SpareCapacityAtDeadline : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (g₀ Ωmax : ℝ) (δ : ℕ → ℕ → ℝ),
    g₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D g₀ δ →
    ∀ U : Workload, U.InDomain D → ∀ (A : Adversary CM) (run : TimedRun Q R S) (td : ℝ),
    Assumptions.HonestClock (CM.adversaries U) A run td →
    ∀ C Y : ℝ, Y ≤ C * (td - run.ts) → ∀ q : ℕ, q ≤ 2 ^ 64 →
    (∀ (H : Q → R) (s : S), run.spentIn H s run.ts td + Y ≤ C * (td - run.ts) ∧
      A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
          AcceptsBy run td x.1 x.2.1 (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
          C * (td - run.ts) - (1 - reductionGap g₀ Ωmax) * (1 - εs) * totalWref P U.layout < Y ∧
          ¬ A.Breaks U x.1 x.2.1)
        ≤ (1 - εs) ^ t + δ q U.layout.N

/-- **Pearl-C's sampled audit by the deadline.** `GGapSampled` and an honest clock give, for every in-domain workload,
adversary, timed run and deadline `td`, every run spending at most `T` after the salt's release and by `td` with at most
`q` oracle calls, and every sample weighted by the tiles' `W_ref`: the audit accepts a transcript received by `td` while
`T` is below `1 − g` of the claimed work less `ε_s` of the layout's with probability at most `(1 − ε_s)^t + δ(q, N)`. -/
def PearlCSampledByDeadline : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : PearlC.CostModel Q R S)
    (P : PearlC.Protocol Q R S) (TR : PearlC.TileRules Q R S) (D : PearlC.Layout → Prop) (g : ℝ)
    (δ : ℕ → ℕ → ℝ), PearlC.GGapSampled CM P TR D g δ →
    ∀ U : PearlC.Workload, U.InDomain P D → ∀ (A : PearlC.Adversary CM) (run : TimedRun Q R S) (td : ℝ),
    Assumptions.HonestClock (CM.adversaries U) A run td →
    ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → A.BoundedBy U run td T q →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ i < (TR.tiling U.layout).NT, (w i : ℝ) = κ * TR.Wref U.layout i) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          AcceptsBy run td x.1 x.2.1 (PearlC.goodTiles P TR U x.1 x.2.1 (A.transcript U x.1 x.2.1) ∪
            PearlC.blankTiles TR U (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - g) * (PearlC.claimedWref TR U (A.transcript U x.1 x.2.1) - εs * PearlC.tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + δ q U.layout.N

end Pouw.Guarantees
