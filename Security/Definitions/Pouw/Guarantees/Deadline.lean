import Definitions.Pouw.Assumptions.GameStatements
import Definitions.Pouw.Assumptions.Deadline
import Definitions.Pouw.Assumptions.PearlC.Deadline
import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
import Definitions.Pouw.Assumptions.PearlC.TTOutW1
import Definitions.Pouw.PearlC.DeviceKernelWref
import Definitions.Pouw.PearlC.DevicePricesLoop
import Definitions.Pouw.PearlC.Tile

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
  `C·(td − ts) − (1 − γ)(1 − ε_s)·W` of it to spare for undeclared work, except with the same probability.
* `PearlCSampledByDeadline` and its instances: the same in Pearl-C's game, from `GγSampled`, and at v1 under rev1 on
  sm_120 at the cap 1/1,000 (FP32 at 8.376, the honest cast at 8.72) on the four layouts its sampled γ is pinned at.
  The instances are stated at W1: their cost model is `W1.w1Priced (W1Device.sm120Loop sha fetch) salt`, and their
  hypotheses are TT_OUT rev1 per audit tile at 1/1,000 at W1 (`TTOutTilePearlCDevRev1W1`) and the honest clock at that
  cost model. `sha`, `fetch`, the salt's bytes `salt` and `sem` are parameters.
-/

namespace Pouw.Guarantees

open Pouw.GameStatements Finset

/-- **Work between the salt and the deadline.** Under TT(γ₀), the `Ω_max` bound and an honest clock, for every in-domain
workload, adversary, timed run, deadline `td` and budget: if the run spends at most `T` after the salt's release `ts` and
by `td`, and makes at most `q` oracle calls, then the audit with `t` work-weighted samples accepts a transcript received
by `td` while `T < (1 − γ)(1 − ε_s)·W`, `γ = 1 − (1 − γ₀)/Ω_max`, with probability at most `(1 − ε_s)^t + ε(q, N)`.
That is, except with that probability, an accepted transcript means at least `(1 − γ)(1 − ε_s)·W` of cost spent between
the salt and the deadline. -/
def WorkBetweenSaltAndDeadline : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D γ₀ ε →
    ∀ U : Workload, U.InDomain D → ∀ (A : Adversary CM) (run : TimedRun Q R S) (td : ℝ),
    Assumptions.HonestClock CM U A run td →
    ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → A.BoundedBy U run td T q → ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
          AcceptsBy run td x.1 x.2.1 (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - reductionGamma γ₀ Ωmax) * (1 - εs) * totalWref P U.layout)
        ≤ (1 - εs) ^ t + ε q U.layout.N

/-- **Remote's exhaustion: the spare capacity at the deadline.** A declared fleet of capacity `C` per unit of time can
spend `C·(td − ts)` between the salt's release and the deadline. If the run's spending in that window and undeclared
work `Y` together fit in it on every oracle and salt, then, except with probability `(1 − ε_s)^t + ε(q, N)`, an
accepted transcript received by `td` leaves `Y ≤ C·(td − ts) − (1 − γ)(1 − ε_s)·W`: the undeclared spare capacity is at
most the fleet's capacity less the work the audit certifies. -/
def SpareCapacityAtDeadline : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D γ₀ ε →
    ∀ U : Workload, U.InDomain D → ∀ (A : Adversary CM) (run : TimedRun Q R S) (td : ℝ),
    Assumptions.HonestClock CM U A run td →
    ∀ C Y : ℝ, Y ≤ C * (td - run.ts) → ∀ q : ℕ, q ≤ 2 ^ 64 →
    (∀ (H : Q → R) (s : S), run.spentIn H s run.ts td + Y ≤ C * (td - run.ts) ∧
      A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
          AcceptsBy run td x.1 x.2.1 (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
          C * (td - run.ts) - (1 - reductionGamma γ₀ Ωmax) * (1 - εs) * totalWref P U.layout < Y)
        ≤ (1 - εs) ^ t + ε q U.layout.N

/-- **Pearl-C's sampled audit by the deadline.** `GγSampled` and an honest clock give, for every in-domain workload,
adversary, timed run and deadline `td`, every run spending at most `T` after the salt's release and by `td` with at most
`q` oracle calls, and every draw weighted by the tiles' `W_ref`: the audit accepts a transcript received by `td` while
`T` is below `1 − γ` of the claimed work less `ε_s` of the layout's with probability at most `(1 − ε_s)^t + η(q, N)`. -/
def PearlCSampledByDeadline : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : PearlC.CostModel Q R S)
    (P : PearlC.Protocol Q R S) (TR : PearlC.TileRules Q R S) (D : PearlC.Layout → Prop) (γ : ℝ)
    (η : ℕ → ℕ → ℝ), PearlC.GγSampled CM P TR D γ η →
    ∀ U : PearlC.Workload, U.InDomain P D → ∀ (A : PearlC.Adversary CM) (run : TimedRun Q R S) (td : ℝ),
    PearlC.Assumptions.HonestClock CM U A run td →
    ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → A.BoundedBy U run td T q →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ g < (TR.tiling U.layout).NT, (w g : ℝ) = κ * TR.Wref U.layout g) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          AcceptsBy run td x.1 x.2.1 (PearlC.goodTiles P TR U x.1 x.2.1 (A.transcript U x.1 x.2.1) ∪
            PearlC.blankTiles TR U (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - γ) * (PearlC.claimedWref TR U (A.transcript U x.1 x.2.1) - εs * PearlC.tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + η q U.layout.N

/-- **Pearl-C v1 under rev1 on sm_120 at the cap 1/1,000 by the deadline, 8192³**: `PearlCSampledByDeadline` at
`pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_8192`'s record, FP32 at 8.376 and the honest cast at 8.72, at
`γ = 1648180439/446072750000` (0.36949%), at W1: the cost model is `W1.w1Priced dv salt`, with `dv` the record's W1
setting `W1Device.sm120Loop sha fetch`. Its hypotheses are TT_OUT rev1 per audit tile at 1/1,000 at W1 and the honest
clock at that cost model; `sha`, `fetch`, `salt` and `sem` are parameters. -/
def PearlCSm120Rev1Cap1000ByDeadline_8192 : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlC.PearlCSem Q R S),
    let d := PearlC.devSm120v1 PearlC.Prices.sm120Loop
    let dv := PearlC.W1.W1Device.sm120Loop sha fetch
    let CM : PearlC.CostModel Q R S := PearlC.W1.w1Priced dv salt
    let P := PearlC.pearlCProtocolDevRev1K d sem (1 / 1000) (218 / 25 - 8953 / 1000)
    let TR := PearlC.pearlCTilesDevRev1K d sem (1 / 1000) (218 / 25 - 8953 / 1000)
    PearlC.Assumptions.TTOutTilePearlCDevRev1W1 dv salt d sem (1 / 1000) →
    ∀ U : PearlC.Workload, U.InDomain P (PearlC.pearlCDomainDevAt d PearlC.sh8192) →
    ∀ (A : PearlC.Adversary CM) (run : TimedRun Q R S) (td : ℝ), PearlC.Assumptions.HonestClock CM U A run td →
    ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → A.BoundedBy U run td T q →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ g < (TR.tiling U.layout).NT, (w g : ℝ) = κ * TR.Wref U.layout g) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          AcceptsBy run td x.1 x.2.1 (PearlC.goodTiles P TR U x.1 x.2.1 (A.transcript U x.1 x.2.1) ∪
            PearlC.blankTiles TR U (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - ((1648180439 / 446072750000 : ℚ) : ℝ)) *
            (PearlC.claimedWref TR U (A.transcript U x.1 x.2.1) - εs * PearlC.tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + PearlC.εPearlC q U.layout.N

/-- **Pearl-C v1 under rev1 on sm_120 at the cap 1/1,000 by the deadline, 16384³**: `PearlCSampledByDeadline` at
`pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_16384`'s record, at `γ = 1058166877/294158650000` (0.35973%), with the
hypotheses of `PearlCSm120Rev1Cap1000ByDeadline_8192`. -/
def PearlCSm120Rev1Cap1000ByDeadline_16384 : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlC.PearlCSem Q R S),
    let d := PearlC.devSm120v1 PearlC.Prices.sm120Loop
    let dv := PearlC.W1.W1Device.sm120Loop sha fetch
    let CM : PearlC.CostModel Q R S := PearlC.W1.w1Priced dv salt
    let P := PearlC.pearlCProtocolDevRev1K d sem (1 / 1000) (218 / 25 - 8953 / 1000)
    let TR := PearlC.pearlCTilesDevRev1K d sem (1 / 1000) (218 / 25 - 8953 / 1000)
    PearlC.Assumptions.TTOutTilePearlCDevRev1W1 dv salt d sem (1 / 1000) →
    ∀ U : PearlC.Workload, U.InDomain P (PearlC.pearlCDomainDevAt d PearlC.sh16384) →
    ∀ (A : PearlC.Adversary CM) (run : TimedRun Q R S) (td : ℝ), PearlC.Assumptions.HonestClock CM U A run td →
    ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → A.BoundedBy U run td T q →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ g < (TR.tiling U.layout).NT, (w g : ℝ) = κ * TR.Wref U.layout g) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          AcceptsBy run td x.1 x.2.1 (PearlC.goodTiles P TR U x.1 x.2.1 (A.transcript U x.1 x.2.1) ∪
            PearlC.blankTiles TR U (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - ((1058166877 / 294158650000 : ℚ) : ℝ)) *
            (PearlC.claimedWref TR U (A.transcript U x.1 x.2.1) - εs * PearlC.tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + PearlC.εPearlC q U.layout.N

/-- **Pearl-C v1 under rev1 on sm_120 at the cap 1/1,000 by the deadline, on Qwen3-8B's served layout**:
`PearlCSampledByDeadline` at `pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_qwen3_8b`'s record, at
`γ = 74663501/19190030000` (0.38907%), with the hypotheses of `PearlCSm120Rev1Cap1000ByDeadline_8192`. -/
def PearlCSm120Rev1Cap1000ByDeadline_qwen3_8b : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlC.PearlCSem Q R S),
    let d := PearlC.devSm120v1 PearlC.Prices.sm120Loop
    let dv := PearlC.W1.W1Device.sm120Loop sha fetch
    let CM : PearlC.CostModel Q R S := PearlC.W1.w1Priced dv salt
    let P := PearlC.pearlCProtocolDevRev1K d sem (1 / 1000) (218 / 25 - 8953 / 1000)
    let TR := PearlC.pearlCTilesDevRev1K d sem (1 / 1000) (218 / 25 - 8953 / 1000)
    PearlC.Assumptions.TTOutTilePearlCDevRev1W1 dv salt d sem (1 / 1000) →
    ∀ U : PearlC.Workload, U.InDomain P (PearlC.pearlCDomainDevShapes d PearlC.qwen3_8bLinears) →
    ∀ (A : PearlC.Adversary CM) (run : TimedRun Q R S) (td : ℝ), PearlC.Assumptions.HonestClock CM U A run td →
    ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → A.BoundedBy U run td T q →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ g < (TR.tiling U.layout).NT, (w g : ℝ) = κ * TR.Wref U.layout g) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          AcceptsBy run td x.1 x.2.1 (PearlC.goodTiles P TR U x.1 x.2.1 (A.transcript U x.1 x.2.1) ∪
            PearlC.blankTiles TR U (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - ((74663501 / 19190030000 : ℚ) : ℝ)) *
            (PearlC.claimedWref TR U (A.transcript U x.1 x.2.1) - εs * PearlC.tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + PearlC.εPearlC q U.layout.N

/-- **Pearl-C v1 under rev1 on sm_120 at the cap 1/1,000 by the deadline, on Llama-3.1-8B's served layout**:
`PearlCSampledByDeadline` at `pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_llama31_8b`'s record, at
`γ = 3045273083/782596025000` (0.38912%), with the hypotheses of `PearlCSm120Rev1Cap1000ByDeadline_8192`. -/
def PearlCSm120Rev1Cap1000ByDeadline_llama31_8b : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlC.PearlCSem Q R S),
    let d := PearlC.devSm120v1 PearlC.Prices.sm120Loop
    let dv := PearlC.W1.W1Device.sm120Loop sha fetch
    let CM : PearlC.CostModel Q R S := PearlC.W1.w1Priced dv salt
    let P := PearlC.pearlCProtocolDevRev1K d sem (1 / 1000) (218 / 25 - 8953 / 1000)
    let TR := PearlC.pearlCTilesDevRev1K d sem (1 / 1000) (218 / 25 - 8953 / 1000)
    PearlC.Assumptions.TTOutTilePearlCDevRev1W1 dv salt d sem (1 / 1000) →
    ∀ U : PearlC.Workload, U.InDomain P (PearlC.pearlCDomainDevShapes d PearlC.llama31_8bLinears) →
    ∀ (A : PearlC.Adversary CM) (run : TimedRun Q R S) (td : ℝ), PearlC.Assumptions.HonestClock CM U A run td →
    ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → A.BoundedBy U run td T q →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ g < (TR.tiling U.layout).NT, (w g : ℝ) = κ * TR.Wref U.layout g) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          AcceptsBy run td x.1 x.2.1 (PearlC.goodTiles P TR U x.1 x.2.1 (A.transcript U x.1 x.2.1) ∪
            PearlC.blankTiles TR U (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - ((3045273083 / 782596025000 : ℚ) : ℝ)) *
            (PearlC.claimedWref TR U (A.transcript U x.1 x.2.1) - εs * PearlC.tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + PearlC.εPearlC q U.layout.N

end Pouw.Guarantees
