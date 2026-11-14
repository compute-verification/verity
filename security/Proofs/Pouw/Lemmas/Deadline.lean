module

public import Definitions.Pouw.Guarantees.Deadline
public import Proofs.Pouw.PearlC.Records

/-!
# Pearl-C's sampled audit by the deadline at the served record: lemmas

`PearlCSampledByDeadline` at v1 under rev1 on sm_120 at the cap 1/1,000 (FP32 at 8.376, the honest cast at 8.72), on the
four layouts its sampled g is pinned at, each a `def G : Prop` under the guarantees' namespace, proved by the theorem
`Pouw.SecurityProofs.G` (`Proofs.Pouw.Deadline`) and audited with the proofs: no record pins it. The listed forms are
over every device record: `PearlCSampledByDeadline` from `PearlC.SampledDevRev1K`'s `GGapSampled`.

Each is stated at W1: its cost model is `W1.w1Priced (W1Device.sm120Loop sha fetch) salt`, and its hypotheses are TT_OUT
rev1 per audit tile at 1/1,000 at W1 (`TTOutTilePearlCDevRev1W1`) and the honest clock at that cost model. `sha`,
`fetch`, the salt's bytes `salt` and `sem` are parameters.
-/

@[expose] public section

namespace Pouw.Guarantees

open Pouw.GameStatements Finset

/-- **Pearl-C v1 under rev1 on sm_120 at the cap 1/1,000 by the deadline, 8192³**: `PearlCSampledByDeadline` at
`pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_8192`'s record, FP32 at 8.376 and the honest cast at 8.72, at
`g = 1648180439/446072750000` (0.36949%), at W1: the cost model is `W1.w1Priced dv salt`, with `dv` the record's W1
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
    ∀ (A : PearlC.Adversary CM) (run : TimedRun Q R S) (td : ℝ),
    Assumptions.HonestClock (CM.adversaries U) A run td →
    ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → A.BoundedBy U run td T q →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ i < (TR.tiling U.layout).NT, (w i : ℝ) = κ * TR.Wref U.layout i) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          AcceptsBy run td x.1 x.2.1 (PearlC.goodTiles P TR U x.1 x.2.1 (A.transcript U x.1 x.2.1) ∪
            PearlC.blankTiles TR U (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - ((1648180439 / 446072750000 : ℚ) : ℝ)) *
            (PearlC.claimedWref TR U (A.transcript U x.1 x.2.1) - εs * PearlC.tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + PearlC.δPearlC q U.layout.N

/-- **Pearl-C v1 under rev1 on sm_120 at the cap 1/1,000 by the deadline, 16384³**: `PearlCSampledByDeadline` at
`pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_16384`'s record, at `g = 1058166877/294158650000` (0.35973%), with the
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
    ∀ (A : PearlC.Adversary CM) (run : TimedRun Q R S) (td : ℝ),
    Assumptions.HonestClock (CM.adversaries U) A run td →
    ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → A.BoundedBy U run td T q →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ i < (TR.tiling U.layout).NT, (w i : ℝ) = κ * TR.Wref U.layout i) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          AcceptsBy run td x.1 x.2.1 (PearlC.goodTiles P TR U x.1 x.2.1 (A.transcript U x.1 x.2.1) ∪
            PearlC.blankTiles TR U (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - ((1058166877 / 294158650000 : ℚ) : ℝ)) *
            (PearlC.claimedWref TR U (A.transcript U x.1 x.2.1) - εs * PearlC.tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + PearlC.δPearlC q U.layout.N

/-- **Pearl-C v1 under rev1 on sm_120 at the cap 1/1,000 by the deadline, on Qwen3-8B's served layout**:
`PearlCSampledByDeadline` at `pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_qwen3_8b`'s record, at
`g = 74663501/19190030000` (0.38907%), with the hypotheses of `PearlCSm120Rev1Cap1000ByDeadline_8192`. -/
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
    ∀ (A : PearlC.Adversary CM) (run : TimedRun Q R S) (td : ℝ),
    Assumptions.HonestClock (CM.adversaries U) A run td →
    ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → A.BoundedBy U run td T q →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ i < (TR.tiling U.layout).NT, (w i : ℝ) = κ * TR.Wref U.layout i) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          AcceptsBy run td x.1 x.2.1 (PearlC.goodTiles P TR U x.1 x.2.1 (A.transcript U x.1 x.2.1) ∪
            PearlC.blankTiles TR U (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - ((74663501 / 19190030000 : ℚ) : ℝ)) *
            (PearlC.claimedWref TR U (A.transcript U x.1 x.2.1) - εs * PearlC.tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + PearlC.δPearlC q U.layout.N

/-- **Pearl-C v1 under rev1 on sm_120 at the cap 1/1,000 by the deadline, on Llama-3.1-8B's served layout**:
`PearlCSampledByDeadline` at `pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_llama31_8b`'s record, at
`g = 3045273083/782596025000` (0.38912%), with the hypotheses of `PearlCSm120Rev1Cap1000ByDeadline_8192`. -/
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
    ∀ (A : PearlC.Adversary CM) (run : TimedRun Q R S) (td : ℝ),
    Assumptions.HonestClock (CM.adversaries U) A run td →
    ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → A.BoundedBy U run td T q →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ i < (TR.tiling U.layout).NT, (w i : ℝ) = κ * TR.Wref U.layout i) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          AcceptsBy run td x.1 x.2.1 (PearlC.goodTiles P TR U x.1 x.2.1 (A.transcript U x.1 x.2.1) ∪
            PearlC.blankTiles TR U (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - ((3045273083 / 782596025000 : ℚ) : ℝ)) *
            (PearlC.claimedWref TR U (A.transcript U x.1 x.2.1) - εs * PearlC.tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + PearlC.δPearlC q U.layout.N

end Pouw.Guarantees
