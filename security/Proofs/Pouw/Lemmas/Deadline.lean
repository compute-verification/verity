module

public import Definitions.Pouw.Guarantees.Deadline
public import Proofs.Pouw.PearlC.Records

/-! Pearl-C's sampled audit by the deadline at the served record, stated at W1: lemmas that no record pins. -/

@[expose] public section

namespace Pouw.Guarantees

open Pouw.GameStatements Finset

def PearlCSm120Rev1InertShare1000ByDeadline_8192 : Prop :=
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
          AcceptsBy run td x.1 x.2.1 (PearlC.goodTiles P TR U x.1 x.2.1 (A.trace U x.1 x.2.1) ∪
            PearlC.blankTiles TR U (A.trace U x.1 x.2.1)) x.2.2 ∧
          T < (1 - ((1648180439 / 446072750000 : ℚ) : ℝ)) *
            (PearlC.claimedWref TR U (A.trace U x.1 x.2.1) - εs * PearlC.tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + PearlC.δPearlC q U.layout.N

def PearlCSm120Rev1InertShare1000ByDeadline_16384 : Prop :=
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
          AcceptsBy run td x.1 x.2.1 (PearlC.goodTiles P TR U x.1 x.2.1 (A.trace U x.1 x.2.1) ∪
            PearlC.blankTiles TR U (A.trace U x.1 x.2.1)) x.2.2 ∧
          T < (1 - ((1058166877 / 294158650000 : ℚ) : ℝ)) *
            (PearlC.claimedWref TR U (A.trace U x.1 x.2.1) - εs * PearlC.tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + PearlC.δPearlC q U.layout.N

def PearlCSm120Rev1InertShare1000ByDeadline_qwen3_8b : Prop :=
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
          AcceptsBy run td x.1 x.2.1 (PearlC.goodTiles P TR U x.1 x.2.1 (A.trace U x.1 x.2.1) ∪
            PearlC.blankTiles TR U (A.trace U x.1 x.2.1)) x.2.2 ∧
          T < (1 - ((74663501 / 19190030000 : ℚ) : ℝ)) *
            (PearlC.claimedWref TR U (A.trace U x.1 x.2.1) - εs * PearlC.tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + PearlC.δPearlC q U.layout.N

def PearlCSm120Rev1InertShare1000ByDeadline_llama31_8b : Prop :=
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
          AcceptsBy run td x.1 x.2.1 (PearlC.goodTiles P TR U x.1 x.2.1 (A.trace U x.1 x.2.1) ∪
            PearlC.blankTiles TR U (A.trace U x.1 x.2.1)) x.2.2 ∧
          T < (1 - ((3045273083 / 782596025000 : ℚ) : ℝ)) *
            (PearlC.claimedWref TR U (A.trace U x.1 x.2.1) - εs * PearlC.tileWrefTotal TR U.layout))
        ≤ (1 - εs) ^ t + PearlC.δPearlC q U.layout.N

end Pouw.Guarantees
