module

public import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeedK
public import Definitions.Pouw.Assumptions.PearlC.HiddenTile
public import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
public import Proofs.Pouw.PearlC.Records
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.DeviceInertBoundRev1Gap
import Proofs.Pouw.PearlC.HiddenGap
import Proofs.Pouw.PearlC.RowSeedCode
import Proofs.Pouw.PearlC.RowSeedGap
public import Proofs.Pouw.PearlC.HiddenGap

/-! Per-row seeds on the served line: rev1 against the honest kernel's `W_ref`. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

section Gap

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

theorem pearlCGapDevRev1KRowSeedAt (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ)
    (s : Shape) (ω g : ℚ) (hω0 : 0 < ω) (hω : wrefDevRev1K d c s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hg : g = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutRowSeed CM (pearlCProtocolDevRev1RowInertBound d sem ρ) (pearlCDomainDev d) (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1KRowInertBound d sem ρ c) (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (g : ℝ)
      (fun q N => δPearlC q (N * s.m)) :=
  pearlCGapDevRev1RowSeedWAt CM d sem ρ (wrefDevRev1K d c) s ω g hω0 hω hg hTT

theorem pearlCSampledDevRev1KRowSeedAt (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ)
    (s : Shape) (ω g : ℚ) (hω0 : 0 < ω) (hw0 : 0 ≤ wrefDevRev1K d c s)
    (hω : wrefDevRev1K d c s = ω * ((1 - ρ) * creditDevRev1 d s)) (hg : g = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowInertBound d sem ρ) (pearlCTilesDevRev1RowInertBound d sem ρ)
      (pearlCDomainDev d) (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1KRowInertBound d sem ρ c) (pearlCTilesDevRev1KRowInertBound d sem ρ c)
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (g : ℝ) (fun q N => δPearlC q (N * s.m)) :=
  pearlCSampledDevRev1RowSeedWAt CM d sem ρ (wrefDevRev1K d c) s ω g hω0 hw0 hω hg hTT

theorem devSm120v1Loop_add : 0 ≤ (devSm120v1 Prices.sm120Loop).prices.add := by
  show (0 : ℚ) ≤ Prices.sm120Loop.add; norm_num [Prices.sm120Loop]

theorem devSm120v1Loop_bf16 : 0 ≤ (devSm120v1 Prices.sm120Loop).prices.bf16 := by
  show (0 : ℚ) ≤ Prices.sm120Loop.bf16; norm_num [Prices.sm120Loop]

theorem wrefDevRev1K_sm120v1LoopCast8p72_8192_inert_share1000 :
    wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) sh8192 =
      8921455 / 8919688 / (1 - 1 / 1000) *
        ((1 - 1 / 1000) * creditDevRev1 (devSm120v1 Prices.sm120Loop) sh8192) := by
  rw [wrefDevRev1K_sm120v1LoopCast8p72_8192]
  ring

variable (CM : CostModel Q R S) {sem sem' semC : PearlCSem Q R S}

theorem pearlCGapSm120v1LoopCast8p72Rev1RowSeedInertShare1000_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) (hN : RowSeedNoise sem sem')
    (hCM : CM.SplitClosed) :
    GGap CM (pearlCProtocolDevRev1KRowInertBound (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192 L ∧ rowsOf L ≤ 2 ^ 64)
      ((1648180439 / 446072750000 : ℚ) : ℝ) (fun q N => δPearlC q (N * 8192)) :=
  pearlCGapDevRev1KRowSeedAt CM (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000) sh8192
    (8921455 / 8919688 / (1 - 1 / 1000)) _ (by norm_num) wrefDevRev1K_sm120v1LoopCast8p72_8192_inert_share1000
    (by norm_num)
    (ttOutRowSeed_of_ttOut hTT hN hCM (devAt_rowLocal _ _ _) (by norm_num) devSm120v1Loop_add devSm120v1Loop_bf16)

theorem pearlCSampledSm120v1LoopCast8p72Rev1RowSeedInertShare1000_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) (hN : RowSeedNoise sem sem')
    (hCM : CM.SplitClosed) :
    GGapSampled CM
      (pearlCProtocolDevRev1KRowInertBound (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1KRowInertBound (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192 L ∧ rowsOf L ≤ 2 ^ 64)
      ((1648180439 / 446072750000 : ℚ) : ℝ) (fun q N => δPearlC q (N * 8192)) :=
  pearlCSampledDevRev1KRowSeedAt CM (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000) sh8192
    (8921455 / 8919688 / (1 - 1 / 1000)) _ (by norm_num) wrefDevRev1K_sm120v1LoopCast8p72_8192_nonneg
    wrefDevRev1K_sm120v1LoopCast8p72_8192_inert_share1000 (by norm_num)
    (ttOutTileRowSeed_of_ttOutTile hTT hN hCM (devAt_rowLocal _ _ _) (by norm_num) devSm120v1Loop_add
      devSm120v1Loop_bf16)

theorem pearlCSampledSm120v1LoopCast8p72Rev1RowSeedCodeInertShare1000_8192 {π : Layout → Q ≃ Q}
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) (hN : RowSeedNoise sem sem')
    (hC : CodeSeedNoise sem' semC π) (hCM : CM.SplitClosed) (hCM' : CM.RelabelClosed) :
    GGapSampled CM
      (pearlCProtocolDevRev1KRowInertBound (devSm120v1 Prices.sm120Loop) semC (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1KRowInertBound (devSm120v1 Prices.sm120Loop) semC (1 / 1000) (218 / 25 - 8953 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192 L ∧ rowsOf L ≤ 2 ^ 64)
      ((1648180439 / 446072750000 : ℚ) : ℝ) (fun q N => δPearlC q (N * 8192)) :=
  pearlCSampledDevRev1KRowSeedAt CM (devSm120v1 Prices.sm120Loop) semC (1 / 1000) (218 / 25 - 8953 / 1000) sh8192
    (8921455 / 8919688 / (1 - 1 / 1000)) _ (by norm_num) wrefDevRev1K_sm120v1LoopCast8p72_8192_nonneg
    wrefDevRev1K_sm120v1LoopCast8p72_8192_inert_share1000 (by norm_num)
    (ttOutTileRowSeedCode_of_ttOutTile hTT hN hC hCM hCM' (devAt_rowLocal _ _ _) (by norm_num) devSm120v1Loop_add
      devSm120v1Loop_bf16)

/-- Conditional as `pearlCHiddenSm120v1LoopCast8p72Rev1InertShare1000_8192`: `HA` and `δzk` are free, and
`TileProofSoundAll` holds at a degenerate hidden audit (an empty class) for any `δzk ≥ 0`, so this says nothing about Flock.
Cite it only as conditional. -/
theorem pearlCHiddenSm120v1LoopCast8p72Rev1RowSeedInertShare1000_8192 (HA : HiddenAudit Q R S) (δzk : ℕ → ℝ)
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) (hN : RowSeedNoise sem sem')
    (hCM : CM.SplitClosed)
    (hzk : TileProofSoundAll
      (pearlCProtocolDevRev1KRowInertBound (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1KRowInertBound (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192 L ∧ rowsOf L ≤ 2 ^ 64) HA δzk) :
    GGapHidden CM
      (pearlCProtocolDevRev1KRowInertBound (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1KRowInertBound (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192 L ∧ rowsOf L ≤ 2 ^ 64) HA
      ((1648180439 / 446072750000 : ℚ) : ℝ) (fun q N => δPearlC q (N * 8192)) δzk :=
  gapHidden_of_sampled_all CM _ _ _ HA _ _ δzk
    (pearlCSampledSm120v1LoopCast8p72Rev1RowSeedInertShare1000_8192 CM hTT hN hCM) hzk

/-- Conditional as `pearlCHiddenSm120v1LoopCast8p72Rev1RowSeedInertShare1000_8192`, with C1-h3 and C5 added. Cite it
only as conditional. -/
theorem pearlCHiddenSm120v1LoopCast8p72Rev1RowSeedCodeInertShare1000_8192 {π : Layout → Q ≃ Q} (HA : HiddenAudit Q R S)
    (δzk : ℕ → ℝ) (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000))
    (hN : RowSeedNoise sem sem') (hC : CodeSeedNoise sem' semC π) (hCM : CM.SplitClosed) (hCM' : CM.RelabelClosed)
    (hzk : TileProofSoundAll
      (pearlCProtocolDevRev1KRowInertBound (devSm120v1 Prices.sm120Loop) semC (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1KRowInertBound (devSm120v1 Prices.sm120Loop) semC (1 / 1000) (218 / 25 - 8953 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192 L ∧ rowsOf L ≤ 2 ^ 64) HA δzk) :
    GGapHidden CM
      (pearlCProtocolDevRev1KRowInertBound (devSm120v1 Prices.sm120Loop) semC (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1KRowInertBound (devSm120v1 Prices.sm120Loop) semC (1 / 1000) (218 / 25 - 8953 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192 L ∧ rowsOf L ≤ 2 ^ 64) HA
      ((1648180439 / 446072750000 : ℚ) : ℝ) (fun q N => δPearlC q (N * 8192)) δzk :=
  gapHidden_of_sampled_all CM _ _ _ HA _ _ δzk
    (pearlCSampledSm120v1LoopCast8p72Rev1RowSeedCodeInertShare1000_8192 CM hTT hN hC hCM hCM') hzk

end Gap

end Pouw.PearlC
