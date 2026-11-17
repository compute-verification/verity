module

public import Proofs.Pouw.PearlC.DeviceSm120KernelGap
public import Proofs.Pouw.Lemmas.PearlCRecords
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.KernelWrefGap
import Proofs.Pouw.PearlC.TTOutW1
public import Proofs.Pouw.PearlC.Records

/-! v1 under rev1 at a stricter cap, with FP32 at the issue-bound add 8.00. -/

@[expose] public section

namespace Pouw.PearlC

open Pouw.Fp8Atom Pouw.PearlC.Assumptions

section Gap

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
  (sem : PearlCSem Q R S)

theorem wrefDevRev1K_sm120v1Cast8p72_8192 :
    wrefDevRev1K (devSm120v1 Prices.sm120) (218 / 25 - 8) sh8192 =
      222443 / 222400 * creditDevRev1 (devSm120v1 Prices.sm120) sh8192 := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120).G = 4 from rfl,
    show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
  norm_num [Prices.sm120, Params.pi, sh8192]

theorem wrefDevRev1K_sm120v1Cast8p72_8192_nonneg :
    0 ≤ wrefDevRev1K (devSm120v1 Prices.sm120) (218 / 25 - 8) sh8192 := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120).G = 4 from rfl,
    show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
  norm_num [Prices.sm120, Params.pi, sh8192]

theorem wrefDevRev1K_sm120v1Cast8p72_16384 :
    wrefDevRev1K (devSm120v1 Prices.sm120) (218 / 25 - 8) sh16384 =
      440043 / 440000 * creditDevRev1 (devSm120v1 Prices.sm120) sh16384 := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120).G = 4 from rfl,
    show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
  norm_num [Prices.sm120, Params.pi, sh16384]

theorem wrefDevRev1K_sm120v1Cast8p72_16384_nonneg :
    0 ≤ wrefDevRev1K (devSm120v1 Prices.sm120) (218 / 25 - 8) sh16384 := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120).G = 4 from rfl,
    show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
  norm_num [Prices.sm120, Params.pi, sh16384]

theorem pearlCGapSm120v1Cast8p72Rev1InertShare_8192 (ρ : ℚ) (hρ : ρ < 1)
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem ρ) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem ρ (218 / 25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192)
      ((1 - (1 - ρ) * (221844 / 222443) : ℚ) : ℝ) δPearlC := by
  have h1 : (1 : ℚ) - ρ ≠ 0 := by linarith
  refine pearlCGapDevRev1KAt CM _ sem ρ _ sh8192 (222443 / 222400 / (1 - ρ)) _
    (div_pos (by norm_num) (by linarith)) ?_ ?_ hTT
  · rw [wrefDevRev1K_sm120v1Cast8p72_8192]
    field_simp
  · field_simp
    ring

theorem pearlCSampledSm120v1Cast8p72Rev1InertShare_8192 (ρ : ℚ) (hρ : ρ < 1)
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem ρ) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem ρ (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem ρ (218 / 25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192)
      ((1 - (1 - ρ) * (221844 / 222443) : ℚ) : ℝ) δPearlC := by
  have h1 : (1 : ℚ) - ρ ≠ 0 := by linarith
  refine pearlCSampledDevRev1KAt CM _ sem ρ _ sh8192 (222443 / 222400 / (1 - ρ)) _
    (div_pos (by norm_num) (by linarith)) wrefDevRev1K_sm120v1Cast8p72_8192_nonneg ?_ ?_ hTT
  · rw [wrefDevRev1K_sm120v1Cast8p72_8192]
    field_simp
  · field_simp
    ring

theorem pearlCGapSm120v1Cast8p72Rev1InertShare_16384 (ρ : ℚ) (hρ : ρ < 1)
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem ρ) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem ρ (218 / 25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384)
      ((1 - (1 - ρ) * (146300 / 146681) : ℚ) : ℝ) δPearlC := by
  have h1 : (1 : ℚ) - ρ ≠ 0 := by linarith
  refine pearlCGapDevRev1KAt CM _ sem ρ _ sh16384 (440043 / 440000 / (1 - ρ)) _
    (div_pos (by norm_num) (by linarith)) ?_ ?_ hTT
  · rw [wrefDevRev1K_sm120v1Cast8p72_16384]
    field_simp
  · field_simp
    ring

theorem pearlCSampledSm120v1Cast8p72Rev1InertShare_16384 (ρ : ℚ) (hρ : ρ < 1)
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem ρ) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem ρ (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem ρ (218 / 25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384)
      ((1 - (1 - ρ) * (146300 / 146681) : ℚ) : ℝ) δPearlC := by
  have h1 : (1 : ℚ) - ρ ≠ 0 := by linarith
  refine pearlCSampledDevRev1KAt CM _ sem ρ _ sh16384 (440043 / 440000 / (1 - ρ)) _
    (div_pos (by norm_num) (by linarith)) wrefDevRev1K_sm120v1Cast8p72_16384_nonneg ?_ ?_ hTT
  · rw [wrefDevRev1K_sm120v1Cast8p72_16384]
    field_simp
  · field_simp
    ring

theorem pearlCGapSm120v1Cast8p72Rev1InertShare1000_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192)
      ((205211 / 55610750 : ℚ) : ℝ) δPearlC := by
  rw [show (205211 / 55610750 : ℚ) = 1 - (1 - 1 / 1000) * (221844 / 222443) by norm_num]
  exact pearlCGapSm120v1Cast8p72Rev1InertShare_8192 CM sem (1 / 1000) (by norm_num) hTT

theorem pearlCSampledSm120v1Cast8p72Rev1InertShare1000_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192)
      ((205211 / 55610750 : ℚ) : ℝ) δPearlC := by
  rw [show (205211 / 55610750 : ℚ) = 1 - (1 - 1 / 1000) * (221844 / 222443) by norm_num]
  exact pearlCSampledSm120v1Cast8p72Rev1InertShare_8192 CM sem (1 / 1000) (by norm_num) hTT

theorem pearlCGapSm120v1Cast8p72Rev1InertShare1000_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384)
      ((5273 / 1466810 : ℚ) : ℝ) δPearlC := by
  rw [show (5273 / 1466810 : ℚ) = 1 - (1 - 1 / 1000) * (146300 / 146681) by norm_num]
  exact pearlCGapSm120v1Cast8p72Rev1InertShare_16384 CM sem (1 / 1000) (by norm_num) hTT

theorem pearlCSampledSm120v1Cast8p72Rev1InertShare1000_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384)
      ((5273 / 1466810 : ℚ) : ℝ) δPearlC := by
  rw [show (5273 / 1466810 : ℚ) = 1 - (1 - 1 / 1000) * (146300 / 146681) by norm_num]
  exact pearlCSampledSm120v1Cast8p72Rev1InertShare_16384 CM sem (1 / 1000) (by norm_num) hTT

end Gap

end Pouw.PearlC

theorem Pouw.SecurityProofs.PearlC.GapSm120v1Cast8p72Rev1InertShare1000_16384 : Pouw.Guarantees.PearlC.GapSm120v1Cast8p72Rev1InertShare1000_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1Cast8p72Rev1InertShare1000_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1Cast8p72Rev1InertShare1000_8192 : Pouw.Guarantees.PearlC.GapSm120v1Cast8p72Rev1InertShare1000_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1Cast8p72Rev1InertShare1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Cast8p72Rev1InertShare1000_16384 : Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1InertShare1000_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1Cast8p72Rev1InertShare1000_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Cast8p72Rev1InertShare1000_8192 : Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1InertShare1000_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1Cast8p72Rev1InertShare1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)
