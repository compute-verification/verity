import Proofs.Pouw.PearlC.DeviceSm120KernelGamma
import Models.Pouw.Guarantees.PearlC

/-!
# v1 under rev1 at a stricter cap, FP32 at the issue-bound add (FP8 security's file)

The FADD-8.00 price twin of v1's restatement at the cap 1/1,000 (`DeviceCapRev1Gamma`, at the in-loop FP32 add 8.376).
* **γ at v1's record with FP32 at 8.00 at any cap `ρ < 1`**: `DeviceSm120KernelGamma`'s v1 under rev1 at
  `devSm120v1 Prices.sm120`, against `W_ref` with the honest cast at 8.72 (`wrefDevRev1K` at `c = 8.72 − 8`), at
  `γ = 1 − (1 − ρ)·221844/222443` at 8192³ and `1 − (1 − ρ)·146300/146681` at 16384³, per unit and per audit tile. At
  `ρ = 1/400` these are that file's `115361/22244300` (0.51861%) and `2987/586724` (0.50910%).
* **At the cap 1/1,000** (`tt-out/pearl-c-sm120-rev1-cap1000` and its tile twin): `205211/55610750` (0.36901%) at
  8192³ and `5273/1466810` (0.35949%) at 16384³, per unit and per audit tile. The in-loop twins are larger (0.36949%,
  0.35973%), so the price-twin ledger publishes those.
-/

namespace Pouw.PearlC

open Pouw.Fp8Atom Pouw.PearlC.Assumptions

section Gamma

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

/-! ## v1 under rev1 at a cap `ρ < 1` -/

/-- **v1 under rev1 at a cap `ρ < 1`, 8192³**, FP32 at 8.00, the honest cast at 8.72:
`γ = 1 − (1 − ρ)·221844/222443`. -/
theorem pearlCGammaSm120v1Cast8p72Rev1Cap_8192 (ρ : ℚ) (hρ : ρ < 1)
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem ρ) :
    Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem ρ (218 / 25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192)
      ((1 - (1 - ρ) * (221844 / 222443) : ℚ) : ℝ) εPearlC := by
  have h1 : (1 : ℚ) - ρ ≠ 0 := by linarith
  refine pearlCGammaDevRev1KAt CM _ sem ρ _ sh8192 (222443 / 222400 / (1 - ρ)) _
    (div_pos (by norm_num) (by linarith)) ?_ ?_ hTT
  · rw [wrefDevRev1K_sm120v1Cast8p72_8192]
    field_simp
  · field_simp
    ring

/-- **v1 under rev1 per audit tile at a cap `ρ < 1`, 8192³**, FP32 at 8.00, the honest cast at 8.72:
`γ = 1 − (1 − ρ)·221844/222443`. -/
theorem pearlCSampledSm120v1Cast8p72Rev1Cap_8192 (ρ : ℚ) (hρ : ρ < 1)
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem ρ) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem ρ (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem ρ (218 / 25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192)
      ((1 - (1 - ρ) * (221844 / 222443) : ℚ) : ℝ) εPearlC := by
  have h1 : (1 : ℚ) - ρ ≠ 0 := by linarith
  refine pearlCSampledDevRev1KAt CM _ sem ρ _ sh8192 (222443 / 222400 / (1 - ρ)) _
    (div_pos (by norm_num) (by linarith)) wrefDevRev1K_sm120v1Cast8p72_8192_nonneg ?_ ?_ hTT
  · rw [wrefDevRev1K_sm120v1Cast8p72_8192]
    field_simp
  · field_simp
    ring

/-- **v1 under rev1 at a cap `ρ < 1`, 16384³**, FP32 at 8.00, the honest cast at 8.72:
`γ = 1 − (1 − ρ)·146300/146681`. -/
theorem pearlCGammaSm120v1Cast8p72Rev1Cap_16384 (ρ : ℚ) (hρ : ρ < 1)
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem ρ) :
    Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem ρ (218 / 25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384)
      ((1 - (1 - ρ) * (146300 / 146681) : ℚ) : ℝ) εPearlC := by
  have h1 : (1 : ℚ) - ρ ≠ 0 := by linarith
  refine pearlCGammaDevRev1KAt CM _ sem ρ _ sh16384 (440043 / 440000 / (1 - ρ)) _
    (div_pos (by norm_num) (by linarith)) ?_ ?_ hTT
  · rw [wrefDevRev1K_sm120v1Cast8p72_16384]
    field_simp
  · field_simp
    ring

/-- **v1 under rev1 per audit tile at a cap `ρ < 1`, 16384³**, FP32 at 8.00, the honest cast at 8.72:
`γ = 1 − (1 − ρ)·146300/146681`. -/
theorem pearlCSampledSm120v1Cast8p72Rev1Cap_16384 (ρ : ℚ) (hρ : ρ < 1)
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem ρ) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem ρ (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem ρ (218 / 25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384)
      ((1 - (1 - ρ) * (146300 / 146681) : ℚ) : ℝ) εPearlC := by
  have h1 : (1 : ℚ) - ρ ≠ 0 := by linarith
  refine pearlCSampledDevRev1KAt CM _ sem ρ _ sh16384 (440043 / 440000 / (1 - ρ)) _
    (div_pos (by norm_num) (by linarith)) wrefDevRev1K_sm120v1Cast8p72_16384_nonneg ?_ ?_ hTT
  · rw [wrefDevRev1K_sm120v1Cast8p72_16384]
    field_simp
  · field_simp
    ring

/-! ## v1 under rev1 at the cap 1/1,000 -/

/-- **v1 under rev1 at the cap 1/1,000, 8192³**, FP32 at 8.00, the honest cast at 8.72:
`γ = 205211/55610750` (0.36901%). -/
theorem pearlCGammaSm120v1Cast8p72Rev1Cap1000_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192)
      ((205211 / 55610750 : ℚ) : ℝ) εPearlC := by
  rw [show (205211 / 55610750 : ℚ) = 1 - (1 - 1 / 1000) * (221844 / 222443) by norm_num]
  exact pearlCGammaSm120v1Cast8p72Rev1Cap_8192 CM sem (1 / 1000) (by norm_num) hTT

/-- **v1 under rev1 per audit tile at the cap 1/1,000, 8192³**, FP32 at 8.00, the honest cast at 8.72:
`γ = 205211/55610750` (0.36901%). -/
theorem pearlCSampledSm120v1Cast8p72Rev1Cap1000_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192)
      ((205211 / 55610750 : ℚ) : ℝ) εPearlC := by
  rw [show (205211 / 55610750 : ℚ) = 1 - (1 - 1 / 1000) * (221844 / 222443) by norm_num]
  exact pearlCSampledSm120v1Cast8p72Rev1Cap_8192 CM sem (1 / 1000) (by norm_num) hTT

/-- **v1 under rev1 at the cap 1/1,000, 16384³**, FP32 at 8.00, the honest cast at 8.72:
`γ = 5273/1466810` (0.35949%). -/
theorem pearlCGammaSm120v1Cast8p72Rev1Cap1000_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384)
      ((5273 / 1466810 : ℚ) : ℝ) εPearlC := by
  rw [show (5273 / 1466810 : ℚ) = 1 - (1 - 1 / 1000) * (146300 / 146681) by norm_num]
  exact pearlCGammaSm120v1Cast8p72Rev1Cap_16384 CM sem (1 / 1000) (by norm_num) hTT

/-- **v1 under rev1 per audit tile at the cap 1/1,000, 16384³**, FP32 at 8.00, the honest cast at 8.72:
`γ = 5273/1466810` (0.35949%). -/
theorem pearlCSampledSm120v1Cast8p72Rev1Cap1000_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384)
      ((5273 / 1466810 : ℚ) : ℝ) εPearlC := by
  rw [show (5273 / 1466810 : ℚ) = 1 - (1 - 1 / 1000) * (146300 / 146681) by norm_num]
  exact pearlCSampledSm120v1Cast8p72Rev1Cap_16384 CM sem (1 / 1000) (by norm_num) hTT

end Gamma

end Pouw.PearlC

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_16384 : Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_16384 := @Pouw.PearlC.pearlCGammaSm120v1Cast8p72Rev1Cap1000_16384

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_8192 : Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_8192 := @Pouw.PearlC.pearlCGammaSm120v1Cast8p72Rev1Cap1000_8192

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_16384 : Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_16384 := @Pouw.PearlC.pearlCSampledSm120v1Cast8p72Rev1Cap1000_16384

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_8192 : Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_8192 := @Pouw.PearlC.pearlCSampledSm120v1Cast8p72Rev1Cap1000_8192
