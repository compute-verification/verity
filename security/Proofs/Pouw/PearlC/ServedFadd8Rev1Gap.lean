module

public import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
public import Definitions.Pouw.PearlC.DeviceKernelWref
public import Proofs.Pouw.PearlC.Records
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.ServedMixedRev1Gap
import Proofs.Pouw.PearlC.ServedRev1Gap
import Proofs.Pouw.PearlC.UOnlyGap
public import Proofs.Pouw.PearlC.ServedMixedRev1Gap

/-! v1 under rev1 at the served linears with FP32 at 8, at any batch; each g is below the in-loop one at the same
linear. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

section Served

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
  (sem : PearlCSem Q R S)

theorem creditDevRev1_sm120v1_nonneg (m n k : ℕ) (hk : 0 < k) (h128 : 128 ∣ k) :
    0 ≤ creditDevRev1 (devSm120v1 Prices.sm120) ⟨m, n, k⟩ :=
  creditDevRev1_nonneg _
    (by rw [show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]; norm_num [Prices.sm120])
    (by rw [show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]; norm_num [Prices.sm120])
    _ hk (Or.inr (by rw [show (devSm120v1 Prices.sm120).G = 4 from rfl]; exact h128))

theorem wrefDevRev1K_sm120v1Cast8p72_n6144k4096 (m : ℕ) :
    wrefDevRev1K (devSm120v1 Prices.sm120) (218 / 25 - 8) ⟨m, 6144, 4096⟩ =
      169543 / 169500 * creditDevRev1 (devSm120v1 Prices.sm120) ⟨m, 6144, 4096⟩ := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120).G = 4 from rfl,
    show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
  norm_num [Prices.sm120, Params.pi]
  ring

theorem wrefDevRev1K_sm120v1Cast8p72_n6144k4096_nonneg (m : ℕ) :
    0 ≤ wrefDevRev1K (devSm120v1 Prices.sm120) (218 / 25 - 8) ⟨m, 6144, 4096⟩ := by
  rw [wrefDevRev1K_sm120v1Cast8p72_n6144k4096]
  exact mul_nonneg (by norm_num) (creditDevRev1_sm120v1_nonneg _ _ _ (by norm_num) (by norm_num))

theorem pearlCGapSm120v1Cast8p72Rev1InertShare1000_n6144k4096
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 6144 4096) ((508661 / 135634400 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KNK CM _ sem _ _ _ _ (169543 / 169500 / (1 - 1 / 1000)) _ (by norm_num)
    (fun m => by rw [wrefDevRev1K_sm120v1Cast8p72_n6144k4096 m]; field_simp) (by norm_num) hTT

theorem pearlCSampledSm120v1Cast8p72Rev1InertShare1000_n6144k4096
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 6144 4096) ((508661 / 135634400 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KNK CM _ sem _ _ _ _ (169543 / 169500 / (1 - 1 / 1000)) _ (by norm_num)
    wrefDevRev1K_sm120v1Cast8p72_n6144k4096_nonneg
    (fun m => by rw [wrefDevRev1K_sm120v1Cast8p72_n6144k4096 m]; field_simp) (by norm_num) hTT

theorem wrefDevRev1K_sm120v1Cast8p72_n4096k4096 (m : ℕ) :
    wrefDevRev1K (devSm120v1 Prices.sm120) (218 / 25 - 8) ⟨m, 4096, 4096⟩ =
      113643 / 113600 * creditDevRev1 (devSm120v1 Prices.sm120) ⟨m, 4096, 4096⟩ := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120).G = 4 from rfl,
    show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
  norm_num [Prices.sm120, Params.pi]
  ring

theorem wrefDevRev1K_sm120v1Cast8p72_n4096k4096_nonneg (m : ℕ) :
    0 ≤ wrefDevRev1K (devSm120v1 Prices.sm120) (218 / 25 - 8) ⟨m, 4096, 4096⟩ := by
  rw [wrefDevRev1K_sm120v1Cast8p72_n4096k4096]
  exact mul_nonneg (by norm_num) (creditDevRev1_sm120v1_nonneg _ _ _ (by norm_num) (by norm_num))

theorem pearlCGapSm120v1Cast8p72Rev1InertShare1000_n4096k4096
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 4096 4096) ((1359 / 350750 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KNK CM _ sem _ _ _ _ (113643 / 113600 / (1 - 1 / 1000)) _ (by norm_num)
    (fun m => by rw [wrefDevRev1K_sm120v1Cast8p72_n4096k4096 m]; field_simp) (by norm_num) hTT

theorem pearlCSampledSm120v1Cast8p72Rev1InertShare1000_n4096k4096
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 4096 4096) ((1359 / 350750 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KNK CM _ sem _ _ _ _ (113643 / 113600 / (1 - 1 / 1000)) _ (by norm_num)
    wrefDevRev1K_sm120v1Cast8p72_n4096k4096_nonneg
    (fun m => by rw [wrefDevRev1K_sm120v1Cast8p72_n4096k4096 m]; field_simp) (by norm_num) hTT

theorem wrefDevRev1K_sm120v1Cast8p72_n28672k4096 (m : ℕ) :
    wrefDevRev1K (devSm120v1 Prices.sm120) (218 / 25 - 8) ⟨m, 28672, 4096⟩ =
      784443 / 784400 * creditDevRev1 (devSm120v1 Prices.sm120) ⟨m, 28672, 4096⟩ := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120).G = 4 from rfl,
    show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
  norm_num [Prices.sm120, Params.pi]
  ring

theorem wrefDevRev1K_sm120v1Cast8p72_n28672k4096_nonneg (m : ℕ) :
    0 ≤ wrefDevRev1K (devSm120v1 Prices.sm120) (218 / 25 - 8) ⟨m, 28672, 4096⟩ := by
  rw [wrefDevRev1K_sm120v1Cast8p72_n28672k4096]
  exact mul_nonneg (by norm_num) (creditDevRev1_sm120v1_nonneg _ _ _ (by norm_num) (by norm_num))

theorem pearlCGapSm120v1Cast8p72Rev1InertShare1000_n28672k4096
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 28672 4096) ((928813 / 261481000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KNK CM _ sem _ _ _ _ (784443 / 784400 / (1 - 1 / 1000)) _ (by norm_num)
    (fun m => by rw [wrefDevRev1K_sm120v1Cast8p72_n28672k4096 m]; field_simp) (by norm_num) hTT

theorem pearlCSampledSm120v1Cast8p72Rev1InertShare1000_n28672k4096
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 28672 4096) ((928813 / 261481000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KNK CM _ sem _ _ _ _ (784443 / 784400 / (1 - 1 / 1000)) _ (by norm_num)
    wrefDevRev1K_sm120v1Cast8p72_n28672k4096_nonneg
    (fun m => by rw [wrefDevRev1K_sm120v1Cast8p72_n28672k4096 m]; field_simp) (by norm_num) hTT

theorem wrefDevRev1K_sm120v1Cast8p72_n4096k14336 (m : ℕ) :
    wrefDevRev1K (devSm120v1 Prices.sm120) (218 / 25 - 8) ⟨m, 4096, 14336⟩ =
      780501 / 780200 * creditDevRev1 (devSm120v1 Prices.sm120) ⟨m, 4096, 14336⟩ := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120).G = 4 from rfl,
    show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
  norm_num [Prices.sm120, Params.pi]
  ring

theorem wrefDevRev1K_sm120v1Cast8p72_n4096k14336_nonneg (m : ℕ) :
    0 ≤ wrefDevRev1K (devSm120v1 Prices.sm120) (218 / 25 - 8) ⟨m, 4096, 14336⟩ := by
  rw [wrefDevRev1K_sm120v1Cast8p72_n4096k14336]
  exact mul_nonneg (by norm_num) (creditDevRev1_sm120v1_nonneg _ _ _ (by norm_num) (by norm_num))

theorem pearlCGapSm120v1Cast8p72Rev1InertShare1000_n4096k14336
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 4096 14336) ((106307 / 27386000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KNK CM _ sem _ _ _ _ (780501 / 780200 / (1 - 1 / 1000)) _ (by norm_num)
    (fun m => by rw [wrefDevRev1K_sm120v1Cast8p72_n4096k14336 m]; field_simp) (by norm_num) hTT

theorem pearlCSampledSm120v1Cast8p72Rev1InertShare1000_n4096k14336
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 4096 14336) ((106307 / 27386000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KNK CM _ sem _ _ _ _ (780501 / 780200 / (1 - 1 / 1000)) _ (by norm_num)
    wrefDevRev1K_sm120v1Cast8p72_n4096k14336_nonneg
    (fun m => by rw [wrefDevRev1K_sm120v1Cast8p72_n4096k14336 m]; field_simp) (by norm_num) hTT

theorem wrefDevRev1K_sm120v1Cast8p72_llama31_8b_nonneg :
    ∀ nk ∈ llama31_8bLinears, ∀ m : ℕ, 0 ≤ wrefDevRev1K (devSm120v1 Prices.sm120) (218 / 25 - 8) ⟨m, nk.1, nk.2⟩ := by
  intro nk hnk m
  simp only [llama31_8bLinears, mem_insert, mem_singleton] at hnk
  rcases hnk with rfl | rfl | rfl | rfl
  · exact wrefDevRev1K_sm120v1Cast8p72_n6144k4096_nonneg m
  · exact wrefDevRev1K_sm120v1Cast8p72_n4096k4096_nonneg m
  · exact wrefDevRev1K_sm120v1Cast8p72_n28672k4096_nonneg m
  · exact wrefDevRev1K_sm120v1Cast8p72_n4096k14336_nonneg m

theorem wrefDevRev1K_sm120v1Cast8p72_llama31_8b_le (ρ : ℚ) (hρ : ρ < 1) :
    ∀ nk ∈ llama31_8bLinears, ∀ m : ℕ,
      wrefDevRev1K (devSm120v1 Prices.sm120) (218 / 25 - 8) ⟨m, nk.1, nk.2⟩ ≤
        780501 / 780200 / (1 - ρ) * ((1 - ρ) * creditDevRev1 (devSm120v1 Prices.sm120) ⟨m, nk.1, nk.2⟩) := by
  intro nk hnk m
  simp only [llama31_8bLinears, mem_insert, mem_singleton] at hnk
  rcases hnk with rfl | rfl | rfl | rfl
  · rw [wrefDevRev1K_sm120v1Cast8p72_n6144k4096 m]
    exact ratio_le_max_div hρ (by norm_num) (creditDevRev1_sm120v1_nonneg _ _ _ (by norm_num) (by norm_num))
  · rw [wrefDevRev1K_sm120v1Cast8p72_n4096k4096 m]
    exact ratio_le_max_div hρ (by norm_num) (creditDevRev1_sm120v1_nonneg _ _ _ (by norm_num) (by norm_num))
  · rw [wrefDevRev1K_sm120v1Cast8p72_n28672k4096 m]
    exact ratio_le_max_div hρ (by norm_num) (creditDevRev1_sm120v1_nonneg _ _ _ (by norm_num) (by norm_num))
  · rw [wrefDevRev1K_sm120v1Cast8p72_n4096k14336 m]
    exact ratio_le_max_div hρ (by norm_num) (creditDevRev1_sm120v1_nonneg _ _ _ (by norm_num) (by norm_num))

theorem pearlCGapSm120v1Cast8p72Rev1InertShare1000_llama31_8b
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120) llama31_8bLinears)
      ((106307 / 27386000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KShapes CM _ sem _ _ _ (780501 / 780200 / (1 - 1 / 1000)) _ (by norm_num)
    (wrefDevRev1K_sm120v1Cast8p72_llama31_8b_le _ (by norm_num)) (by norm_num) hTT

theorem pearlCSampledSm120v1Cast8p72Rev1InertShare1000_llama31_8b
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120) llama31_8bLinears)
      ((106307 / 27386000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KShapes CM _ sem _ _ _ (780501 / 780200 / (1 - 1 / 1000)) _ (by norm_num)
    wrefDevRev1K_sm120v1Cast8p72_llama31_8b_nonneg
    (wrefDevRev1K_sm120v1Cast8p72_llama31_8b_le _ (by norm_num)) (by norm_num) hTT

end Served

end Pouw.PearlC
