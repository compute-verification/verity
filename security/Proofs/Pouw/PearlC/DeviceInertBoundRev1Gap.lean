module

public import Proofs.Pouw.PearlC.DeviceSm120KernelGap
public import Proofs.Pouw.PearlC.UOnlyGap
public import Proofs.Pouw.Lemmas.PearlCRecords
public import Proofs.Pouw.PearlC.TTOutW1
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.KernelWrefGap
import Proofs.Pouw.Prob

/-! v1 under rev1 at a stricter cap, and g there (staged, nothing pinned); per audit tile the gap at 1/1,000 does
not follow from rev1 at 1/400. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

section
variable {Q R S : Type}

open Classical in
theorem inert_bound_credit_sum_le_rev1 (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρ' : ℚ) (hadd : 0 ≤ d.prices.add)
    (hbf : 0 ≤ d.prices.bf16) (hle : ρ' ≤ ρ) (hρ1 : ρ ≤ 1) (U : Workload) (hD : pearlCDomainDev d U.layout)
    (H : Q → R) (s : S) (τ : Trace) :
    ∑ u ∈ correctSet (pearlCProtocolDevRev1 d sem ρ') U H s τ,
        (pearlCProtocolDevRev1 d sem ρ').credit H s U u (actOf τ u) ≤
      ∑ u ∈ correctSet (pearlCProtocolDevRev1 d sem ρ) U H s τ,
        (pearlCProtocolDevRev1 d sem ρ).credit H s U u (actOf τ u) := by
  have hcr : ∀ u ∈ range U.layout.N, 0 ≤ creditDevRev1 d (U.layout.shape u) := fun u hu =>
    have hs := hD u (mem_range.mp hu)
    creditDevRev1_nonneg d hadd hbf _ hs.2.2.1 hs.2.2.2.2.1
  have hsub : correctSet (pearlCProtocolDevRev1 d sem ρ') U H s τ ⊆
      correctSet (pearlCProtocolDevRev1 d sem ρ) U H s τ := by
    intro u hu
    unfold correctSet at hu ⊢
    rw [mem_filter] at hu ⊢
    obtain ⟨hu, c, hc, hact, hcap, hw⟩ := hu
    refine ⟨hu, c, hc, hact, ?_, hw⟩
    have h1 : sem.unitInertWorkRev1 d H s U u c.act ≤ ρ' * creditDevRev1 d (U.layout.shape u) := hcap
    show sem.unitInertWorkRev1 d H s U u c.act ≤ ρ * creditDevRev1 d (U.layout.shape u)
    have := mul_le_mul_of_nonneg_right hle (hcr u hu)
    linarith
  refine sum_le_sum_of_subset_of_nonneg hsub fun u hu _ => ?_
  unfold correctSet at hu
  rw [mem_filter] at hu
  obtain ⟨hu, c, hc, _, hcap, _⟩ := hu
  have hact : actOf τ u = c.act := by simp [actOf, hc]
  rw [hact]
  have h1 : sem.unitInertWorkRev1 d H s U u c.act ≤ ρ * creditDevRev1 d (U.layout.shape u) := hcap
  have h2 := hcr u hu
  show (0 : ℝ) ≤ ((creditDevRev1 d (U.layout.shape u) - sem.unitInertWorkRev1 d H s U u c.act : ℚ) : ℝ)
  have : (0 : ℚ) ≤ creditDevRev1 d (U.layout.shape u) - sem.unitInertWorkRev1 d H s U u c.act := by nlinarith
  exact_mod_cast this

theorem ttOutPearlCDevRev1_inert_bound_anti [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρ' : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16)
    (hle : ρ' ≤ ρ) (hρ1 : ρ ≤ 1) (h : TTOutPearlCDevRev1 CM d sem ρ) : TTOutPearlCDevRev1 CM d sem ρ' := by
  refine ⟨h.1, fun U hU T hT q hq A hA => le_trans (Pouw.pr_mono ?_) (h.2 U hU T hT q hq A hA)⟩
  intro x hx
  exact lt_of_lt_of_le hx (inert_bound_credit_sum_le_rev1 d sem ρ ρ' hadd hbf hle hρ1 U hU.2 x.1 x.2 _)

end

section Gap

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
  (sem : PearlCSem Q R S)

theorem wrefDevRev1K_sm120v1LoopCast8p72_8192 :
    wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) sh8192 =
      8921455 / 8919688 * creditDevRev1 (devSm120v1 Prices.sm120Loop) sh8192 := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
    show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
  norm_num [Prices.sm120Loop, Params.pi, sh8192]

theorem wrefDevRev1K_sm120v1LoopCast8p72_8192_nonneg :
    0 ≤ wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) sh8192 := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
    show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
  norm_num [Prices.sm120Loop, Params.pi, sh8192]

theorem wrefDevRev1K_sm120v1LoopCast8p72_16384 :
    wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) sh16384 =
      5883173 / 5882584 * creditDevRev1 (devSm120v1 Prices.sm120Loop) sh16384 := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
    show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
  norm_num [Prices.sm120Loop, Params.pi, sh16384]

theorem wrefDevRev1K_sm120v1LoopCast8p72_16384_nonneg :
    0 ≤ wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) sh16384 := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
    show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
  norm_num [Prices.sm120Loop, Params.pi, sh16384]

theorem pearlCGapSm120v1LoopCast8p72Rev1InertShare_8192 (ρ : ℚ) (hρ : ρ < 1)
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem ρ) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem ρ (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
      ((1 - (1 - ρ) * (444869439 / 446072750) : ℚ) : ℝ) δPearlC := by
  have h1 : (1 : ℚ) - ρ ≠ 0 := by linarith
  refine pearlCGapDevRev1KAt CM _ sem ρ _ sh8192 (8921455 / 8919688 / (1 - ρ)) _
    (div_pos (by norm_num) (by linarith)) ?_ ?_ hTT
  · rw [wrefDevRev1K_sm120v1LoopCast8p72_8192]
    field_simp
  · field_simp
    ring

theorem pearlCSampledSm120v1LoopCast8p72Rev1InertShare_8192 (ρ : ℚ) (hρ : ρ < 1)
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem ρ) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem ρ (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem ρ (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
      ((1 - (1 - ρ) * (444869439 / 446072750) : ℚ) : ℝ) δPearlC := by
  have h1 : (1 : ℚ) - ρ ≠ 0 := by linarith
  refine pearlCSampledDevRev1KAt CM _ sem ρ _ sh8192 (8921455 / 8919688 / (1 - ρ)) _
    (div_pos (by norm_num) (by linarith)) wrefDevRev1K_sm120v1LoopCast8p72_8192_nonneg ?_ ?_ hTT
  · rw [wrefDevRev1K_sm120v1LoopCast8p72_8192]
    field_simp
  · field_simp
    ring

theorem pearlCGapSm120v1LoopCast8p72Rev1InertShare_16384 (ρ : ℚ) (hρ : ρ < 1)
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem ρ) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem ρ (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
      ((1 - (1 - ρ) * (293393877 / 294158650) : ℚ) : ℝ) δPearlC := by
  have h1 : (1 : ℚ) - ρ ≠ 0 := by linarith
  refine pearlCGapDevRev1KAt CM _ sem ρ _ sh16384 (5883173 / 5882584 / (1 - ρ)) _
    (div_pos (by norm_num) (by linarith)) ?_ ?_ hTT
  · rw [wrefDevRev1K_sm120v1LoopCast8p72_16384]
    field_simp
  · field_simp
    ring

theorem pearlCSampledSm120v1LoopCast8p72Rev1InertShare_16384 (ρ : ℚ) (hρ : ρ < 1)
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem ρ) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem ρ (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem ρ (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
      ((1 - (1 - ρ) * (293393877 / 294158650) : ℚ) : ℝ) δPearlC := by
  have h1 : (1 : ℚ) - ρ ≠ 0 := by linarith
  refine pearlCSampledDevRev1KAt CM _ sem ρ _ sh16384 (5883173 / 5882584 / (1 - ρ)) _
    (div_pos (by norm_num) (by linarith)) wrefDevRev1K_sm120v1LoopCast8p72_16384_nonneg ?_ ?_ hTT
  · rw [wrefDevRev1K_sm120v1LoopCast8p72_16384]
    field_simp
  · field_simp
    ring

theorem pearlCGapSm120v1LoopCast8p72Rev1InertShare1000_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
      ((1648180439 / 446072750000 : ℚ) : ℝ) δPearlC := by
  rw [show (1648180439 / 446072750000 : ℚ) = 1 - (1 - 1 / 1000) * (444869439 / 446072750) by norm_num]
  exact pearlCGapSm120v1LoopCast8p72Rev1InertShare_8192 CM sem (1 / 1000) (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast8p72Rev1InertShare1000_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
      ((1648180439 / 446072750000 : ℚ) : ℝ) δPearlC := by
  rw [show (1648180439 / 446072750000 : ℚ) = 1 - (1 - 1 / 1000) * (444869439 / 446072750) by norm_num]
  exact pearlCSampledSm120v1LoopCast8p72Rev1InertShare_8192 CM sem (1 / 1000) (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast8p72Rev1InertShare1000_8192_of_inert_share400
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
      ((1648180439 / 446072750000 : ℚ) : ℝ) δPearlC :=
  pearlCGapSm120v1LoopCast8p72Rev1InertShare1000_8192 CM sem
    (ttOutPearlCDevRev1_inert_bound_anti CM _ sem (1 / 400) (1 / 1000)
      (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
      (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
      (by norm_num) (by norm_num) hTT)

theorem pearlCGapSm120v1LoopCast8p72Rev1InertShare1000_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
      ((1058166877 / 294158650000 : ℚ) : ℝ) δPearlC := by
  rw [show (1058166877 / 294158650000 : ℚ) = 1 - (1 - 1 / 1000) * (293393877 / 294158650) by norm_num]
  exact pearlCGapSm120v1LoopCast8p72Rev1InertShare_16384 CM sem (1 / 1000) (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast8p72Rev1InertShare1000_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
      ((1058166877 / 294158650000 : ℚ) : ℝ) δPearlC := by
  rw [show (1058166877 / 294158650000 : ℚ) = 1 - (1 - 1 / 1000) * (293393877 / 294158650) by norm_num]
  exact pearlCSampledSm120v1LoopCast8p72Rev1InertShare_16384 CM sem (1 / 1000) (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast8p72Rev1InertShare1000_16384_of_inert_share400
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
      ((1058166877 / 294158650000 : ℚ) : ℝ) δPearlC :=
  pearlCGapSm120v1LoopCast8p72Rev1InertShare1000_16384 CM sem
    (ttOutPearlCDevRev1_inert_bound_anti CM _ sem (1 / 400) (1 / 1000)
      (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
      (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
      (by norm_num) (by norm_num) hTT)

end Gap

end Pouw.PearlC

theorem Pouw.SecurityProofs.PearlC.GapSm120v1LoopCast8p72Rev1InertShare1000_16384OfInertShare400 : Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1InertShare1000_16384OfInertShare400 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1LoopCast8p72Rev1InertShare1000_16384_of_inert_share400 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1LoopCast8p72Rev1InertShare1000_8192OfInertShare400 : Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1InertShare1000_8192OfInertShare400 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1LoopCast8p72Rev1InertShare1000_8192_of_inert_share400 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)
