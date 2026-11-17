module

public import Proofs.Pouw.PearlC.DeviceRev1Gap
public import Proofs.Pouw.Lemmas.PearlCRecords
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.DeviceGap
import Proofs.Pouw.PearlC.TTOutW1
import Proofs.Pouw.Prob
public import Proofs.Pouw.PearlC.Records

/-! The unpromoted record, the cap, and g at sm_120's prices (staged, nothing pinned); a stricter cap asserts less
per unit, and per tile there is no such theorem. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

theorem pearlCDomainDev_unpromoted (d : PearlCDevice) (hG : d.G = 0) :
    pearlCDomainDev d ⟨1, fun _ => ⟨1, 64, 128⟩, fun _ => 0⟩ := by
  intro u _
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, Or.inl hG, by norm_num⟩

section
variable {Q R S : Type}

open Classical in
theorem inert_bound_credit_sum_le (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρ' : ℚ) (hadd : 0 ≤ d.prices.add)
    (hbf : 0 ≤ d.prices.bf16) (hle : ρ' ≤ ρ) (hρ1 : ρ ≤ 1) (U : Workload) (H : Q → R) (s : S)
    (τ : Trace) :
    ∑ u ∈ correctSet (pearlCProtocolDev d sem ρ') U H s τ, (pearlCProtocolDev d sem ρ').credit H s U u (actOf τ u) ≤
      ∑ u ∈ correctSet (pearlCProtocolDev d sem ρ) U H s τ, (pearlCProtocolDev d sem ρ).credit H s U u (actOf τ u) := by
  have hsub : correctSet (pearlCProtocolDev d sem ρ') U H s τ ⊆ correctSet (pearlCProtocolDev d sem ρ) U H s τ := by
    intro u hu
    unfold correctSet at hu ⊢
    rw [mem_filter] at hu ⊢
    obtain ⟨hu, c, hc, hact, hcap, hw⟩ := hu
    refine ⟨hu, c, hc, hact, ?_, hw⟩
    have h1 : inertWorkDev d (sem.unitInertWorkDev d H s U u c.act) ≤ ρ' * creditDev d (U.layout.shape u) := hcap
    show inertWorkDev d (sem.unitInertWorkDev d H s U u c.act) ≤ ρ * creditDev d (U.layout.shape u)
    have := mul_le_mul_of_nonneg_right hle (creditDev_nonneg d hadd hbf (U.layout.shape u))
    linarith
  refine sum_le_sum_of_subset_of_nonneg hsub fun u hu _ => ?_
  unfold correctSet at hu
  rw [mem_filter] at hu
  obtain ⟨_, c, hc, _, hcap, _⟩ := hu
  have hact : actOf τ u = c.act := by simp [actOf, hc]
  rw [hact]
  have h1 : inertWorkDev d (sem.unitInertWorkDev d H s U u c.act) ≤ ρ * creditDev d (U.layout.shape u) := hcap
  have h2 := creditDev_nonneg d hadd hbf (U.layout.shape u)
  show (0 : ℝ) ≤ ((creditDev d (U.layout.shape u) - inertWorkDev d (sem.unitInertWorkDev d H s U u c.act) : ℚ) : ℝ)
  have : (0 : ℚ) ≤ creditDev d (U.layout.shape u) - inertWorkDev d (sem.unitInertWorkDev d H s U u c.act) := by nlinarith
  exact_mod_cast this

theorem ttOutPearlCDev_inert_bound_anti [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρ' : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16)
    (hle : ρ' ≤ ρ) (hρ1 : ρ ≤ 1) (h : TTOutPearlCDev CM d sem ρ) : TTOutPearlCDev CM d sem ρ' := by
  refine ⟨h.1, fun U hU T hT q hq A hA => le_trans (Pouw.pr_mono ?_) (h.2 U hU T hT q hq A hA)⟩
  intro x hx
  exact lt_of_lt_of_le hx (inert_bound_credit_sum_le d sem ρ ρ' hadd hbf hle hρ1 U x.1 x.2 _)

end

theorem pearlCGapUnpromotedInertShare1000_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120)
    (hTT : TTOutPearlCDev CM d sem (1 / 1000)) :
    GGap CM (pearlCProtocolDev d sem (1 / 1000)) (pearlCDomainDevAt d sh8192) ((216793 / 59950000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevAt CM d sem _ _ (1049125 / 1047951) _ (by norm_num)
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192]) (by norm_num) hTT

theorem pearlCGapUnpromotedInertShare1000_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120)
    (hTT : TTOutPearlCDev CM d sem (1 / 1000)) :
    GGap CM (pearlCProtocolDev d sem (1 / 1000)) (pearlCDomainDevAt d sh16384) ((2950127 / 829250000 : ℚ) : ℝ)
      δPearlC :=
  pearlCGapDevAt CM d sem _ _ (2073125 / 2070927) _ (by norm_num)
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh16384]) (by norm_num) hTT

theorem pearlCSampledUnpromotedInertShare1000_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120)
    (hTT : TTOutTilePearlCDev CM d sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDev d sem (1 / 1000)) (pearlCTilesDev d sem (1 / 1000)) (pearlCDomainDevAt d sh8192)
      ((216793 / 59950000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevAt CM d sem _ _ (1049125 / 1047951) _ (by norm_num)
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192]) (by norm_num) hTT

theorem pearlCSampledUnpromotedInertShare1000_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120)
    (hTT : TTOutTilePearlCDev CM d sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDev d sem (1 / 1000)) (pearlCTilesDev d sem (1 / 1000)) (pearlCDomainDevAt d sh16384)
      ((2950127 / 829250000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevAt CM d sem _ _ (2073125 / 2070927) _ (by norm_num)
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh16384]) (by norm_num) hTT

theorem pearlCGapSm120Rev1_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4) (hp : d.prices = Prices.sm120)
    (hTT : TTOutPearlCDevRev1 CM d sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCDomainDevAt d sh8192) ((16223 / 3177500 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1At CM d sem _ _ (31775 / 31692) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120Rev1_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4) (hp : d.prices = Prices.sm120)
    (hTT : TTOutPearlCDevRev1 CM d sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCDomainDevAt d sh16384) ((2963 / 586700 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1At CM d sem _ _ (5867 / 5852) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120Rev1_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4) (hp : d.prices = Prices.sm120)
    (hTT : TTOutTilePearlCDevRev1 CM d sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCTilesDevRev1 d sem (1 / 400))
      (pearlCDomainDevAt d sh8192) ((16223 / 3177500 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1At CM d sem _ _ (31775 / 31692) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120Rev1_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4) (hp : d.prices = Prices.sm120)
    (hTT : TTOutTilePearlCDevRev1 CM d sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCTilesDevRev1 d sem (1 / 400))
      (pearlCDomainDevAt d sh16384) ((2963 / 586700 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1At CM d sem _ _ (5867 / 5852) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

end Pouw.PearlC

theorem Pouw.SecurityProofs.PearlC.GapUnpromotedInertShare1000_8192 : Pouw.Guarantees.PearlC.GapUnpromotedInertShare1000_8192 :=
  fun sha fetch salt d sem hG hp hTT =>
    @Pouw.PearlC.pearlCGapUnpromotedInertShare1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) d sem hG hp
      (Pouw.PearlC.TTOutW1.ttOutPearlCDev_of_w1 hTT)
