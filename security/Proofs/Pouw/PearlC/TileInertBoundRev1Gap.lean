module

public import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
public import Definitions.Pouw.PearlC.DeviceKernelWref
public import Proofs.Pouw.PearlC.Records
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.KernelWrefGap
import Proofs.Pouw.PearlC.TileGap
import Proofs.Pouw.PearlC.UOnlyGap
import Proofs.Pouw.Prob
public import Proofs.Pouw.PearlC.DeviceInertBoundRev1Gap

/-! v1 under rev1 per audit tile at a stricter cap, which costs `g₀` per tile and nothing per unit. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

section
variable {Q R S : Type}

open Classical in
theorem rev1_tile_inert_bound_credit_sum_le (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρ' : ℚ) (hadd : 0 ≤ d.prices.add)
    (hbf : 0 ≤ d.prices.bf16) (hle : ρ' ≤ ρ) (hρ1 : ρ < 1) (U : Workload) (hD : pearlCDomainDev d U.layout)
    (H : Q → R) (s : S) (τ : Trace) :
    ∑ g ∈ goodTiles (pearlCProtocolDevRev1 d sem ρ') (pearlCTilesDevRev1 d sem ρ') U H s τ,
        (pearlCTilesDevRev1 d sem ρ').credit H s U g
          (actOf τ (((pearlCTilesDevRev1 d sem ρ').tiling U.layout).unit g)) ≤
      (((1 - ρ') / (1 - ρ) : ℚ) : ℝ) *
        ∑ g ∈ goodTiles (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) U H s τ,
          (pearlCTilesDevRev1 d sem ρ).credit H s U g
            (actOf τ (((pearlCTilesDevRev1 d sem ρ).tiling U.layout).unit g)) := by
  set tl := auditTiling 64 64 U.layout
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hC : ∀ g < tl.NT, 0 ≤ creditDevRev1 d (U.layout.shape (tl.unit g)) := fun g hg =>
    have hs := hD _ (auditTiling_unit_lt 64 64 U.layout hg)
    creditDevRev1_nonneg d hadd hbf _ hs.2.2.1 hs.2.2.2.2.1
  have hsub : goodTiles (pearlCProtocolDevRev1 d sem ρ') (pearlCTilesDevRev1 d sem ρ') U H s τ ⊆
      goodTiles (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) U H s τ := by
    intro g hg
    unfold goodTiles at hg ⊢
    rw [mem_filter] at hg ⊢
    obtain ⟨hg, c, hc, hcap, hw⟩ := hg
    refine ⟨hg, c, hc, ?_, hw⟩
    have h1 : sem.tileInertWorkRev1 d H s U (tl.unit g) (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g))
        (tl.cols g) c.act ≤ ρ' * creditDevRev1 d (U.layout.shape (tl.unit g)) *
          tileShare (U.layout.shape (tl.unit g)) (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g))
            (tl.cols g) := hcap
    show sem.tileInertWorkRev1 d H s U (tl.unit g) (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g))
        (tl.cols g) c.act ≤ ρ * creditDevRev1 d (U.layout.shape (tl.unit g)) *
          tileShare (U.layout.shape (tl.unit g)) (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g))
            (tl.cols g)
    have hsh := tileShare_nonneg (U.layout.shape (tl.unit g))
      (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g)) (tl.cols g)
    have := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hle (hC g (mem_range.mp hg))) hsh
    linarith
  rw [mul_sum]
  calc _ = ∑ g ∈ goodTiles (pearlCProtocolDevRev1 d sem ρ') (pearlCTilesDevRev1 d sem ρ') U H s τ,
          (((1 - ρ') / (1 - ρ) : ℚ) : ℝ) * (pearlCTilesDevRev1 d sem ρ).credit H s U g (actOf τ (tl.unit g)) := by
        refine sum_congr rfl fun g _ => ?_
        simp only [pearlCTilesDevRev1]
        have : (1 - ρ') * creditDevRev1 d (U.layout.shape (tl.unit g)) *
            tileShare (U.layout.shape (tl.unit g))
              (sem.passRows (U.layout.shape (tl.unit g)) (actOf τ (tl.unit g)) (tl.rows g)) (tl.cols g) =
            (1 - ρ') / (1 - ρ) * ((1 - ρ) * creditDevRev1 d (U.layout.shape (tl.unit g)) *
              tileShare (U.layout.shape (tl.unit g))
                (sem.passRows (U.layout.shape (tl.unit g)) (actOf τ (tl.unit g)) (tl.rows g)) (tl.cols g)) := by
          field_simp
        exact_mod_cast this
    _ ≤ _ := by
        refine sum_le_sum_of_subset_of_nonneg hsub fun g hg _ => ?_
        have hgr : g < tl.NT := by
          unfold goodTiles at hg
          exact mem_range.mp (mem_filter.mp hg).1
        simp only [pearlCTilesDevRev1]
        have hsh := tileShare_nonneg (U.layout.shape (tl.unit g))
          (sem.passRows (U.layout.shape (tl.unit g)) (actOf τ (tl.unit g)) (tl.rows g)) (tl.cols g)
        have hq : (0 : ℚ) ≤ (1 - ρ') / (1 - ρ) := div_nonneg (by linarith) hr.le
        have : (0 : ℚ) ≤ (1 - ρ') / (1 - ρ) * ((1 - ρ) * creditDevRev1 d (U.layout.shape (tl.unit g)) *
            tileShare (U.layout.shape (tl.unit g))
              (sem.passRows (U.layout.shape (tl.unit g)) (actOf τ (tl.unit g)) (tl.rows g)) (tl.cols g)) :=
          mul_nonneg hq (mul_nonneg (mul_nonneg hr.le (hC g hgr)) hsh)
        exact_mod_cast this


theorem ttOutTilePearlCDevRev1_inert_bound_transfer [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρ' : ℚ) (g₀ : ℝ) (δ : ℕ → ℕ → ℝ)
    (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hle : ρ' ≤ ρ) (hρ1 : ρ < 1)
    (h : TTOutTile CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDev d) g₀ δ) :
    TTOutTile CM (pearlCProtocolDevRev1 d sem ρ') (pearlCTilesDevRev1 d sem ρ') (pearlCDomainDev d)
      (1 - (1 - g₀) * (((1 - ρ) / (1 - ρ') : ℚ) : ℝ)) δ := by
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hr' : (0 : ℚ) < 1 - ρ' := by linarith
  have ha : (0 : ℝ) < (((1 - ρ) / (1 - ρ') : ℚ) : ℝ) := by exact_mod_cast div_pos hr hr'
  have hab : (((1 - ρ') / (1 - ρ) : ℚ) : ℝ) * (((1 - ρ) / (1 - ρ') : ℚ) : ℝ) = 1 := by
    rw [← Rat.cast_mul, show (1 - ρ') / (1 - ρ) * ((1 - ρ) / (1 - ρ')) = 1 by field_simp, Rat.cast_one]
  refine ⟨h.1, fun U hU T hT q hq A hA => le_trans (Pouw.pr_mono ?_) (h.2 U hU T hT q hq A hA)⟩
  intro x hx
  have hsum := rev1_tile_inert_bound_credit_sum_le d sem ρ ρ' hadd hbf hle hρ1 U hU.2 x.1 x.2 (A.trace U x.1 x.2)
  rw [show 1 - (1 - (1 - g₀) * (((1 - ρ) / (1 - ρ') : ℚ) : ℝ)) = (1 - g₀) * (((1 - ρ) / (1 - ρ') : ℚ) : ℝ) by ring,
    div_mul_eq_div_div] at hx
  have h2 := lt_of_lt_of_le hx hsum
  rw [div_lt_iff₀ ha] at h2
  calc T / (1 - g₀) < _ := h2
    _ = _ := by rw [mul_comm, ← mul_assoc, mul_comm (((1 - ρ) / (1 - ρ') : ℚ) : ℝ), hab, one_mul]

theorem pearlCSampledDevRev1KAt_gen [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (s : Shape) (ω : ℚ) (g₀ : ℝ)
    (hg₀ : g₀ < 1) (hω0 : 0 < ω)
    (hw0 : 0 ≤ wrefDevRev1K d c s) (hω : wrefDevRev1K d c s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hTT : TTOutTile CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDev d) g₀ δPearlC) :
    GGapSampled CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCTilesDevRev1K d sem ρ c) (pearlCDomainDevAt d s)
      (1 - (1 - g₀) / (ω : ℝ)) δPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevAt d s) →
      ∀ (H : Q → R) (s' : S), ∀ g < ((pearlCTilesDevRev1K d sem ρ c).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevRev1K d sem ρ c).inertBoundOK H s' U g act →
        (pearlCTilesDevRev1K d sem ρ c).Wcred U.layout g act ≤
          (ω : ℝ) * (pearlCTilesDevRev1K d sem ρ c).credit H s' U g act := by
    intro U hU H s' g hg act _
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.2 _ hu
    simp only [pearlCTilesDevRev1K, pearlCTilesDevRev1]
    rw [hsh, hω]
    push_cast
    exact le_of_eq (by ring)
  have hWc : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevAt d s) →
      ∀ g < ((pearlCTilesDevRev1K d sem ρ c).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevRev1K d sem ρ c).Wcred U.layout g act ≤ (pearlCTilesDevRev1K d sem ρ c).Wref U.layout g := by
    intro U hU g hg act
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.2 _ hu
    simp only [pearlCTilesDevRev1K, pearlCTilesDevRev1]
    rw [hsh]
    exact_mod_cast mul_le_mul_of_nonneg_left (tileShare_mono _ (passRows_subset sem _ _ _) _) hw0
  exact gapSampled CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCTilesDevRev1K d sem ρ c)
    (pearlCDomainDevAt d s) g₀ (ω : ℝ) δPearlC hg₀ (by exact_mod_cast hω0) hW hWc
    (ttOutTile_mono_domain (fun _ h => h.1) (ttOutTile_wref _ _ _ hTT))

theorem pearlCSampledDevRev1KAt_of_inert_bound [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρ' c : ℚ) (s : Shape) (ω g : ℚ)
    (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hle : ρ' ≤ ρ) (hρ1 : ρ < 1) (hω0 : 0 < ω)
    (hw0 : 0 ≤ wrefDevRev1K d c s) (hω : wrefDevRev1K d c s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hg : g = 1 - (1 - 1 / 400) / ω) (hTT : TTOutTilePearlCDevRev1 CM d sem ρ) :
    GGapSampled CM (pearlCProtocolDevRev1K d sem ρ' c) (pearlCTilesDevRev1K d sem ρ' c) (pearlCDomainDevAt d s)
      (g : ℝ) δPearlC := by
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hr' : (0 : ℚ) < 1 - ρ' := by linarith
  have key := pearlCSampledDevRev1KAt_gen CM d sem ρ' c s (ω * ((1 - ρ) / (1 - ρ'))) _
    (by
      have : (0 : ℝ) < (((1 - ρ) / (1 - ρ') : ℚ) : ℝ) := by exact_mod_cast div_pos hr hr'
      nlinarith)
    (mul_pos hω0 (div_pos hr hr')) hw0
    (by rw [hω]; field_simp)
    (ttOutTilePearlCDevRev1_inert_bound_transfer CM d sem ρ ρ' (1 / 400) δPearlC hadd hbf hle hρ1 hTT)
  have e : (g : ℝ) = 1 - (1 - (1 - (1 - 1 / 400) * (((1 - ρ) / (1 - ρ') : ℚ) : ℝ))) /
      ((ω * ((1 - ρ) / (1 - ρ')) : ℚ) : ℝ) := by
    rw [hg]
    have h1 : ((1 - ρ : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hr.ne'
    have h2 : ((1 - ρ' : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hr'.ne'
    have h3 : ((ω : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hω0.ne'
    push_cast at h1 h2 ⊢
    field_simp
    ring
  rw [e]
  exact key

end

section Gap

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
  (sem : PearlCSem Q R S)

theorem pearlCSampledSm120v1LoopCast8p72Rev1InertShare1000_8192_of_inert_share400
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
      ((926193839 / 178429100000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt_of_inert_bound CM _ sem (1 / 400) (1 / 1000) _ sh8192 (8921455 / 8919688 / (1 - 1 / 400)) _
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by norm_num) (by norm_num) (by norm_num) wrefDevRev1K_sm120v1LoopCast8p72_8192_nonneg
    (by rw [wrefDevRev1K_sm120v1LoopCast8p72_8192]; field_simp) (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast8p72Rev1InertShare1000_16384_of_inert_share400
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
      ((599303077 / 117663460000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt_of_inert_bound CM _ sem (1 / 400) (1 / 1000) _ sh16384 (5883173 / 5882584 / (1 - 1 / 400)) _
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by norm_num) (by norm_num) (by norm_num) wrefDevRev1K_sm120v1LoopCast8p72_16384_nonneg
    (by rw [wrefDevRev1K_sm120v1LoopCast8p72_16384]; field_simp) (by norm_num) hTT

end Gap

end Pouw.PearlC
