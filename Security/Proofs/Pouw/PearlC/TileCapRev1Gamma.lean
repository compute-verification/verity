import Proofs.Pouw.PearlC.DeviceCapRev1Gamma

/-!
# v1 under rev1 per audit tile at a stricter cap (compute accounting's file)

* **A stricter cap per tile costs `γ₀`:** `ttOutTilePearlCDevRev1_cap_transfer` gives TT_OUT per tile under rev1 at
  `ρ' ≤ ρ < 1` from it at `ρ`, with `1 − γ₀' = (1 − γ₀)·(1 − ρ)/(1 − ρ')`, at any record with non-negative add and BF16
  prices. A good tile at `ρ'` is good at `ρ`, but its credit is the cap's worst case, `(1 − ρ)·C` times its share, not
  its replayed debit, so it grows by `(1 − ρ')/(1 − ρ)`. Per unit the credit is debited, and the stricter cap costs
  nothing (`ttOutPearlCDevRev1_cap_anti`).
* **The sampled γ at `ρ'` from TT_OUT per tile at `ρ`** (`pearlCSampledDevRev1KAt_of_cap`) is the γ at `ρ`,
  `1 − (399/400)/ω` with `wrefDevRev1K = ω·(1 − ρ)·creditDevRev1`.
* From `TTOutTilePearlCDevRev1` at 1/400 the sampled γ at 1/1,000 equals 1/400's (0.51908% at 8192³, 0.50934% at
  16384³), so the headline 0.36949% rests on `TTOutTilePearlCDevRev1` at 1/1,000
  (`pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_8192_of_cap400`, `_16384_of_cap400`).
-/

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

section
variable {Q R S : Type}

open Classical in
theorem rev1_tile_cap_credit_sum_le (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρ' : ℚ) (hadd : 0 ≤ d.prices.add)
    (hbf : 0 ≤ d.prices.bf16) (hle : ρ' ≤ ρ) (hρ1 : ρ < 1) (U : Workload) (hD : pearlCDomainDev d U.layout)
    (H : Q → R) (s : S) (τ : CommitLog) :
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
    have h1 : sem.tileDebitRev1 d H s U (tl.unit g) (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g))
        (tl.cols g) c.act ≤ ρ' * creditDevRev1 d (U.layout.shape (tl.unit g)) *
          tileShare (U.layout.shape (tl.unit g)) (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g))
            (tl.cols g) := hcap
    show sem.tileDebitRev1 d H s U (tl.unit g) (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g))
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


/-- **A stricter cap per audit tile under rev1, at a larger `γ₀`.** TT_OUT per tile under rev1 at a cap `ρ < 1` and any
`γ₀` gives it at any cap `ρ' ≤ ρ` with `1 − γ₀' = (1 − γ₀)·(1 − ρ)/(1 − ρ')`, at any record with non-negative add and
BF16 prices. -/
theorem ttOutTilePearlCDevRev1_cap_transfer [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρ' : ℚ) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ)
    (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hle : ρ' ≤ ρ) (hρ1 : ρ < 1)
    (h : TTOutTile CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDev d) γ₀ ε) :
    TTOutTile CM (pearlCProtocolDevRev1 d sem ρ') (pearlCTilesDevRev1 d sem ρ') (pearlCDomainDev d)
      (1 - (1 - γ₀) * (((1 - ρ) / (1 - ρ') : ℚ) : ℝ)) ε := by
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hr' : (0 : ℚ) < 1 - ρ' := by linarith
  have ha : (0 : ℝ) < (((1 - ρ) / (1 - ρ') : ℚ) : ℝ) := by exact_mod_cast div_pos hr hr'
  have hab : (((1 - ρ') / (1 - ρ) : ℚ) : ℝ) * (((1 - ρ) / (1 - ρ') : ℚ) : ℝ) = 1 := by
    rw [← Rat.cast_mul, show (1 - ρ') / (1 - ρ) * ((1 - ρ) / (1 - ρ')) = 1 by field_simp, Rat.cast_one]
  refine ⟨h.1, fun U hU T hT q hq A hA => le_trans (Pouw.pr_mono ?_) (h.2 U hU T hT q hq A hA)⟩
  intro x hx
  have hsum := rev1_tile_cap_credit_sum_le d sem ρ ρ' hadd hbf hle hρ1 U hU.2 x.1 x.2 (A.transcript U x.1 x.2)
  rw [show 1 - (1 - (1 - γ₀) * (((1 - ρ) / (1 - ρ') : ℚ) : ℝ)) = (1 - γ₀) * (((1 - ρ) / (1 - ρ') : ℚ) : ℝ) by ring,
    div_mul_eq_div_div] at hx
  have h2 := lt_of_lt_of_le hx hsum
  rw [div_lt_iff₀ ha] at h2
  calc T / (1 - γ₀) < _ := h2
    _ = _ := by rw [mul_comm, ← mul_assoc, mul_comm (((1 - ρ) / (1 - ρ') : ℚ) : ℝ), hab, one_mul]

/-- `pearlCSampledDevRev1KAt` at any `γ₀ < 1`. -/
theorem pearlCSampledDevRev1KAt_gen [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (s : Shape) (ω : ℚ) (γ₀ : ℝ)
    (hγ₀ : γ₀ < 1) (hω0 : 0 < ω)
    (hw0 : 0 ≤ wrefDevRev1K d c s) (hω : wrefDevRev1K d c s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hTT : TTOutTile CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDev d) γ₀ εPearlC) :
    GγSampled CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCTilesDevRev1K d sem ρ c) (pearlCDomainDevAt d s)
      (1 - (1 - γ₀) / (ω : ℝ)) εPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevAt d s) →
      ∀ (H : Q → R) (s' : S), ∀ g < ((pearlCTilesDevRev1K d sem ρ c).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevRev1K d sem ρ c).capOK H s' U g act →
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
  exact gammaSampled CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCTilesDevRev1K d sem ρ c)
    (pearlCDomainDevAt d s) γ₀ (ω : ℝ) εPearlC hγ₀ (by exact_mod_cast hω0) hW hWc
    (ttOutTile_mono_domain (fun _ h => h.1) (ttOutTile_wref _ _ _ hTT))

/-- The sampled γ at a cap `ρ' ≤ ρ < 1` from rev1's TT_OUT per tile at `ρ`, on the one-shape domain: the γ at `ρ`,
`1 − (399/400)/ω` where `wrefDevRev1K d c = ω·(1 − ρ)·creditDevRev1 ≥ 0`. -/
theorem pearlCSampledDevRev1KAt_of_cap [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρ' c : ℚ) (s : Shape) (ω γ : ℚ)
    (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hle : ρ' ≤ ρ) (hρ1 : ρ < 1) (hω0 : 0 < ω)
    (hw0 : 0 ≤ wrefDevRev1K d c s) (hω : wrefDevRev1K d c s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hγ : γ = 1 - (1 - 1 / 400) / ω) (hTT : TTOutTilePearlCDevRev1 CM d sem ρ) :
    GγSampled CM (pearlCProtocolDevRev1K d sem ρ' c) (pearlCTilesDevRev1K d sem ρ' c) (pearlCDomainDevAt d s)
      (γ : ℝ) εPearlC := by
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hr' : (0 : ℚ) < 1 - ρ' := by linarith
  have key := pearlCSampledDevRev1KAt_gen CM d sem ρ' c s (ω * ((1 - ρ) / (1 - ρ'))) _
    (by
      have : (0 : ℝ) < (((1 - ρ) / (1 - ρ') : ℚ) : ℝ) := by exact_mod_cast div_pos hr hr'
      nlinarith)
    (mul_pos hω0 (div_pos hr hr')) hw0
    (by rw [hω]; field_simp)
    (ttOutTilePearlCDevRev1_cap_transfer CM d sem ρ ρ' (1 / 400) εPearlC hadd hbf hle hρ1 hTT)
  have e : (γ : ℝ) = 1 - (1 - (1 - (1 - 1 / 400) * (((1 - ρ) / (1 - ρ') : ℚ) : ℝ))) /
      ((ω * ((1 - ρ) / (1 - ρ')) : ℚ) : ℝ) := by
    rw [hγ]
    have h1 : ((1 - ρ : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hr.ne'
    have h2 : ((1 - ρ' : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hr'.ne'
    have h3 : ((ω : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hω0.ne'
    push_cast at h1 h2 ⊢
    field_simp
    ring
  rw [e]
  exact key

end

section Gamma

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
  (sem : PearlCSem Q R S)

/-- **v1 under rev1 per audit tile at the cap 1/1,000 from rev1 per tile at 1/400, 8192³**, FP32 at 8.376, the honest
cast at 8.72: `γ = 926193839/178429100000` (0.51908%), the γ at 1/400. -/
theorem pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_8192_of_cap400
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
      ((926193839 / 178429100000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1KAt_of_cap CM _ sem (1 / 400) (1 / 1000) _ sh8192 (8921455 / 8919688 / (1 - 1 / 400)) _
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by norm_num) (by norm_num) (by norm_num) wrefDevRev1K_sm120v1LoopCast8p72_8192_nonneg
    (by rw [wrefDevRev1K_sm120v1LoopCast8p72_8192]; field_simp) (by norm_num) hTT

/-- **v1 under rev1 per audit tile at the cap 1/1,000 from rev1 per tile at 1/400, 16384³**, FP32 at 8.376, the honest
cast at 8.72: `γ = 599303077/117663460000` (0.50934%), the γ at 1/400. -/
theorem pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_16384_of_cap400
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
      ((599303077 / 117663460000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1KAt_of_cap CM _ sem (1 / 400) (1 / 1000) _ sh16384 (5883173 / 5882584 / (1 - 1 / 400)) _
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by norm_num) (by norm_num) (by norm_num) wrefDevRev1K_sm120v1LoopCast8p72_16384_nonneg
    (by rw [wrefDevRev1K_sm120v1LoopCast8p72_16384]; field_simp) (by norm_num) hTT

end Gamma

end Pouw.PearlC
