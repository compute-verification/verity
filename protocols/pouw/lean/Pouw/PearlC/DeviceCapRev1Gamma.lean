import Pouw.PearlC.DeviceSm120Gamma

/-!
# v1 under rev1 at a stricter cap, and γ there (FP8 security's file)

Staged; nothing pinned.
* **A stricter cap asserts less under rev1, per unit:** `ttOutPearlCDevRev1_cap_anti` gives `TTOutPearlCDevRev1` at `ρ'`
  from it at `ρ`, for `ρ' ≤ ρ ≤ 1` at any record with non-negative add and BF16 prices: `DeviceCapGamma`'s
  `ttOutPearlCDev_cap_anti` with rev1's credit and cap. rev1's credit can be negative off the domain, so the sum's bound
  (`cap_credit_sum_le_rev1`) takes the workload's layout in `pearlCDomainDev`, where `creditDevRev1_nonneg` holds. Per
  tile there is no such theorem, for `DeviceCapGamma`'s reason.
* **γ at v1's record under rev1 at any cap `ρ < 1`** (`devSm120v1 Prices.sm120`): `1 − (1 − ρ)·31692/31775` at 8192³ and
  `1 − (1 − ρ)·5852/5867` at 16384³, per unit and per audit tile. At `ρ = 1/400` these are `DeviceSm120Gamma`'s
  `16223/3177500` and `2963/586700`.
* **At the cap 1/1,000** (`tt-out/pearl-c-sm120-rev1-cap1000` and its tile twin): `28673/7943750` (0.36095%) at 8192³ and
  `5213/1466750` (0.35541%) at 16384³, per unit and per audit tile. Per unit the γ also follows from rev1 at 1/400
  (`tt-out/pearl-c-sm120-rev1`) alone (`pearlCGammaSm120v1Rev1Cap1000_8192_of_cap400`, `_16384_of_cap400`).
-/

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

section
variable {Q R S : Type}

open Classical in
theorem cap_credit_sum_le_rev1 (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρ' : ℚ) (hadd : 0 ≤ d.prices.add)
    (hbf : 0 ≤ d.prices.bf16) (hle : ρ' ≤ ρ) (hρ1 : ρ ≤ 1) (U : Workload) (hD : pearlCDomainDev d U.layout)
    (H : Q → R) (s : S) (τ : Transcript) :
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
    have h1 : sem.unitDebitRev1 d H s U u c.act ≤ ρ' * creditDevRev1 d (U.layout.shape u) := hcap
    show sem.unitDebitRev1 d H s U u c.act ≤ ρ * creditDevRev1 d (U.layout.shape u)
    have := mul_le_mul_of_nonneg_right hle (hcr u hu)
    linarith
  refine sum_le_sum_of_subset_of_nonneg hsub fun u hu _ => ?_
  unfold correctSet at hu
  rw [mem_filter] at hu
  obtain ⟨hu, c, hc, _, hcap, _⟩ := hu
  have hact : actOf τ u = c.act := by simp [actOf, hc]
  rw [hact]
  have h1 : sem.unitDebitRev1 d H s U u c.act ≤ ρ * creditDevRev1 d (U.layout.shape u) := hcap
  have h2 := hcr u hu
  show (0 : ℝ) ≤ ((creditDevRev1 d (U.layout.shape u) - sem.unitDebitRev1 d H s U u c.act : ℚ) : ℝ)
  have : (0 : ℚ) ≤ creditDevRev1 d (U.layout.shape u) - sem.unitDebitRev1 d H s U u c.act := by nlinarith
  exact_mod_cast this

/-- **A stricter cap asserts less under rev1**: rev1's TT_OUT at a device record at cap `ρ` gives it at any cap
`ρ' ≤ ρ ≤ 1`. -/
theorem ttOutPearlCDevRev1_cap_anti [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρ' : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16)
    (hle : ρ' ≤ ρ) (hρ1 : ρ ≤ 1) (h : TTOutPearlCDevRev1 CM d sem ρ) : TTOutPearlCDevRev1 CM d sem ρ' := by
  refine ⟨h.1, fun U hU T hT q hq A hA => le_trans (Pouw.pr_mono ?_) (h.2 U hU T hT q hq A hA)⟩
  intro x hx
  exact lt_of_lt_of_le hx (cap_credit_sum_le_rev1 d sem ρ ρ' hadd hbf hle hρ1 U hU.2.1 x.1 x.2 _)

end

section Gamma

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
  (sem : PearlCSem Q R S)

theorem wrefDevRev1_sm120v1_8192 :
    wrefDevRev1 (devSm120v1 Prices.sm120) sh8192 = 8897 / 8896 * creditDevRev1 (devSm120v1 Prices.sm120) sh8192 := by
  rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devSm120v1_G, devSm120v1_prices]
  norm_num [Prices.sm120, Params.pi, sh8192]

theorem wrefDevRev1_sm120v1_16384 :
    wrefDevRev1 (devSm120v1 Prices.sm120) sh16384 =
      17601 / 17600 * creditDevRev1 (devSm120v1 Prices.sm120) sh16384 := by
  rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devSm120v1_G, devSm120v1_prices]
  norm_num [Prices.sm120, Params.pi, sh16384]

theorem wrefDevRev1_sm120v1_8192_nonneg : 0 ≤ wrefDevRev1 (devSm120v1 Prices.sm120) sh8192 := by
  rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devSm120v1_G, devSm120v1_prices]
  norm_num [Prices.sm120, Params.pi, sh8192]

theorem wrefDevRev1_sm120v1_16384_nonneg : 0 ≤ wrefDevRev1 (devSm120v1 Prices.sm120) sh16384 := by
  rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devSm120v1_G, devSm120v1_prices]
  norm_num [Prices.sm120, Params.pi, sh16384]

/-! ## v1 under rev1 at a cap `ρ < 1` -/

/-- **v1 under rev1 at a cap `ρ < 1`, 8192³**: `γ = 1 − (1 − ρ)·31692/31775`. -/
theorem pearlCGammaSm120v1Rev1Cap_8192 (ρ : ℚ) (hρ : ρ < 1)
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem ρ) :
    Gγ CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem ρ)
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((1 - (1 - ρ) * (31692 / 31775) : ℚ) : ℝ) εPearlC := by
  have h1 : (1 : ℚ) - ρ ≠ 0 := by linarith
  refine pearlCGammaDevRev1At CM _ sem ρ sh8192 (8897 / 8896 / (1 - ρ)) _ (div_pos (by norm_num) (by linarith)) ?_ ?_ hTT
  · rw [wrefDevRev1_sm120v1_8192]
    field_simp
  · field_simp
    ring

/-- **v1 under rev1 at a cap `ρ < 1`, 16384³**: `γ = 1 − (1 − ρ)·5852/5867`. -/
theorem pearlCGammaSm120v1Rev1Cap_16384 (ρ : ℚ) (hρ : ρ < 1)
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem ρ) :
    Gγ CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem ρ)
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((1 - (1 - ρ) * (5852 / 5867) : ℚ) : ℝ) εPearlC := by
  have h1 : (1 : ℚ) - ρ ≠ 0 := by linarith
  refine pearlCGammaDevRev1At CM _ sem ρ sh16384 (17601 / 17600 / (1 - ρ)) _ (div_pos (by norm_num) (by linarith)) ?_ ?_
    hTT
  · rw [wrefDevRev1_sm120v1_16384]
    field_simp
  · field_simp
    ring

/-- **v1 under rev1 per audit tile at a cap `ρ < 1`, 8192³**: `γ = 1 − (1 − ρ)·31692/31775`. -/
theorem pearlCSampledSm120v1Rev1Cap_8192 (ρ : ℚ) (hρ : ρ < 1)
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem ρ) :
    GγSampled CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem ρ)
      (pearlCTilesDevRev1 (devSm120v1 Prices.sm120) sem ρ)
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((1 - (1 - ρ) * (31692 / 31775) : ℚ) : ℝ) εPearlC := by
  have h1 : (1 : ℚ) - ρ ≠ 0 := by linarith
  refine pearlCSampledDevRev1At CM _ sem ρ sh8192 (8897 / 8896 / (1 - ρ)) _ (div_pos (by norm_num) (by linarith))
    wrefDevRev1_sm120v1_8192_nonneg ?_ ?_ hTT
  · rw [wrefDevRev1_sm120v1_8192]
    field_simp
  · field_simp
    ring

/-- **v1 under rev1 per audit tile at a cap `ρ < 1`, 16384³**: `γ = 1 − (1 − ρ)·5852/5867`. -/
theorem pearlCSampledSm120v1Rev1Cap_16384 (ρ : ℚ) (hρ : ρ < 1)
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem ρ) :
    GγSampled CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem ρ)
      (pearlCTilesDevRev1 (devSm120v1 Prices.sm120) sem ρ)
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((1 - (1 - ρ) * (5852 / 5867) : ℚ) : ℝ) εPearlC := by
  have h1 : (1 : ℚ) - ρ ≠ 0 := by linarith
  refine pearlCSampledDevRev1At CM _ sem ρ sh16384 (17601 / 17600 / (1 - ρ)) _ (div_pos (by norm_num) (by linarith))
    wrefDevRev1_sm120v1_16384_nonneg ?_ ?_ hTT
  · rw [wrefDevRev1_sm120v1_16384]
    field_simp
  · field_simp
    ring

/-! ## v1 under rev1 at the cap 1/1,000 -/

/-- **v1 under rev1 at the cap 1/1,000, 8192³**: `γ = 28673/7943750`. -/
theorem pearlCGammaSm120v1Rev1Cap1000_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((28673 / 7943750 : ℚ) : ℝ) εPearlC := by
  rw [show (28673 / 7943750 : ℚ) = 1 - (1 - 1 / 1000) * (31692 / 31775) by norm_num]
  exact pearlCGammaSm120v1Rev1Cap_8192 CM sem (1 / 1000) (by norm_num) hTT

/-- **v1 under rev1 at the cap 1/1,000, 16384³**: `γ = 5213/1466750`. -/
theorem pearlCGammaSm120v1Rev1Cap1000_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((5213 / 1466750 : ℚ) : ℝ) εPearlC := by
  rw [show (5213 / 1466750 : ℚ) = 1 - (1 - 1 / 1000) * (5852 / 5867) by norm_num]
  exact pearlCGammaSm120v1Rev1Cap_16384 CM sem (1 / 1000) (by norm_num) hTT

/-- **v1 under rev1 per audit tile at the cap 1/1,000, 8192³**: `γ = 28673/7943750`. -/
theorem pearlCSampledSm120v1Rev1Cap1000_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 1000))
      (pearlCTilesDevRev1 (devSm120v1 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((28673 / 7943750 : ℚ) : ℝ) εPearlC := by
  rw [show (28673 / 7943750 : ℚ) = 1 - (1 - 1 / 1000) * (31692 / 31775) by norm_num]
  exact pearlCSampledSm120v1Rev1Cap_8192 CM sem (1 / 1000) (by norm_num) hTT

/-- **v1 under rev1 per audit tile at the cap 1/1,000, 16384³**: `γ = 5213/1466750`. -/
theorem pearlCSampledSm120v1Rev1Cap1000_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 1000))
      (pearlCTilesDevRev1 (devSm120v1 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((5213 / 1466750 : ℚ) : ℝ) εPearlC := by
  rw [show (5213 / 1466750 : ℚ) = 1 - (1 - 1 / 1000) * (5852 / 5867) by norm_num]
  exact pearlCSampledSm120v1Rev1Cap_16384 CM sem (1 / 1000) (by norm_num) hTT

/-- **v1 under rev1 at the cap 1/1,000 from rev1 at 1/400, 8192³**: `γ = 28673/7943750`. -/
theorem pearlCGammaSm120v1Rev1Cap1000_8192_of_cap400
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    Gγ CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((28673 / 7943750 : ℚ) : ℝ) εPearlC :=
  pearlCGammaSm120v1Rev1Cap1000_8192 CM sem
    (ttOutPearlCDevRev1_cap_anti CM _ sem (1 / 400) (1 / 1000)
      (by rw [devSm120v1_prices]; norm_num [Prices.sm120]) (by rw [devSm120v1_prices]; norm_num [Prices.sm120])
      (by norm_num) (by norm_num) hTT)

/-- **v1 under rev1 at the cap 1/1,000 from rev1 at 1/400, 16384³**: `γ = 5213/1466750`. -/
theorem pearlCGammaSm120v1Rev1Cap1000_16384_of_cap400
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    Gγ CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((5213 / 1466750 : ℚ) : ℝ) εPearlC :=
  pearlCGammaSm120v1Rev1Cap1000_16384 CM sem
    (ttOutPearlCDevRev1_cap_anti CM _ sem (1 / 400) (1 / 1000)
      (by rw [devSm120v1_prices]; norm_num [Prices.sm120]) (by rw [devSm120v1_prices]; norm_num [Prices.sm120])
      (by norm_num) (by norm_num) hTT)

end Gamma

end Pouw.PearlC
