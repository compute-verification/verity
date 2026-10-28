import Proofs.Pouw.PearlC.DeviceRev1Gamma
import Definitions.Pouw.PearlC.DevicePrices
import Definitions.Pouw.Guarantees.PearlC
import Proofs.Pouw.PearlC.TTOutW1

/-!
# The unpromoted record, the cap, and γ at sm_120's prices

Staged; nothing pinned.
* **The unpromoted domain is not empty:** `pearlCDomainDev_unpromoted` puts a `1 × 64 × 128` layout in `pearlCDomainDev d`
  at any record with `G = 0`. Before the domain's group condition became `G = 0 ∨ 32·G ∣ k`, `32·0 ∣ k` forced `k = 0`,
  so every statement at `devSm120v2` was vacuous.
* **A stricter cap asserts less, per unit:** `ttOutPearlCDev_cap_anti` gives TT_OUT at `ρ'` from TT_OUT at `ρ` for
  `ρ' ≤ ρ ≤ 1` at any record with non-negative prices. The credit doesn't read the cap, and a unit within `ρ'` is
  within `ρ`. So `tt-out/pearl-c-sm120-unpromoted-cap1000` is weaker than `tt-out/pearl-c-sm120-unpromoted`. Per tile
  there is no such theorem: a good tile's fixed credit `(1 − ρ)·creditDev·share` grows as the cap tightens.
* **v2 at the cap 1/1,000**, at any record with `G = 0` and `Prices.sm120`: `216793/59950000` (0.36162%) at 8192³ and
  `2950127/829250000` (0.35576%) at 16384³, per unit and per audit tile.
* **v1 under rev1**, at any record with `G = 4` and `Prices.sm120`, cap 1/400: `16223/3177500` (0.51056%) at 8192³ and
  `2963/586700` (0.50503%) at 16384³, per unit and per audit tile.

`devSm120v1` and `devSm120v2` meet the hypotheses by `rfl` once the flags twin defines them.
-/

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- A record without promotion admits a `1 × 64 × 128` unit (`m n k`). -/
theorem pearlCDomainDev_unpromoted (d : PearlCDevice) (hG : d.G = 0) :
    pearlCDomainDev d ⟨1, fun _ => ⟨1, 64, 128⟩, fun _ => 0⟩ := by
  intro u _
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, Or.inl hG, by norm_num⟩

section
variable {Q R S : Type}

open Classical in
theorem cap_credit_sum_le (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρ' : ℚ) (hadd : 0 ≤ d.prices.add)
    (hbf : 0 ≤ d.prices.bf16) (hle : ρ' ≤ ρ) (hρ1 : ρ ≤ 1) (U : Workload) (H : Q → R) (s : S)
    (τ : CommitLog) :
    ∑ u ∈ correctSet (pearlCProtocolDev d sem ρ') U H s τ, (pearlCProtocolDev d sem ρ').credit H s U u (actOf τ u) ≤
      ∑ u ∈ correctSet (pearlCProtocolDev d sem ρ) U H s τ, (pearlCProtocolDev d sem ρ).credit H s U u (actOf τ u) := by
  have hsub : correctSet (pearlCProtocolDev d sem ρ') U H s τ ⊆ correctSet (pearlCProtocolDev d sem ρ) U H s τ := by
    intro u hu
    unfold correctSet at hu ⊢
    rw [mem_filter] at hu ⊢
    obtain ⟨hu, c, hc, hact, hcap, hw⟩ := hu
    refine ⟨hu, c, hc, hact, ?_, hw⟩
    have h1 : debitDev d (sem.unitDebitDev d H s U u c.act) ≤ ρ' * creditDev d (U.layout.shape u) := hcap
    show debitDev d (sem.unitDebitDev d H s U u c.act) ≤ ρ * creditDev d (U.layout.shape u)
    have := mul_le_mul_of_nonneg_right hle (creditDev_nonneg d hadd hbf (U.layout.shape u))
    linarith
  refine sum_le_sum_of_subset_of_nonneg hsub fun u hu _ => ?_
  unfold correctSet at hu
  rw [mem_filter] at hu
  obtain ⟨_, c, hc, _, hcap, _⟩ := hu
  have hact : actOf τ u = c.act := by simp [actOf, hc]
  rw [hact]
  have h1 : debitDev d (sem.unitDebitDev d H s U u c.act) ≤ ρ * creditDev d (U.layout.shape u) := hcap
  have h2 := creditDev_nonneg d hadd hbf (U.layout.shape u)
  show (0 : ℝ) ≤ ((creditDev d (U.layout.shape u) - debitDev d (sem.unitDebitDev d H s U u c.act) : ℚ) : ℝ)
  have : (0 : ℚ) ≤ creditDev d (U.layout.shape u) - debitDev d (sem.unitDebitDev d H s U u c.act) := by nlinarith
  exact_mod_cast this

/-- **A stricter cap asserts less**: TT_OUT at a device record at cap `ρ` gives it at any cap `ρ' ≤ ρ ≤ 1`. -/
theorem ttOutPearlCDev_cap_anti [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρ' : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16)
    (hle : ρ' ≤ ρ) (hρ1 : ρ ≤ 1) (h : TTOutPearlCDev CM d sem ρ) : TTOutPearlCDev CM d sem ρ' := by
  refine ⟨h.1, fun U hU T hT q hq A hA => le_trans (Pouw.pr_mono ?_) (h.2 U hU T hT q hq A hA)⟩
  intro x hx
  exact lt_of_lt_of_le hx (cap_credit_sum_le d sem ρ ρ' hadd hbf hle hρ1 U x.1 x.2 _)

end

/-- **v2 at the cap 1/1,000, 8192³**, at a record with no promotion and sm_120's prices: `γ = 216793/59950000`. -/
theorem pearlCGammaUnpromotedCap1000_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120)
    (hTT : TTOutPearlCDev CM d sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDev d sem (1 / 1000)) (pearlCDomainDevAt d sh8192) ((216793 / 59950000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaDevAt CM d sem _ _ (1049125 / 1047951) _ (by norm_num)
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192]) (by norm_num) hTT

/-- **v2 at the cap 1/1,000, 16384³**, at a record with no promotion and sm_120's prices: `γ = 2950127/829250000`. -/
theorem pearlCGammaUnpromotedCap1000_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120)
    (hTT : TTOutPearlCDev CM d sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDev d sem (1 / 1000)) (pearlCDomainDevAt d sh16384) ((2950127 / 829250000 : ℚ) : ℝ)
      εPearlC :=
  pearlCGammaDevAt CM d sem _ _ (2073125 / 2070927) _ (by norm_num)
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh16384]) (by norm_num) hTT

/-- **v2 at the cap 1/1,000 per audit tile, 8192³**: `GγSampled` at `γ = 216793/59950000`. -/
theorem pearlCSampledUnpromotedCap1000_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120)
    (hTT : TTOutTilePearlCDev CM d sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDev d sem (1 / 1000)) (pearlCTilesDev d sem (1 / 1000)) (pearlCDomainDevAt d sh8192)
      ((216793 / 59950000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevAt CM d sem _ _ (1049125 / 1047951) _ (by norm_num)
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192]) (by norm_num) hTT

/-- **v2 at the cap 1/1,000 per audit tile, 16384³**: `GγSampled` at `γ = 2950127/829250000`. -/
theorem pearlCSampledUnpromotedCap1000_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120)
    (hTT : TTOutTilePearlCDev CM d sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDev d sem (1 / 1000)) (pearlCTilesDev d sem (1 / 1000)) (pearlCDomainDevAt d sh16384)
      ((2950127 / 829250000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevAt CM d sem _ _ (2073125 / 2070927) _ (by norm_num)
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh16384]) (by norm_num) hTT

/-- **v1 under rev1, 8192³**, at a record promoted every 4 atoms with sm_120's prices, cap 1/400: `γ = 16223/3177500`. -/
theorem pearlCGammaSm120Rev1_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4) (hp : d.prices = Prices.sm120)
    (hTT : TTOutPearlCDevRev1 CM d sem (1 / 400)) :
    Gγ CM (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCDomainDevAt d sh8192) ((16223 / 3177500 : ℚ) : ℝ) εPearlC :=
  pearlCGammaDevRev1At CM d sem _ _ (31775 / 31692) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

/-- **v1 under rev1, 16384³**, at a record promoted every 4 atoms with sm_120's prices, cap 1/400: `γ = 2963/586700`. -/
theorem pearlCGammaSm120Rev1_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4) (hp : d.prices = Prices.sm120)
    (hTT : TTOutPearlCDevRev1 CM d sem (1 / 400)) :
    Gγ CM (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCDomainDevAt d sh16384) ((2963 / 586700 : ℚ) : ℝ) εPearlC :=
  pearlCGammaDevRev1At CM d sem _ _ (5867 / 5852) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

/-- **v1 under rev1 per audit tile, 8192³**, at sm_120's prices: `GγSampled` at `γ = 16223/3177500`. -/
theorem pearlCSampledSm120Rev1_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4) (hp : d.prices = Prices.sm120)
    (hTT : TTOutTilePearlCDevRev1 CM d sem (1 / 400)) :
    GγSampled CM (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCTilesDevRev1 d sem (1 / 400))
      (pearlCDomainDevAt d sh8192) ((16223 / 3177500 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1At CM d sem _ _ (31775 / 31692) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

/-- **v1 under rev1 per audit tile, 16384³**, at sm_120's prices: `GγSampled` at `γ = 2963/586700`. -/
theorem pearlCSampledSm120Rev1_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4) (hp : d.prices = Prices.sm120)
    (hTT : TTOutTilePearlCDevRev1 CM d sem (1 / 400)) :
    GγSampled CM (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCTilesDevRev1 d sem (1 / 400))
      (pearlCDomainDevAt d sh16384) ((2963 / 586700 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1At CM d sem _ _ (5867 / 5852) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

end Pouw.PearlC

theorem Pouw.SecurityProofs.PearlC.GammaUnpromotedCap1000_8192 : Pouw.Guarantees.PearlC.GammaUnpromotedCap1000_8192 :=
  fun sha fetch salt d sem hG hp hTT =>
    @Pouw.PearlC.pearlCGammaUnpromotedCap1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) d sem hG hp
      (Pouw.PearlC.TTOutW1.ttOutPearlCDev_of_w1 hTT)
