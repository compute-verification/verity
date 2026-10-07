import Proofs.Pouw.PearlC.UOnlyGamma
import Models.Pouw.Assumptions.PearlC.TTOutTileUOnly

/-!
# TT_OUT over U alone per audit tile: it asserts more than the (C̃, U) tile row, and γ from it (the cheap-binding lane's file)

Staged; nothing pinned.
* `goodTiles_subset_goodTilesU`: a tile good under (C̃, U) binding is good under U alone.
* `ttOutTilePearlCDevRev1_of_uOnly`: the per-tile row over U gives rev1's (C̃, U) tile row at the same record, for
  non-negative prices and `ρ ≤ 1`.
* `gammaSampledU_implies_gammaSampled`: the sampled audit's γ under U-only binding gives it under (C̃, U) binding, since a
  tile the (C̃, U) audit accepts is one the U-only audit accepts.
* `gammaSampledU`: `gammaSampled` over `goodTilesU` (both are `sampledOfGood`), and at a record `pearlCSampledDevUOnlyAt`,
  with the same ω as rev1: `pearlCSampledUOnlySm120Rev1_8192` (0.51056%) and `pearlCSampledUOnlyUnpromotedCap1000_8192`
  (0.36162%) at sm_120's prices.
-/

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

section

variable {Q R S : Type}

open Classical in
theorem goodTiles_subset_goodTilesU (P : Protocol Q R S) (TR : TileRules Q R S) (U : Workload) (H : Q → R) (s : S)
    (τ : Transcript) : goodTiles P TR U H s τ ⊆ goodTilesU P TR U H s τ := by
  intro g hg
  unfold goodTiles at hg
  unfold goodTilesU
  rw [mem_filter] at hg ⊢
  obtain ⟨hg, c, hc, hcap, hB, hw⟩ := hg
  exact ⟨hg, c, hc, hcap, hB, fun i hi hr j hj => (hw i hi hr j hj).2⟩

theorem goodTilesU_subset (P : Protocol Q R S) (TR : TileRules Q R S) (U : Workload) (H : Q → R) (s : S)
    (τ : Transcript) : goodTilesU P TR U H s τ ⊆ range (TR.tiling U.layout).NT \ blankTiles TR U τ := by
  classical
  intro g hg
  unfold goodTilesU at hg
  rw [mem_filter] at hg
  obtain ⟨hr, c, hc, _⟩ := hg
  rw [mem_sdiff]
  refine ⟨hr, fun hb => ?_⟩
  unfold blankTiles at hb
  rw [mem_filter, hc] at hb
  exact Option.some_ne_none _ hb.2

/-- `GγSampled` under U-only binding gives it under (C̃, U) binding. -/
theorem gammaSampledU_implies_gammaSampled [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ)
    (h : GγSampledU CM P TR D γ η) : GγSampled CM P TR D γ η := by
  intro U hU T hT q hq A hA w κ hκ hw t εs hεs
  refine le_trans (Pouw.pr_mono ?_) (h U hU T hT q hq A hA w κ hκ hw t εs hεs)
  intro x ⟨hacc, hT'⟩
  refine ⟨fun i => ?_, hT'⟩
  have := hacc i
  rw [mem_union] at this ⊢
  exact this.imp (fun hg => goodTiles_subset_goodTilesU P TR U _ _ _ hg) id

theorem gammaSampledU [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (γ₀ Ω : ℝ) (ε : ℕ → ℕ → ℝ) (hγ₀ : γ₀ < 1)
    (hΩ : 0 < Ω)
    (hW : ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S), ∀ g < (TR.tiling U.layout).NT, ∀ act : Codes,
      TR.capOK H s U g act → TR.Wcred U.layout g act ≤ Ω * TR.credit H s U g act)
    (hWc : ∀ U : Workload, U.InDomain P D → ∀ g < (TR.tiling U.layout).NT, ∀ act : Codes,
      TR.Wcred U.layout g act ≤ TR.Wref U.layout g)
    (hTT : TTOutTileU CM P TR D γ₀ ε) : GγSampledU CM P TR D (1 - (1 - γ₀) / Ω) ε := by
  intro U hU T hT q hq A hA w κ hκ hw t εs hεs
  classical
  refine sampledOfGood TR U (fun H s => A.transcript U H s) (fun H s => goodTilesU P TR U H s (A.transcript U H s))
    γ₀ Ω T (ε q U.layout.N) hγ₀ hΩ (fun H s => goodTilesU_subset P TR U H s _) ?_ (hWc U hU)
    (hTT.2 U hU T hT q hq A hA) w κ hκ hw t εs hεs
  intro H s g hg
  unfold goodTilesU at hg
  rw [mem_filter] at hg
  obtain ⟨hgr, c, hc, hgc⟩ := hg
  have hact : actOf (A.transcript U H s) ((TR.tiling U.layout).unit g) = c.act := by simp [actOf, hc]
  rw [hact]
  exact hW U hU H s g (mem_range.mp hgr) c.act hgc.1

open Classical in
/-- **TT_OUT over U per tile asserts more than rev1's (C̃, U) tile row** at the same record, for non-negative prices and
`ρ ≤ 1`: a good tile's credit is a non-negative share of `creditDevRev1`. -/
theorem ttOutTilePearlCDevRev1_of_uOnly [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16)
    (hρ1 : ρ ≤ 1) (h : TTOutTilePearlCDevUOnly CM d sem ρ) : TTOutTilePearlCDevRev1 CM d sem ρ := by
  refine ⟨h.1, fun U hU T hT q hq A hA => le_trans (Pouw.pr_mono ?_) (h.2 U hU T hT q hq A hA)⟩
  intro x hx
  refine lt_of_lt_of_le hx (sum_le_sum_of_subset_of_nonneg (goodTiles_subset_goodTilesU _ _ U _ _ _) ?_)
  intro g hg _
  unfold goodTilesU at hg
  rw [mem_filter] at hg
  have hu := auditTiling_unit_lt 64 64 U.layout (mem_range.mp hg.1)
  obtain ⟨_, _, hk, _, hgd, _⟩ := hU.2 _ hu
  have hcr := creditDevRev1_nonneg d hadd hbf _ hk hgd
  simp only [pearlCTilesDevRev1]
  have hsh := tileShare_nonneg (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
    (sem.passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
      (actOf (A.transcript U x.1 x.2) ((auditTiling 64 64 U.layout).unit g)) ((auditTiling 64 64 U.layout).rows g))
    ((auditTiling 64 64 U.layout).cols g)
  have h1 : (0 : ℚ) ≤ 1 - ρ := by linarith
  exact_mod_cast mul_nonneg (mul_nonneg h1 hcr) hsh

end

/-- TT_OUT over U per tile at a device record gives `GγSampledU` on the one-shape domain at `γ = 1 − (399/400)/ω`, where
`wrefDevRev1 = ω·(1 − ρ)·creditDevRev1`. -/
theorem pearlCSampledDevUOnlyAt {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω)
    (hw0 : 0 ≤ wrefDevRev1 d s) (hω : wrefDevRev1 d s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hγ : γ = 1 - (1 - 1 / 400) / ω) (hTT : TTOutTilePearlCDevUOnly CM d sem ρ) :
    GγSampledU CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDevAt d s) (γ : ℝ)
      εPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDevAt d s) →
      ∀ (H : Q → R) (s' : S), ∀ g < ((pearlCTilesDevRev1 d sem ρ).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevRev1 d sem ρ).capOK H s' U g act →
        (pearlCTilesDevRev1 d sem ρ).Wcred U.layout g act ≤ (ω : ℝ) * (pearlCTilesDevRev1 d sem ρ).credit H s' U g act := by
    intro U hU H s' g hg act _
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.2 _ hu
    simp only [pearlCTilesDevRev1]
    rw [hsh, hω]
    push_cast
    exact le_of_eq (by ring)
  have hWc : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDevAt d s) →
      ∀ g < ((pearlCTilesDevRev1 d sem ρ).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevRev1 d sem ρ).Wcred U.layout g act ≤ (pearlCTilesDevRev1 d sem ρ).Wref U.layout g := by
    intro U hU g hg act
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.2 _ hu
    simp only [pearlCTilesDevRev1]
    rw [hsh]
    exact_mod_cast mul_le_mul_of_nonneg_left (tileShare_mono _ (passRows_subset sem _ _ _) _) hw0
  have hmono : TTOutTileU CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDevAt d s)
      (1 / 400) εPearlC :=
    ⟨hTT.1, fun U hU => hTT.2 U ⟨hU.1, hU.2.1⟩⟩
  have key := gammaSampledU CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDevAt d s)
    (1 / 400) (ω : ℝ) εPearlC (by norm_num) (by exact_mod_cast hω0) hW hWc hmono
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  exact key

/-- **v1 under rev1 per audit tile with U-only binding, 8192³**, at sm_120's prices, cap 1/400: `γ = 16223/3177500`. -/
theorem pearlCSampledUOnlySm120Rev1_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4) (hp : d.prices = Prices.sm120)
    (hTT : TTOutTilePearlCDevUOnly CM d sem (1 / 400)) :
    GγSampledU CM (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCTilesDevRev1 d sem (1 / 400))
      (pearlCDomainDevAt d sh8192) ((16223 / 3177500 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevUOnlyAt CM d sem _ _ (31775 / 31692) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

/-- **v2 at the cap 1/1,000 per audit tile with U-only binding, 8192³**, at sm_120's prices: `γ = 216793/59950000`. -/
theorem pearlCSampledUOnlyUnpromotedCap1000_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120)
    (hTT : TTOutTilePearlCDevUOnly CM d sem (1 / 1000)) :
    GγSampledU CM (pearlCProtocolDevRev1 d sem (1 / 1000)) (pearlCTilesDevRev1 d sem (1 / 1000))
      (pearlCDomainDevAt d sh8192) ((216793 / 59950000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevUOnlyAt CM d sem _ _ (1049125 / 1047951) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

end Pouw.PearlC
