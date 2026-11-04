import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeedK
import Proofs.Pouw.PearlC.RowSeedCode
import Proofs.Pouw.PearlC.DeviceGamma

/-!
# Per-row seeds at a census k: the tile cap `ρ` and a wider row cap `ρr` (compute accounting's file)

By ruling (option (a)): at a census `k < 1024` the served check caps each credited row of a tile at
`ρr = 4ρ` and the tile at `ρ` (`pearlCTilesDevRev1RowCapAt`, `TTOutRowSeed.lean`). The row-seeded γ rests on rev1's
TT_OUT per tile at `ρr`, through (C2) at `ρr`:

* **The reduction** (`ttOutTileRowSeed_rowCapAt`): a tile good under both caps is good under the row cap `ρr` alone, and
  its credit is `(1 − ρ)·C` times its credited rows' share, `(1 − ρ)/(1 − ρr)` times (C2)'s at `ρr`. So the row-seeded
  TT_OUT per tile at (C2, `ρr`) and `γ₀` gives it at both caps with `1 − γ₀' = (1 − γ₀)·(1 − ρr)/(1 − ρ)`, at any record
  with non-negative add and BF16 prices.
* **The sampled γ** (`pearlCSampledDevRev1RowCapAtW`, at any per-shape `W_ref`; `pearlCSampledDevRev1KRowCapAt` at the
  honest kernel's): `γ = 1 − (1 − γ₀')/ω = 1 − (399/400)·(1 − ρr)/((1 − ρ)·ω)`, where `W_ref = ω·(1 − ρ)·creditDevRev1`.
* **GLM-5.3's down projection on the H100 record**, `m × 6144 × 256` at any `m`, `ρ = 1/400`, `ρr = 1/100`, from
  `TTOutTilePearlCH100Rev1` at 1/100, C1 and C3 (`pearlCSampledH100v1RowSeedRowCap100_6144x256`), and its code form
  (`-h3`) with C1-h3 and C5 (`pearlCSampledH100v1RowSeedCodeRowCap100_6144x256`): `ω = 25210/25137` and
  `γ = 32437/2521000` (1.28667%), with the error at the `m·N` rows. C0 is `devAt_rowLocal`, by `rfl`.
-/

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

section
variable {Q R S : Type}

open Classical in
theorem rowCapAt_credit_sum_le (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr : ℚ) (hadd : 0 ≤ d.prices.add)
    (hbf : 0 ≤ d.prices.bf16) (hρ : ρ ≤ 1) (hρr : ρr < 1) (U : Workload) (hD : pearlCDomainDev d U.layout)
    (H : Q → R) (s : S) (τ : Transcript) :
    ∑ g ∈ goodTiles (pearlCProtocolDevRev1RowCapAt d sem ρ ρr) (pearlCTilesDevRev1RowCapAt d sem ρ ρr) U H s τ,
        (pearlCTilesDevRev1RowCapAt d sem ρ ρr).credit H s U g
          (actOf τ (((pearlCTilesDevRev1RowCapAt d sem ρ ρr).tiling U.layout).unit g)) ≤
      (((1 - ρ) / (1 - ρr) : ℚ) : ℝ) *
        ∑ g ∈ goodTiles (pearlCProtocolDevRev1RowCap d sem ρr) (pearlCTilesDevRev1RowCap d sem ρr) U H s τ,
          (pearlCTilesDevRev1RowCap d sem ρr).credit H s U g
            (actOf τ (((pearlCTilesDevRev1RowCap d sem ρr).tiling U.layout).unit g)) := by
  set tl := auditTiling 64 64 U.layout
  have hr : (0 : ℚ) < 1 - ρr := by linarith
  have hC : ∀ g < tl.NT, 0 ≤ creditDevRev1 d (U.layout.shape (tl.unit g)) := fun g hg =>
    have hs := hD _ (auditTiling_unit_lt 64 64 U.layout hg)
    creditDevRev1_nonneg d hadd hbf _ hs.2.2.1 hs.2.2.2.2.1
  have hsub : goodTiles (pearlCProtocolDevRev1RowCapAt d sem ρ ρr) (pearlCTilesDevRev1RowCapAt d sem ρ ρr) U H s τ ⊆
      goodTiles (pearlCProtocolDevRev1RowCap d sem ρr) (pearlCTilesDevRev1RowCap d sem ρr) U H s τ := by
    intro g hg
    unfold goodTiles at hg ⊢
    rw [mem_filter] at hg ⊢
    obtain ⟨hg, c, hc, hcap, hw⟩ := hg
    exact ⟨hg, c, hc, hcap.2, hw⟩
  rw [mul_sum]
  calc _ = ∑ g ∈ goodTiles (pearlCProtocolDevRev1RowCapAt d sem ρ ρr) (pearlCTilesDevRev1RowCapAt d sem ρ ρr) U H s τ,
          (((1 - ρ) / (1 - ρr) : ℚ) : ℝ) *
            (pearlCTilesDevRev1RowCap d sem ρr).credit H s U g (actOf τ (tl.unit g)) := by
        refine sum_congr rfl fun g _ => ?_
        simp only [pearlCTilesDevRev1RowCapAt, pearlCTilesDevRev1RowCap, pearlCTilesDevRev1]
        have : (1 - ρ) * creditDevRev1 d (U.layout.shape (tl.unit g)) *
            tileShare (U.layout.shape (tl.unit g))
              (sem.passRows (U.layout.shape (tl.unit g)) (actOf τ (tl.unit g)) (tl.rows g)) (tl.cols g) =
            (1 - ρ) / (1 - ρr) * ((1 - ρr) * creditDevRev1 d (U.layout.shape (tl.unit g)) *
              tileShare (U.layout.shape (tl.unit g))
                (sem.passRows (U.layout.shape (tl.unit g)) (actOf τ (tl.unit g)) (tl.rows g)) (tl.cols g)) := by
          field_simp
        exact_mod_cast this
    _ ≤ _ := by
        refine sum_le_sum_of_subset_of_nonneg hsub fun g hg _ => ?_
        have hgr : g < tl.NT := by
          unfold goodTiles at hg
          exact mem_range.mp (mem_filter.mp hg).1
        simp only [pearlCTilesDevRev1RowCap, pearlCTilesDevRev1]
        have hsh := tileShare_nonneg (U.layout.shape (tl.unit g))
          (sem.passRows (U.layout.shape (tl.unit g)) (actOf τ (tl.unit g)) (tl.rows g)) (tl.cols g)
        have hq : (0 : ℚ) ≤ (1 - ρ) / (1 - ρr) := div_nonneg (by linarith) hr.le
        have : (0 : ℚ) ≤ (1 - ρ) / (1 - ρr) * ((1 - ρr) * creditDevRev1 d (U.layout.shape (tl.unit g)) *
            tileShare (U.layout.shape (tl.unit g))
              (sem.passRows (U.layout.shape (tl.unit g)) (actOf τ (tl.unit g)) (tl.rows g)) (tl.cols g)) :=
          mul_nonneg hq (mul_nonneg (mul_nonneg hr.le (hC g hgr)) hsh)
        exact_mod_cast this

/-- **The tile cap `ρ` with a wider row cap `ρr`, from the row cap `ρr` alone.** The row-seeded TT_OUT per tile at
(C2) with the cap `ρr` and any `γ₀` gives it at the tile cap `ρ < 1` and the row cap `ρr < 1` together, with
`1 − γ₀' = (1 − γ₀)·(1 − ρr)/(1 − ρ)`, at any record with non-negative add and BF16 prices. -/
theorem ttOutTileRowSeed_rowCapAt [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S}
    {d : PearlCDevice} {sem : PearlCSem Q R S} {ρ ρr : ℚ} {γ₀ : ℝ} (hadd : 0 ≤ d.prices.add)
    (hbf : 0 ≤ d.prices.bf16) (hρ : ρ < 1) (hρr : ρr < 1)
    (h : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCap d sem ρr) (pearlCTilesDevRev1RowCap d sem ρr)
      (pearlCDomainDev d) γ₀) :
    TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCapAt d sem ρ ρr) (pearlCTilesDevRev1RowCapAt d sem ρ ρr)
      (pearlCDomainDev d) (1 - (1 - γ₀) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ)) := by
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hr' : (0 : ℚ) < 1 - ρr := by linarith
  have ha : (0 : ℝ) < (((1 - ρr) / (1 - ρ) : ℚ) : ℝ) := by exact_mod_cast div_pos hr' hr
  have hab : (((1 - ρ) / (1 - ρr) : ℚ) : ℝ) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ) = 1 := by
    rw [← Rat.cast_mul, show (1 - ρ) / (1 - ρr) * ((1 - ρr) / (1 - ρ)) = 1 by field_simp, Rat.cast_one]
  refine ⟨h.1, fun U hU hR T hT q hq A hA => le_trans (Pouw.pr_mono ?_) (h.2 U hU hR T hT q hq A hA)⟩
  intro x hx
  have hsum := rowCapAt_credit_sum_le d sem ρ ρr hadd hbf hρ.le hρr U hU.2 x.1 x.2 (A.transcript U x.1 x.2)
  rw [show 1 - (1 - (1 - γ₀) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ)) = (1 - γ₀) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ) by ring,
    div_mul_eq_div_div] at hx
  have h2 := lt_of_lt_of_le hx hsum
  rw [div_lt_iff₀ ha] at h2
  calc T / (1 - γ₀) < _ := h2
    _ = _ := by rw [mul_comm, ← mul_assoc, mul_comm (((1 - ρr) / (1 - ρ) : ℚ) : ℝ), hab, one_mul]

end

section Gamma

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

/-- **The row-seeded TT_OUT per audit tile at both caps gives `GγSampled` at any per-shape `W_ref`** `W`: rev1's tiles
with the tile cap `ρ`, the row cap `ρr` and `W_ref` and the credited rows' `W_ref` at `W`, on the one-shape domain at
`γ = 1 − (1 − γ₀)/ω`, where `W s = ω·(1 − ρ)·creditDevRev1 ≥ 0`, with the error at the row count (`gammaSampled`). -/
theorem pearlCSampledDevRev1RowCapAtW (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr : ℚ)
    (W : Shape → ℚ) (s : Shape) (ω : ℚ) (γ₀ γ : ℝ) (hγ₀ : γ₀ < 1) (hω0 : 0 < ω) (hw0 : 0 ≤ W s)
    (hω : W s = ω * ((1 - ρ) * creditDevRev1 d s)) (hγ : γ = 1 - (1 - γ₀) / (ω : ℝ))
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCapAt d sem ρ ρr) (pearlCTilesDevRev1RowCapAt d sem ρ ρr)
      (pearlCDomainDev d) γ₀) :
    GγSampled CM { pearlCProtocolDevRev1RowCapAt d sem ρ ρr with Wref := fun L u => (W (L.shape u) : ℝ) }
      { pearlCTilesDevRev1RowCapAt d sem ρ ρr with
        Wref := fun L g =>
          let tl := auditTiling 64 64 L
          let sh := L.shape (tl.unit g)
          ((W sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ)
        Wcred := fun L g act =>
          let tl := auditTiling 64 64 L
          let sh := L.shape (tl.unit g)
          ((W sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ) }
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) γ (fun q N => εPearlC q (N * s.m)) := by
  rw [hγ]
  refine gammaSampled CM _ _ _ γ₀ (ω : ℝ) _ hγ₀ (by exact_mod_cast hω0) ?_ ?_
    (ttOutTile_of_ttOutTileRowSeed (ttOutTileRowSeed_wref (fun L u => (W (L.shape u) : ℝ))
      (fun L g =>
        let tl := auditTiling 64 64 L
        let sh := L.shape (tl.unit g)
        ((W sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ))
      (fun L g act =>
        let tl := auditTiling 64 64 L
        let sh := L.shape (tl.unit g)
        ((W sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ)) hTT))
  · intro U hU H s' g hg act _
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.1.2 _ hu
    simp only [pearlCTilesDevRev1RowCapAt, pearlCTilesDevRev1]
    rw [hsh, hω]
    push_cast
    exact le_of_eq (by ring)
  · intro U hU g hg act
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.1.2 _ hu
    dsimp only
    rw [hsh]
    exact_mod_cast mul_le_mul_of_nonneg_left (tileShare_mono _ (passRows_subset sem _ _ _) _) hw0

/-- `pearlCSampledDevRev1RowCapAtW` from the row-seeded TT_OUT per tile at (C2) with the row cap `ρr` and `γ₀ = 1/400`
(`ttOutTileRowSeed_rowCapAt`), at rev1's `W_ref`: `γ = 1 − (399/400)·(1 − ρr)/((1 − ρ)·ω)`, where
`wrefDevRev1 = ω·(1 − ρ)·creditDevRev1 ≥ 0`. -/
theorem pearlCSampledDevRev1RowCapAt (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr : ℚ)
    (s : Shape) (ω γ : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hρ : ρ < 1) (hρr : ρr < 1)
    (hω0 : 0 < ω) (hw0 : 0 ≤ wrefDevRev1 d s) (hω : wrefDevRev1 d s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hγ : γ = 1 - (1 - 1 / 400) * (1 - ρr) / ((1 - ρ) * ω))
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCap d sem ρr) (pearlCTilesDevRev1RowCap d sem ρr)
      (pearlCDomainDev d) (1 / 400)) :
    GγSampled CM (pearlCProtocolDevRev1RowCapAt d sem ρ ρr) (pearlCTilesDevRev1RowCapAt d sem ρ ρr)
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (γ : ℝ) (fun q N => εPearlC q (N * s.m)) := by
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hr' : (0 : ℚ) < 1 - ρr := by linarith
  refine pearlCSampledDevRev1RowCapAtW CM d sem ρ ρr (wrefDevRev1 d) s ω _ (γ : ℝ) ?_ hω0 hw0 hω ?_
    (ttOutTileRowSeed_rowCapAt hadd hbf hρ hρr hTT)
  · have : (0 : ℝ) < (1 - 1 / 400) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ) :=
      mul_pos (by norm_num) (by exact_mod_cast div_pos hr' hr)
    linarith
  · rw [hγ]
    have h1 : ((1 - ρ : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hr.ne'
    have h3 : ((ω : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hω0.ne'
    push_cast at h1 h3 ⊢
    field_simp
    ring

/-- As `pearlCSampledDevRev1RowCapAt`, against the honest kernel's `W_ref`, where
`wrefDevRev1K d c = ω·(1 − ρ)·creditDevRev1 ≥ 0`. -/
theorem pearlCSampledDevRev1KRowCapAt (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S)
    (ρ ρr c : ℚ) (s : Shape) (ω γ : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hρ : ρ < 1)
    (hρr : ρr < 1) (hω0 : 0 < ω) (hw0 : 0 ≤ wrefDevRev1K d c s)
    (hω : wrefDevRev1K d c s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hγ : γ = 1 - (1 - 1 / 400) * (1 - ρr) / ((1 - ρ) * ω))
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCap d sem ρr) (pearlCTilesDevRev1RowCap d sem ρr)
      (pearlCDomainDev d) (1 / 400)) :
    GγSampled CM (pearlCProtocolDevRev1KRowCapAt d sem ρ ρr c) (pearlCTilesDevRev1KRowCapAt d sem ρ ρr c)
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (γ : ℝ) (fun q N => εPearlC q (N * s.m)) := by
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hr' : (0 : ℚ) < 1 - ρr := by linarith
  refine pearlCSampledDevRev1RowCapAtW CM d sem ρ ρr (wrefDevRev1K d c) s ω _ (γ : ℝ) ?_ hω0 hw0 hω ?_
    (ttOutTileRowSeed_rowCapAt hadd hbf hρ hρr hTT)
  · have : (0 : ℝ) < (1 - 1 / 400) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ) :=
      mul_pos (by norm_num) (by exact_mod_cast div_pos hr' hr)
    linarith
  · rw [hγ]
    have h1 : ((1 - ρ : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hr.ne'
    have h3 : ((ω : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hω0.ne'
    push_cast at h1 h3 ⊢
    field_simp
    ring

/-! ## GLM-5.3's down projection on the H100 record -/

theorem devH100_add : 0 ≤ devH100.prices.add := by
  show (0 : ℚ) ≤ Prices.h100.add; norm_num [Prices.h100]

theorem devH100_bf16 : 0 ≤ devH100.prices.bf16 := by
  show (0 : ℚ) ≤ Prices.h100.bf16; norm_num [Prices.h100]

theorem devH100_rowLocal : devH100.RowLocal := by
  rw [← devAt_hopper]; exact devAt_rowLocal _ _ _

theorem creditDevRev1_h100_n6144k256 (m : ℕ) : creditDevRev1 devH100 ⟨m, 6144, 256⟩ = 2580480 * m := by
  norm_num [creditDevRev1, firstAddDev, creditDev, devH100, Prices.h100, Params.pi, Costs.adopted]
  ring

theorem wrefDevRev1_h100_n6144k256 (m : ℕ) : wrefDevRev1 devH100 ⟨m, 6144, 256⟩ = 2581504 * m := by
  rw [wrefDevRev1, creditDevRev1_h100_n6144k256]
  norm_num [devH100, Prices.h100, Costs.adopted]
  ring

variable (CM : CostModel Q R S) {sem sem' semC : PearlCSem Q R S}

/-- **GLM-5.3's down projection, `m × 6144 × 256` at any `m`, on the H100 record with per-row seeds per audit tile, the
tile cap 1/400 and the row cap 1/100**, from `TTOutTilePearlCH100Rev1` at 1/100, C1 and C3:
`γ = 1 − (399/400)·(99/100)/((399/400)·ω) = 32437/2521000` (1.28667%) at `ω = 25210/25137`, with the error at the
`m·N` rows. -/
theorem pearlCSampledH100v1RowSeedRowCap100_6144x256 (m : ℕ)
    (hTT : TTOutTilePearlCH100Rev1 CM sem (1 / 100)) (hN : RowSeedNoise sem sem') (hCM : CM.SplitClosed) :
    GγSampled CM (pearlCProtocolDevRev1RowCapAt devH100 sem' (1 / 400) (1 / 100))
      (pearlCTilesDevRev1RowCapAt devH100 sem' (1 / 400) (1 / 100))
      (fun L => pearlCDomainDevAt devH100 ⟨m, 6144, 256⟩ L ∧ rowsOf L ≤ 2 ^ 64)
      ((32437 / 2521000 : ℚ) : ℝ) (fun q N => εPearlC q (N * m)) :=
  pearlCSampledDevRev1RowCapAt CM devH100 sem' (1 / 400) (1 / 100) ⟨m, 6144, 256⟩ (25210 / 25137) _ devH100_add
    devH100_bf16 (by norm_num) (by norm_num) (by norm_num)
    (by rw [wrefDevRev1_h100_n6144k256]; positivity)
    (by rw [wrefDevRev1_h100_n6144k256, creditDevRev1_h100_n6144k256]; ring)
    (by norm_num)
    (ttOutTileRowSeed_of_ttOutTile hTT hN hCM devH100_rowLocal (by norm_num) devH100_add devH100_bf16)

/-- **The code-form seed (`-h3`) of `pearlCSampledH100v1RowSeedRowCap100_6144x256`**, the served record
`pearl-c-h100-v1-h3` at a census k, adding C1-h3 and C5: `γ = 32437/2521000` (1.28667%), with the error at the `m·N`
rows. -/
theorem pearlCSampledH100v1RowSeedCodeRowCap100_6144x256 {π : Layout → Q ≃ Q} (m : ℕ)
    (hTT : TTOutTilePearlCH100Rev1 CM sem (1 / 100)) (hN : RowSeedNoise sem sem') (hC : CodeSeedNoise sem' semC π)
    (hCM : CM.SplitClosed) (hCM' : CM.RelabelClosed) :
    GγSampled CM (pearlCProtocolDevRev1RowCapAt devH100 semC (1 / 400) (1 / 100))
      (pearlCTilesDevRev1RowCapAt devH100 semC (1 / 400) (1 / 100))
      (fun L => pearlCDomainDevAt devH100 ⟨m, 6144, 256⟩ L ∧ rowsOf L ≤ 2 ^ 64)
      ((32437 / 2521000 : ℚ) : ℝ) (fun q N => εPearlC q (N * m)) :=
  pearlCSampledDevRev1RowCapAt CM devH100 semC (1 / 400) (1 / 100) ⟨m, 6144, 256⟩ (25210 / 25137) _ devH100_add
    devH100_bf16 (by norm_num) (by norm_num) (by norm_num)
    (by rw [wrefDevRev1_h100_n6144k256]; positivity)
    (by rw [wrefDevRev1_h100_n6144k256, creditDevRev1_h100_n6144k256]; ring)
    (by norm_num)
    (ttOutTileRowSeedCode_of_ttOutTile hTT hN hC hCM hCM' devH100_rowLocal (by norm_num) devH100_add devH100_bf16)

end Gamma

end Pouw.PearlC
