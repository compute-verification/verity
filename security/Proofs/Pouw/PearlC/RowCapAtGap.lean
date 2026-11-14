module

public import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeedK
public import Proofs.Pouw.PearlC.RowSeedCode
public import Proofs.Pouw.PearlC.DeviceGap
public import Proofs.Pouw.PearlC.TileDrawGap
public import Proofs.Pouw.PearlC.Records

/-!
# Per-row seeds at a census k: the tile cap `ρ` and a wider row cap `ρr` (compute accounting's file)

By ruling (option (a)): at a census `k < 1024` the served check caps each credited row of a tile at
`ρr = 4ρ` and the tile at `ρ` (`pearlCTilesDevRev1RowCapAt`, `TTOutRowSeed.lean`). The row-seeded g rests on rev1's
TT_OUT per tile at `ρr`, through (C2) at `ρr`:

* **The reduction** (`ttOutTileRowSeed_rowCapAt`): a tile good under both caps is good under the row cap `ρr` alone, and
  its credit is `(1 − ρ)·C` times its credited rows' share, `(1 − ρ)/(1 − ρr)` times (C2)'s at `ρr`. So the row-seeded
  TT_OUT per tile at (C2, `ρr`) and `g₀` gives it at both caps with `1 − g₀' = (1 − g₀)·(1 − ρr)/(1 − ρ)`, at any record
  with non-negative add and BF16 prices.
* **The sampled g** (`pearlCSampledDevRev1RowCapAtW`, at any per-shape `W_ref`; `pearlCSampledDevRev1KRowCapAt` at the
  honest kernel's): `g = 1 − (1 − g₀')/ω = 1 − (399/400)·(1 − ρr)/((1 − ρ)·ω)`, where `W_ref = ω·(1 − ρ)·creditDevRev1`.
* **The tolerant audit** (`…Tol`, `GGapSampledTol`): the same g from the same premises when the audit accepts a sampled tile
  over its inert-work bound and credits it nothing. The strict forms are its case `u 0 = ε_s`, `u x = 1` otherwise
  (`gapSampled_of_tol`).
* **The audit sampled as the run samples** (`pearlCSampledDrawDevRev1RowCapAtTol`, `GGapSampledDrawTol`): each transcript sampled
  by its records' tickets (`pearlCDrawRev1`), at the same g, since a declaration consistent with a tile weighs it at most
  its credited rows' `W_ref` (`pearlCDrawRev1_le`, `wrefRowRev1_le`).
* **GLM-5.3's down projection on the H100 record**, `m × 6144 × 256` at any `m`, `ρ = 1/400`, `ρr = 1/100`, sampled as
  the run samples, from `TTOutTilePearlCH100Rev1` at 1/100, C1 and C3 (`pearlCSampledH100v1RowSeedRowCap100_6144x256Tol`),
  and its code form (`-h3`) with C1-h3 and C5 (`pearlCSampledH100v1RowSeedCodeRowCap100_6144x256Tol`), both at the
  deployed table `u 0 = ε_s`, `u x = 1` otherwise (`GGapSampledDrawTolThreshold`): `ω = 25210/25137` and
  `g = 32437/2521000` (1.28667%), with the error at the `m·N` rows; the forms without `Tol` are the audit that accepts
  only checked tiles (`gapSampledDraw_of_threshold`). C0 is `devAt_rowLocal`, by `rfl`.
-/

@[expose] public section

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
(C2) with the cap `ρr` and any `g₀` gives it at the tile cap `ρ < 1` and the row cap `ρr < 1` together, with
`1 − g₀' = (1 − g₀)·(1 − ρr)/(1 − ρ)`, at any record with non-negative add and BF16 prices. -/
theorem ttOutTileRowSeed_rowCapAt [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S}
    {d : PearlCDevice} {sem : PearlCSem Q R S} {ρ ρr : ℚ} {g₀ : ℝ} (hadd : 0 ≤ d.prices.add)
    (hbf : 0 ≤ d.prices.bf16) (hρ : ρ < 1) (hρr : ρr < 1)
    (h : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCap d sem ρr) (pearlCTilesDevRev1RowCap d sem ρr)
      (pearlCDomainDev d) g₀) :
    TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCapAt d sem ρ ρr) (pearlCTilesDevRev1RowCapAt d sem ρ ρr)
      (pearlCDomainDev d) (1 - (1 - g₀) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ)) := by
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hr' : (0 : ℚ) < 1 - ρr := by linarith
  have ha : (0 : ℝ) < (((1 - ρr) / (1 - ρ) : ℚ) : ℝ) := by exact_mod_cast div_pos hr' hr
  have hab : (((1 - ρ) / (1 - ρr) : ℚ) : ℝ) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ) = 1 := by
    rw [← Rat.cast_mul, show (1 - ρ) / (1 - ρr) * ((1 - ρr) / (1 - ρ)) = 1 by field_simp, Rat.cast_one]
  refine ⟨h.1, fun U hU hR T hT q hq A hA => le_trans (Pouw.pr_mono ?_) (h.2 U hU hR T hT q hq A hA)⟩
  intro x hx
  have hsum := rowCapAt_credit_sum_le d sem ρ ρr hadd hbf hρ.le hρr U hU.2 x.1 x.2 (A.transcript U x.1 x.2)
  rw [show 1 - (1 - (1 - g₀) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ)) = (1 - g₀) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ) by ring,
    div_mul_eq_div_div] at hx
  have h2 := lt_of_lt_of_le hx hsum
  rw [div_lt_iff₀ ha] at h2
  calc T / (1 - g₀) < _ := h2
    _ = _ := by rw [mul_comm, ← mul_assoc, mul_comm (((1 - ρr) / (1 - ρ) : ℚ) : ℝ), hab, one_mul]

theorem wrefRowRev1_nonneg (d : PearlCDevice) (hbf : 0 ≤ d.prices.bf16) (k n : ℕ) : 0 ≤ wrefRowRev1 d k n := by
  unfold wrefRowRev1
  split_ifs <;> positivity

/-- **One row's `W_ref` at a kept width within `k` is at most a row's share of the unit's**, when `32·G` divides `k > 0`:
the cut chain part is at most `wrefDevRev1`'s, whose promotions `k/(32·G) ≥ 1` per word cover the first one taken out,
and every other term grows with the width. -/
theorem wrefRowRev1_le (d : PearlCDevice) (hadd : 0 ≤ d.prices.add) (s : Shape) (hk : 0 < s.k)
    (hg : d.G = 0 ∨ 32 * d.G ∣ s.k) {k' : ℕ} (hk' : k' ≤ s.k) :
    (s.m : ℚ) * wrefRowRev1 d k' s.n ≤ wrefDevRev1 d s := by
  have hk'q : (k' : ℚ) ≤ s.k := by exact_mod_cast hk'
  have hm : (0 : ℚ) ≤ s.m := by positivity
  have hn : (0 : ℚ) ≤ s.n := by positivity
  have hrest : (k' : ℚ) * Params.pi.r + 2 * d.prices.bf16 * Params.pi.r * s.n +
      ((d.prices.costs.fs : ℚ) + d.prices.costs.qa) * k' ≤
      (s.k : ℚ) * Params.pi.r + 2 * d.prices.bf16 * Params.pi.r * s.n +
        ((d.prices.costs.fs : ℚ) + d.prices.costs.qa) * s.k := by
    have h1 : (k' : ℚ) * Params.pi.r ≤ s.k * Params.pi.r := mul_le_mul_of_nonneg_right hk'q (by positivity)
    have h2 : ((d.prices.costs.fs : ℚ) + d.prices.costs.qa) * k' ≤ ((d.prices.costs.fs : ℚ) + d.prices.costs.qa) * s.k :=
      mul_le_mul_of_nonneg_left hk'q (by positivity)
    linarith
  by_cases hG : d.G = 0
  · have hY : wrefDevRev1 d s = (s.m : ℚ) * ((s.n : ℚ) * s.k + ((s.k : ℚ) * Params.pi.r +
        2 * d.prices.bf16 * Params.pi.r * s.n + ((d.prices.costs.fs : ℚ) + d.prices.costs.qa) * s.k)) := by
      unfold wrefDevRev1 creditDevRev1 creditDev firstAddDev
      simp only [hG, ite_true]
      ring
    have hX : wrefRowRev1 d k' s.n = (s.n : ℚ) * k' + ((k' : ℚ) * Params.pi.r + 2 * d.prices.bf16 * Params.pi.r * s.n +
        ((d.prices.costs.fs : ℚ) + d.prices.costs.qa) * k') := by
      unfold wrefRowRev1
      simp only [hG, ite_true]
      ring
    rw [hY, hX]
    exact mul_le_mul_of_nonneg_left (add_le_add (mul_le_mul_of_nonneg_left hk'q hn) hrest) hm
  · have hpos : (0 : ℚ) < 32 * d.G := by
      have : 0 < d.G := Nat.pos_of_ne_zero hG
      positivity
    have hle : (32 * d.G : ℚ) ≤ s.k := by exact_mod_cast Nat.le_of_dvd hk (hg.resolve_left hG)
    have hY : wrefDevRev1 d s = (s.m : ℚ) * (((s.n : ℚ) * s.k + d.prices.add * s.n * ((s.k : ℚ) / (32 * d.G)) -
        d.prices.add * s.n) + ((s.k : ℚ) * Params.pi.r + 2 * d.prices.bf16 * Params.pi.r * s.n +
          ((d.prices.costs.fs : ℚ) + d.prices.costs.qa) * s.k)) := by
      unfold wrefDevRev1 creditDevRev1 creditDev firstAddDev
      simp only [hG, ite_false]
      ring
    have hX : wrefRowRev1 d k' s.n = max ((s.n : ℚ) * k' + d.prices.add * s.n * ((k' / (32 * d.G) : ℕ) : ℚ) -
        d.prices.add * s.n) 0 + ((k' : ℚ) * Params.pi.r + 2 * d.prices.bf16 * Params.pi.r * s.n +
          ((d.prices.costs.fs : ℚ) + d.prices.costs.qa) * k') := by
      unfold wrefRowRev1
      simp only [hG, ite_false]
      ring
    have hfl : ((k' / (32 * d.G) : ℕ) : ℚ) ≤ (s.k : ℚ) / (32 * d.G) := by
      calc ((k' / (32 * d.G) : ℕ) : ℚ) ≤ (k' : ℚ) / ((32 * d.G : ℕ) : ℚ) := Nat.cast_div_le
        _ ≤ (s.k : ℚ) / (32 * d.G) := by
            push_cast
            exact div_le_div_of_nonneg_right hk'q hpos.le
    have h1 : (1 : ℚ) ≤ (s.k : ℚ) / (32 * d.G) := by
      rw [le_div_iff₀ hpos]
      linarith
    have hpre : 0 ≤ d.prices.add * s.n := mul_nonneg hadd hn
    have h2 : d.prices.add * s.n * ((k' / (32 * d.G) : ℕ) : ℚ) ≤ d.prices.add * s.n * ((s.k : ℚ) / (32 * d.G)) :=
      mul_le_mul_of_nonneg_left hfl hpre
    have h3 : d.prices.add * s.n ≤ d.prices.add * s.n * ((s.k : ℚ) / (32 * d.G)) := le_mul_of_one_le_right hpre h1
    have h4 : (s.n : ℚ) * k' ≤ s.n * s.k := mul_le_mul_of_nonneg_left hk'q hn
    have h5 : (0 : ℚ) ≤ s.n * s.k := by positivity
    rw [hY, hX]
    exact mul_le_mul_of_nonneg_left (add_le_add (max_le (by linarith) (by linarith)) hrest) hm

theorem pearlCDrawRev1_nonneg (d : PearlCDevice) (hbf : 0 ≤ d.prices.bf16) (L : Layout) (g : ℕ) (dc : Decl) :
    0 ≤ pearlCDrawRev1 d L g dc := by
  unfold pearlCDrawRev1
  have := wrefRowRev1_nonneg d hbf dc.kept (L.shape ((auditTiling 64 64 L).unit g)).n
  positivity

/-- **A consistent declaration weighs a tile at most its credited rows' `W_ref`**: with the declared kept width within
the unit's `k` (`32·G` dividing `k > 0`) and the declared credited rows passing Π's rules, the sample's weight under rev1
is at most `wrefDevRev1` times the tile's share of its passing rows and columns. -/
theorem pearlCDrawRev1_le (d : PearlCDevice) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16)
    (sem : PearlCSem Q R S) (L : Layout) (g : ℕ) (dc : Decl) (act : Codes)
    (hm : 0 < (L.shape ((auditTiling 64 64 L).unit g)).m) (hk : 0 < (L.shape ((auditTiling 64 64 L).unit g)).k)
    (hg : d.G = 0 ∨ 32 * d.G ∣ (L.shape ((auditTiling 64 64 L).unit g)).k)
    (hkept : dc.kept ≤ (L.shape ((auditTiling 64 64 L).unit g)).k)
    (hrows : ∀ i ∈ (auditTiling 64 64 L).rows g, i ∈ dc.rows →
      sem.RowOK (L.shape ((auditTiling 64 64 L).unit g)).m (L.shape ((auditTiling 64 64 L).unit g)).k act i) :
    pearlCDrawRev1 d L g dc ≤ wrefDevRev1 d (L.shape ((auditTiling 64 64 L).unit g)) *
      tileShare (L.shape ((auditTiling 64 64 L).unit g))
        (sem.passRows (L.shape ((auditTiling 64 64 L).unit g)) act ((auditTiling 64 64 L).rows g))
        ((auditTiling 64 64 L).cols g) := by
  classical
  set sh := L.shape ((auditTiling 64 64 L).unit g) with hsh
  have hsubr : (auditTiling 64 64 L).rows g ∩ dc.rows ⊆ sem.passRows sh act ((auditTiling 64 64 L).rows g) := by
    intro i hi
    rw [mem_inter] at hi
    unfold PearlCSem.passRows
    exact mem_filter.mpr ⟨hi.1, hrows i hi.1 hi.2⟩
  have hA : (((auditTiling 64 64 L).rows g ∩ dc.rows).card : ℚ) ≤
      (sem.passRows sh act ((auditTiling 64 64 L).rows g)).card := by exact_mod_cast card_le_card hsubr
  have hB : (((auditTiling 64 64 L).cols g ∩ dc.cols).card : ℚ) ≤ ((auditTiling 64 64 L).cols g).card := by
    exact_mod_cast card_le_card inter_subset_left
  have hw := wrefRowRev1_le d hadd sh hk hg hkept
  have hw0 := wrefRowRev1_nonneg d hbf dc.kept sh.n
  have hmq : (0 : ℚ) < sh.m := by exact_mod_cast hm
  unfold pearlCDrawRev1 tileShare
  rw [← hsh]
  rcases Nat.eq_zero_or_pos sh.n with hn | hn
  · simp [hn]
  · have hnq : (0 : ℚ) < sh.n := by exact_mod_cast hn
    rw [div_le_iff₀ hnq]
    calc wrefRowRev1 d dc.kept sh.n *
          ((((auditTiling 64 64 L).rows g ∩ dc.rows).card : ℚ) * (((auditTiling 64 64 L).cols g ∩ dc.cols).card : ℚ))
        ≤ wrefRowRev1 d dc.kept sh.n * (((sem.passRows sh act ((auditTiling 64 64 L).rows g)).card : ℚ) *
            (((auditTiling 64 64 L).cols g).card : ℚ)) :=
          mul_le_mul_of_nonneg_left (mul_le_mul hA hB (by positivity) (by positivity)) hw0
      _ = (sh.m * wrefRowRev1 d dc.kept sh.n) * ((((sem.passRows sh act ((auditTiling 64 64 L).rows g)).card : ℚ) *
            (((auditTiling 64 64 L).cols g).card : ℚ)) / (sh.m * sh.n)) * sh.n := by
          field_simp
      _ ≤ wrefDevRev1 d sh * ((((sem.passRows sh act ((auditTiling 64 64 L).rows g)).card : ℚ) *
            (((auditTiling 64 64 L).cols g).card : ℚ)) / (sh.m * sh.n)) * sh.n := by
          gcongr

end

section Gap

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

/-- **The row-seeded TT_OUT per audit tile at both caps gives `GGapSampledTol` at any per-shape `W_ref`** `W`: rev1's
tiles with the tile cap `ρ`, the row cap `ρr` and `W_ref` and the credited rows' `W_ref` at `W`, on the one-shape domain
at `γ = 1 − (1 − γ₀)/ω`, where `W s = ω·(1 − ρ)·creditDevRev1 ≥ 0`, with the error at the row count
(`gapSampledTol`). -/
theorem pearlCSampledDevRev1RowCapAtWTol (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr : ℚ)
    (W : Shape → ℚ) (s : Shape) (ω : ℚ) (γ₀ γ : ℝ) (hγ₀ : γ₀ < 1) (hω0 : 0 < ω) (hw0 : 0 ≤ W s)
    (hω : W s = ω * ((1 - ρ) * creditDevRev1 d s)) (hγ : γ = 1 - (1 - γ₀) / (ω : ℝ))
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCapAt d sem ρ ρr) (pearlCTilesDevRev1RowCapAt d sem ρ ρr)
      (pearlCDomainDev d) γ₀) :
    GGapSampledTol CM { pearlCProtocolDevRev1RowCapAt d sem ρ ρr with Wref := fun L u => (W (L.shape u) : ℝ) }
      { pearlCTilesDevRev1RowCapAt d sem ρ ρr with
        Wref := fun L g =>
          let tl := auditTiling 64 64 L
          let sh := L.shape (tl.unit g)
          ((W sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ)
        Wcred := fun L g act =>
          let tl := auditTiling 64 64 L
          let sh := L.shape (tl.unit g)
          ((W sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ) }
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) γ (fun q N => δPearlC q (N * s.m)) := by
  rw [hγ]
  refine gapSampledTol CM _ _ _ γ₀ (ω : ℝ) _ hγ₀ (by exact_mod_cast hω0) ?_ ?_
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

/-- `pearlCSampledDevRev1RowCapAtWTol` under the audit that accepts only checked tiles (`gapSampled_of_tol`). -/
theorem pearlCSampledDevRev1RowCapAtW (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr : ℚ)
    (W : Shape → ℚ) (s : Shape) (ω : ℚ) (γ₀ γ : ℝ) (hγ₀ : γ₀ < 1) (hω0 : 0 < ω) (hw0 : 0 ≤ W s)
    (hω : W s = ω * ((1 - ρ) * creditDevRev1 d s)) (hγ : γ = 1 - (1 - γ₀) / (ω : ℝ))
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCapAt d sem ρ ρr) (pearlCTilesDevRev1RowCapAt d sem ρ ρr)
      (pearlCDomainDev d) γ₀) :
    GGapSampled CM { pearlCProtocolDevRev1RowCapAt d sem ρ ρr with Wref := fun L u => (W (L.shape u) : ℝ) }
      { pearlCTilesDevRev1RowCapAt d sem ρ ρr with
        Wref := fun L g =>
          let tl := auditTiling 64 64 L
          let sh := L.shape (tl.unit g)
          ((W sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ)
        Wcred := fun L g act =>
          let tl := auditTiling 64 64 L
          let sh := L.shape (tl.unit g)
          ((W sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ) }
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) γ (fun q N => δPearlC q (N * s.m)) :=
  gapSampled_of_tol (pearlCSampledDevRev1RowCapAtWTol CM d sem ρ ρr W s ω γ₀ γ hγ₀ hω0 hw0 hω hγ hTT)

/-- `pearlCSampledDevRev1RowCapAtWTol` from the row-seeded TT_OUT per tile at (C2) with the row cap `ρr` and
`g₀ = 1/400` (`ttOutTileRowSeed_rowCapAt`), at rev1's `W_ref`: `g = 1 − (399/400)·(1 − ρr)/((1 − ρ)·ω)`, where
`wrefDevRev1 = ω·(1 − ρ)·creditDevRev1 ≥ 0`. -/
theorem pearlCSampledDevRev1RowCapAtTol (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr : ℚ)
    (s : Shape) (ω g : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hρ : ρ < 1) (hρr : ρr < 1)
    (hω0 : 0 < ω) (hw0 : 0 ≤ wrefDevRev1 d s) (hω : wrefDevRev1 d s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hg : g = 1 - (1 - 1 / 400) * (1 - ρr) / ((1 - ρ) * ω))
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCap d sem ρr) (pearlCTilesDevRev1RowCap d sem ρr)
      (pearlCDomainDev d) (1 / 400)) :
    GGapSampledTol CM (pearlCProtocolDevRev1RowCapAt d sem ρ ρr) (pearlCTilesDevRev1RowCapAt d sem ρ ρr)
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (g : ℝ) (fun q N => δPearlC q (N * s.m)) := by
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hr' : (0 : ℚ) < 1 - ρr := by linarith
  refine pearlCSampledDevRev1RowCapAtWTol CM d sem ρ ρr (wrefDevRev1 d) s ω _ (g : ℝ) ?_ hω0 hw0 hω ?_
    (ttOutTileRowSeed_rowCapAt hadd hbf hρ hρr hTT)
  · have : (0 : ℝ) < (1 - 1 / 400) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ) :=
      mul_pos (by norm_num) (by exact_mod_cast div_pos hr' hr)
    linarith
  · rw [hg]
    have h1 : ((1 - ρ : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hr.ne'
    have h3 : ((ω : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hω0.ne'
    push_cast at h1 h3 ⊢
    field_simp
    ring

/-- `pearlCSampledDevRev1RowCapAtTol` under the audit that accepts only checked tiles (`gapSampled_of_tol`). -/
theorem pearlCSampledDevRev1RowCapAt (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr : ℚ)
    (s : Shape) (ω g : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hρ : ρ < 1) (hρr : ρr < 1)
    (hω0 : 0 < ω) (hw0 : 0 ≤ wrefDevRev1 d s) (hω : wrefDevRev1 d s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hg : g = 1 - (1 - 1 / 400) * (1 - ρr) / ((1 - ρ) * ω))
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCap d sem ρr) (pearlCTilesDevRev1RowCap d sem ρr)
      (pearlCDomainDev d) (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1RowCapAt d sem ρ ρr) (pearlCTilesDevRev1RowCapAt d sem ρ ρr)
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (g : ℝ) (fun q N => δPearlC q (N * s.m)) :=
  gapSampled_of_tol
    (pearlCSampledDevRev1RowCapAtTol CM d sem ρ ρr s ω g hadd hbf hρ hρr hω0 hw0 hω hg hTT)

/-- As `pearlCSampledDevRev1RowCapAt`, against the honest kernel's `W_ref`, where
`wrefDevRev1K d c = ω·(1 − ρ)·creditDevRev1 ≥ 0`. -/
theorem pearlCSampledDevRev1KRowCapAt (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S)
    (ρ ρr c : ℚ) (s : Shape) (ω g : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hρ : ρ < 1)
    (hρr : ρr < 1) (hω0 : 0 < ω) (hw0 : 0 ≤ wrefDevRev1K d c s)
    (hω : wrefDevRev1K d c s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hg : g = 1 - (1 - 1 / 400) * (1 - ρr) / ((1 - ρ) * ω))
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCap d sem ρr) (pearlCTilesDevRev1RowCap d sem ρr)
      (pearlCDomainDev d) (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1KRowCapAt d sem ρ ρr c) (pearlCTilesDevRev1KRowCapAt d sem ρ ρr c)
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (g : ℝ) (fun q N => δPearlC q (N * s.m)) := by
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hr' : (0 : ℚ) < 1 - ρr := by linarith
  refine pearlCSampledDevRev1RowCapAtW CM d sem ρ ρr (wrefDevRev1K d c) s ω _ (g : ℝ) ?_ hω0 hw0 hω ?_
    (ttOutTileRowSeed_rowCapAt hadd hbf hρ hρr hTT)
  · have : (0 : ℝ) < (1 - 1 / 400) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ) :=
      mul_pos (by norm_num) (by exact_mod_cast div_pos hr' hr)
    linarith
  · rw [hg]
    have h1 : ((1 - ρ : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hr.ne'
    have h3 : ((ω : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hω0.ne'
    push_cast at h1 h3 ⊢
    field_simp
    ring

/-- **The row-seeded TT_OUT per audit tile at both caps gives `GGapSampledDrawTol`**, the audit sampled by rev1's tickets
(`pearlCDrawRev1`), at `γ = 1 − (399/400)·(1 − ρr)/((1 − ρ)·ω)` where `wrefDevRev1 = ω·(1 − ρ)·creditDevRev1`
(`gapSampledDrawTol`): a consistent declaration weighs a tile at most its credited rows' `W_ref`
(`pearlCDrawRev1_le`), which is `ω` times its credit. -/
theorem pearlCSampledDrawDevRev1RowCapAtTol (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S)
    (ρ ρr : ℚ) (s : Shape) (ω γ : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hρ : ρ < 1) (hρr : ρr < 1)
    (hω0 : 0 < ω) (hω : wrefDevRev1 d s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hγ : γ = 1 - (1 - 1 / 400) * (1 - ρr) / ((1 - ρ) * ω))
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCap d sem ρr) (pearlCTilesDevRev1RowCap d sem ρr)
      (pearlCDomainDev d) (1 / 400)) :
    GGapSampledDrawTol CM (pearlCProtocolDevRev1RowCapAt d sem ρ ρr) (pearlCTilesDevRev1RowCapAt d sem ρ ρr)
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (pearlCDrawRev1 d) (γ : ℝ)
      (fun q N => δPearlC q (N * s.m)) := by
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hr' : (0 : ℚ) < 1 - ρr := by linarith
  have hγ' : (γ : ℝ) = 1 - (1 - (1 - (1 - 1 / 400) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ))) / (ω : ℝ) := by
    rw [hγ]
    have h1 : ((1 - ρ : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hr.ne'
    have h3 : ((ω : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hω0.ne'
    push_cast at h1 h3 ⊢
    field_simp
    ring
  rw [hγ']
  refine gapSampledDrawTol CM _ _ _ _ _ (ω : ℝ) _ ?_ (by exact_mod_cast hω0) ?_ ?_ ?_
    (ttOutTile_of_ttOutTileRowSeed (ttOutTileRowSeed_rowCapAt hadd hbf hρ hρr hTT))
  · have : (0 : ℝ) < (1 - 1 / 400) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ) :=
      mul_pos (by norm_num) (by exact_mod_cast div_pos hr' hr)
    linarith
  · intro U _ g _ dc
    exact pearlCDrawRev1_nonneg d hbf U.layout g dc
  · intro U hU H s' g hg act _
    have hD := hU.2.1.1 _ (auditTiling_unit_lt 64 64 U.layout hg)
    have hc := creditDevRev1_nonneg d hadd hbf _ hD.2.2.1 hD.2.2.2.2.1
    simp only [pearlCTilesDevRev1RowCapAt, pearlCTilesDevRev1]
    have := tileShare_nonneg (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
      (sem.passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) act ((auditTiling 64 64 U.layout).rows g))
      ((auditTiling 64 64 U.layout).cols g)
    exact_mod_cast mul_nonneg (mul_nonneg hr.le hc) this
  · intro U hU H s' g hg act dc _ hkept hrows
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hD := hU.2.1.1 _ hu
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.1.2 _ hu
    have hle := pearlCDrawRev1_le d hadd hbf sem U.layout g dc act hD.1 hD.2.2.1 hD.2.2.2.2.1 hkept hrows
    simp only [pearlCTilesDevRev1RowCapAt, pearlCTilesDevRev1]
    rw [hsh] at hle ⊢
    rw [hω] at hle
    have hle' : pearlCDrawRev1 d U.layout g dc ≤ ω * ((1 - ρ) * creditDevRev1 d s *
        tileShare s (sem.passRows s act ((auditTiling 64 64 U.layout).rows g)) ((auditTiling 64 64 U.layout).cols g)) :=
      hle.trans_eq (by ring)
    exact_mod_cast hle'

/-- `pearlCSampledDrawDevRev1RowCapAtTol` under the audit that accepts only checked tiles (`gapSampledDraw_of_tol`). -/
theorem pearlCSampledDrawDevRev1RowCapAt (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S)
    (ρ ρr : ℚ) (s : Shape) (ω g : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hρ : ρ < 1) (hρr : ρr < 1)
    (hω0 : 0 < ω) (hω : wrefDevRev1 d s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hg : g = 1 - (1 - 1 / 400) * (1 - ρr) / ((1 - ρ) * ω))
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCap d sem ρr) (pearlCTilesDevRev1RowCap d sem ρr)
      (pearlCDomainDev d) (1 / 400)) :
    GGapSampledDraw CM (pearlCProtocolDevRev1RowCapAt d sem ρ ρr) (pearlCTilesDevRev1RowCapAt d sem ρ ρr)
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (pearlCDrawRev1 d) (g : ℝ)
      (fun q N => δPearlC q (N * s.m)) :=
  gapSampledDraw_of_tol
    (pearlCSampledDrawDevRev1RowCapAtTol CM d sem ρ ρr s ω g hadd hbf hρ hρr hω0 hω hg hTT)

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
tile cap 1/400 and the row cap 1/100, under the tolerant audit sampled as the run samples, at the deployed table**
(`pearlCDrawRev1`, the records' tickets; `u 0 = ε_s`, `u x = 1` otherwise), from `TTOutTilePearlCH100Rev1` at 1/100, C1
and C3: `g = 1 − (399/400)·(99/100)/((399/400)·ω) = 32437/2521000` (1.28667%) at `ω = 25210/25137`, with the error at
the `m·N` rows. -/
theorem pearlCSampledH100v1RowSeedRowCap100_6144x256Tol (m : ℕ)
    (hTT : TTOutTilePearlCH100Rev1 CM sem (1 / 100)) (hN : RowSeedNoise sem sem') (hCM : CM.SplitClosed) :
    GGapSampledDrawTolThreshold CM (pearlCProtocolDevRev1RowCapAt devH100 sem' (1 / 400) (1 / 100))
      (pearlCTilesDevRev1RowCapAt devH100 sem' (1 / 400) (1 / 100))
      (fun L => pearlCDomainDevAt devH100 ⟨m, 6144, 256⟩ L ∧ rowsOf L ≤ 2 ^ 64) (pearlCDrawRev1 devH100)
      ((32437 / 2521000 : ℚ) : ℝ) (fun q N => δPearlC q (N * m)) :=
  gapSampledDrawTolThreshold_of_tol
    (pearlCSampledDrawDevRev1RowCapAtTol CM devH100 sem' (1 / 400) (1 / 100) ⟨m, 6144, 256⟩ (25210 / 25137) _
      devH100_add devH100_bf16 (by norm_num) (by norm_num) (by norm_num)
      (by rw [wrefDevRev1_h100_n6144k256, creditDevRev1_h100_n6144k256]; ring)
      (by norm_num)
      (ttOutTileRowSeed_of_ttOutTile hTT hN hCM devH100_rowLocal (by norm_num) devH100_add devH100_bf16))

/-- **GLM-5.3's down projection on the H100 record under the audit sampled as the run samples that accepts only checked
tiles**: `pearlCSampledH100v1RowSeedRowCap100_6144x256Tol`'s case (`gapSampledDraw_of_threshold`),
`g = 32437/2521000` (1.28667%), with the error at the `m·N` rows. -/
theorem pearlCSampledH100v1RowSeedRowCap100_6144x256 (m : ℕ)
    (hTT : TTOutTilePearlCH100Rev1 CM sem (1 / 100)) (hN : RowSeedNoise sem sem') (hCM : CM.SplitClosed) :
    GGapSampledDraw CM (pearlCProtocolDevRev1RowCapAt devH100 sem' (1 / 400) (1 / 100))
      (pearlCTilesDevRev1RowCapAt devH100 sem' (1 / 400) (1 / 100))
      (fun L => pearlCDomainDevAt devH100 ⟨m, 6144, 256⟩ L ∧ rowsOf L ≤ 2 ^ 64) (pearlCDrawRev1 devH100)
      ((32437 / 2521000 : ℚ) : ℝ) (fun q N => δPearlC q (N * m)) :=
  gapSampledDraw_of_threshold (pearlCSampledH100v1RowSeedRowCap100_6144x256Tol CM m hTT hN hCM)

/-- **The code-form seed (`-h3`) of `pearlCSampledH100v1RowSeedRowCap100_6144x256Tol`**, the served record
`pearl-c-h100-v1-h3` at a census k, adding C1-h3 and C5: under the tolerant audit sampled as the run samples, at the
deployed table, `g = 32437/2521000` (1.28667%), with the error at the `m·N` rows. -/
theorem pearlCSampledH100v1RowSeedCodeRowCap100_6144x256Tol {π : Layout → Q ≃ Q} (m : ℕ)
    (hTT : TTOutTilePearlCH100Rev1 CM sem (1 / 100)) (hN : RowSeedNoise sem sem') (hC : CodeSeedNoise sem' semC π)
    (hCM : CM.SplitClosed) (hCM' : CM.RelabelClosed) :
    GGapSampledDrawTolThreshold CM (pearlCProtocolDevRev1RowCapAt devH100 semC (1 / 400) (1 / 100))
      (pearlCTilesDevRev1RowCapAt devH100 semC (1 / 400) (1 / 100))
      (fun L => pearlCDomainDevAt devH100 ⟨m, 6144, 256⟩ L ∧ rowsOf L ≤ 2 ^ 64) (pearlCDrawRev1 devH100)
      ((32437 / 2521000 : ℚ) : ℝ) (fun q N => δPearlC q (N * m)) :=
  gapSampledDrawTolThreshold_of_tol
    (pearlCSampledDrawDevRev1RowCapAtTol CM devH100 semC (1 / 400) (1 / 100) ⟨m, 6144, 256⟩ (25210 / 25137) _
      devH100_add devH100_bf16 (by norm_num) (by norm_num) (by norm_num)
      (by rw [wrefDevRev1_h100_n6144k256, creditDevRev1_h100_n6144k256]; ring)
      (by norm_num)
      (ttOutTileRowSeedCode_of_ttOutTile hTT hN hC hCM hCM' devH100_rowLocal (by norm_num) devH100_add devH100_bf16))

/-- **The code-form seed (`-h3`) of `pearlCSampledH100v1RowSeedRowCap100_6144x256`**, the served record
`pearl-c-h100-v1-h3` at a census k, adding C1-h3 and C5: `g = 32437/2521000` (1.28667%), with the error at the `m·N`
rows (`pearlCSampledH100v1RowSeedCodeRowCap100_6144x256Tol`'s case, `gapSampledDraw_of_threshold`). -/
theorem pearlCSampledH100v1RowSeedCodeRowCap100_6144x256 {π : Layout → Q ≃ Q} (m : ℕ)
    (hTT : TTOutTilePearlCH100Rev1 CM sem (1 / 100)) (hN : RowSeedNoise sem sem') (hC : CodeSeedNoise sem' semC π)
    (hCM : CM.SplitClosed) (hCM' : CM.RelabelClosed) :
    GGapSampledDraw CM (pearlCProtocolDevRev1RowCapAt devH100 semC (1 / 400) (1 / 100))
      (pearlCTilesDevRev1RowCapAt devH100 semC (1 / 400) (1 / 100))
      (fun L => pearlCDomainDevAt devH100 ⟨m, 6144, 256⟩ L ∧ rowsOf L ≤ 2 ^ 64) (pearlCDrawRev1 devH100)
      ((32437 / 2521000 : ℚ) : ℝ) (fun q N => δPearlC q (N * m)) :=
  gapSampledDraw_of_threshold (pearlCSampledH100v1RowSeedCodeRowCap100_6144x256Tol CM m hTT hN hC hCM hCM')

end Gap

end Pouw.PearlC
