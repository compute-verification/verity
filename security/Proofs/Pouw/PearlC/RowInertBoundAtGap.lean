module

public import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeedK
public import Proofs.Pouw.PearlC.RowSeedCode
public import Proofs.Pouw.PearlC.DeviceGap
public import Proofs.Pouw.PearlC.TileSampleGap
public import Proofs.Pouw.PearlC.Records

/-! Per-row seeds at a census k: the tile cap `ρ` and a wider row cap `ρr`, the row-seeded g resting on rev1's
TT_OUT per tile at `ρr`. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

section
variable {Q R S : Type}

open Classical in
theorem rowInertBoundAt_credit_sum_le (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr : ℚ) (hadd : 0 ≤ d.prices.add)
    (hbf : 0 ≤ d.prices.bf16) (hρ : ρ ≤ 1) (hρr : ρr < 1) (U : Workload) (hD : pearlCDomainDev d U.layout)
    (H : Q → R) (s : S) (τ : Trace) :
    ∑ g ∈ goodTiles (pearlCProtocolDevRev1RowInertBoundAt d sem ρ ρr) (pearlCTilesDevRev1RowInertBoundAt d sem ρ ρr) U H s τ,
        (pearlCTilesDevRev1RowInertBoundAt d sem ρ ρr).credit H s U g
          (actOf τ (((pearlCTilesDevRev1RowInertBoundAt d sem ρ ρr).tiling U.layout).unit g)) ≤
      (((1 - ρ) / (1 - ρr) : ℚ) : ℝ) *
        ∑ g ∈ goodTiles (pearlCProtocolDevRev1RowInertBound d sem ρr) (pearlCTilesDevRev1RowInertBound d sem ρr) U H s τ,
          (pearlCTilesDevRev1RowInertBound d sem ρr).credit H s U g
            (actOf τ (((pearlCTilesDevRev1RowInertBound d sem ρr).tiling U.layout).unit g)) := by
  set tl := auditTiling 64 64 U.layout
  have hr : (0 : ℚ) < 1 - ρr := by linarith
  have hC : ∀ g < tl.NT, 0 ≤ creditDevRev1 d (U.layout.shape (tl.unit g)) := fun g hg =>
    have hs := hD _ (auditTiling_unit_lt 64 64 U.layout hg)
    creditDevRev1_nonneg d hadd hbf _ hs.2.2.1 hs.2.2.2.2.1
  have hsub : goodTiles (pearlCProtocolDevRev1RowInertBoundAt d sem ρ ρr) (pearlCTilesDevRev1RowInertBoundAt d sem ρ ρr) U H s τ ⊆
      goodTiles (pearlCProtocolDevRev1RowInertBound d sem ρr) (pearlCTilesDevRev1RowInertBound d sem ρr) U H s τ := by
    intro g hg
    unfold goodTiles at hg ⊢
    rw [mem_filter] at hg ⊢
    obtain ⟨hg, c, hc, hcap, hw⟩ := hg
    exact ⟨hg, c, hc, hcap.2, hw⟩
  rw [mul_sum]
  calc _ = ∑ g ∈ goodTiles (pearlCProtocolDevRev1RowInertBoundAt d sem ρ ρr) (pearlCTilesDevRev1RowInertBoundAt d sem ρ ρr) U H s τ,
          (((1 - ρ) / (1 - ρr) : ℚ) : ℝ) *
            (pearlCTilesDevRev1RowInertBound d sem ρr).credit H s U g (actOf τ (tl.unit g)) := by
        refine sum_congr rfl fun g _ => ?_
        simp only [pearlCTilesDevRev1RowInertBoundAt, pearlCTilesDevRev1RowInertBound, pearlCTilesDevRev1]
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
        simp only [pearlCTilesDevRev1RowInertBound, pearlCTilesDevRev1]
        have hsh := tileShare_nonneg (U.layout.shape (tl.unit g))
          (sem.passRows (U.layout.shape (tl.unit g)) (actOf τ (tl.unit g)) (tl.rows g)) (tl.cols g)
        have hq : (0 : ℚ) ≤ (1 - ρ) / (1 - ρr) := div_nonneg (by linarith) hr.le
        have : (0 : ℚ) ≤ (1 - ρ) / (1 - ρr) * ((1 - ρr) * creditDevRev1 d (U.layout.shape (tl.unit g)) *
            tileShare (U.layout.shape (tl.unit g))
              (sem.passRows (U.layout.shape (tl.unit g)) (actOf τ (tl.unit g)) (tl.rows g)) (tl.cols g)) :=
          mul_nonneg hq (mul_nonneg (mul_nonneg hr.le (hC g hgr)) hsh)
        exact_mod_cast this

theorem ttOutTileRowSeed_rowInertBoundAt [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S}
    {d : PearlCDevice} {sem : PearlCSem Q R S} {ρ ρr : ℚ} {g₀ : ℝ} (hadd : 0 ≤ d.prices.add)
    (hbf : 0 ≤ d.prices.bf16) (hρ : ρ < 1) (hρr : ρr < 1)
    (h : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowInertBound d sem ρr) (pearlCTilesDevRev1RowInertBound d sem ρr)
      (pearlCDomainDev d) g₀) :
    TTOutTileRowSeed CM (pearlCProtocolDevRev1RowInertBoundAt d sem ρ ρr) (pearlCTilesDevRev1RowInertBoundAt d sem ρ ρr)
      (pearlCDomainDev d) (1 - (1 - g₀) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ)) := by
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hr' : (0 : ℚ) < 1 - ρr := by linarith
  have ha : (0 : ℝ) < (((1 - ρr) / (1 - ρ) : ℚ) : ℝ) := by exact_mod_cast div_pos hr' hr
  have hab : (((1 - ρ) / (1 - ρr) : ℚ) : ℝ) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ) = 1 := by
    rw [← Rat.cast_mul, show (1 - ρ) / (1 - ρr) * ((1 - ρr) / (1 - ρ)) = 1 by field_simp, Rat.cast_one]
  refine ⟨h.1, fun U hU hR T hT q hq A hA => le_trans (Pouw.pr_mono ?_) (h.2 U hU hR T hT q hq A hA)⟩
  intro x hx
  have hsum := rowInertBoundAt_credit_sum_le d sem ρ ρr hadd hbf hρ.le hρr U hU.2 x.1 x.2 (A.trace U x.1 x.2)
  rw [show 1 - (1 - (1 - g₀) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ)) = (1 - g₀) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ) by ring,
    div_mul_eq_div_div] at hx
  have h2 := lt_of_lt_of_le hx hsum
  rw [div_lt_iff₀ ha] at h2
  calc T / (1 - g₀) < _ := h2
    _ = _ := by rw [mul_comm, ← mul_assoc, mul_comm (((1 - ρr) / (1 - ρ) : ℚ) : ℝ), hab, one_mul]

theorem wrefRowRev1_nonneg (d : PearlCDevice) (hbf : 0 ≤ d.prices.bf16) (k n : ℕ) : 0 ≤ wrefRowRev1 d k n := by
  unfold wrefRowRev1
  split_ifs <;> positivity

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

theorem pearlCSampledDevRev1RowInertBoundAtWTol (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr : ℚ)
    (W : Shape → ℚ) (s : Shape) (ω : ℚ) (γ₀ γ : ℝ) (hγ₀ : γ₀ < 1) (hω0 : 0 < ω) (hw0 : 0 ≤ W s)
    (hω : W s = ω * ((1 - ρ) * creditDevRev1 d s)) (hγ : γ = 1 - (1 - γ₀) / (ω : ℝ))
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowInertBoundAt d sem ρ ρr) (pearlCTilesDevRev1RowInertBoundAt d sem ρ ρr)
      (pearlCDomainDev d) γ₀) :
    GGapSampledTol CM { pearlCProtocolDevRev1RowInertBoundAt d sem ρ ρr with Wref := fun L u => (W (L.shape u) : ℝ) }
      { pearlCTilesDevRev1RowInertBoundAt d sem ρ ρr with
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
    simp only [pearlCTilesDevRev1RowInertBoundAt, pearlCTilesDevRev1]
    rw [hsh, hω]
    push_cast
    exact le_of_eq (by ring)
  · intro U hU g hg act
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.1.2 _ hu
    dsimp only
    rw [hsh]
    exact_mod_cast mul_le_mul_of_nonneg_left (tileShare_mono _ (passRows_subset sem _ _ _) _) hw0

theorem pearlCSampledDevRev1RowInertBoundAtW (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr : ℚ)
    (W : Shape → ℚ) (s : Shape) (ω : ℚ) (γ₀ γ : ℝ) (hγ₀ : γ₀ < 1) (hω0 : 0 < ω) (hw0 : 0 ≤ W s)
    (hω : W s = ω * ((1 - ρ) * creditDevRev1 d s)) (hγ : γ = 1 - (1 - γ₀) / (ω : ℝ))
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowInertBoundAt d sem ρ ρr) (pearlCTilesDevRev1RowInertBoundAt d sem ρ ρr)
      (pearlCDomainDev d) γ₀) :
    GGapSampled CM { pearlCProtocolDevRev1RowInertBoundAt d sem ρ ρr with Wref := fun L u => (W (L.shape u) : ℝ) }
      { pearlCTilesDevRev1RowInertBoundAt d sem ρ ρr with
        Wref := fun L g =>
          let tl := auditTiling 64 64 L
          let sh := L.shape (tl.unit g)
          ((W sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ)
        Wcred := fun L g act =>
          let tl := auditTiling 64 64 L
          let sh := L.shape (tl.unit g)
          ((W sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ) }
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) γ (fun q N => δPearlC q (N * s.m)) :=
  gapSampled_of_tol (pearlCSampledDevRev1RowInertBoundAtWTol CM d sem ρ ρr W s ω γ₀ γ hγ₀ hω0 hw0 hω hγ hTT)

theorem pearlCSampledDevRev1RowInertBoundAtTol (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr : ℚ)
    (s : Shape) (ω g : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hρ : ρ < 1) (hρr : ρr < 1)
    (hω0 : 0 < ω) (hw0 : 0 ≤ wrefDevRev1 d s) (hω : wrefDevRev1 d s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hg : g = 1 - (1 - 1 / 400) * (1 - ρr) / ((1 - ρ) * ω))
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowInertBound d sem ρr) (pearlCTilesDevRev1RowInertBound d sem ρr)
      (pearlCDomainDev d) (1 / 400)) :
    GGapSampledTol CM (pearlCProtocolDevRev1RowInertBoundAt d sem ρ ρr) (pearlCTilesDevRev1RowInertBoundAt d sem ρ ρr)
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (g : ℝ) (fun q N => δPearlC q (N * s.m)) := by
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hr' : (0 : ℚ) < 1 - ρr := by linarith
  refine pearlCSampledDevRev1RowInertBoundAtWTol CM d sem ρ ρr (wrefDevRev1 d) s ω _ (g : ℝ) ?_ hω0 hw0 hω ?_
    (ttOutTileRowSeed_rowInertBoundAt hadd hbf hρ hρr hTT)
  · have : (0 : ℝ) < (1 - 1 / 400) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ) :=
      mul_pos (by norm_num) (by exact_mod_cast div_pos hr' hr)
    linarith
  · rw [hg]
    have h1 : ((1 - ρ : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hr.ne'
    have h3 : ((ω : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hω0.ne'
    push_cast at h1 h3 ⊢
    field_simp
    ring

theorem pearlCSampledDevRev1RowInertBoundAt (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr : ℚ)
    (s : Shape) (ω g : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hρ : ρ < 1) (hρr : ρr < 1)
    (hω0 : 0 < ω) (hw0 : 0 ≤ wrefDevRev1 d s) (hω : wrefDevRev1 d s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hg : g = 1 - (1 - 1 / 400) * (1 - ρr) / ((1 - ρ) * ω))
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowInertBound d sem ρr) (pearlCTilesDevRev1RowInertBound d sem ρr)
      (pearlCDomainDev d) (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1RowInertBoundAt d sem ρ ρr) (pearlCTilesDevRev1RowInertBoundAt d sem ρ ρr)
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (g : ℝ) (fun q N => δPearlC q (N * s.m)) :=
  gapSampled_of_tol
    (pearlCSampledDevRev1RowInertBoundAtTol CM d sem ρ ρr s ω g hadd hbf hρ hρr hω0 hw0 hω hg hTT)

theorem pearlCSampledDevRev1KRowInertBoundAt (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S)
    (ρ ρr c : ℚ) (s : Shape) (ω g : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hρ : ρ < 1)
    (hρr : ρr < 1) (hω0 : 0 < ω) (hw0 : 0 ≤ wrefDevRev1K d c s)
    (hω : wrefDevRev1K d c s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hg : g = 1 - (1 - 1 / 400) * (1 - ρr) / ((1 - ρ) * ω))
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowInertBound d sem ρr) (pearlCTilesDevRev1RowInertBound d sem ρr)
      (pearlCDomainDev d) (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1KRowInertBoundAt d sem ρ ρr c) (pearlCTilesDevRev1KRowInertBoundAt d sem ρ ρr c)
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (g : ℝ) (fun q N => δPearlC q (N * s.m)) := by
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hr' : (0 : ℚ) < 1 - ρr := by linarith
  refine pearlCSampledDevRev1RowInertBoundAtW CM d sem ρ ρr (wrefDevRev1K d c) s ω _ (g : ℝ) ?_ hω0 hw0 hω ?_
    (ttOutTileRowSeed_rowInertBoundAt hadd hbf hρ hρr hTT)
  · have : (0 : ℝ) < (1 - 1 / 400) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ) :=
      mul_pos (by norm_num) (by exact_mod_cast div_pos hr' hr)
    linarith
  · rw [hg]
    have h1 : ((1 - ρ : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hr.ne'
    have h3 : ((ω : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hω0.ne'
    push_cast at h1 h3 ⊢
    field_simp
    ring

theorem pearlCSampledDrawDevRev1RowInertBoundAtTol (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S)
    (ρ ρr : ℚ) (s : Shape) (ω γ : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hρ : ρ < 1) (hρr : ρr < 1)
    (hω0 : 0 < ω) (hω : wrefDevRev1 d s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hγ : γ = 1 - (1 - 1 / 400) * (1 - ρr) / ((1 - ρ) * ω))
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowInertBound d sem ρr) (pearlCTilesDevRev1RowInertBound d sem ρr)
      (pearlCDomainDev d) (1 / 400)) :
    GGapSampledDrawTol CM (pearlCProtocolDevRev1RowInertBoundAt d sem ρ ρr) (pearlCTilesDevRev1RowInertBoundAt d sem ρ ρr)
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
    (ttOutTile_of_ttOutTileRowSeed (ttOutTileRowSeed_rowInertBoundAt hadd hbf hρ hρr hTT))
  · have : (0 : ℝ) < (1 - 1 / 400) * (((1 - ρr) / (1 - ρ) : ℚ) : ℝ) :=
      mul_pos (by norm_num) (by exact_mod_cast div_pos hr' hr)
    linarith
  · intro U _ g _ dc
    exact pearlCDrawRev1_nonneg d hbf U.layout g dc
  · intro U hU H s' g hg act _
    have hD := hU.2.1.1 _ (auditTiling_unit_lt 64 64 U.layout hg)
    have hc := creditDevRev1_nonneg d hadd hbf _ hD.2.2.1 hD.2.2.2.2.1
    simp only [pearlCTilesDevRev1RowInertBoundAt, pearlCTilesDevRev1]
    have := tileShare_nonneg (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
      (sem.passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) act ((auditTiling 64 64 U.layout).rows g))
      ((auditTiling 64 64 U.layout).cols g)
    exact_mod_cast mul_nonneg (mul_nonneg hr.le hc) this
  · intro U hU H s' g hg act dc _ hkept hrows
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hD := hU.2.1.1 _ hu
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.1.2 _ hu
    have hle := pearlCDrawRev1_le d hadd hbf sem U.layout g dc act hD.1 hD.2.2.1 hD.2.2.2.2.1 hkept hrows
    simp only [pearlCTilesDevRev1RowInertBoundAt, pearlCTilesDevRev1]
    rw [hsh] at hle ⊢
    rw [hω] at hle
    have hle' : pearlCDrawRev1 d U.layout g dc ≤ ω * ((1 - ρ) * creditDevRev1 d s *
        tileShare s (sem.passRows s act ((auditTiling 64 64 U.layout).rows g)) ((auditTiling 64 64 U.layout).cols g)) :=
      hle.trans_eq (by ring)
    exact_mod_cast hle'

theorem pearlCSampledDrawDevRev1RowInertBoundAt (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S)
    (ρ ρr : ℚ) (s : Shape) (ω g : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hρ : ρ < 1) (hρr : ρr < 1)
    (hω0 : 0 < ω) (hω : wrefDevRev1 d s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hg : g = 1 - (1 - 1 / 400) * (1 - ρr) / ((1 - ρ) * ω))
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowInertBound d sem ρr) (pearlCTilesDevRev1RowInertBound d sem ρr)
      (pearlCDomainDev d) (1 / 400)) :
    GGapSampledDraw CM (pearlCProtocolDevRev1RowInertBoundAt d sem ρ ρr) (pearlCTilesDevRev1RowInertBoundAt d sem ρ ρr)
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (pearlCDrawRev1 d) (g : ℝ)
      (fun q N => δPearlC q (N * s.m)) :=
  gapSampledDraw_of_tol
    (pearlCSampledDrawDevRev1RowInertBoundAtTol CM d sem ρ ρr s ω g hadd hbf hρ hρr hω0 hω hg hTT)

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

theorem pearlCSampledH100v1RowSeedRowInertShare100_6144x256Tol (m : ℕ)
    (hTT : TTOutTilePearlCH100Rev1 CM sem (1 / 100)) (hN : RowSeedNoise sem sem') (hCM : CM.SplitClosed) :
    GGapSampledDrawTolThreshold CM (pearlCProtocolDevRev1RowInertBoundAt devH100 sem' (1 / 400) (1 / 100))
      (pearlCTilesDevRev1RowInertBoundAt devH100 sem' (1 / 400) (1 / 100))
      (fun L => pearlCDomainDevAt devH100 ⟨m, 6144, 256⟩ L ∧ rowsOf L ≤ 2 ^ 64) (pearlCDrawRev1 devH100)
      ((32437 / 2521000 : ℚ) : ℝ) (fun q N => δPearlC q (N * m)) :=
  gapSampledDrawTolThreshold_of_tol
    (pearlCSampledDrawDevRev1RowInertBoundAtTol CM devH100 sem' (1 / 400) (1 / 100) ⟨m, 6144, 256⟩ (25210 / 25137) _
      devH100_add devH100_bf16 (by norm_num) (by norm_num) (by norm_num)
      (by rw [wrefDevRev1_h100_n6144k256, creditDevRev1_h100_n6144k256]; ring)
      (by norm_num)
      (ttOutTileRowSeed_of_ttOutTile hTT hN hCM devH100_rowLocal (by norm_num) devH100_add devH100_bf16))

theorem pearlCSampledH100v1RowSeedRowInertShare100_6144x256 (m : ℕ)
    (hTT : TTOutTilePearlCH100Rev1 CM sem (1 / 100)) (hN : RowSeedNoise sem sem') (hCM : CM.SplitClosed) :
    GGapSampledDraw CM (pearlCProtocolDevRev1RowInertBoundAt devH100 sem' (1 / 400) (1 / 100))
      (pearlCTilesDevRev1RowInertBoundAt devH100 sem' (1 / 400) (1 / 100))
      (fun L => pearlCDomainDevAt devH100 ⟨m, 6144, 256⟩ L ∧ rowsOf L ≤ 2 ^ 64) (pearlCDrawRev1 devH100)
      ((32437 / 2521000 : ℚ) : ℝ) (fun q N => δPearlC q (N * m)) :=
  gapSampledDraw_of_threshold (pearlCSampledH100v1RowSeedRowInertShare100_6144x256Tol CM m hTT hN hCM)

theorem pearlCSampledH100v1RowSeedCodeRowInertShare100_6144x256Tol {π : Layout → Q ≃ Q} (m : ℕ)
    (hTT : TTOutTilePearlCH100Rev1 CM sem (1 / 100)) (hN : RowSeedNoise sem sem') (hC : CodeSeedNoise sem' semC π)
    (hCM : CM.SplitClosed) (hCM' : CM.RelabelClosed) :
    GGapSampledDrawTolThreshold CM (pearlCProtocolDevRev1RowInertBoundAt devH100 semC (1 / 400) (1 / 100))
      (pearlCTilesDevRev1RowInertBoundAt devH100 semC (1 / 400) (1 / 100))
      (fun L => pearlCDomainDevAt devH100 ⟨m, 6144, 256⟩ L ∧ rowsOf L ≤ 2 ^ 64) (pearlCDrawRev1 devH100)
      ((32437 / 2521000 : ℚ) : ℝ) (fun q N => δPearlC q (N * m)) :=
  gapSampledDrawTolThreshold_of_tol
    (pearlCSampledDrawDevRev1RowInertBoundAtTol CM devH100 semC (1 / 400) (1 / 100) ⟨m, 6144, 256⟩ (25210 / 25137) _
      devH100_add devH100_bf16 (by norm_num) (by norm_num) (by norm_num)
      (by rw [wrefDevRev1_h100_n6144k256, creditDevRev1_h100_n6144k256]; ring)
      (by norm_num)
      (ttOutTileRowSeedCode_of_ttOutTile hTT hN hC hCM hCM' devH100_rowLocal (by norm_num) devH100_add devH100_bf16))

theorem pearlCSampledH100v1RowSeedCodeRowInertShare100_6144x256 {π : Layout → Q ≃ Q} (m : ℕ)
    (hTT : TTOutTilePearlCH100Rev1 CM sem (1 / 100)) (hN : RowSeedNoise sem sem') (hC : CodeSeedNoise sem' semC π)
    (hCM : CM.SplitClosed) (hCM' : CM.RelabelClosed) :
    GGapSampledDraw CM (pearlCProtocolDevRev1RowInertBoundAt devH100 semC (1 / 400) (1 / 100))
      (pearlCTilesDevRev1RowInertBoundAt devH100 semC (1 / 400) (1 / 100))
      (fun L => pearlCDomainDevAt devH100 ⟨m, 6144, 256⟩ L ∧ rowsOf L ≤ 2 ^ 64) (pearlCDrawRev1 devH100)
      ((32437 / 2521000 : ℚ) : ℝ) (fun q N => δPearlC q (N * m)) :=
  gapSampledDraw_of_threshold (pearlCSampledH100v1RowSeedCodeRowInertShare100_6144x256Tol CM m hTT hN hC hCM hCM')

end Gap

end Pouw.PearlC
