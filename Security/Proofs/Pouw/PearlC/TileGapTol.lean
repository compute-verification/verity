module

public import Proofs.Pouw.PearlC.TileGap

/-!
# The gap under the tolerant sampled audit, from TT_OUT per tile

The tolerant audit accepts a sampled tile that is good, of a blank unit, or over its inert-work bound
(`overCapTiles`), and credits the last nothing.
- An accepted run certifies `T ≥ (1 − g)·(S − u(X_O)·W)`, where `S` is the credit (the claimed work; the type `S` in
  the signatures below is the salt's), `W` is the layout's work, `g` is the gap (the statements' `γ`), `X_O` is the
  number of samples on tiles over their bound, and `u` is a table with a miss bound at `δ` (`MissBound`).
- That is `S ≤ (1 + ε)·T`, at the advantage `1 + ε = (1 + ε_a)/(1 − ε_s)`, with `1 + ε_a = 1/(1 − g)` and the sampling
  tolerance `ε_s = u(X_O)·W/S`, except with failure probability `δ + η`.
- TT_OUT's credit set is unchanged, since a good tile is within its bound (`TileGood`), so no premise is added.

* `sampledOfGoodTol`: `sampledOfGood` with `u(X_O)` in place of `ε_s`. Pointwise, an accepted run either has more than
  `u(X_O)` of the layout's tickets outside the good and blank tiles, and since no sample landed on a tile that is bad and
  within its bound, `X_O` counts the samples outside them, which the miss bound gives with probability at most `δ`; or
  its good tiles carry the claimed work less `u(X_O)·W`, and then TT_OUT's event holds.
* `gapSampledTol`: `TTOutTile` at the gap `γ₀` gives `GGapSampledTol` at the gap `1 − (1 − γ₀)/Ω`, under
  `gapSampled`'s hypotheses.
* `missBound_strict` and `gapSampled_of_tol`: the audit that accepts only good and blank tiles is the table `u 0 = ε_s`,
  `u x = 1` otherwise, at `δ = (1 − ε_s)^t` (`workWeightedSampling`), so `GGapSampledTol` gives `GGapSampled`.
-/

@[expose] public section

namespace Pouw.PearlC

open Finset _root_.Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- A tile over its inert-work bound is neither good, whose bound holds, nor of a blank unit. -/
theorem overCapTiles_disjoint {Q R S : Type} (P : Protocol Q R S) (TR : TileRules Q R S) (U : Workload) (H : Q → R)
    (s : S) (τ : Transcript) : Disjoint (goodTiles P TR U H s τ ∪ blankTiles TR U τ) (overCapTiles TR U H s τ) := by
  classical
  rw [Finset.disjoint_left]
  intro g hg ho
  unfold overCapTiles at ho
  rw [mem_filter] at ho
  obtain ⟨_, c, hc, hcap⟩ := ho
  rcases mem_union.mp hg with hg | hg
  · unfold goodTiles at hg
    rw [mem_filter] at hg
    obtain ⟨_, c', hc', hgood⟩ := hg
    obtain rfl := Option.some.inj (hc.symm.trans hc')
    exact hcap hgood.1
  · unfold blankTiles at hg
    rw [mem_filter, hc] at hg
    exact Option.some_ne_none _ hg.2

/-- On a tolerant acceptance with `O` disjoint from `K`, the samples in `O` are those outside `K`. -/
theorem drawsIn_eq_of_acceptsTol {w : ℕ → ℕ} {N t : ℕ} {K O : Finset ℕ} (c : Fin t → Ticket w N) (hdisj : Disjoint K O)
    (hacc : AuditAcceptsTol K O c) : drawsIn O c = drawsIn (range N \ K) c := by
  unfold drawsIn
  congr 1
  ext i
  simp only [mem_filter, mem_univ, true_and, mem_sdiff, mem_range]
  constructor
  · exact fun hO => ⟨(c i).1.isLt, fun hK => Finset.disjoint_left.mp hdisj hK hO⟩
  · exact fun hS => (hacc i).resolve_left hS.2

/-- **The audit that accepts only checked units is a miss-bound table**: `u 0 = ε_s` and `u x = 1` otherwise, at
`δ = (1 − ε_s)^t`. A set holding more than `ε_s` of the tickets is missed by all `t` samples with at most that
probability (`workWeightedSampling`), and no set holds more than all of them. -/
theorem missBound_strict (t : ℕ) (εs : ℝ) (hεs : εs ≤ 1) :
    MissBound t (fun x => if x = 0 then εs else 1) ((1 - εs) ^ t) := by
  classical
  intro N w S hS
  have hS' : range N \ (range N \ S) = S := by
    ext g
    simp only [mem_sdiff]
    constructor
    · rintro ⟨h1, h2⟩
      by_contra h
      exact h2 ⟨h1, h⟩
    · exact fun h => ⟨hS h, fun h' => h'.2 h⟩
  have key : ∀ c : Fin t → Ticket w N,
      (if drawsIn S c = 0 then εs else 1) * ∑ g ∈ range N, (w g : ℝ) < ∑ g ∈ S, (w g : ℝ) →
        AuditAccepts (range N \ S) c ∧
          εs * ∑ g ∈ range N, (w g : ℝ) < ∑ g ∈ range N \ (range N \ S), (w g : ℝ) := by
    intro c hc
    rw [hS']
    by_cases h0 : drawsIn S c = 0
    · rw [ite_eq_left h0] at hc
      refine ⟨fun i => mem_sdiff.mpr ⟨mem_range.mpr (c i).1.isLt, fun hi => ?_⟩, hc⟩
      have h := Finset.card_eq_zero.mp h0
      rw [Finset.filter_eq_empty_iff] at h
      exact h (mem_univ i) hi
    · rw [ite_eq_right h0, one_mul] at hc
      exact absurd (sum_le_sum_of_subset_of_nonneg hS fun _ _ _ => Nat.cast_nonneg _) (not_le.mpr hc)
  refine (pr_mono key).trans ?_
  have h := Pouw.Proofs.workWeightedSampling Unit w N t (fun _ => range N \ S) εs hεs
  rw [← pr_equiv (Equiv.punitProd (Fin t → Ticket w N)).symm] at h
  exact h

/-- **The tolerant sampled audit's bound over any set of accepted tiles.** As `sampledOfGood`, with a set `over` of
committed tiles, disjoint from the good and blank ones, that the audit accepts and credits nothing, and a table `u` with
a miss bound at `δ`: the audit that accepts `good ∪ blank` and `over` does so while `T` is below `1 − γ` (the gap) of
the claimed work less `u(X_O)` of the layout's, `X_O` the samples in `over`, with probability at most `δ + e`. -/
theorem sampledOfGoodTol {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (TR : TileRules Q R S)
    (U : Workload) (τ : (Q → R) → S → Transcript) (good over : (Q → R) → S → Finset ℕ) (γ₀ Ω T e : ℝ)
    (hγ₀ : γ₀ < 1) (hΩ : 0 < Ω) (hsub : ∀ H s, good H s ⊆ range (TR.tiling U.layout).NT \ blankTiles TR U (τ H s))
    (hdisj : ∀ H s, Disjoint (good H s ∪ blankTiles TR U (τ H s)) (over H s))
    (hW : ∀ H s, ∀ g ∈ good H s, TR.Wcred U.layout g (actOf (τ H s) ((TR.tiling U.layout).unit g)) ≤
      Ω * TR.credit H s U g (actOf (τ H s) ((TR.tiling U.layout).unit g)))
    (hWc : ∀ g < (TR.tiling U.layout).NT, ∀ act : Codes, TR.Wcred U.layout g act ≤ TR.Wref U.layout g)
    (hTT : pr (fun x : (Q → R) × S => T / (1 - γ₀) < ∑ g ∈ good x.1 x.2,
        TR.credit x.1 x.2 U g (actOf (τ x.1 x.2) ((TR.tiling U.layout).unit g))) ≤ e)
    (w : ℕ → ℕ) (κ : ℝ) (hκ : 0 < κ) (hw : ∀ g < (TR.tiling U.layout).NT, (w g : ℝ) = κ * TR.Wref U.layout g)
    (t : ℕ) (u : ℕ → ℝ) (δ : ℝ) (hu : MissBound t u δ) :
    pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
        AuditAcceptsTol (good x.1 x.2.1 ∪ blankTiles TR U (τ x.1 x.2.1)) (over x.1 x.2.1) x.2.2 ∧
        T < (1 - (1 - (1 - γ₀) / Ω)) *
          (claimedWref TR U (τ x.1 x.2.1) - u (drawsIn (over x.1 x.2.1) x.2.2) * tileWrefTotal TR U.layout))
      ≤ δ + e := by
  classical
  have h1γ₀ : 0 < 1 - γ₀ := by linarith
  let K : (Q → R) × S → Finset ℕ := fun y => good y.1 y.2 ∪ blankTiles TR U (τ y.1 y.2)
  let O : (Q → R) × S → Finset ℕ := fun y => over y.1 y.2
  let E1 : ((Q → R) × S) × (Fin t → Ticket w (TR.tiling U.layout).NT) → Prop := fun y =>
    AuditAcceptsTol (K y.1) (O y.1) y.2 ∧
      u (drawsIn (O y.1) y.2) * ∑ g ∈ range (TR.tiling U.layout).NT, (w g : ℝ) <
        ∑ g ∈ range (TR.tiling U.layout).NT \ K y.1, (w g : ℝ)
  let E2 : (Q → R) × S → Prop := fun y =>
    T / (1 - γ₀) < ∑ g ∈ good y.1 y.2, TR.credit y.1 y.2 U g (actOf (τ y.1 y.2) ((TR.tiling U.layout).unit g))
  have hsumw : ∀ F : Finset ℕ, F ⊆ range (TR.tiling U.layout).NT →
      ∑ g ∈ F, (w g : ℝ) = κ * ∑ g ∈ F, TR.Wref U.layout g := by
    intro F hF
    rw [mul_sum]
    exact sum_congr rfl fun g hg => hw g (mem_range.mp (hF hg))
  have hstep : ∀ x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT),
      (AuditAcceptsTol (K (x.1, x.2.1)) (O (x.1, x.2.1)) x.2.2 ∧
        T < (1 - (1 - (1 - γ₀) / Ω)) *
          (claimedWref TR U (τ x.1 x.2.1) - u (drawsIn (O (x.1, x.2.1)) x.2.2) * tileWrefTotal TR U.layout)) →
      E1 ((x.1, x.2.1), x.2.2) ∨ E2 (x.1, x.2.1) := by
    intro x ⟨hacc, hTlt⟩
    by_cases h1 : E1 ((x.1, x.2.1), x.2.2)
    · exact Or.inl h1
    right
    set τ' := τ x.1 x.2.1 with hτ
    set gd := good x.1 x.2.1 with hgd
    set blank := blankTiles TR U τ' with hblank
    set uX := u (drawsIn (O (x.1, x.2.1)) x.2.2) with huX
    have hK : K (x.1, x.2.1) = gd ∪ blank := rfl
    have hbadw : ∑ g ∈ range (TR.tiling U.layout).NT \ K (x.1, x.2.1), (w g : ℝ) ≤
        uX * ∑ g ∈ range (TR.tiling U.layout).NT, (w g : ℝ) := by
      by_contra h
      exact h1 ⟨hacc, lt_of_not_ge h⟩
    rw [hsumw _ sdiff_subset, hsumw _ (Subset.refl _)] at hbadw
    have hbad : ∑ g ∈ range (TR.tiling U.layout).NT \ K (x.1, x.2.1), TR.Wref U.layout g ≤
        uX * tileWrefTotal TR U.layout := by
      unfold tileWrefTotal
      have h' : κ * ∑ g ∈ range (TR.tiling U.layout).NT \ K (x.1, x.2.1), TR.Wref U.layout g ≤
          κ * (uX * ∑ g ∈ range (TR.tiling U.layout).NT, TR.Wref U.layout g) := by linarith
      exact le_of_mul_le_mul_left h' hκ
    have hset : (range (TR.tiling U.layout).NT \ blank) \ gd = range (TR.tiling U.layout).NT \ K (x.1, x.2.1) := by
      rw [hK]
      ext g
      simp only [mem_sdiff, mem_union]
      tauto
    let f : ℕ → ℝ := fun g => TR.Wcred U.layout g (actOf τ' ((TR.tiling U.layout).unit g))
    have hsplit := sum_sdiff (f := f) (hsub x.1 x.2.1)
    rw [hset] at hsplit
    have hbadc : ∑ g ∈ range (TR.tiling U.layout).NT \ K (x.1, x.2.1), f g ≤ uX * tileWrefTotal TR U.layout :=
      le_trans (sum_le_sum fun g hg => hWc g (mem_range.mp (mem_sdiff.mp hg).1) _) hbad
    have hclaim : claimedWref TR U τ' - uX * tileWrefTotal TR U.layout ≤ ∑ g ∈ gd, f g := by
      show ∑ g ∈ range (TR.tiling U.layout).NT \ blank, f g - uX * tileWrefTotal TR U.layout ≤ ∑ g ∈ gd, f g
      rw [← hsplit]
      linarith
    have hcred : ∑ g ∈ gd, f g ≤
        Ω * ∑ g ∈ gd, TR.credit x.1 x.2.1 U g (actOf τ' ((TR.tiling U.layout).unit g)) := by
      rw [mul_sum]
      exact sum_le_sum fun g hg => hW x.1 x.2.1 g hg
    have hden : 1 - (1 - (1 - γ₀) / Ω) = (1 - γ₀) / Ω := by ring
    rw [hden] at hTlt
    show T / (1 - γ₀) < ∑ g ∈ gd, TR.credit x.1 x.2.1 U g (actOf τ' ((TR.tiling U.layout).unit g))
    rw [div_lt_iff₀ h1γ₀]
    have hpos : 0 < (1 - γ₀) / Ω := div_pos h1γ₀ hΩ
    calc T < (1 - γ₀) / Ω * (claimedWref TR U τ' - uX * tileWrefTotal TR U.layout) := hTlt
      _ ≤ (1 - γ₀) / Ω * ∑ g ∈ gd, f g := mul_le_mul_of_nonneg_left hclaim hpos.le
      _ ≤ (1 - γ₀) / Ω * (Ω * ∑ g ∈ gd, TR.credit x.1 x.2.1 U g (actOf τ' ((TR.tiling U.layout).unit g))) :=
          mul_le_mul_of_nonneg_left hcred hpos.le
      _ = (∑ g ∈ gd, TR.credit x.1 x.2.1 U g (actOf τ' ((TR.tiling U.layout).unit g))) * (1 - γ₀) := by
          field_simp
  have hδ : 0 ≤ δ := (pr_nonneg _).trans (hu 0 w ∅ (empty_subset _))
  have hmiss : ∀ a : (Q → R) × S, pr (fun c : Fin t → Ticket w (TR.tiling U.layout).NT => E1 (a, c)) ≤ δ := by
    intro a
    refine (pr_mono fun c hc => ?_).trans (hu _ w (range (TR.tiling U.layout).NT \ K a) sdiff_subset)
    obtain ⟨hacc, hlt⟩ : AuditAcceptsTol (K a) (O a) c ∧ _ := hc
    rw [← drawsIn_eq_of_acceptsTol (K := K a) (O := O a) c (hdisj a.1 a.2) hacc]
    exact hlt
  calc _ ≤ pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          E1 ((x.1, x.2.1), x.2.2) ∨ E2 (x.1, x.2.1)) := pr_mono hstep
    _ ≤ pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) => E1 ((x.1, x.2.1), x.2.2)) +
        pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) => E2 (x.1, x.2.1)) := pr_or_le _ _
    _ ≤ δ + e := by
        refine add_le_add ?_ ((pr_fst2_le E2).trans hTT)
        calc _ = pr E1 := pr_equiv (Equiv.prodAssoc _ _ _).symm E1
          _ ≤ δ := pr_prod_le E1 δ hδ hmiss

theorem gapSampledTol {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (γ₀ Ω : ℝ) (ε : ℕ → ℕ → ℝ) (hγ₀ : γ₀ < 1)
    (hΩ : 0 < Ω)
    (hW : ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S), ∀ g < (TR.tiling U.layout).NT, ∀ act : Codes,
      TR.capOK H s U g act → TR.Wcred U.layout g act ≤ Ω * TR.credit H s U g act)
    (hWc : ∀ U : Workload, U.InDomain P D → ∀ g < (TR.tiling U.layout).NT, ∀ act : Codes,
      TR.Wcred U.layout g act ≤ TR.Wref U.layout g)
    (hTT : TTOutTile CM P TR D γ₀ ε) : GGapSampledTol CM P TR D (1 - (1 - γ₀) / Ω) ε := by
  intro U hU T hT q hq A hA w κ hκ hw t u δ hu
  classical
  refine sampledOfGoodTol TR U (fun H s => A.transcript U H s) (fun H s => goodTiles P TR U H s (A.transcript U H s))
    (fun H s => overCapTiles TR U H s (A.transcript U H s)) γ₀ Ω T (ε q U.layout.N) hγ₀ hΩ
    (fun H s => goodTiles_subset P TR U H s _) (fun H s => overCapTiles_disjoint P TR U H s _) ?_ (hWc U hU)
    (hTT.2 U hU T hT q hq A hA) w κ hκ hw t u δ hu
  intro H s g hg
  unfold goodTiles at hg
  rw [mem_filter] at hg
  obtain ⟨hgr, c, hc, hgc⟩ := hg
  have hact : actOf (A.transcript U H s) ((TR.tiling U.layout).unit g) = c.act := by simp [actOf, hc]
  rw [hact]
  exact hW U hU H s g (mem_range.mp hgr) c.act hgc.1

/-- **The audit that accepts only checked tiles is the tolerant one's strict case**: `GGapSampledTol` at the table
`u 0 = ε_s`, `u x = 1` otherwise (`missBound_strict`) gives `GGapSampled`, since a run every sample of which lands on a
good or blank tile has none on a tile over its inert-work bound. -/
theorem gapSampled_of_tol {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S}
    {P : Protocol Q R S} {TR : TileRules Q R S} {D : Layout → Prop} {g : ℝ} {δ : ℕ → ℕ → ℝ}
    (h : GGapSampledTol CM P TR D g δ) : GGapSampled CM P TR D g δ := by
  intro U hU T hT q hq A hA w κ hκ hw t εs hεs
  classical
  refine le_trans (pr_mono fun x hx => ?_) (h U hU T hT q hq A hA w κ hκ hw t _ _ (missBound_strict t εs hεs))
  obtain ⟨hacc, hTlt⟩ := hx
  have h0 : drawsIn (overCapTiles TR U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 = 0 := by
    unfold drawsIn
    rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
    exact fun i _ ho => Finset.disjoint_left.mp (overCapTiles_disjoint P TR U x.1 x.2.1 _) (hacc i) ho
  have hu0 : (if drawsIn (overCapTiles TR U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 = 0 then εs else 1) = εs := by
    rw [h0, ite_eq_left rfl]
  refine ⟨fun i => Or.inl (hacc i), ?_⟩
  show T < (1 - g) * (claimedWref TR U (A.transcript U x.1 x.2.1) -
    (if drawsIn (overCapTiles TR U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 = 0 then εs else 1) *
      tileWrefTotal TR U.layout)
  rw [hu0]
  exact hTlt

end Pouw.PearlC
