module

public import Definitions.Pouw.PearlC.TileSample
public import Proofs.Flock.Claims.Prob
public import Proofs.Pouw.PearlC.TileGapTol

/-! g under the sampled audit sampled as the run samples, from TT_OUT per tile. -/

@[expose] public section

namespace Pouw.PearlC

open Finset _root_.Pouw.PearlC.Assumptions Pouw.Fp8Atom

theorem ex_le_add_pr {α : Type} [Fintype α] (f : α → ℝ) (E : α → Prop) (δ : ℝ) (hδ : 0 ≤ δ)
    (h1 : ∀ a, f a ≤ δ + 1) (h2 : ∀ a, ¬ E a → f a ≤ δ) : Claims.Ex f ≤ δ + pr E := by
  classical
  unfold Claims.Ex pr Claims.Pr Claims.prUniform
  rcases Nat.eq_zero_or_pos (Fintype.card α) with h0 | hpos
  · rw [h0, Nat.cast_zero, div_zero, div_zero, add_zero]
    exact hδ
  · have hc : (0 : ℝ) < Fintype.card α := by exact_mod_cast hpos
    rw [div_le_iff₀ hc, add_mul, div_mul_cancel₀ _ hc.ne', ← sum_filter_add_sum_filter_not univ E f]
    have hA : ∑ a ∈ univ.filter E, f a ≤ (univ.filter E).card • (δ + 1) := by
      rw [← sum_const]
      exact sum_le_sum fun a _ => h1 a
    have hB : ∑ a ∈ univ.filter (fun a => ¬ E a), f a ≤ (univ.filter (fun a => ¬ E a)).card • δ := by
      rw [← sum_const]
      exact sum_le_sum fun a ha => h2 a (mem_filter.mp ha).2
    rw [nsmul_eq_mul, mul_add, mul_one] at hA
    rw [nsmul_eq_mul] at hB
    have hcard := Finset.card_filter_add_card_filter_not (s := (univ : Finset α)) E
    rw [card_univ] at hcard
    have hcard' : ((univ.filter E).card : ℝ) * δ + ((univ.filter (fun a => ¬ E a)).card : ℝ) * δ =
        δ * Fintype.card α := by
      rw [← add_mul, mul_comm]
      congr 1
      exact_mod_cast hcard
    linarith

theorem lcm_den_pos (NT : ℕ) (v : ℕ → ℚ) : 0 < (range NT).lcm (fun i => (v i).den) := by
  refine Nat.pos_of_ne_zero fun h => ?_
  rw [Finset.lcm_eq_zero_iff] at h
  obtain ⟨i, _, hi⟩ := h
  exact (v i).den_nz hi

theorem ticketsOf_cast (NT : ℕ) (v : ℕ → ℚ) {g : ℕ} (hg : g < NT) (hv : 0 ≤ v g) :
    (ticketsOf NT v g : ℝ) = ((range NT).lcm (fun i => (v i).den) : ℕ) * (v g : ℝ) := by
  have hd : (v g).den ∣ (range NT).lcm (fun i => (v i).den) := Finset.dvd_lcm (mem_range.mpr hg)
  have hnum : (((v g).num.toNat : ℕ) : ℤ) = (v g).num := Int.toNat_of_nonneg (Rat.num_nonneg.mpr hv)
  have hden : ((v g).den : ℝ) ≠ 0 := by exact_mod_cast (v g).den_nz
  have hq : (v g : ℝ) = ((v g).num.toNat : ℝ) / (v g).den := by
    rw [Rat.cast_def]
    congr 1
    exact_mod_cast hnum.symm
  unfold ticketsOf
  rw [ite_eq_left hg]
  obtain ⟨k, hk⟩ := hd
  rw [hk, Nat.mul_div_cancel_left _ (Nat.pos_of_ne_zero (v g).den_nz), hq]
  push_cast
  field_simp

theorem goodDeclTiles_subset {Q R S : Type} (P : Protocol Q R S) (TR : TileRules Q R S) (U : Workload) (H : Q → R)
    (s : S) (τ : Trace) (dec : ℕ → Decl) : goodDeclTiles P TR U H s τ dec ⊆ goodTiles P TR U H s τ := by
  classical
  intro g hg
  unfold goodDeclTiles at hg
  exact (mem_filter.mp hg).1

theorem sampledOfGoodDrawTol {Q R S : Type} (TR : TileRules Q R S) (Wd : Layout → ℕ → Decl → ℚ) (U : Workload)
    (τ : Trace) (dec : ℕ → Decl) (good O : Finset ℕ) (credit : ℕ → ℝ) (γ₀ Ω T : ℝ) (hγ₀ : γ₀ < 1)
    (hΩ : 0 < Ω) (hsub : good ⊆ range (TR.tiling U.layout).NT \ blankTiles TR U τ)
    (hdisj : Disjoint (good ∪ blankTiles TR U τ) O)
    (hW0 : ∀ g < (TR.tiling U.layout).NT, 0 ≤ drawWeight TR Wd U τ dec g)
    (hW : ∀ g ∈ good, (drawWeight TR Wd U τ dec g : ℝ) ≤ Ω * credit g)
    (t : ℕ) (u : ℕ → ℝ) (δ : ℝ) (hu : MissBound t u δ) (hE : ¬ T / (1 - γ₀) < ∑ g ∈ good, credit g) :
    pr (fun c : Fin t → Ticket (drawTickets TR Wd U τ dec) (TR.tiling U.layout).NT =>
        AuditAcceptsTol (good ∪ blankTiles TR U τ) O c ∧
        T < (1 - (1 - (1 - γ₀) / Ω)) * (1 - u (drawsIn O c)) * claimedDraw TR Wd U τ dec) ≤ δ := by
  classical
  set NT := (TR.tiling U.layout).NT with hNT
  set v := drawWeight TR Wd U τ dec with hv
  set w := drawTickets TR Wd U τ dec with hw
  set K := good ∪ blankTiles TR U τ with hK
  set L := (range NT).lcm (fun i => (v i).den) with hL
  have hLpos : (0 : ℝ) < L := by exact_mod_cast lcm_den_pos NT v
  have hwv : ∀ g < NT, (w g : ℝ) = L * (v g : ℝ) := fun g hg => ticketsOf_cast NT v hg (hW0 g hg)
  have hsumw : ∀ F ⊆ range NT, ∑ g ∈ F, (w g : ℝ) = L * ∑ g ∈ F, (v g : ℝ) := by
    intro F hF
    rw [mul_sum]
    exact sum_congr rfl fun g hg => hwv g (mem_range.mp (hF hg))
  have hblank0 : ∀ g ∈ blankTiles TR U τ, (v g : ℝ) = 0 := by
    intro g hg
    unfold blankTiles at hg
    rw [mem_filter] at hg
    simp [hv, drawWeight, hg.2]
  have hKsub : K ⊆ range NT := union_subset (hsub.trans sdiff_subset) (filter_subset _ _)
  have hgb : Disjoint good (blankTiles TR U τ) := by
    rw [Finset.disjoint_left]
    exact fun g hg hb => (mem_sdiff.mp (hsub hg)).2 hb
  have hKv : ∑ g ∈ K, (v g : ℝ) = ∑ g ∈ good, (v g : ℝ) := by
    rw [hK, sum_union hgb, sum_eq_zero hblank0, add_zero]
  have hclaim : claimedDraw TR Wd U τ dec = ∑ g ∈ range NT \ K, (v g : ℝ) + ∑ g ∈ good, (v g : ℝ) := by
    rw [← hKv]
    exact (sum_sdiff hKsub).symm
  let E1 : (Fin t → Ticket w NT) → Prop := fun c =>
    AuditAcceptsTol K O c ∧ u (drawsIn O c) * ∑ g ∈ range NT, (w g : ℝ) < ∑ g ∈ range NT \ K, (w g : ℝ)
  have hstep : ∀ c : Fin t → Ticket w NT, (AuditAcceptsTol K O c ∧
      T < (1 - (1 - (1 - γ₀) / Ω)) * (1 - u (drawsIn O c)) * claimedDraw TR Wd U τ dec) → E1 c := by
    intro c ⟨hacc, hT⟩
    by_contra h1
    refine hE ?_
    have hbad : ∑ g ∈ range NT \ K, (w g : ℝ) ≤ u (drawsIn O c) * ∑ g ∈ range NT, (w g : ℝ) := by
      by_contra h
      exact h1 ⟨hacc, lt_of_not_ge h⟩
    rw [hsumw _ sdiff_subset, hsumw _ (Subset.refl _)] at hbad
    have hbad' : ∑ g ∈ range NT \ K, (v g : ℝ) ≤ u (drawsIn O c) * claimedDraw TR Wd U τ dec := by
      have h' : L * ∑ g ∈ range NT \ K, (v g : ℝ) ≤ L * (u (drawsIn O c) * claimedDraw TR Wd U τ dec) := by
        calc _ ≤ u (drawsIn O c) * (L * ∑ g ∈ range NT, (v g : ℝ)) := hbad
          _ = L * (u (drawsIn O c) * claimedDraw TR Wd U τ dec) := by
              rw [show claimedDraw TR Wd U τ dec = ∑ g ∈ range NT, (v g : ℝ) from rfl]
              ring
      exact le_of_mul_le_mul_left h' hLpos
    have hgood : (1 - u (drawsIn O c)) * claimedDraw TR Wd U τ dec ≤ ∑ g ∈ good, (v g : ℝ) := by
      rw [sub_mul, one_mul]
      linarith
    have hcred : ∑ g ∈ good, (v g : ℝ) ≤ Ω * ∑ g ∈ good, credit g := by
      rw [mul_sum]
      exact sum_le_sum fun g hg => hW g hg
    have h1γ₀ : 0 < 1 - γ₀ := by linarith
    have hpos : 0 < (1 - γ₀) / Ω := div_pos h1γ₀ hΩ
    rw [show 1 - (1 - (1 - γ₀) / Ω) = (1 - γ₀) / Ω by ring, mul_assoc] at hT
    rw [div_lt_iff₀ h1γ₀]
    calc T < (1 - γ₀) / Ω * ((1 - u (drawsIn O c)) * claimedDraw TR Wd U τ dec) := hT
      _ ≤ (1 - γ₀) / Ω * (Ω * ∑ g ∈ good, credit g) := mul_le_mul_of_nonneg_left (hgood.trans hcred) hpos.le
      _ = (∑ g ∈ good, credit g) * (1 - γ₀) := by field_simp
  refine (pr_mono hstep).trans ((pr_mono fun c hc => ?_).trans (hu NT w (range NT \ K) sdiff_subset))
  obtain ⟨hacc, hlt⟩ := hc
  rw [← drawsIn_eq_of_acceptsTol c hdisj hacc]
  exact hlt

theorem gapSampledDrawTol {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (Wd : Layout → ℕ → Decl → ℚ) (γ₀ Ω : ℝ)
    (ε : ℕ → ℕ → ℝ) (hγ₀ : γ₀ < 1) (hΩ : 0 < Ω)
    (hW0 : ∀ U : Workload, U.InDomain P D → ∀ g < (TR.tiling U.layout).NT, ∀ dc : Decl, 0 ≤ Wd U.layout g dc)
    (hc0 : ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S), ∀ g < (TR.tiling U.layout).NT, ∀ act : Codes,
      TR.inertBoundOK H s U g act → 0 ≤ TR.credit H s U g act)
    (hW : ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S), ∀ g < (TR.tiling U.layout).NT, ∀ (act : Codes)
      (dc : Decl), TR.inertBoundOK H s U g act → dc.kept ≤ (U.layout.shape ((TR.tiling U.layout).unit g)).k →
      (∀ i ∈ (TR.tiling U.layout).rows g, i ∈ dc.rows →
        TR.rowOK (U.layout.shape ((TR.tiling U.layout).unit g)) act i) →
      (Wd U.layout g dc : ℝ) ≤ Ω * TR.credit H s U g act)
    (hTT : TTOutTile CM P TR D γ₀ ε) : GGapSampledDrawTol CM P TR D Wd (1 - (1 - γ₀) / Ω) ε := by
  intro U hU T hT q hq A hA dec t u δ hu
  classical
  have hδ : 0 ≤ δ := (pr_nonneg _).trans (hu 0 (fun _ => 0) ∅ (empty_subset _))
  have hTT' := hTT.2 U hU T hT q (hq.trans (by norm_num)) A hA
  refine (ex_le_add_pr _ _ δ hδ (fun x => (pr_le_one _).trans (by linarith)) (fun x hx => ?_)).trans
    (add_le_add le_rfl hTT')
  set τ := A.trace U x.1 x.2 with hτ
  have hgood : ∀ g ∈ goodDeclTiles P TR U x.1 x.2 τ (dec x.1 x.2), ∃ c,
      τ ((TR.tiling U.layout).unit g) = some c ∧ TileGood P TR U x.1 x.2 g c ∧
        g < (TR.tiling U.layout).NT ∧ declOK TR U τ (dec x.1 x.2) g := by
    intro g hg
    unfold goodDeclTiles goodTiles at hg
    rw [mem_filter, mem_filter] at hg
    obtain ⟨⟨hgr, c, hc, hgc⟩, hdc⟩ := hg
    exact ⟨c, hc, hgc, mem_range.mp hgr, hdc⟩
  refine sampledOfGoodDrawTol TR Wd U τ (dec x.1 x.2) (goodDeclTiles P TR U x.1 x.2 τ (dec x.1 x.2))
    (overInertBoundTiles TR U x.1 x.2 τ) (fun g => TR.credit x.1 x.2 U g (actOf τ ((TR.tiling U.layout).unit g))) γ₀ Ω T hγ₀
    hΩ ((goodDeclTiles_subset P TR U x.1 x.2 τ _).trans (goodTiles_subset P TR U x.1 x.2 τ))
    ((overInertBoundTiles_disjoint P TR U x.1 x.2 τ).mono_left
      (union_subset_union (goodDeclTiles_subset P TR U x.1 x.2 τ _) (Subset.refl _)))
    ?_ ?_ t u δ hu ?_
  · intro g hg
    unfold drawWeight
    split_ifs
    · exact le_refl _
    · exact hW0 U hU g hg _
  · intro g hg
    obtain ⟨c, hc, hgc, hgr, hdc⟩ := hgood g hg
    have hact : actOf τ ((TR.tiling U.layout).unit g) = c.act := by simp [actOf, hc]
    have hdw : drawWeight TR Wd U τ (dec x.1 x.2) g =
        Wd U.layout g (dec x.1 x.2 ((TR.tiling U.layout).unit g)) := by
      simp [drawWeight, hc]
    rw [hdw]
    refine hW U hU x.1 x.2 g hgr _ _ ?_ hdc.1 hdc.2
    rw [hact]
    exact hgc.1
  · intro hlt
    refine hx (lt_of_lt_of_le hlt
      (sum_le_sum_of_subset_of_nonneg (goodDeclTiles_subset P TR U x.1 x.2 τ _) fun g hg _ => ?_))
    unfold goodTiles at hg
    rw [mem_filter] at hg
    obtain ⟨hgr, c, hc, hgc⟩ := hg
    have hact : actOf τ ((TR.tiling U.layout).unit g) = c.act := by simp [actOf, hc]
    show 0 ≤ TR.credit x.1 x.2 U g (actOf τ ((TR.tiling U.layout).unit g))
    rw [hact]
    exact hc0 U hU x.1 x.2 g (mem_range.mp hgr) c.act hgc.1

theorem gapSampledDrawTolThreshold_of_tol {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    {CM : CostModel Q R S} {P : Protocol Q R S} {TR : TileRules Q R S} {D : Layout → Prop}
    {Wd : Layout → ℕ → Decl → ℚ} {g : ℝ} {δ : ℕ → ℕ → ℝ} (h : GGapSampledDrawTol CM P TR D Wd g δ) :
    GGapSampledDrawTolThreshold CM P TR D Wd g δ := by
  intro U hU T hT q hq A hA dec t εs hεs
  exact h U hU T hT q hq A hA dec t (fun x => if x = 0 then εs else 1) ((1 - εs) ^ t) (missBound_strict t εs hεs)

theorem gapSampledDraw_of_threshold {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    {CM : CostModel Q R S} {P : Protocol Q R S} {TR : TileRules Q R S} {D : Layout → Prop}
    {Wd : Layout → ℕ → Decl → ℚ} {g : ℝ} {δ : ℕ → ℕ → ℝ} (h : GGapSampledDrawTolThreshold CM P TR D Wd g δ) :
    GGapSampledDraw CM P TR D Wd g δ := by
  intro U hU T hT q hq A hA dec t εs hεs
  classical
  refine le_trans (Claims.Ex_mono fun x => pr_mono fun c hc => ?_) (h U hU T hT q hq A hA dec t εs hεs)
  obtain ⟨hacc, hTlt⟩ := hc
  have hdisj := (overInertBoundTiles_disjoint P TR U x.1 x.2 (A.trace U x.1 x.2)).mono_left
    (union_subset_union (goodDeclTiles_subset P TR U x.1 x.2 (A.trace U x.1 x.2) (dec x.1 x.2)) (Subset.refl _))
  have h0 : drawsIn (overInertBoundTiles TR U x.1 x.2 (A.trace U x.1 x.2)) c = 0 := by
    unfold drawsIn
    rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
    exact fun i _ ho => Finset.disjoint_left.mp hdisj (hacc i) ho
  refine ⟨fun i => Or.inl (hacc i), ?_⟩
  simpa [h0] using hTlt

theorem gapSampledDraw_of_tol {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    {CM : CostModel Q R S} {P : Protocol Q R S} {TR : TileRules Q R S} {D : Layout → Prop}
    {Wd : Layout → ℕ → Decl → ℚ} {g : ℝ} {δ : ℕ → ℕ → ℝ} (h : GGapSampledDrawTol CM P TR D Wd g δ) :
    GGapSampledDraw CM P TR D Wd g δ :=
  gapSampledDraw_of_threshold (gapSampledDrawTolThreshold_of_tol h)

end Pouw.PearlC
