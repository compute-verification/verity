import Mathlib.Analysis.Real.Sqrt
import Proofs.Nci.LoomisWhitneyBase
import Proofs.Nci.Cover.PartialSumRead

/-!
# The partial-sum bounds: one block's products in the cube

`block_counts` bounds what one inference block pays for a set `B` of the counted products it computes whole, split by
whether their token is low for it (fewer than `f · a` of its gates handed). Its updated parameters cost `b` each
(`UpdateHardness`), and the tokens' reads are disjoint (`tokReads_sum_le`). A high token costs `f a` reads, and a
product is determined by its parameter and token (`cell_inj`), so the high products number at most
`(k / b) (Ah / (f a))`. The low products are cells of the cube, and Loomis–Whitney (`card_sq_le_cube`) bounds them by
the weights, the entries and the activations they use, the activations at `f` reads each (`ActivationReads`); and
by `aOut` per activation (`Keyed.reach`). `PartialSumIntensity` is `block_counts` on all of a block's whole products.
-/

namespace Nci.Cover.SecurityProofs

/-- If `G` determines `F` on `s`, then `F` takes at most as many values on `s` as `G`. -/
theorem card_image_le_of_determines {α β γ : Type*} [DecidableEq β] [DecidableEq γ] {s : Finset α} (F : α → β)
    (G : α → γ) (h : ∀ x ∈ s, ∀ y ∈ s, G x = G y → F x = F y) : (s.image F).card ≤ (s.image G).card := by
  have e1 : (s.image fun x => (G x, F x)).image Prod.fst = s.image G := by
    rw [Finset.image_image]
    rfl
  have e2 : (s.image fun x => (G x, F x)).image Prod.snd = s.image F := by
    rw [Finset.image_image]
    rfl
  have hinj : Set.InjOn Prod.fst ((s.image fun x => (G x, F x) : Finset (γ × β)) : Set (γ × β)) := by
    intro u hu v hv huv
    obtain ⟨x, hx, rfl⟩ := Finset.mem_image.1 (Finset.mem_coe.1 hu)
    obtain ⟨y, hy, rfl⟩ := Finset.mem_image.1 (Finset.mem_coe.1 hv)
    exact Prod.ext huv (h x hx y hy huv)
  rw [← e1, ← e2, Finset.card_image_of_injOn hinj]
  exact Finset.card_image_le

section

variable {C : Circuit} {J L Tok Val Mat Row Col : Type} [DecidableEq Val] {M : MulBlocks C J L} {d aOut c V w : ℕ}
  {K : Keyed M Tok Val d aOut c V w}

/-- Distinct tokens have disjoint gates. -/
theorem token_gates_disjoint [DecidableEq Tok] {t t' : Tok} (h : t ≠ t') :
    Disjoint ((K.vals.filter fun v => K.tok v = t).biUnion K.gates)
      ((K.vals.filter fun v => K.tok v = t').biUnion K.gates) := by
  rw [Finset.disjoint_left]
  intro g hg hg'
  obtain ⟨v, hv, hgv⟩ := Finset.mem_biUnion.1 hg
  obtain ⟨v', hv', hgv'⟩ := Finset.mem_biUnion.1 hg'
  obtain ⟨hvV, hvt⟩ := Finset.mem_filter.1 hv
  obtain ⟨hv'V, hv't⟩ := Finset.mem_filter.1 hv'
  have hne : v ≠ v' := fun he => h (by rw [← hvt, ← hv't, he])
  have hd : Disjoint (K.gates v) (K.gates v') := K.gates_disjoint hvV hv'V hne
  exact Finset.disjoint_left.1 hd hgv hgv'

/-- A block's reads of distinct tokens are distinct inputs outside `θ`. -/
theorem tokReads_sum_le [DecidableEq Tok] (I : Finset (Fin C.N)) (T : Finset Tok) :
    ∑ t ∈ T, K.tokReads I t ≤ (C.inputs I \ C.θ).card := by
  simp only [Keyed.tokReads]
  rw [← Finset.card_biUnion (s := T)
    (t := fun t => C.inputs I ∩ (K.vals.filter fun v => K.tok v = t).biUnion K.gates)
    fun t _ t' _ h => Finset.disjoint_of_subset_left Finset.inter_subset_right
      (Finset.disjoint_of_subset_right Finset.inter_subset_right (token_gates_disjoint h))]
  refine Finset.card_le_card fun g hg => ?_
  obtain ⟨t, -, hg⟩ := Finset.mem_biUnion.1 hg
  obtain ⟨hgi, hgt⟩ := Finset.mem_inter.1 hg
  obtain ⟨v, hv, hgv⟩ := Finset.mem_biUnion.1 hgt
  exact Finset.mem_sdiff.2 ⟨hgi, Finset.disjoint_left.1 (K.gates_θ v (Finset.mem_filter.1 hv).1) hgv⟩

/-- A counted product is determined by its parameter and its token (`cell_inj`). -/
theorem card_le_params_mul_toks [DecidableEq J] [DecidableEq L] [DecidableEq Tok] (Q : Cube K Mat Row Col)
    {B : Finset (J × L)} (hB : B ⊆ M.P) :
    B.card ≤ (B.image Prod.snd).card * (B.image fun p => K.tok (K.act p)).card := by
  rw [← Finset.card_product]
  refine Finset.card_le_card_of_injOn (fun p => (p.2, K.tok (K.act p)))
    (fun p hp => Finset.mem_product.2 ⟨Finset.mem_image_of_mem _ hp, Finset.mem_image_of_mem _ hp⟩) ?_
  intro p hp p' hp' h
  simp only [Prod.mk.injEq] at h
  exact Q.cell_inj p (hB hp) p' (hB hp') (congrArg Q.mat h.1) (congrArg Q.row h.1) (congrArg Q.col h.1) h.2

/-- **Loomis–Whitney on the cube**: counted products, squared, number at most their parameters times their entries
times their activations. -/
theorem card_sq_le_cube [DecidableEq J] [DecidableEq L] [DecidableEq Tok] [DecidableEq Mat] [DecidableEq Row]
    [DecidableEq Col] (Q : Cube K Mat Row Col) {B : Finset (J × L)} (hB : B ⊆ M.P) :
    B.card ^ 2 ≤ (B.image Prod.snd).card * (B.image Q.entry).card * (B.image K.act).card := by
  have hTB : (B.image fun p => ((Q.mat p.2, Q.row p.2), Q.col p.2, K.tok (K.act p))).card = B.card := by
    refine Finset.card_image_of_injOn fun p hp p' hp' h => ?_
    simp only [Prod.mk.injEq] at h
    exact Q.cell_inj p (hB hp) p' (hB hp') h.1.1 h.1.2 h.2.1 h.2.2
  have h1 : ((B.image fun p => ((Q.mat p.2, Q.row p.2), Q.col p.2, K.tok (K.act p))).image
      fun x => (x.1, x.2.1)).card ≤ (B.image Prod.snd).card := by
    rw [Finset.image_image]
    exact card_image_le_of_determines _ _ fun p _ p' _ h => by simp only [Function.comp_apply, h]
  have h2 : ((B.image fun p => ((Q.mat p.2, Q.row p.2), Q.col p.2, K.tok (K.act p))).image
      fun x => (x.1, x.2.2)).card ≤ (B.image Q.entry).card := by
    rw [Finset.image_image]
    refine card_image_le_of_determines _ _ fun p _ p' _ h => ?_
    simp only [Cube.entry, Prod.mk.injEq] at h
    simp only [Function.comp_apply, h.1, h.2.1, h.2.2]
  have h3 : ((B.image fun p => ((Q.mat p.2, Q.row p.2), Q.col p.2, K.tok (K.act p))).image
      fun x => (x.2.1, x.2.2)).card ≤ (B.image K.act).card := by
    rw [Finset.image_image]
    refine card_image_le_of_determines _ _ fun p hp p' hp' h => ?_
    simp only [Function.comp_apply, (Q.stack p (hB hp) p' (hB hp') h).2, h]
  rw [← hTB]
  exact (Nci.SecurityProofs.loomis_whitney _).trans (Nat.mul_le_mul (Nat.mul_le_mul h1 h2) h3)

/-- **One block's counts.** For an inference block `I` and a set `B` of counted products it computes whole, split into
`Hi` and `Lo` by whether `I` is handed at least `f · a` gates of their token, there are `X` (the parameters' bits),
`Ah` and `Al` (the reads of `Hi`'s and `Lo`'s tokens) with `X ≤ k`, `Ah + Al ≤ k`, `b f a #Hi ≤ X Ah`,
`b f #Lo² ≤ X Al #(Lo's entries)` and `f #Lo ≤ aOut Al`. -/
theorem block_counts [DecidableEq J] [DecidableEq L] [DecidableEq Tok] [DecidableEq Mat] [DecidableEq Row]
    (Q : Cube K Mat Row Col) {s k f a : ℕ} {b : ℝ} (hb : 0 < b) (hH : Assumptions.UpdateHardness M s b)
    (hX : Assumptions.ActivationReads Q s k f a) {I : Finset (Fin C.N)} (hI : C.InferenceBlock s k I)
    {B : Finset (J × L)} (hB : B ⊆ M.whole I) :
    ∃ X Ah Al : ℝ, 0 ≤ X ∧ 0 ≤ Ah ∧ 0 ≤ Al ∧ X ≤ k ∧ Ah + Al ≤ k ∧
      b * f * a * (B.filter fun p => ¬ K.tokReads I (K.tok (K.act p)) < f * a).card ≤ X * Ah ∧
      b * f * ((B.filter fun p => K.tokReads I (K.tok (K.act p)) < f * a).card : ℝ) ^ 2 ≤
        X * Al * ((B.filter fun p => K.tokReads I (K.tok (K.act p)) < f * a).image Q.entry).card ∧
      f * ((B.filter fun p => K.tokReads I (K.tok (K.act p)) < f * a).card : ℝ) ≤ aOut * Al := by
  classical
  set Lo := B.filter fun p => K.tokReads I (K.tok (K.act p)) < f * a with hLo
  set Hi := B.filter fun p => ¬ K.tokReads I (K.tok (K.act p)) < f * a with hHi
  have hBP : B ⊆ M.P := fun p hp => (Finset.mem_filter.1 (hB hp)).1
  have hLoP : Lo ⊆ M.P := fun p hp => hBP (Finset.mem_filter.1 hp).1
  have hHiP : Hi ⊆ M.P := fun p hp => hBP (Finset.mem_filter.1 hp).1
  have hcap : ((C.inputs I \ C.θ).card : ℝ) ≤ k := by exact_mod_cast hI.2
  set w := (B.image Prod.snd).card with hwdef
  have hw : b * w ≤ k :=
    (hH I hI.1 B (fun p hp => M.P_subset (hBP hp)) fun p hp => (Finset.mem_filter.1 (hB hp)).2).trans hcap
  set TH := Hi.image fun p => K.tok (K.act p) with hTH
  set TL := Lo.image fun p => K.tok (K.act p) with hTL
  have hdisjT : Disjoint TH TL := by
    rw [Finset.disjoint_left]
    intro t ht ht'
    obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 ht
    obtain ⟨p', hp', hpt⟩ := Finset.mem_image.1 ht'
    exact (Finset.mem_filter.1 hp).2 (hpt ▸ (Finset.mem_filter.1 hp').2)
  have hsumT : ∑ t ∈ TH, K.tokReads I t + ∑ t ∈ TL, K.tokReads I t ≤ k := by
    rw [← Finset.sum_union hdisjT]
    exact (tokReads_sum_le I _).trans hI.2
  have hHtok : f * a * TH.card ≤ ∑ t ∈ TH, K.tokReads I t := by
    rw [mul_comm, ← smul_eq_mul, ← Finset.sum_const]
    refine Finset.sum_le_sum fun t ht => ?_
    obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 ht
    exact not_lt.1 (Finset.mem_filter.1 hp).2
  have hHicard : Hi.card ≤ w * TH.card :=
    (card_le_params_mul_toks Q hHiP).trans
      (Nat.mul_le_mul_right _ (Finset.card_le_card (Finset.image_subset_image (Finset.filter_subset _ _))))
  have hq : f * (Lo.image K.act).card ≤ ∑ t ∈ TL, K.tokReads I t := by
    have hsub : Lo.image K.act ⊆
        TL.biUnion fun t => ((M.whole I).filter fun p => K.tok (K.act p) = t).image K.act := by
      intro v hv
      obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hv
      exact Finset.mem_biUnion.2 ⟨_, Finset.mem_image_of_mem _ hp,
        Finset.mem_image_of_mem _ (Finset.mem_filter.2 ⟨hB (Finset.mem_filter.1 hp).1, rfl⟩)⟩
    calc f * (Lo.image K.act).card
        ≤ f * ∑ t ∈ TL, (((M.whole I).filter fun p => K.tok (K.act p) = t).image K.act).card :=
          Nat.mul_le_mul_left _ ((Finset.card_le_card hsub).trans Finset.card_biUnion_le)
      _ = ∑ t ∈ TL, f * (((M.whole I).filter fun p => K.tok (K.act p) = t).image K.act).card :=
          Finset.mul_sum _ _ _
      _ ≤ ∑ t ∈ TL, K.tokReads I t := Finset.sum_le_sum fun t ht => by
          obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 ht
          exact (hX I hI _ (Finset.mem_filter.1 hp).2).2
  have hLW := card_sq_le_cube Q hLoP
  have hwL : (Lo.image Prod.snd).card ≤ w :=
    Finset.card_le_card (Finset.image_subset_image (Finset.filter_subset _ _))
  have hreach : Lo.card ≤ aOut * (Lo.image K.act).card :=
    Finset.card_le_mul_card_image Lo aOut fun v _ =>
      K.reach v _ (fun p hp => hLoP (Finset.mem_filter.1 hp).1) fun p hp => by
        rw [(Finset.mem_filter.1 hp).2]
        exact Follows.refl v
  have hHicard' : (Hi.card : ℝ) ≤ w * TH.card := by exact_mod_cast hHicard
  have hHtok' : (f : ℝ) * a * TH.card ≤ ∑ t ∈ TH, (K.tokReads I t : ℝ) := by exact_mod_cast hHtok
  have hq' : (f : ℝ) * (Lo.image K.act).card ≤ ∑ t ∈ TL, (K.tokReads I t : ℝ) := by exact_mod_cast hq
  have hLW' : (Lo.card : ℝ) ^ 2 ≤ (Lo.image Prod.snd).card * (Lo.image Q.entry).card * (Lo.image K.act).card := by
    exact_mod_cast hLW
  have hwL' : ((Lo.image Prod.snd).card : ℝ) ≤ w := by exact_mod_cast hwL
  have hreach' : (Lo.card : ℝ) ≤ aOut * (Lo.image K.act).card := by exact_mod_cast hreach
  have hsumT' : ∑ t ∈ TH, (K.tokReads I t : ℝ) + ∑ t ∈ TL, (K.tokReads I t : ℝ) ≤ k := by exact_mod_cast hsumT
  have hf : (0 : ℝ) ≤ f := Nat.cast_nonneg _
  have ha : (0 : ℝ) ≤ a := Nat.cast_nonneg _
  have hY : (0 : ℝ) ≤ (Lo.image Q.entry).card := Nat.cast_nonneg _
  have hqn : (0 : ℝ) ≤ (Lo.image K.act).card := Nat.cast_nonneg _
  have hwn : (0 : ℝ) ≤ w := Nat.cast_nonneg _
  refine ⟨b * w, ∑ t ∈ TH, (K.tokReads I t : ℝ), ∑ t ∈ TL, (K.tokReads I t : ℝ), by positivity, by positivity,
    by positivity, hw, hsumT', ?_, ?_, ?_⟩
  · calc b * f * a * Hi.card ≤ b * f * a * (w * TH.card) := mul_le_mul_of_nonneg_left hHicard' (by positivity)
      _ = b * w * (f * a * TH.card) := by ring
      _ ≤ b * w * ∑ t ∈ TH, (K.tokReads I t : ℝ) := mul_le_mul_of_nonneg_left hHtok' (by positivity)
  · calc b * f * (Lo.card : ℝ) ^ 2
        ≤ b * f * ((Lo.image Prod.snd).card * (Lo.image Q.entry).card * (Lo.image K.act).card) :=
          mul_le_mul_of_nonneg_left hLW' (by positivity)
      _ ≤ b * f * (w * (Lo.image Q.entry).card * (Lo.image K.act).card) := by
          refine mul_le_mul_of_nonneg_left ?_ (by positivity)
          exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hwL' hY) hqn
      _ = b * w * (f * (Lo.image K.act).card) * (Lo.image Q.entry).card := by ring
      _ ≤ b * w * (∑ t ∈ TL, (K.tokReads I t : ℝ)) * (Lo.image Q.entry).card := by
          refine mul_le_mul_of_nonneg_right ?_ hY
          exact mul_le_mul_of_nonneg_left hq' (by positivity)
  · calc (f : ℝ) * Lo.card ≤ f * (aOut * (Lo.image K.act).card) := mul_le_mul_of_nonneg_left hreach' hf
      _ = aOut * (f * (Lo.image K.act).card) := by ring
      _ ≤ aOut * ∑ t ∈ TL, (K.tokReads I t : ℝ) := mul_le_mul_of_nonneg_left hq' (Nat.cast_nonneg _)

end

theorem PartialSumIntensity : Guarantees.PartialSumIntensity := by
  intro C J L Tok Val Mat Row Col _ _ _ _ _ _ M d aOut c V w K Q s k f a b hb hH hX I hI
  classical
  obtain ⟨X, Ah, Al, hX0, hAh0, hAl0, hXk, hAk, hHi, hLo, -⟩ := block_counts Q hb hH hX hI (subset_refl (M.whole I))
  set Lo := (M.whole I).filter fun p => K.tokReads I (K.tok (K.act p)) < f * a with hLodef
  set Hi := (M.whole I).filter fun p => ¬ K.tokReads I (K.tok (K.act p)) < f * a with hHidef
  set Y : ℝ := ((Q.crossings I f a).card : ℝ) with hYdef
  have hYLo : (Q.crossings I f a).card = (Lo.image Q.entry).card := rfl
  rw [hYLo] at hYdef
  have hsplit : (M.finished I).card = (M.finished I \ M.whole I).card + (Hi.card + Lo.card) := by
    have h1 := Finset.card_sdiff_add_card_eq_card (whole_subset_finished M I)
    have h2 : Lo.card + Hi.card = (M.whole I).card := Finset.card_filter_add_card_filter_not _
    omega
  have hrest : ((M.finished I \ M.whole I).card : ℝ) ≤ k := by
    exact_mod_cast (finished_sdiff_whole_card_le M (I := I)).trans hI.2
  have hk : (0 : ℝ) ≤ k := Nat.cast_nonneg _
  have hf : (0 : ℝ) ≤ f := Nat.cast_nonneg _
  have ha : (0 : ℝ) ≤ a := Nat.cast_nonneg _
  have hY : 0 ≤ Y := Nat.cast_nonneg _
  have hHk : b * f * a * Hi.card ≤ k * k :=
    hHi.trans (mul_le_mul hXk (by linarith) hAh0 hk)
  have hLsq : b * f * (b * f * (Lo.card : ℝ) ^ 2) ≤ b * f * (k ^ 2 * Y) := by
    refine mul_le_mul_of_nonneg_left (hLo.trans ?_) (by positivity)
    rw [← hYdef]
    calc X * Al * Y ≤ k * k * Y := mul_le_mul_of_nonneg_right (mul_le_mul hXk (by linarith) hAl0 hk) hY
      _ = k ^ 2 * Y := by ring
  have hLk : b * f * Lo.card ≤ k * √(b * f * Y) := by
    have hsq : (b * f * Lo.card) ^ 2 ≤ (k : ℝ) ^ 2 * (b * f * Y) :=
      calc (b * f * Lo.card) ^ 2 = b * f * (b * f * (Lo.card : ℝ) ^ 2) := by ring
        _ ≤ b * f * (k ^ 2 * Y) := hLsq
        _ = (k : ℝ) ^ 2 * (b * f * Y) := by ring
    have h := Real.le_sqrt_of_sq_le hsq
    rwa [Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq hk] at h
  have hLa : b * f * a * Lo.card ≤ a * k * √(b * f * Y) := by
    calc b * f * a * Lo.card = a * (b * f * Lo.card) := by ring
      _ ≤ a * (k * √(b * f * Y)) := mul_le_mul_of_nonneg_left hLk ha
      _ = a * k * √(b * f * Y) := by ring
  have hRa : b * f * a * (M.finished I \ M.whole I).card ≤ b * f * a * k :=
    mul_le_mul_of_nonneg_left hrest (by positivity)
  calc b * f * a * ((M.finished I).card : ℝ)
      = b * f * a * (M.finished I \ M.whole I).card + b * f * a * Hi.card + b * f * a * Lo.card := by
        rw [hsplit]
        push_cast
        ring
    _ ≤ b * f * a * k + k * k + a * k * √(b * f * Y) := by linarith
    _ = k * k + a * k * √(b * f * Y) + b * f * a * k := by ring

end Nci.Cover.SecurityProofs
