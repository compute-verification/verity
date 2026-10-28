import Proofs.Nci.Chain

/-!
# Incompressibility in support form gives the draft's

Every value an IU `R` holds is a function of the values on `in(R)` (`eval_agree_of_into`). Over the runs with the
baseline at its committed values, the values on `in(R) ∩ θ` are fixed, so the gates of `R` take at most
`2^novelReads(R)` distinct values (`support_le`). Adding one gate to a set adds at most one weight entry and one input
entry to its charge (`charge_insert_le`), so a set of `R`'s gates charged more than `R` reads has a subset charged
between `novelReads(R)` and `novelReads(R) + b_w + b_a` (`exists_charge_window`). Support form at that cap bounds the
subset's charge by `novelReads(R)`, a contradiction: `activationsIncompressible_of_support`. With it, the reads Lemmas
1 and 2 give (`iu_reads`) rest on support form (`IuReads`).
-/

namespace Nci.SecurityProofs

open Verity Finset

/-- The values on `R` are determined by the values on `in(R)`. -/
theorem eval_agree_of_into {D : Circuit} {R : Finset (Fin D.N)} (hR : R ⊆ D.computed) {a b : D.Transcript}
    (h : ∀ g ∈ D.into R, D.eval a g = D.eval b g) : ∀ g ∈ R, D.eval a g = D.eval b g := by
  intro g
  induction g using WellFoundedLT.induction with
  | _ g ih => ?_
  intro hg
  have hc : D.isInput g = false := by simpa [Circuit.computed] using hR hg
  rw [D.eval_op a hc, D.eval_op b hc]
  refine D.op_congr g _ _ fun x hx => ?_
  by_cases hxR : x ∈ R
  · exact ih x (D.reads_lt g x hx) hxR
  · exact h x (by simpa [Circuit.into] using ⟨hxR, g, hg, hx⟩)

theorem card_bitVec (n : ℕ) : Fintype.card (BitVec n) = 2 ^ n :=
  (Fintype.card_congr (⟨BitVec.toFin, BitVec.ofFin, fun _ => by simp, fun _ => by simp⟩ : BitVec n ≃ Fin (2 ^ n))).trans
    (Fintype.card_fin _)

/-- `f`'s image has at most as many elements as `r`'s when `f` is a function of `r` on `s`. -/
theorem card_image_le_of_factor {α β γ : Type} [DecidableEq β] [DecidableEq γ] {s : Finset α} {f : α → β} {r : α → γ}
    (h : ∀ a ∈ s, ∀ b ∈ s, r a = r b → f a = f b) : (s.image f).card ≤ (s.image r).card := by
  classical
  calc (s.image f).card = ((s.image fun a => (r a, f a)).image Prod.snd).card := by rw [image_image]; rfl
    _ ≤ (s.image fun a => (r a, f a)).card := card_image_le
    _ ≤ (s.image r).card := by
      refine card_le_card_of_injOn Prod.fst (fun x hx => ?_) fun x hx y hy hxy => ?_
      · obtain ⟨a, ha, rfl⟩ := mem_image.mp hx
        exact mem_image_of_mem r ha
      · obtain ⟨a, ha, rfl⟩ := mem_image.mp (mem_coe.mp hx)
        obtain ⟨b, hb, rfl⟩ := mem_image.mp (mem_coe.mp hy)
        exact Prod.ext hxy (h a ha b hb hxy)

/-- **The counting bound**: the gates of an IU's subset `T` take at most `2^novelReads(R)` distinct values over the runs
with the baseline at `θv`. -/
theorem support_le (d : Registered) (θv : d.D.Transcript) {R T : Finset (Fin d.D.N)} (hR : R ⊆ d.D.computed)
    (hT : T ⊆ R) : support d θv T ≤ 2 ^ novelReads d R := by
  classical
  set U := d.D.into R \ d.θ
  let r : d.D.Transcript → (g : U) → BitVec (d.D.w g) := fun a g => d.D.eval a g
  calc support d θv T ≤ ((admissible d θv).image r).card := by
        unfold support
        refine card_image_le_of_factor fun a ha b hb hab => funext fun g => ?_
        have ha' := (mem_filter.mp ha).2
        have hb' := (mem_filter.mp hb).2
        refine eval_agree_of_into hR (fun x hx => ?_) g (hT g.2)
        by_cases hθ : x ∈ d.θ
        · rw [d.D.eval_input a (d.θ_input x hθ), d.D.eval_input b (d.θ_input x hθ), ha' x hθ, hb' x hθ]
        · exact congrFun hab ⟨x, mem_sdiff.mpr ⟨hx, hθ⟩⟩
    _ ≤ Fintype.card ((g : U) → BitVec (d.D.w g)) := card_le_univ _
    _ = 2 ^ novelReads d R := by
        rw [Fintype.card_pi]
        simp only [card_bitVec]
        rw [prod_coe_sort U fun g => 2 ^ d.D.w g, prod_pow_eq_pow_sum]
        rfl

/-- A set of gates takes at most `2^width` distinct values. -/
theorem support_le_width (d : Registered) (θv : d.D.Transcript) (T : Finset (Fin d.D.N)) :
    support d θv T ≤ 2 ^ d.D.width T := by
  classical
  calc support d θv T ≤ Fintype.card ((g : T) → BitVec (d.D.w g)) := card_le_univ _
    _ = 2 ^ d.D.width T := by
        rw [Fintype.card_pi]
        simp only [card_bitVec]
        rw [prod_coe_sort T fun g => 2 ^ d.D.w g, prod_pow_eq_pow_sum]
        rfl

theorem card_le_succ_of_subset {α : Type} [DecidableEq α] {s t u : Finset α} (h : s ⊆ t ∪ u) (hu : u.card ≤ 1) :
    s.card ≤ t.card + 1 :=
  (card_le_card h).trans ((card_union_le t u).trans (by omega))

section Charge

variable {D : Circuit} (c : Chain D)

/-- At most one product is computed by a given gate. -/
theorem card_prods_le_one (g : Fin D.N) :
    (univ.filter fun p : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
      c.prod p.1 p.2.1 p.2.2.1 p.2.2.2 = g).card ≤ 1 :=
  card_le_one.mpr fun _ hp _ hq => c.prod_inj ((mem_filter.mp hp).2.trans (mem_filter.mp hq).2.symm)

theorem card_weightsUsed_insert_le (g : Fin D.N) (T : Finset (Fin D.N)) :
    (c.weightsUsed (insert g T)).card ≤ (c.weightsUsed T).card + 1 := by
  classical
  refine card_le_succ_of_subset (u := (univ.filter fun p : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
      c.prod p.1 p.2.1 p.2.2.1 p.2.2.2 = g).image fun p => ⟨p.1, p.2.2.1, p.2.2.2⟩) (fun w hw => ?_)
    (card_image_le.trans (card_prods_le_one c g))
  obtain ⟨t, ht⟩ := (mem_filter.mp hw).2
  rcases mem_insert.mp ht with h | h
  · exact mem_union_right _ (mem_image.mpr ⟨⟨w.1, t, w.2.1, w.2.2⟩, mem_filter.mpr ⟨mem_univ _, h⟩, rfl⟩)
  · exact mem_union_left _ (mem_filter.mpr ⟨mem_univ _, t, h⟩)

theorem card_inputsUsed_insert_le (P : (ℓ : Fin c.L) × Fin c.N × Fin (c.d ℓ) → Prop) [DecidablePred P]
    (g : Fin D.N) (T : Finset (Fin D.N)) :
    ((c.inputsUsed (insert g T)).filter P).card ≤ ((c.inputsUsed T).filter P).card + 1 := by
  classical
  refine card_le_succ_of_subset (u := (univ.filter fun p : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
      c.prod p.1 p.2.1 p.2.2.1 p.2.2.2 = g).image fun p => ⟨p.1, p.2.1, p.2.2.2⟩) (fun x hx => ?_)
    (card_image_le.trans (card_prods_le_one c g))
  obtain ⟨hx, hP⟩ := mem_filter.mp hx
  obtain ⟨i, hi⟩ := (mem_filter.mp hx).2
  rcases mem_insert.mp hi with h | h
  · exact mem_union_right _ (mem_image.mpr ⟨⟨x.1, x.2.1, i, x.2.2⟩, mem_filter.mpr ⟨mem_univ _, h⟩, rfl⟩)
  · exact mem_union_left _ (mem_filter.mpr ⟨mem_filter.mpr ⟨mem_univ _, i, h⟩, hP⟩)

/-- Adding one gate adds at most `b_w + b_a` to a charge. -/
theorem charge_insert_le {b_w b_a : ℝ} (hw : 0 ≤ b_w) (ha : 0 ≤ b_a) (R : Finset (Fin D.N)) (g : Fin D.N)
    (T : Finset (Fin D.N)) : c.charge R b_w b_a (insert g T) ≤ c.charge R b_w b_a T + b_w + b_a := by
  classical
  have h1 : ((c.weightsUsed (insert g T)).card : ℝ) ≤ (c.weightsUsed T).card + 1 := by
    exact_mod_cast card_weightsUsed_insert_le c g T
  have h2 : (((c.inputsUsed (insert g T)).filter fun x => ¬ c.LinkedWhole R x.1 x.2.1 x.2.2).card : ℝ) ≤
      ((c.inputsUsed T).filter fun x => ¬ c.LinkedWhole R x.1 x.2.1 x.2.2).card + 1 := by
    exact_mod_cast card_inputsUsed_insert_le c _ g T
  unfold Chain.charge
  nlinarith [mul_le_mul_of_nonneg_left h1 hw, mul_le_mul_of_nonneg_left h2 ha]

/-- A set charged more than `N` has a subset charged more than `N` and at most `N + b_w + b_a`. -/
theorem exists_charge_window {b_w b_a : ℝ} (hw : 0 ≤ b_w) (ha : 0 ≤ b_a) (R : Finset (Fin D.N)) {N : ℝ} (hN : 0 ≤ N)
    (T : Finset (Fin D.N)) :
    c.charge R b_w b_a T ≤ N ∨ ∃ T' ⊆ T, N < c.charge R b_w b_a T' ∧ c.charge R b_w b_a T' ≤ N + b_w + b_a := by
  classical
  induction T using Finset.induction_on with
  | empty => left; simpa [Chain.charge, Chain.weightsUsed, Chain.inputsUsed] using hN
  | insert g T _ ih =>
    rcases ih with h | ⟨T', hT', h1, h2⟩
    · by_cases hc : c.charge R b_w b_a (insert g T) ≤ N
      · exact Or.inl hc
      · exact Or.inr ⟨insert g T, subset_refl _, not_le.mp hc, (charge_insert_le c hw ha R g T).trans (by linarith)⟩
    · exact Or.inr ⟨T', hT'.trans (subset_insert _ _), h1, h2⟩

/-- A product of the first multiplication is charged `b_w + b_a`: one weight entry and one input entry, which links
to no entry. -/
theorem charge_first_product {b_w b_a : ℝ} (R : Finset (Fin D.N)) (hL : 0 < c.L) (t : Fin c.N)
    (i : Fin (c.e ⟨0, hL⟩)) (j : Fin (c.d ⟨0, hL⟩)) :
    c.charge R b_w b_a {c.prod ⟨0, hL⟩ t i j} = b_w + b_a := by
  classical
  have hw : (c.weightsUsed {c.prod ⟨0, hL⟩ t i j}).card = 1 := by
    refine le_antisymm ?_ (card_pos.mpr ⟨⟨⟨0, hL⟩, i, j⟩, mem_filter.mpr ⟨mem_univ _, t, mem_singleton_self _⟩⟩)
    simpa [Chain.weightsUsed] using card_weightsUsed_insert_le c (c.prod ⟨0, hL⟩ t i j) ∅
  have ha : ((c.inputsUsed {c.prod ⟨0, hL⟩ t i j}).filter fun x => ¬ c.LinkedWhole R x.1 x.2.1 x.2.2).card = 1 := by
    refine le_antisymm ?_ (card_pos.mpr ⟨⟨⟨0, hL⟩, t, j⟩, mem_filter.mpr
      ⟨mem_filter.mpr ⟨mem_univ _, i, mem_singleton_self _⟩, by simp [Chain.LinkedWhole]⟩⟩)
    simpa [Chain.inputsUsed] using card_inputsUsed_insert_le c (fun x => ¬ c.LinkedWhole R x.1 x.2.1 x.2.2)
      (c.prod ⟨0, hL⟩ t i j) ∅
  simp [Chain.charge, hw, ha]

end Charge

/-- **What falsifies support form at its rates**: a product of the first multiplication takes at most `2^width` values,
so `ActivationsSupport` at a cap of at least `b_w + b_a` bounds `b_w + b_a` by that product gate's width. Measured rates
above a product's support falsify it. -/
theorem rates_le_width_of_support {D : Registered} {θv : D.D.Transcript} {c : Chain D.D} {K b_w b_a : ℝ}
    (h : Assumptions.ActivationsSupport D θv c K b_w b_a) (hK : b_w + b_a ≤ K) (hL : 0 < c.L) (t : Fin c.N)
    (i : Fin (c.e ⟨0, hL⟩)) (j : Fin (c.d ⟨0, hL⟩)) : b_w + b_a ≤ D.D.w (c.prod ⟨0, hL⟩ t i j) := by
  set g := c.prod ⟨0, hL⟩ t i j
  have hc := charge_first_product c (b_w := b_w) (b_a := b_a) {g} hL t i j
  have hs := h {g} (singleton_subset_iff.mpr (by simpa [Circuit.computed] using c.prod_computed _ t i j)) {g}
    (subset_refl _) (hc ▸ hK)
  rw [hc] at hs
  have hu : (support D θv {g} : ℝ) ≤ (2 : ℝ) ^ (D.D.w g : ℝ) := by
    rw [Real.rpow_natCast]
    have := support_le_width D θv {g}
    simp only [Circuit.width, sum_singleton] at this
    exact_mod_cast this
  exact (Real.rpow_le_rpow_left_iff (by norm_num : (1 : ℝ) < 2)).mp (hs.trans hu)

/-- **Support form gives the draft's incompressibility**: `ActivationsSupport` at cap `K + b_w + b_a`, for any committed
values of the baseline, gives `ActivationsIncompressible` at cap `K`, for IUs of any size. -/
theorem activationsIncompressible_of_support {D : Registered} {θv : D.D.Transcript} {c : Chain D.D} {Γ : ℕ}
    {K b_w b_a : ℝ} (hw : 0 ≤ b_w) (ha : 0 ≤ b_a) (h : Assumptions.ActivationsSupport D θv c (K + b_w + b_a) b_w b_a) :
    Assumptions.ActivationsIncompressible D c Γ K b_w b_a := by
  intro R hR _ hK
  show c.charge R b_w b_a R ≤ _
  by_contra hlt
  rcases exists_charge_window c hw ha R (N := novelReads D R) (Nat.cast_nonneg _) R with h1 | ⟨T, hT, h1, h2⟩
  · exact hlt h1
  · have hs := h R hR T hT (by linarith)
    have hc : (support D θv T : ℝ) ≤ (2 : ℝ) ^ (novelReads D R : ℝ) := by
      rw [Real.rpow_natCast]; exact_mod_cast support_le D θv hR hT
    have := (Real.rpow_le_rpow_left_iff (by norm_num : (1 : ℝ) < 2)).mp (hs.trans hc)
    linarith

end Nci.SecurityProofs

theorem Nci.SecurityProofs.IuReads : Nci.Guarantees.IuReads :=
  fun D c Γ K b_w b_a d hd hb_a hb_w _ h R hR hΓ hK =>
    Nci.SecurityProofs.iu_reads D c Γ K b_w b_a d hd hb_a
      (Nci.SecurityProofs.activationsIncompressible_of_support hb_w hb_a h) R hR hΓ hK
