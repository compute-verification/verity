import Mathlib

/-!
Scratch probe (vacuity sweep, not for merge): an hm96 commitment `(d r + M·y, c y)` under a uniform salt `y` has an
opening to any other row `r'` except with probability `4 · hidingGap M c` over `y`. `hidingGap` is copied verbatim from
`Security/Proofs/Flock/Soundness/CROnly/Hm96Defs.lean`. So a statement whose only public pins are hm96 digests, with the
salts free in its witness (`HmOut.regions`), is satisfiable whatever rows were committed, and the recursion guarantees'
`¬ holds` event (`∃ z, Satisfies z`) is empty at such statements up to that probability per row.
-/

namespace VacuityProbe

open Finset

variable {Y B C : Type} [Fintype Y] [Fintype B] [DecidableEq B] [DecidableEq C]

noncomputable def hidingGap (My : Y → B) (c : Y → C) : ℝ :=
  1 / 2 * ∑ v ∈ (univ : Finset B) ×ˢ univ.image c,
    |(#{y | (My y, c y) = v} : ℝ) / Fintype.card Y - (#{p : B × Y | (p.1, c p.2) = v} : ℝ) / Fintype.card (B × Y)|

theorem card_filter_eq_sum {α β : Type} [Fintype α] [DecidableEq β] (f : α → β) (T : Finset β)
    (hT : ∀ a, f a ∈ T) (E : β → Prop) [DecidablePred E] :
    (#{a | E (f a)} : ℝ) = ∑ v ∈ T with E v, (#{a | f a = v} : ℝ) := by
  rw [← Nat.cast_sum]
  congr 1
  rw [card_eq_sum_card_fiberwise (f := f) (t := T.filter E)]
  · refine sum_congr rfl fun v hv => ?_
    congr 1
    ext a
    simp only [mem_filter, mem_univ, true_and]
    exact ⟨fun h => h.2, fun h => ⟨h ▸ (mem_filter.1 hv).2, h⟩⟩
  · intro a ha
    simp only [coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at ha
    exact mem_filter.2 ⟨hT a, ha⟩

/-- The two distributions' probabilities of any event differ by at most twice the gap, each way. -/
theorem pr_le_pr_add (My : Y → B) (c : Y → C) (E : B × C → Prop) [DecidablePred E] :
    (#{y | E (My y, c y)} : ℝ) / Fintype.card Y ≤
        (#{p : B × Y | E (p.1, c p.2)} : ℝ) / Fintype.card (B × Y) + 2 * hidingGap My c ∧
      (#{p : B × Y | E (p.1, c p.2)} : ℝ) / Fintype.card (B × Y) ≤
        (#{y | E (My y, c y)} : ℝ) / Fintype.card Y + 2 * hidingGap My c := by
  set T := (univ : Finset B) ×ˢ univ.image c
  have hP : ∀ y, (My y, c y) ∈ T := fun y => mem_product.2 ⟨mem_univ _, mem_image_of_mem _ (mem_univ _)⟩
  have hQ : ∀ p : B × Y, (p.1, c p.2) ∈ T := fun p => mem_product.2 ⟨mem_univ _, mem_image_of_mem _ (mem_univ _)⟩
  rw [card_filter_eq_sum (fun y => (My y, c y)) T hP E, card_filter_eq_sum (fun p : B × Y => (p.1, c p.2)) T hQ E,
    sum_div, sum_div]
  set a := fun v => (#{y | (My y, c y) = v} : ℝ) / Fintype.card Y
  set b := fun v => (#{p : B × Y | (p.1, c p.2) = v} : ℝ) / Fintype.card (B × Y)
  have hg : 2 * hidingGap My c = ∑ v ∈ T, |a v - b v| := by
    unfold hidingGap; ring
  have hsub : ∑ v ∈ T with E v, |a v - b v| ≤ ∑ v ∈ T, |a v - b v| :=
    sum_le_sum_of_subset_of_nonneg (filter_subset _ _) fun _ _ _ => abs_nonneg _
  rw [hg]
  constructor
  · have : ∑ v ∈ T with E v, a v ≤ ∑ v ∈ T with E v, b v + ∑ v ∈ T with E v, |a v - b v| := by
      rw [← sum_add_distrib]
      exact sum_le_sum fun v _ => by linarith [le_abs_self (a v - b v)]
    linarith
  · have : ∑ v ∈ T with E v, b v ≤ ∑ v ∈ T with E v, a v + ∑ v ∈ T with E v, |a v - b v| := by
      rw [← sum_add_distrib]
      exact sum_le_sum fun v _ => by linarith [neg_abs_le (a v - b v)]
    linarith

/-- **A shifted value is reachable**: for a uniform salt `y`, some salt `y'` with the same `c` has `M·y' = M·y + Δ`,
except with probability `4 · hidingGap`. -/
theorem shift_reachable [AddGroup B] (My : Y → B) (c : Y → C) (Δ : B) :
    (#{y | ¬ ∃ y', c y' = c y ∧ My y' = My y + Δ} : ℝ) / Fintype.card Y ≤ 4 * hidingGap My c := by
  classical
  let E : B × C → Prop := fun v => ¬ ∃ y', c y' = v.2 ∧ My y' = v.1 + Δ
  let Bad : B × C → Prop := fun v => ¬ ∃ y', c y' = v.2 ∧ My y' = v.1
  have h1 := (pr_le_pr_add My c E).1
  have h3 := (pr_le_pr_add My c Bad).2
  have h2 : (#{p : B × Y | E (p.1, c p.2)} : ℕ) = #{p : B × Y | Bad (p.1, c p.2)} := by
    refine card_bij (fun p _ => (p.1 + Δ, p.2)) ?_ ?_ ?_
    · intro p hp
      simpa [E, Bad] using hp
    · intro p _ q _ h
      simp only [Prod.mk.injEq, add_left_inj] at h
      exact Prod.ext h.1 h.2
    · intro q hq
      refine ⟨(q.1 - Δ, q.2), ?_, ?_⟩
      · simpa [E, Bad] using hq
      · simp
  have h4 : (#{y | Bad (My y, c y)} : ℕ) = 0 := by
    rw [card_eq_zero, filter_eq_empty_iff]
    intro y _ h
    exact h ⟨y, rfl, rfl⟩
  have hE : (#{y | ¬ ∃ y', c y' = c y ∧ My y' = My y + Δ} : ℕ) = #{y | E (My y, c y)} := rfl
  rw [hE]
  rw [h2] at h1
  rw [h4] at h3
  simp only [Nat.cast_zero, zero_div, zero_add] at h3
  linarith

/-- **hm96 opens to any row**: the commitment `(d r + M·y, c y)` to row `r` under a uniform salt `y` equals the
commitment to any row `r'` under some salt `y'`, except with probability `4 · hidingGap M c` over `y`. -/
theorem opens_to_any_row [AddCommGroup B] {R : Type} (d : R → B) (My : Y → B) (c : Y → C) (r r' : R) :
    (#{y | ¬ ∃ y', (d r' + My y', c y') = (d r + My y, c y)} : ℝ) / Fintype.card Y ≤ 4 * hidingGap My c := by
  classical
  have hsub : (#{y | ¬ ∃ y', (d r' + My y', c y') = (d r + My y, c y)} : ℕ) ≤
      #{y | ¬ ∃ y', c y' = c y ∧ My y' = My y + (d r - d r')} := by
    refine card_le_card fun y hy => ?_
    simp only [mem_filter, mem_univ, true_and] at hy ⊢
    rintro ⟨y', hc, hm⟩
    exact hy ⟨y', by rw [Prod.mk.injEq]; exact ⟨by rw [hm]; abel, hc⟩⟩
  calc (#{y | ¬ ∃ y', (d r' + My y', c y') = (d r + My y, c y)} : ℝ) / Fintype.card Y
      ≤ (#{y | ¬ ∃ y', c y' = c y ∧ My y' = My y + (d r - d r')} : ℝ) / Fintype.card Y := by
        gcongr
    _ ≤ 4 * hidingGap My c := shift_reachable My c _

end VacuityProbe

#print axioms VacuityProbe.shift_reachable
#print axioms VacuityProbe.opens_to_any_row
