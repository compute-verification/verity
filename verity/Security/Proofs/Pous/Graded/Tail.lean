import Proofs.Pous.Thm1.Theorem1iiSeq

/-!
# The tail rule for sequential rounds

`pr_seq_le` bounds the probability that all `k` rounds succeed by `c^k` when each succeeds with conditional
probability at most `c` given the history. `pr_hits_ge_le_binomial` is its tail form: the probability that at most
`t` of the `k` rounds fail is at most `binTail k t c = P[Bin(k, c) ≥ k − t] = Σ_{j ≤ t} C(k, j)·(1 − c)^j·c^(k − j)`,
for every `c ≤ 1`. It is one induction on `k`, as for `pr_seq_le`. In the step, a first round that succeeds with
probability `q ≤ c` leaves at most `q·T(t) + (1 − q)·T(t − 1)`, which is at most `c·T(t) + (1 − c)·T(t − 1)` because
the bound grows with `t` (`binTail_le_succ`, `mix_le`). The bound also grows with `c` on `[0, 1]` (`binTail_mono`), so
a certified `p₀ ≤ r` may be replaced by `r`.

Pure probability: no game and no primitive. A round is any function of the history, through a state `x : X` that
`nxt` advances and `ok` judges.
-/

namespace PousGraded

open Pous PousThm1
open scoped ENNReal

/-! ### The binomial tail -/

/-- `P[Bin(k, c) ≥ k − t]`: at most `t` failures among `k` independent rounds that each succeed with probability
`c`. -/
noncomputable def binTail (k t : ℕ) (c : ℝ≥0∞) : ℝ≥0∞ :=
  ∑ j ∈ Finset.range (t + 1), (k.choose j : ℝ≥0∞) * (1 - c) ^ j * c ^ (k - j)

theorem binTail_zero_left (t : ℕ) (c : ℝ≥0∞) : binTail 0 t c = 1 := by
  unfold binTail
  rw [Finset.sum_range_succ']
  simp

theorem binTail_succ_zero (k : ℕ) (c : ℝ≥0∞) : binTail (k + 1) 0 c = c * binTail k 0 c := by
  simp [binTail, pow_succ']

theorem binTail_succ_succ (k t : ℕ) (c : ℝ≥0∞) :
    binTail (k + 1) (t + 1) c = c * binTail k (t + 1) c + (1 - c) * binTail k t c := by
  have h1 : ∀ j, ((k + 1).choose (j + 1) : ℝ≥0∞) * (1 - c) ^ (j + 1) * c ^ (k + 1 - (j + 1)) =
      (1 - c) * ((k.choose j : ℝ≥0∞) * (1 - c) ^ j * c ^ (k - j)) +
        c * ((k.choose (j + 1) : ℝ≥0∞) * (1 - c) ^ (j + 1) * c ^ (k - (j + 1))) := by
    intro j
    rw [Nat.choose_succ_succ, Nat.cast_add, Nat.add_sub_add_right]
    rcases Nat.lt_or_ge k (j + 1) with h | h
    · rw [Nat.choose_eq_zero_of_lt h]
      ring
    · have hk : k - j = k - (j + 1) + 1 := by omega
      rw [hk, pow_succ]
      ring
  unfold binTail
  rw [Finset.sum_range_succ' _ (t + 1), Finset.sum_range_succ' (fun j => (k.choose j : ℝ≥0∞) * (1 - c) ^ j *
    c ^ (k - j)) (t + 1)]
  simp only [h1, Finset.sum_add_distrib, Finset.mul_sum, mul_add]
  simp only [Nat.choose_zero_right, Nat.cast_one, pow_zero, one_mul, mul_one, Nat.sub_zero, pow_succ']
  ring

/-- More tolerated failures, a larger bound. -/
theorem binTail_le_succ (k t : ℕ) (c : ℝ≥0∞) : binTail k t c ≤ binTail k (t + 1) c := by
  unfold binTail
  rw [Finset.sum_range_succ _ (t + 1)]
  exact le_self_add

/-- A mixture of `a` and `b ≤ a` grows with the weight on `a`. -/
theorem mix_le {q c a b : ℝ≥0∞} (hqc : q ≤ c) (hc : c ≤ 1) (hba : b ≤ a) :
    q * a + (1 - q) * b ≤ c * a + (1 - c) * b := by
  obtain ⟨e, rfl⟩ : ∃ e, c = q + e := ⟨c - q, (add_tsub_cancel_of_le hqc).symm⟩
  have he : e ≤ 1 - q :=
    ENNReal.le_sub_of_add_le_left (ne_top_of_le_ne_top ENNReal.one_ne_top (le_self_add.trans hc)) hc
  have h1 : 1 - q = (1 - (q + e)) + e := by
    rw [tsub_add_eq_tsub_tsub, tsub_add_cancel_of_le he]
  calc q * a + (1 - q) * b = q * a + (1 - (q + e)) * b + e * b := by rw [h1, add_mul, ← add_assoc]
    _ ≤ q * a + (1 - (q + e)) * b + e * a := by gcongr
    _ = (q + e) * a + (1 - (q + e)) * b := by rw [add_mul]; exact add_right_comm _ _ _

/-- **The bound grows with `c` on `[0, 1]`.** -/
theorem binTail_mono {c c' : ℝ≥0∞} (hcc : c ≤ c') (hc' : c' ≤ 1) : ∀ k t, binTail k t c ≤ binTail k t c'
  | 0, t => by rw [binTail_zero_left, binTail_zero_left]
  | k + 1, 0 => by
      rw [binTail_succ_zero, binTail_succ_zero]
      exact mul_le_mul' hcc (binTail_mono hcc hc' k 0)
  | k + 1, t + 1 => by
      rw [binTail_succ_succ, binTail_succ_succ]
      calc c * binTail k (t + 1) c + (1 - c) * binTail k t c
          ≤ c * binTail k (t + 1) c' + (1 - c) * binTail k t c' := by
            gcongr
            · exact binTail_mono hcc hc' k (t + 1)
            · exact binTail_mono hcc hc' k t
        _ ≤ c' * binTail k (t + 1) c' + (1 - c') * binTail k t c' :=
            mix_le hcc hc' (binTail_le_succ k t c')

/-! ### Probability -/

theorem pr_add_pr_not {α : Type} [Fintype α] [Nonempty α] (P : α → Prop) :
    pr P + pr (fun x => ¬ P x) = 1 := by
  classical
  rw [pr_def, pr_def, ← ENNReal.add_div, ← Nat.cast_add, Finset.card_filter_add_card_filter_not,
    Finset.card_univ]
  exact ENNReal.div_self (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)

theorem pr_not {α : Type} [Fintype α] [Nonempty α] (P : α → Prop) : pr (fun x => ¬ P x) = 1 - pr P :=
  ENNReal.eq_sub_of_add_eq (ne_top_of_le_ne_top ENNReal.one_ne_top (pr_le_one P))
    ((add_comm _ _).trans (pr_add_pr_not P))

/-- **A first-coordinate branch.** On `P`, an event of conditional probability at most `a`; off `P`, one of
conditional probability at most `b`. -/
theorem pr_ite_le {α β : Type} [Fintype α] [Fintype β] [Nonempty α] [Nonempty β]
    (P : α → Prop) (R₁ R₂ : α → β → Prop) (a b : ℝ≥0∞) (h₁ : ∀ x, pr (R₁ x) ≤ a) (h₂ : ∀ x, pr (R₂ x) ≤ b) :
    pr (fun y : α × β => (P y.1 ∧ R₁ y.1 y.2) ∨ (¬ P y.1 ∧ R₂ y.1 y.2)) ≤
      pr P * a + pr (fun x => ¬ P x) * b := by
  classical
  have hα0 : (Fintype.card α : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hβ0 : (Fintype.card β : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hαt : (Fintype.card α : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have hβt : (Fintype.card β : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have hfib : ∀ x, ((Finset.univ.filter fun y => (P x ∧ R₁ x y) ∨ (¬ P x ∧ R₂ x y)).card : ℝ≥0∞) ≤
      (if P x then 1 else 0) * (a * Fintype.card β) + (if ¬ P x then 1 else 0) * (b * Fintype.card β) := by
    intro x
    by_cases hx : P x
    · simp only [hx, true_and, not_true_eq_false, false_and, or_false, ite_true, ite_false, one_mul, zero_mul,
        add_zero]
      have := h₁ x
      rwa [pr_def, ENNReal.div_le_iff hβ0 hβt] at this
    · simp only [hx, false_and, not_false_eq_true, true_and, false_or, ite_true, ite_false, one_mul, zero_mul,
        zero_add]
      have := h₂ x
      rwa [pr_def, ENNReal.div_le_iff hβ0 hβt] at this
  have hcount : ((Finset.univ.filter fun y : α × β => (P y.1 ∧ R₁ y.1 y.2) ∨ (¬ P y.1 ∧ R₂ y.1 y.2)).card : ℝ≥0∞) =
      ∑ x, ((Finset.univ.filter fun y => (P x ∧ R₁ x y) ∨ (¬ P x ∧ R₂ x y)).card : ℝ≥0∞) := by
    rw [Finset.natCast_card_filter, Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun x _ => ?_
    rw [Finset.natCast_card_filter]
  have hP : ((Finset.univ.filter P).card : ℝ≥0∞) = ∑ x, (if P x then 1 else 0 : ℝ≥0∞) :=
    Finset.natCast_card_filter _ _
  have hnP : ((Finset.univ.filter fun x => ¬ P x).card : ℝ≥0∞) = ∑ x, (if ¬ P x then 1 else 0 : ℝ≥0∞) :=
    Finset.natCast_card_filter _ _
  have e1 : ((Finset.univ.filter P).card : ℝ≥0∞) / Fintype.card α * Fintype.card α =
      (Finset.univ.filter P).card := ENNReal.div_mul_cancel hα0 hαt
  have e2 : ((Finset.univ.filter fun x => ¬ P x).card : ℝ≥0∞) / Fintype.card α * Fintype.card α =
      (Finset.univ.filter fun x => ¬ P x).card := ENNReal.div_mul_cancel hα0 hαt
  have e3 : ∀ x y u v A C : ℝ≥0∞,
      (x / A * u + y / A * v) * (A * C) = x / A * A * (u * C) + y / A * A * (v * C) := by
    intros; ring
  rw [pr_def (fun y : α × β => (P y.1 ∧ R₁ y.1 y.2) ∨ (¬ P y.1 ∧ R₂ y.1 y.2)), hcount, Fintype.card_prod,
    Nat.cast_mul, ENNReal.div_le_iff (mul_ne_zero hα0 hβ0) (ENNReal.mul_ne_top hαt hβt)]
  calc ∑ x, ((Finset.univ.filter fun y => (P x ∧ R₁ x y) ∨ (¬ P x ∧ R₂ x y)).card : ℝ≥0∞)
      ≤ ∑ x, ((if P x then 1 else 0) * (a * Fintype.card β) + (if ¬ P x then 1 else 0) * (b * Fintype.card β)) :=
        Finset.sum_le_sum fun x _ => hfib x
    _ = ((Finset.univ.filter P).card : ℝ≥0∞) * (a * Fintype.card β) +
          ((Finset.univ.filter fun x => ¬ P x).card : ℝ≥0∞) * (b * Fintype.card β) := by
        rw [Finset.sum_add_distrib, ← Finset.sum_mul, ← Finset.sum_mul, ← hP, ← hnP]
    _ = (pr P * a + pr (fun x => ¬ P x) * b) * (Fintype.card α * Fintype.card β) := by
        rw [pr_def P, pr_def (fun x => ¬ P x), e3, e1, e2]

/-! ### Rounds -/

/-- The state after the history `h`, from round `j` in state `x`. -/
def after {X : Type} {B : ℕ} (nxt : ℕ → X → Fin B → X) : ℕ → List (Fin B) → X → X
  | _, [], x => x
  | j, i :: is, x => after nxt (j + 1) is (nxt j x i)

open Classical in
/-- The rounds of `is` that fail, from round `j` in state `x`: round `j` at index `i` succeeds iff `ok j x i`, and
moves to the state `nxt j x i`. -/
noncomputable def misses {X : Type} {B : ℕ} (nxt : ℕ → X → Fin B → X) (ok : ℕ → X → Fin B → Prop) :
    ℕ → List (Fin B) → X → ℕ
  | _, [], _ => 0
  | j, i :: is, x => (if ok j x i then 0 else 1) + misses nxt ok (j + 1) is (nxt j x i)

/-- **The tail rule.** If after every history of fewer than `k` indices a uniform next index succeeds with
probability at most `c ≤ 1`, then at most `t` of the `k` rounds fail with probability at most
`P[Bin(k, c) ≥ k − t]`. At `t = 0` it is `pr_seq_le`'s `c^k`. -/
theorem pr_hits_ge_le_binomial {X : Type} {B : ℕ} [NeZero B] (nxt : ℕ → X → Fin B → X) (ok : ℕ → X → Fin B → Prop)
    (c : ℝ≥0∞) (hc : c ≤ 1) :
    ∀ (k t j : ℕ) (x : X),
      (∀ h : List (Fin B), h.length < k → pr (ok (j + h.length) (after nxt j h x)) ≤ c) →
      pr (fun I : Fin k → Fin B => misses nxt ok j (List.ofFn I) x ≤ t) ≤ binTail k t c := by
  intro k
  induction k with
  | zero => intro t _ _ _; rw [binTail_zero_left]; exact pr_le_one _
  | succ k ih =>
      intro t j x hyp
      classical
      have : Nonempty (Fin B) := ⟨⟨0, Nat.pos_of_ne_zero (NeZero.ne B)⟩⟩
      rw [← pr_equiv (Fin.consEquiv fun _ : Fin (k + 1) => Fin B :
        Fin B × (Fin k → Fin B) ≃ (Fin (k + 1) → Fin B))]
      simp only [Fin.consEquiv_apply, List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, misses]
      have h0 : pr (ok j x) ≤ c := by simpa [after] using hyp [] (Nat.succ_pos k)
      have hrest : ∀ (i₀ : Fin B) (t' : ℕ), pr (fun I : Fin k → Fin B =>
          misses nxt ok (j + 1) (List.ofFn fun i => I i) (nxt j x i₀) ≤ t') ≤ binTail k t' c :=
        fun i₀ t' => ih t' (j + 1) (nxt j x i₀) fun h hh => by
          have := hyp (i₀ :: h) (by simp only [List.length_cons]; omega)
          simpa [after, Nat.add_assoc, Nat.add_comm 1] using this
      cases t with
      | zero =>
        rw [binTail_succ_zero]
        have hand := pr_and_le (α := Fin B) (β := Fin k → Fin B) (ok j x)
          (fun i₀ I => misses nxt ok (j + 1) (List.ofFn fun i => I i) (nxt j x i₀) ≤ 0) c (binTail k 0 c) h0
          fun i₀ => hrest i₀ 0
        refine le_trans (pr_mono ?_) hand
        rintro ⟨i₀, I⟩ hy
        dsimp only at hy ⊢
        by_cases hok : ok j x i₀
        · simp only [hok, ↓reduceIte] at hy
          exact ⟨hok, by omega⟩
        · simp only [hok, ↓reduceIte] at hy
          omega
      | succ t =>
        rw [binTail_succ_succ]
        have hite := pr_ite_le (α := Fin B) (β := Fin k → Fin B) (ok j x)
          (fun i₀ I => misses nxt ok (j + 1) (List.ofFn fun i => I i) (nxt j x i₀) ≤ t + 1)
          (fun i₀ I => misses nxt ok (j + 1) (List.ofFn fun i => I i) (nxt j x i₀) ≤ t) _ _
          (fun i₀ => hrest i₀ (t + 1)) (fun i₀ => hrest i₀ t)
        rw [pr_not] at hite
        refine le_trans (pr_mono ?_) (hite.trans (mix_le h0 hc (binTail_le_succ k t c)))
        rintro ⟨i₀, I⟩ hy
        dsimp only at hy ⊢
        by_cases hok : ok j x i₀
        · simp only [hok, ↓reduceIte] at hy
          exact Or.inl ⟨hok, by omega⟩
        · simp only [hok, ↓reduceIte] at hy
          exact Or.inr ⟨hok, by omega⟩

end PousGraded
