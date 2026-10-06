import Proofs.Flock.Soundness.Game.Basic
import Proofs.Flock.Soundness.Rewinding

/-!
# Expectations against a strategy, and the rewinding lemma for any run distribution

`expect f g s` is the expectation of `f` over the verifier's coins when the prover plays `s`: the
real-valued version of `prob` (`prob_toReal`). Two independent runs of the same strategy give
`expect₂`.

`expect_bad_le` is the rewinding lemma of `Rewinding.prBad_le` for any strategy's run
distribution: if each run opens at most `Q` of `N` positions, the probability that a run opens a
position off the plurality table is at most `√(N · Q · Pr₂[two runs open a position differently])`.
The proof is `Rewinding`'s counting argument with weights: at each position, a run opening it to
`v` conflicts with every run opening it to anything else, which by plurality is at least the
off-plurality mass `b_p`, so the conflict mass at `p` is at least `b_p²`; and Cauchy–Schwarz over
positions.
-/

namespace FlockSoundness

open scoped ENNReal

namespace Game

variable {α : Type}

/-- The expectation of `f` over the verifier's uniform coins, against the strategy `s`. -/
noncomputable def expect (f : α → ℝ) : (g : Game α) → Strategy g → ℝ
  | .ret a, _ => f a
  | .send _ k, s => expect f (k s.1) s.2
  | @Game.coin _ C iF _ k, s => (∑ c : C, expect f (k c) (s c)) / (@Fintype.card C iF)

/-- The expectation over two independent runs of the same strategy. -/
noncomputable def expect₂ (F : α → α → ℝ) (g : Game α) (s : Strategy g) : ℝ :=
  expect (fun a => expect (F a) g s) g s

theorem expect_add (f f' : α → ℝ) : ∀ (g : Game α) (s : Strategy g),
    expect (fun a => f a + f' a) g s = expect f g s + expect f' g s
  | .ret _, _ => rfl
  | .send _ k, s => expect_add f f' (k s.1) s.2
  | @Game.coin _ C _ _ k, s => by
      simp only [expect, expect_add f f' (k _) (s _), Finset.sum_add_distrib, add_div]

theorem expect_const_mul (c : ℝ) (f : α → ℝ) : ∀ (g : Game α) (s : Strategy g),
    expect (fun a => c * f a) g s = c * expect f g s
  | .ret _, _ => rfl
  | .send _ k, s => expect_const_mul c f (k s.1) s.2
  | @Game.coin _ C _ _ k, s => by
      simp only [expect, expect_const_mul c f (k _) (s _), ← Finset.mul_sum, mul_div_assoc]

theorem expect_const (c : ℝ) : ∀ (g : Game α) (s : Strategy g), expect (fun _ => c) g s = c
  | .ret _, _ => rfl
  | .send _ k, s => expect_const c (k s.1) s.2
  | @Game.coin _ C _ _ k, s => by
      simp only [expect, expect_const c (k _) (s _), Finset.sum_const, Finset.card_univ,
        nsmul_eq_mul]
      have : (Fintype.card C : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
      field_simp

theorem expect_sub (f f' : α → ℝ) (g : Game α) (s : Strategy g) :
    expect (fun a => f a - f' a) g s = expect f g s - expect f' g s := by
  have h := expect_add f (fun a => -f' a) g s
  have h' := expect_const_mul (-1) f' g s
  simp only [neg_one_mul] at h'
  simp only [sub_eq_add_neg, h, h']

theorem expect_sum {ι : Type} (t : Finset ι) (f : ι → α → ℝ) (g : Game α) (s : Strategy g) :
    expect (fun a => ∑ i ∈ t, f i a) g s = ∑ i ∈ t, expect (f i) g s := by
  classical
  induction t using Finset.induction_on with
  | empty => simp [expect_const]
  | insert j t hj ih =>
      simp only [Finset.sum_insert hj]
      rw [expect_add, ih]

theorem expect_mono {f f' : α → ℝ} (h : ∀ a, f a ≤ f' a) : ∀ (g : Game α) (s : Strategy g),
    expect f g s ≤ expect f' g s
  | .ret a, _ => h a
  | .send _ k, s => expect_mono h (k s.1) s.2
  | @Game.coin _ C _ _ k, s => by
      simp only [expect]
      exact div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => expect_mono h (k c) (s c))
        (Nat.cast_nonneg _)

theorem expect_nonneg {f : α → ℝ} (h : ∀ a, 0 ≤ f a) (g : Game α) (s : Strategy g) :
    0 ≤ expect f g s := by
  have := expect_mono h g s
  rwa [expect_const] at this

open Classical in
/-- The indicator of an event, as a real function. -/
noncomputable def ind (p : α → Prop) (a : α) : ℝ := if p a then 1 else 0

theorem ind_nonneg (p : α → Prop) (a : α) : 0 ≤ ind p a := by
  unfold ind; split_ifs <;> norm_num

theorem ind_le_one (p : α → Prop) (a : α) : ind p a ≤ 1 := by
  unfold ind; split_ifs <;> norm_num

theorem ind_mono {p q : α → Prop} (h : ∀ a, p a → q a) (a : α) : ind p a ≤ ind q a := by
  unfold ind
  by_cases hp : p a
  · rw [if_pos hp, if_pos (h a hp)]
  · rw [if_neg hp]; split_ifs <;> norm_num

theorem expect_ind_le_one (p : α → Prop) (g : Game α) (s : Strategy g) :
    expect (ind p) g s ≤ 1 := by
  have := expect_mono (ind_le_one p) g s
  rwa [expect_const] at this

open Classical in
/-- `prob` is the expectation of the indicator. -/
theorem prob_toReal (p : α → Prop) : ∀ (g : Game α) (s : Strategy g),
    (prob p g s).toReal = expect (ind p) g s
  | .ret a, _ => by simp only [prob, expect, ind]; split_ifs <;> simp
  | .send _ k, s => prob_toReal p (k s.1) s.2
  | @Game.coin _ C iF _ k, s => by
      simp only [prob, expect]
      have hC : (Fintype.card C : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
      have hfin : ∀ c, prob p (k c) (s c) ≠ ⊤ := fun c =>
        ne_top_of_le_ne_top ENNReal.one_ne_top ((prob_le_value p _ _).trans (value_le_one p _))
      rw [ENNReal.toReal_sum fun c _ => ENNReal.mul_ne_top (ENNReal.inv_ne_top.2 hC) (hfin c)]
      simp only [ENNReal.toReal_mul, ENNReal.toReal_inv, ENNReal.toReal_natCast,
        prob_toReal p (k _) (s _)]
      rw [← Finset.mul_sum, inv_mul_eq_div]

end Game

/-! ## The rewinding lemma for any run distribution -/

namespace Rewinding

open Game Finset

variable {α : Type} {P V : Type} [Fintype P] [Fintype V] [DecidableEq V]

variable (op : α → P → Option V)

/-- `vstar` is a plurality table for the strategy's run distribution: at each position no value is
opened with larger probability. -/
def IsPluralityE (g : Game α) (s : Strategy g) (vstar : P → V) : Prop :=
  ∀ p v, expect (ind fun a => op a p = some v) g s ≤ expect (ind fun a => op a p = some (vstar p)) g s

/-- Every run distribution has a plurality table. -/
theorem exists_pluralityE [Nonempty V] (g : Game α) (s : Strategy g) :
    ∃ vstar : P → V, IsPluralityE op g s vstar := by
  have h : ∀ p, ∃ v : V, ∀ w, expect (ind fun a => op a p = some w) g s ≤
      expect (ind fun a => op a p = some v) g s := by
    intro p
    obtain ⟨v, -, hv⟩ := Finset.exists_max_image (univ : Finset V)
      (fun w => expect (ind fun a => op a p = some w) g s) univ_nonempty
    exact ⟨v, fun w => hv w (mem_univ w)⟩
  exact ⟨fun p => (h p).choose, fun p => (h p).choose_spec⟩

omit [Fintype P] [Fintype V] [DecidableEq V] in
open Classical in
theorem ind_isSome_eq_sum [Fintype V] (p : P) (a : α) :
    ind (fun a => (op a p).isSome) a = ∑ v : V, ind (fun a => op a p = some v) a := by
  unfold ind
  rcases h : op a p with _ | w
  · simp [h]
  · simp only [h, Option.isSome_some, if_true, Option.some.injEq]
    rw [Finset.sum_ite_eq]
    simp

open Classical in
/-- **The rewinding lemma for any run distribution.** -/
theorem expect_bad_le (g : Game α) (s : Strategy g) (vstar : P → V)
    (hv : IsPluralityE op g s vstar) (Q : ℕ) (hQ : ∀ a, #(univ.filter fun p => (op a p).isSome) ≤ Q) :
    expect (ind fun a => ∃ p, Off op vstar a p) g s ≤
      Real.sqrt (Fintype.card P * Q *
        expect₂ (fun a a' => ind (fun x : α × α => ∃ p, ConflictAt op x p) (a, a')) g s) := by
  set π : P → V → ℝ := fun p v => expect (ind fun a => op a p = some v) g s
  set o : P → ℝ := fun p => expect (ind fun a => (op a p).isSome) g s
  set b : P → ℝ := fun p => expect (ind fun a => Off op vstar a p) g s
  set c : P → ℝ := fun p =>
    expect₂ (fun a a' => ind (fun x : α × α => ConflictAt op x p) (a, a')) g s
  have ho : ∀ p, o p = ∑ v, π p v := fun p => by
    simp only [o, π]
    rw [← expect_sum]
    exact congrArg (fun f => expect f g s) (funext fun a => ind_isSome_eq_sum op p a)
  -- the off-plurality mass
  have hb : ∀ p, b p = o p - π p (vstar p) := fun p => by
    simp only [b, o, π]
    rw [← expect_sub]
    refine congrArg (fun f => expect f g s) (funext fun a => ?_)
    unfold ind Off
    by_cases h1 : (op a p).isSome <;> by_cases h2 : op a p = some (vstar p) <;> simp_all
  -- the conflict mass at `p`
  have hc : ∀ p, c p = ∑ v, π p v * (o p - π p v) := fun p => by
    simp only [c, expect₂]
    have hin : ∀ a, expect (fun a' => ind (fun x : α × α => ConflictAt op x p) (a, a')) g s =
        ∑ v, ind (fun a => op a p = some v) a * (o p - π p v) := by
      intro a
      cases hsa : op a p with
      | none =>
          have : (fun a' => ind (fun x : α × α => ConflictAt op x p) (a, a')) = fun _ => 0 := by
            funext a'
            simp [ind, ConflictAt, hsa]
          rw [this, expect_const]
          simp [ind, hsa]
      | some w =>
          have : (fun a' => ind (fun x : α × α => ConflictAt op x p) (a, a')) =
              fun a' => ind (fun a' => (op a' p).isSome) a' - ind (fun a' => op a' p = some w) a' := by
            funext a'
            unfold ind ConflictAt
            simp only [hsa, Option.isSome_some, true_and]
            by_cases h1 : (op a' p).isSome
            · by_cases h2 : op a' p = some w
              · rw [if_neg (fun h => h.2 h2.symm), if_pos h1, if_pos h2, sub_self]
              · rw [if_pos ⟨h1, fun h => h2 h.symm⟩, if_pos h1, if_neg h2, sub_zero]
            · have h2 : op a' p ≠ some w := fun h => by rw [h] at h1; simp at h1
              rw [if_neg (fun h => h1 h.1), if_neg h1, if_neg h2, sub_zero]
          rw [this, expect_sub]
          rw [Finset.sum_eq_single w]
          · simp [ind, hsa, o, π]
          · intro v _ hvw
            simp [ind, hsa, Ne.symm hvw]
          · simp
    rw [show (fun a => expect (fun a' => ind (fun x : α × α => ConflictAt op x p) (a, a')) g s) =
      fun a => ∑ v, ind (fun a => op a p = some v) a * (o p - π p v) from funext hin, expect_sum]
    refine Finset.sum_congr rfl fun v _ => ?_
    rw [show (fun a => ind (fun a => op a p = some v) a * (o p - π p v)) =
      fun a => (o p - π p v) * ind (fun a => op a p = some v) a from funext fun a => mul_comm _ _,
      expect_const_mul, mul_comm]
  have hπ0 : ∀ p v, 0 ≤ π p v := fun p v => expect_nonneg (ind_nonneg _) g s
  have hb0 : ∀ p, 0 ≤ b p := fun p => expect_nonneg (ind_nonneg _) g s
  -- plurality: every value's complement mass is at least `b_p`, so `c_p ≥ o_p·b_p ≥ b_p²`
  have hcb : ∀ p, b p ^ 2 ≤ c p := fun p => by
    have h1 : ∀ v, π p v * b p ≤ π p v * (o p - π p v) := fun v =>
      mul_le_mul_of_nonneg_left (by rw [hb]; linarith [hv p v]) (hπ0 p v)
    have h2 : b p ≤ o p := by rw [hb]; linarith [hπ0 p (vstar p)]
    calc b p ^ 2 = b p * b p := sq _
      _ ≤ o p * b p := mul_le_mul_of_nonneg_right h2 (hb0 p)
      _ = ∑ v, π p v * b p := by rw [ho, Finset.sum_mul]
      _ ≤ ∑ v, π p v * (o p - π p v) := Finset.sum_le_sum fun v _ => h1 v
      _ = c p := (hc p).symm
  -- the union bound over positions
  have hunion : expect (ind fun a => ∃ p, Off op vstar a p) g s ≤ ∑ p, b p := by
    rw [← expect_sum]
    refine expect_mono (fun a => ?_) g s
    by_cases h : ∃ p, Off op vstar a p
    · obtain ⟨p, hp⟩ := h
      calc ind (fun a => ∃ p, Off op vstar a p) a = 1 := by
            simp only [ind]; exact if_pos ⟨p, hp⟩
        _ = ind (fun a => Off op vstar a p) a := by simp only [ind]; exact (if_pos hp).symm
        _ ≤ ∑ p, ind (fun a => Off op vstar a p) a :=
            Finset.single_le_sum (f := fun p => ind (fun a => Off op vstar a p) a)
              (fun p _ => ind_nonneg _ _) (mem_univ p)
    · rw [show ind (fun a => ∃ p, Off op vstar a p) a = 0 by simp [ind, h]]
      exact Finset.sum_nonneg fun p _ => ind_nonneg _ _
  -- each conflicting pair counted at no more than `Q` positions
  have hQc : ∑ p, c p ≤ Q * expect₂ (fun a a' =>
      ind (fun x : α × α => ∃ p, ConflictAt op x p) (a, a')) g s := by
    simp only [c, expect₂]
    rw [← expect_sum, ← expect_const_mul]
    refine expect_mono (fun a => ?_) g s
    rw [← expect_sum, ← expect_const_mul]
    refine expect_mono (fun a' => ?_) g s
    by_cases h : ∃ p, ConflictAt op (a, a') p
    · calc ∑ p, ind (fun x : α × α => ConflictAt op x p) (a, a')
          = (#(univ.filter fun p => ConflictAt op (a, a') p) : ℝ) := by
            rw [Finset.card_filter]; push_cast; exact Finset.sum_congr rfl fun p _ => by
              simp only [ind]; split_ifs <;> simp
        _ ≤ Q := by
            exact_mod_cast (card_le_card fun p => by
              simp only [mem_filter, mem_univ, true_and]; exact fun h => h.1).trans (hQ a)
        _ = Q * ind (fun x : α × α => ∃ p, ConflictAt op x p) (a, a') := by simp [ind, h]
    · push Not at h
      rw [Finset.sum_eq_zero fun p _ => by simp [ind, h p]]
      exact mul_nonneg (Nat.cast_nonneg _) (ind_nonneg _ _)
  -- Cauchy–Schwarz over positions
  have hcs : (∑ p, b p) ^ 2 ≤ Fintype.card P * ∑ p, b p ^ 2 := by
    have := sq_sum_le_card_mul_sum_sq (s := (univ : Finset P)) (f := b)
    simpa using this
  refine hunion.trans (Real.le_sqrt_of_sq_le ?_)
  calc (∑ p, b p) ^ 2 ≤ Fintype.card P * ∑ p, b p ^ 2 := hcs
    _ ≤ Fintype.card P * ∑ p, c p :=
        mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun p _ => hcb p) (Nat.cast_nonneg _)
    _ ≤ Fintype.card P * (Q * expect₂ (fun a a' =>
          ind (fun x : α × α => ∃ p, ConflictAt op x p) (a, a')) g s) :=
        mul_le_mul_of_nonneg_left hQc (Nat.cast_nonneg _)
    _ = _ := by ring

end Rewinding

end FlockSoundness
