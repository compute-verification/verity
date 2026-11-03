import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Nat.Factorial.Basic
import Definitions.Pouw.Basic.Prob
import Proofs.Pouw.Prob

/-!
# Sequentially dominated events are rarely many

`count_tail`: over a uniform product of finite coordinates, let each event `E i` read only coordinates `≤ i`, and let
it hold on at most a `p i` share of every cylinder that fixes the coordinates `< i`. Then at least `r` of the events
hold on at most a `(Σ p)^r / r!` share of the coins. This is the multiplicative (Poisson-type) Chernoff bound for
sequentially dominated indicators.

The proof inducts on the prefix of events `i < k`, for every `r` at once. With `N_k` the prefix's count and `μ_k` its
`Σ p`, the event `N_k ≥ r` reads only coordinates `< k`, so it keeps event `k` to a `p k` share
(`pr_and_le_of_dom`, cylinder by cylinder), and `Pr[N_{k+1} ≥ r + 1] ≤ Pr[N_k ≥ r + 1] + p k · Pr[N_k ≥ r]`;
then `μ^{r+1}/(r+1)! + p · μ^r/r! ≤ (μ + p)^{r+1}/(r+1)!` (`pow_succ_add_le`).

The lead's coin-branch tail (`Tail.Repeats`) reads it. An additive Hoeffding bound (Mathlib's
`ProbabilityTheory.HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun`, or `Tail.Hoeffding` over coin blocks) doesn't
serve: its deviation grows as the square root of the number of blocks, which at T 512 × 64 exceeds the coin-branch
slack, and it needs independent blocks, which the lead's coins don't give (`Tail.Repeats`).
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Finset

/-- **One dominated event**: if `E` holds on at most a `p` share of every cylinder that fixes the coordinates `< i`, and
`A` reads only those coordinates, then `A ∧ E` holds on at most a `p` share of `A`. -/
theorem pr_and_le_of_dom {B : ℕ} {X : Fin B → Type} [∀ i, Fintype (X i)] (E : (∀ i, X i) → Prop) (i : Fin B)
    (p : ℝ) (hp : 0 ≤ p)
    (hdom : ∀ ω : ∀ i, X i,
      Pouw.pr (fun ω' : ∀ i, X i => (∀ j, j < i → ω' j = ω j) ∧ E ω') ≤
        p * Pouw.pr fun ω' : ∀ i, X i => ∀ j, j < i → ω' j = ω j)
    (A : (∀ i, X i) → Prop) (hA : ∀ ω ω' : ∀ i, X i, (∀ j, j < i → ω j = ω' j) → (A ω ↔ A ω')) :
    Pouw.pr (fun ω => A ω ∧ E ω) ≤ p * Pouw.pr A := by
  classical
  rcases isEmpty_or_nonempty (∀ i, X i) with hΩ | hΩ
  · rw [pr_eq_zero fun ω => isEmptyElim ω, pr_eq_zero fun ω => isEmptyElim ω, mul_zero]
  have hn : (0 : ℝ) < Fintype.card (∀ i, X i) := Nat.cast_pos.2 Fintype.card_pos
  let res : (∀ i, X i) → ∀ j : {j : Fin B // j < i}, X j := fun ω j => ω j
  have hres : ∀ ω ω' : ∀ i, X i, res ω = res ω' ↔ ∀ j, j < i → ω j = ω' j := fun ω ω' =>
    ⟨fun h j hj => congr_fun h ⟨j, hj⟩, fun h => funext fun j => h j j.2⟩
  have hcyl : ∀ ω₀, (#{ω | res ω = res ω₀ ∧ E ω} : ℝ) ≤ p * #{ω | res ω = res ω₀} := by
    intro ω₀
    have h := hdom ω₀
    rw [pr_eq_card (fun ω' : ∀ i, X i => (∀ j, j < i → ω' j = ω₀ j) ∧ E ω') {ω | res ω = res ω₀ ∧ E ω}
        fun ω => by simp [hres],
      pr_eq_card (fun ω' : ∀ i, X i => ∀ j, j < i → ω' j = ω₀ j) {ω | res ω = res ω₀} fun ω => by simp [hres],
      mul_div_assoc'] at h
    exact (div_le_div_iff_of_pos_right hn).1 h
  rw [pr_eq_card (fun ω => A ω ∧ E ω) {ω | A ω ∧ E ω} fun ω => by simp, pr_eq_card A {ω | A ω} fun ω => by simp,
    mul_div_assoc']
  refine div_le_div_of_nonneg_right ?_ hn.le
  have H : ∀ s : Finset (∀ i, X i), (s : Set (∀ i, X i)).MapsTo res (univ.image res) := fun _ ω _ =>
    mem_coe.2 (mem_image_of_mem res (mem_univ ω))
  rw [card_eq_sum_card_fiberwise (H {ω | A ω ∧ E ω}), card_eq_sum_card_fiberwise (H {ω | A ω}), Nat.cast_sum,
    Nat.cast_sum, mul_sum]
  refine sum_le_sum fun a ha => ?_
  obtain ⟨ω₀, -, rfl⟩ := mem_image.1 ha
  have hAω : ∀ ω, res ω = res ω₀ → (A ω ↔ A ω₀) := fun ω hr => hA ω ω₀ ((hres ω ω₀).1 hr)
  by_cases h0 : A ω₀
  · refine Eq.trans_le ?_ ((hcyl ω₀).trans_eq ?_)
    · refine congrArg (fun s : Finset (∀ i, X i) => (#s : ℝ)) (Finset.ext fun ω => ?_)
      simp only [mem_filter, mem_univ, true_and]
      exact ⟨fun h => ⟨h.2, h.1.2⟩, fun h => ⟨⟨(hAω ω h.1).2 h0, h.2⟩, h.1⟩⟩
    · refine congrArg (fun s : Finset (∀ i, X i) => p * (#s : ℝ)) (Finset.ext fun ω => ?_)
      simp only [mem_filter, mem_univ, true_and]
      exact ⟨fun h => ⟨(hAω ω h).2 h0, h⟩, fun h => h.2⟩
  · refine Eq.trans_le ?_ (mul_nonneg hp (Nat.cast_nonneg _))
    rw [Nat.cast_eq_zero, card_eq_zero]
    refine filter_false_of_mem fun ω hω hr => ?_
    simp only [mem_filter, mem_univ, true_and] at hω
    exact h0 ((hAω ω hr).1 hω.1)

/-- `μ^(n+1) + (n+1)·p·μ^n ≤ (μ + p)^(n+1)` for `μ, p ≥ 0`: the binomial expansion's first two terms. -/
theorem pow_succ_add_le {μ p : ℝ} (hμ : 0 ≤ μ) (hp : 0 ≤ p) (n : ℕ) :
    μ ^ (n + 1) + (n + 1) * p * μ ^ n ≤ (μ + p) ^ (n + 1) := by
  induction n with
  | zero => simp
  | succ n ih =>
    have h1 : 0 ≤ (n + 1 : ℝ) * p ^ 2 * μ ^ n := by positivity
    have h2 := mul_le_mul_of_nonneg_left ih (add_nonneg hμ hp)
    have e1 : (μ + p) * (μ ^ (n + 1) + (n + 1) * p * μ ^ n) =
        μ ^ (n + 1 + 1) + ((n + 1 : ℕ) + 1 : ℝ) * p * μ ^ (n + 1) + (n + 1) * p ^ 2 * μ ^ n := by
      push_cast; ring
    have e2 : (μ + p) * (μ + p) ^ (n + 1) = (μ + p) ^ (n + 1 + 1) := by ring
    linarith

open Classical in
/-- **Sequentially dominated events are rarely many**: `r` or more of them hold on at most a `(Σ p)^r / r!` share of
the coins. -/
theorem count_tail {B : ℕ} {X : Fin B → Type} [∀ i, Fintype (X i)] (E : Fin B → (∀ i, X i) → Prop)
    (p : Fin B → ℝ) (hp : ∀ i, 0 ≤ p i)
    (hread : ∀ i (ω ω' : ∀ i, X i), (∀ j, j ≤ i → ω j = ω' j) → (E i ω ↔ E i ω'))
    (hdom : ∀ i (ω : ∀ i, X i),
      Pouw.pr (fun ω' : ∀ i, X i => (∀ j, j < i → ω' j = ω j) ∧ E i ω') ≤
        p i * Pouw.pr fun ω' : ∀ i, X i => ∀ j, j < i → ω' j = ω j)
    (r : ℕ) :
    Pouw.pr (fun ω : ∀ i, X i => r ≤ (Finset.univ.filter fun i => E i ω).card) ≤
      (∑ i, p i) ^ r / r.factorial := by
  have key : ∀ k, k ≤ B → ∀ r : ℕ,
      Pouw.pr (fun ω : ∀ i, X i => r ≤ #{i : Fin B | (i : ℕ) < k ∧ E i ω}) ≤
        (∑ i ∈ {i : Fin B | (i : ℕ) < k}, p i) ^ r / r.factorial := by
    intro k
    induction k with
    | zero =>
      intro _ r
      cases r with
      | zero =>
        rw [pow_zero, Nat.factorial_zero, Nat.cast_one, div_one]
        exact pr_le_one _
      | succ r =>
        refine (pr_eq_zero fun ω h => ?_).trans_le
          (div_nonneg (pow_nonneg (sum_nonneg fun i _ => hp i) _) (Nat.cast_nonneg _))
        have h0 : #{i : Fin B | (i : ℕ) < 0 ∧ E i ω} = 0 :=
          card_eq_zero.2 (filter_false_of_mem fun i _ hi => Nat.not_lt_zero _ hi.1)
        omega
    | succ k ih =>
      intro hk r
      have hk' : k < B := hk
      have ih := ih hk'.le
      let i₀ : Fin B := ⟨k, hk'⟩
      have hset : (univ.filter fun i : Fin B => (i : ℕ) < k + 1) =
          insert i₀ (univ.filter fun i : Fin B => (i : ℕ) < k) := by
        ext i
        simp only [mem_filter, mem_univ, true_and, mem_insert]
        constructor
        · intro h
          rcases Nat.lt_succ_iff_lt_or_eq.1 h with h | h
          · exact Or.inr h
          · exact Or.inl (Fin.ext h)
        · rintro (rfl | h)
          · exact Nat.lt_succ_self k
          · exact Nat.lt_succ_of_lt h
      rw [hset, sum_insert (by simp [i₀]), add_comm (p i₀)]
      set μ := ∑ i ∈ {i : Fin B | (i : ℕ) < k}, p i with hμdef
      have hμ0 : 0 ≤ μ := sum_nonneg fun i _ => hp i
      cases r with
      | zero =>
        rw [pow_zero, Nat.factorial_zero, Nat.cast_one, div_one]
        exact pr_le_one _
      | succ r =>
        have hsub : ∀ ω : ∀ i, X i, r + 1 ≤ #{i : Fin B | (i : ℕ) < k + 1 ∧ E i ω} →
            r + 1 ≤ #{i : Fin B | (i : ℕ) < k ∧ E i ω} ∨ (r ≤ #{i : Fin B | (i : ℕ) < k ∧ E i ω} ∧ E i₀ ω) := by
          intro ω h
          by_cases hE : E i₀ ω
          · have hle : #{i : Fin B | (i : ℕ) < k + 1 ∧ E i ω} ≤ #{i : Fin B | (i : ℕ) < k ∧ E i ω} + 1 := by
              refine le_trans (card_le_card ?_) (card_insert_le i₀ _)
              intro i hi
              simp only [mem_filter, mem_univ, true_and, mem_insert] at hi ⊢
              rcases Nat.lt_succ_iff_lt_or_eq.1 hi.1 with h' | h'
              · exact Or.inr ⟨h', hi.2⟩
              · exact Or.inl (Fin.ext h')
            exact Or.inr ⟨by omega, hE⟩
          · refine Or.inl (h.trans (card_le_card ?_))
            intro i hi
            simp only [mem_filter, mem_univ, true_and] at hi ⊢
            rcases Nat.lt_succ_iff_lt_or_eq.1 hi.1 with h' | h'
            · exact ⟨h', hi.2⟩
            · exact (hE ((Fin.ext h' : i = i₀) ▸ hi.2)).elim
        have hA : ∀ ω ω' : ∀ i, X i, (∀ j, j < i₀ → ω j = ω' j) →
            (r ≤ #{i : Fin B | (i : ℕ) < k ∧ E i ω} ↔ r ≤ #{i : Fin B | (i : ℕ) < k ∧ E i ω'}) := by
          intro ω ω' h
          have e : (univ.filter fun i : Fin B => (i : ℕ) < k ∧ E i ω) =
              univ.filter fun i : Fin B => (i : ℕ) < k ∧ E i ω' := by
            refine filter_congr fun i _ => and_congr_right fun hi => hread i ω ω' fun j hj => ?_
            exact h j (Fin.lt_def.2 (Nat.lt_of_le_of_lt (Fin.le_iff_val_le_val.1 hj) hi))
          rw [e]
        refine (pr_mono hsub).trans ((pr_or_le _ _).trans ?_)
        refine (add_le_add (ih (r + 1)) ((pr_and_le_of_dom (E i₀) i₀ (p i₀) (hp i₀) (hdom i₀) _ hA).trans
          (mul_le_mul_of_nonneg_left (ih r) (hp i₀)))).trans ?_
        have hF : (r.factorial : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (Nat.factorial_ne_zero r)
        have hr : (r + 1 : ℝ) ≠ 0 := by positivity
        calc μ ^ (r + 1) / ((r + 1).factorial : ℕ) + p i₀ * (μ ^ r / r.factorial)
            = (μ ^ (r + 1) + (r + 1) * p i₀ * μ ^ r) / ((r + 1).factorial : ℕ) := by
              rw [Nat.factorial_succ, Nat.cast_mul, Nat.cast_succ]
              field_simp
          _ ≤ (μ + p i₀) ^ (r + 1) / ((r + 1).factorial : ℕ) :=
              div_le_div_of_nonneg_right (pow_succ_add_le hμ0 (hp i₀) r) (Nat.cast_nonneg _)
  refine le_trans (pr_mono fun ω h => ?_) ((key B le_rfl r).trans_eq ?_)
  · refine h.trans (card_le_card ?_)
    intro i hi
    simp only [mem_filter, mem_univ, true_and] at hi ⊢
    exact ⟨i.isLt, hi⟩
  · rw [filter_true_of_mem fun (i : Fin B) _ => i.isLt]

end Pouw.SecurityProofs.Lead.Tail
