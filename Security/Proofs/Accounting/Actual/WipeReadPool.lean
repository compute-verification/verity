import Mathlib
import Definitions.Accounting.Guarantees.Actual.WipeBridge

/-!
# The lifetime pool per read, on the challenger's verdicts

`Accounting.Guarantees.Actual.WipeReadPool`, by Ville's inequality over the reads, proved directly on the coins
`r = (ω₀, c)` rather than through `Accounting.WipePlay` (whose store is a `Type`, where this one is any `Type*`).

Ville's product up to read `m` (`ville`) is `∏_{j < m} [read j accepted] · exp(x_j)`, `x_j` read `j`'s un-answered
fraction (`Actual.unanswered`). Its sum over the coins never grows with `m` (`ville_succ_le`): averaging read `m`'s index
(`sum_update_mul`) leaves the product up to `m` and `x_m` unchanged, since both are fixed by the coins before read `m`'s
index (`Assumptions.LiveVerdicts`), and read `m` is accepted on `|answered| = |B| (1 − x_m)` of the indices, so the
average gains the factor `exp(x_m) (1 − x_m) ≤ 1`. At `m = 0` the sum is the number of coins, and on the event the
product is `exp(Σ_j x_j) ≥ exp(H)`, so the event has at most `exp(−H)` of the coins (Markov).
-/

namespace Accounting.SecurityProofs.Actual

open Finset

variable {Ω₀ B : Type*} [Fintype Ω₀] [Fintype B] {n : ℕ}

/-- **Averaging read `j`'s index**: each coin `r` is `update r j i` for one coin per index `i`. -/
theorem sum_update_mul (j : Fin n) (φ : Ω₀ × (Fin n → B) → ℝ) :
    (∑ r : Ω₀ × (Fin n → B), φ r) * Fintype.card B =
      ∑ r : Ω₀ × (Fin n → B), ∑ i : B, φ (r.1, Function.update r.2 j i) := by
  let f : (Ω₀ × (Fin n → B)) × B → (Ω₀ × (Fin n → B)) × B :=
    fun q => ((q.1.1, Function.update q.1.2 j q.2), q.1.2 j)
  have hf : Function.Involutive f := fun q => by simp [f]
  have h : ∑ q : (Ω₀ × (Fin n → B)) × B, φ (Function.Involutive.toPerm f hf q).1 =
      ∑ q : (Ω₀ × (Fin n → B)) × B, φ q.1 :=
    Equiv.sum_comp (Function.Involutive.toPerm f hf) (fun q => φ q.1)
  have h1 : ∑ q : (Ω₀ × (Fin n → B)) × B, φ (Function.Involutive.toPerm f hf q).1 =
      ∑ r : Ω₀ × (Fin n → B), ∑ i : B, φ (r.1, Function.update r.2 j i) := Fintype.sum_prod_type _
  have h2 : ∑ q : (Ω₀ × (Fin n → B)) × B, φ q.1 = ∑ r : Ω₀ × (Fin n → B), ∑ _i : B, φ r :=
    Fintype.sum_prod_type _
  rw [← h1, h, h2, Finset.sum_mul]
  refine Finset.sum_congr rfl fun r _ => ?_
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_comm]

variable {acc : Ω₀ × (Fin n → B) → Fin n → Prop}

omit [Fintype Ω₀] in
/-- Read `j`'s answered blocks are fixed by `ω₀` and the indices before `j`. -/
theorem answered_congr (hlive : Assumptions.LiveVerdicts acc) {j : Fin n} {r r' : Ω₀ × (Fin n → B)}
    (h1 : r.1 = r'.1) (h2 : ∀ i < j, r.2 i = r'.2 i) : Actual.answered acc j r = Actual.answered acc j r' := by
  ext b
  simp only [Actual.answered, mem_filter, mem_univ, true_and]
  refine hlive j (r.1, Function.update r.2 j b) (r'.1, Function.update r'.2 j b) h1 fun i hi => ?_
  rcases eq_or_ne i j with rfl | hne
  · simp
  · simp only [Function.update_of_ne hne]
    exact h2 i (lt_of_le_of_ne hi hne)

omit [Fintype Ω₀] in
theorem unanswered_congr (hlive : Assumptions.LiveVerdicts acc) {j : Fin n} {r r' : Ω₀ × (Fin n → B)}
    (h1 : r.1 = r'.1) (h2 : ∀ i < j, r.2 i = r'.2 i) :
    Actual.unanswered acc j r = Actual.unanswered acc j r' := by
  unfold Actual.unanswered
  rw [answered_congr hlive h1 h2]

/-- `exp(x) (1 − x) ≤ 1` at `x = (N − a)/N`, times `N`. -/
theorem exp_unanswered_mul_le (N a : ℕ) (ha : a ≤ N) (hN : 0 < N) :
    Real.exp (((N - a : ℕ) : ℝ) / N) * a ≤ N := by
  have hN' : (0 : ℝ) < N := by exact_mod_cast hN
  have hN0 : (N : ℝ) ≠ 0 := hN'.ne'
  set x : ℝ := ((N - a : ℕ) : ℝ) / N with hx_def
  have hx : (a : ℝ) = N * (1 - x) := by
    rw [hx_def, Nat.cast_sub ha]
    field_simp
    ring
  rw [hx]
  calc Real.exp x * (N * (1 - x)) ≤ Real.exp x * (N * Real.exp (-x)) := by
        gcongr
        exact Real.one_sub_le_exp_neg x
    _ = N := by
        rw [mul_left_comm, ← Real.exp_add]
        simp

open Classical in
/-- **Ville's product** up to read `m`: `∏_{j < m} [read j accepted] · exp(x_j)`. -/
noncomputable def ville (acc : Ω₀ × (Fin n → B) → Fin n → Prop) (m : ℕ) (r : Ω₀ × (Fin n → B)) : ℝ :=
  ∏ j ∈ univ.filter (fun j : Fin n => (j : ℕ) < m),
    if acc r j then Real.exp (Actual.unanswered acc j r) else 0

omit [Fintype Ω₀] in
theorem ville_nonneg (m : ℕ) (r : Ω₀ × (Fin n → B)) : 0 ≤ ville acc m r :=
  prod_nonneg fun j _ => by
    split_ifs
    exacts [(Real.exp_pos _).le, le_rfl]

theorem sum_ville_zero : ∑ r, ville acc 0 r = Fintype.card (Ω₀ × (Fin n → B)) := by
  simp [ville]

/-- **Ville's product is a supermartingale**: its sum over the coins never grows from read to read. -/
theorem ville_succ_le (hlive : Assumptions.LiveVerdicts acc) (m : ℕ) :
    ∑ r, ville acc (m + 1) r ≤ ∑ r, ville acc m r := by
  classical
  by_cases hm : m < n
  swap
  · refine le_of_eq (sum_congr rfl fun r _ => ?_)
    unfold ville
    congr 1
    ext j
    simp only [mem_filter, mem_univ, true_and]
    have := j.isLt
    omega
  obtain ⟨j0, rfl⟩ : ∃ j0 : Fin n, (j0 : ℕ) = m := ⟨⟨m, hm⟩, rfl⟩
  have hsplit : ∀ r, ville acc (j0 + 1) r =
      ville acc j0 r * (if acc r j0 then Real.exp (Actual.unanswered acc j0 r) else 0) := by
    intro r
    have hfilt : (univ.filter fun j : Fin n => (j : ℕ) < j0 + 1) =
        insert j0 (univ.filter fun j : Fin n => (j : ℕ) < j0) := by
      ext j
      simp only [mem_filter, mem_univ, true_and, mem_insert, Fin.ext_iff]
      constructor <;> intro h <;> omega
    unfold ville
    rw [hfilt, prod_insert (by simp)]
    exact mul_comm _ _
  rcases isEmpty_or_nonempty B with hB | hB
  · have : IsEmpty (Fin n → B) := ⟨fun c => hB.false (c j0)⟩
    simp
  have hN : (0 : ℝ) < Fintype.card B := by exact_mod_cast Fintype.card_pos
  refine le_of_mul_le_mul_right ?_ hN
  rw [sum_update_mul j0 (ville acc (j0 + 1)), sum_mul]
  refine sum_le_sum fun r _ => ?_
  have hinv : ∀ i, ville acc j0 (r.1, Function.update r.2 j0 i) = ville acc j0 r := by
    intro i
    unfold ville
    refine prod_congr rfl fun j hj => ?_
    have hjm : (j : ℕ) < j0 := (mem_filter.1 hj).2
    have hagree : ∀ i' : Fin n, i' ≤ j → Function.update r.2 j0 i i' = r.2 i' := fun i' hi' =>
      Function.update_of_ne (lt_of_le_of_lt hi' (Fin.lt_def.2 hjm)).ne _ _
    rw [unanswered_congr hlive (r := (r.1, Function.update r.2 j0 i)) (r' := r) rfl
      (fun i' hi' => hagree i' hi'.le)]
    exact if_congr (hlive j (r.1, Function.update r.2 j0 i) r rfl hagree) rfl rfl
  have hx : ∀ i, Actual.unanswered acc j0 (r.1, Function.update r.2 j0 i) = Actual.unanswered acc j0 r :=
    fun i => unanswered_congr hlive rfl fun i' hi' => Function.update_of_ne hi'.ne _ _
  simp only [hsplit, hinv, hx]
  rw [← mul_sum]
  refine mul_le_mul_of_nonneg_left ?_ (ville_nonneg _ r)
  have hcount : ∑ i : B, (if acc (r.1, Function.update r.2 j0 i) j0 then
      Real.exp (Actual.unanswered acc j0 r) else 0) =
      ((Actual.answered acc j0 r).card : ℝ) * Real.exp (Actual.unanswered acc j0 r) := by
    rw [Actual.answered, card_filter, Nat.cast_sum, sum_mul]
    refine sum_congr rfl fun i _ => ?_
    split_ifs <;> simp
  rw [hcount, mul_comm, Actual.unanswered]
  exact exp_unanswered_mul_le _ _ (card_le_univ _) Fintype.card_pos

/-- Markov's step: a set of coins on each of which every read is accepted and the un-answered fractions sum to at
least `H` has at most `exp(−H)` of the coins, as a product with `exp(H)`. -/
theorem card_mul_exp_le (hlive : Assumptions.LiveVerdicts acc) (H : ℝ) (s : Finset (Ω₀ × (Fin n → B)))
    (hs : ∀ r ∈ s, (∀ j, acc r j) ∧ H ≤ ∑ j, Actual.unanswered acc j r) :
    (s.card : ℝ) * Real.exp H ≤ Fintype.card (Ω₀ × (Fin n → B)) := by
  classical
  have hsum : ∀ m, ∑ r, ville acc m r ≤ Fintype.card (Ω₀ × (Fin n → B)) := by
    intro m
    induction m with
    | zero => exact sum_ville_zero.le
    | succ m ih => exact (ville_succ_le hlive m).trans ih
  have hev : ∀ r ∈ s, Real.exp H ≤ ville acc n r := by
    intro r hr
    obtain ⟨hall, hH⟩ := hs r hr
    have hf : (univ.filter fun j : Fin n => (j : ℕ) < n) = univ := filter_true_of_mem fun j _ => j.isLt
    have : ville acc n r = Real.exp (∑ j, Actual.unanswered acc j r) := by
      rw [ville, hf, Real.exp_sum]
      exact prod_congr rfl fun j _ => by simp [hall j]
    rw [this]
    exact Real.exp_le_exp.2 hH
  calc (s.card : ℝ) * Real.exp H = ∑ r ∈ s, Real.exp H := by rw [sum_const, nsmul_eq_mul]
    _ ≤ ∑ r ∈ s, ville acc n r := sum_le_sum hev
    _ ≤ ∑ r, ville acc n r :=
        sum_le_sum_of_subset_of_nonneg (subset_univ _) fun r _ _ => ville_nonneg n r
    _ ≤ _ := hsum n

open Classical in
theorem WipeReadPool : Guarantees.Actual.WipeReadPool := by
  intro Ω₀ B _ _ n acc H hlive
  rw [Real.exp_neg, inv_mul_eq_div, le_div_iff₀ (Real.exp_pos H)]
  refine card_mul_exp_le hlive H _ fun r hr => ?_
  exact (mem_filter.1 hr).2

end Accounting.SecurityProofs.Actual
