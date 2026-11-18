module

public import Definitions.Poub.Remote.Sample
import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# The sample's pooled bound: below the receipts, and lowered by capping the sizes

`ResourceAccounting.Remote.Sample.pooled` is a supremum of `Σ_t f_t·U_t` over shares `f_t ≤ 1`, so at most `Σ_t U_t`
(`pooled_le_sum`). Its budget `S·(1 − δ^(1/S))` grows with `S` (`budget_mono`, `exp` convex), so capping the windows'
sizes, each still drawing `min(s, U_t)`, lowers it (`pooled_mono_sizes`): a share `f'` of the capped windows is the share
`f'·U'/U` of the uncapped ones, with the same wrong receipts and no more of the budget.
-/

@[expose] public section

namespace Properties.Poub.Subprotocol.SecurityProofs

open ResourceAccounting.Remote.Sample

theorem share_sum_le {n : ℕ} {U d : Fin n → ℕ} {f : Fin n → ℝ}
    (hf : ∀ t, 0 ≤ f t ∧ f t ≤ 1 - (d t : ℝ) / U t) : ∑ t, f t * U t ≤ ∑ t, (U t : ℝ) := by
  apply Finset.sum_le_sum
  intro t _
  have hd : (0 : ℝ) ≤ (d t : ℝ) / U t := by positivity
  exact mul_le_of_le_one_left (Nat.cast_nonneg _) ((hf t).2.trans (by linarith))

/-- The pooled bound is at most the windows' receipts. -/
theorem pooled_le_sum (n : ℕ) (U d : Fin n → ℕ) (δ : ℝ) : pooled n U d δ ≤ ∑ c, (U c : ℝ) := by
  unfold pooled
  apply Real.sSup_le
  · rintro x ⟨f, ⟨hf, -⟩, rfl⟩
    exact share_sum_le hf
  · positivity

/-- `S·(1 − δ^(1/S))` grows with `S` for `δ ∈ (0, 1]`: `(1 − e^(−x))/x` falls in `x > 0`, `exp` being convex. -/
theorem budget_mono {δ : ℝ} (h0 : 0 < δ) (h1 : δ ≤ 1) {S T : ℕ} (hST : S ≤ T) : budget S δ ≤ budget T δ := by
  unfold budget
  rcases Nat.eq_zero_or_pos S with rfl | hS
  · simp only [Nat.cast_zero, zero_mul]
    apply mul_nonneg (Nat.cast_nonneg _)
    have : δ ^ ((T : ℝ)⁻¹) ≤ 1 := Real.rpow_le_one h0.le h1 (by positivity)
    linarith
  have hT : 0 < T := lt_of_lt_of_le hS hST
  obtain ⟨a, ha⟩ : ∃ a, a = -Real.log δ := ⟨_, rfl⟩
  have ha0 : 0 ≤ a := by rw [ha]; linarith [Real.log_nonpos h0.le h1]
  have hpow : ∀ N : ℕ, δ ^ ((N : ℝ)⁻¹) = Real.exp (-(a / N)) := by
    intro N
    rw [Real.rpow_def_of_pos h0, ha]
    congr 1
    ring
  rw [hpow S, hpow T]
  rcases eq_or_lt_of_le ha0 with ha' | ha'
  · rw [← ha']
    simp
  have hSR : (0 : ℝ) < S := by exact_mod_cast hS
  have hTR : (0 : ℝ) < T := by exact_mod_cast hT
  have hx : 0 < a / T := div_pos ha' hTR
  have hy : a / T ≤ a / S := div_le_div_of_nonneg_left ha0 hSR (by exact_mod_cast hST)
  have key := convexOn_exp.secant_mono (a := 0) (x := -(a / S)) (y := -(a / T)) (Set.mem_univ _)
    (Set.mem_univ _) (Set.mem_univ _) (by linarith) (by linarith) (by linarith)
  rw [Real.exp_zero, sub_zero, sub_zero, div_neg_eq_neg_div, div_neg_eq_neg_div, neg_le_neg_iff,
    div_div_eq_mul_div, div_div_eq_mul_div, div_le_div_iff_of_pos_right ha'] at key
  linarith

/-- **Capping the sizes lowers the pooled bound**: windows of sizes `U' ≤ U`, each drawing `min(s, U'_t)`, have a
pooled bound at most that of sizes `U` drawing `min(s, U_t)`, at any `δ ∈ (0, 1]`. -/
theorem pooled_mono_sizes {n : ℕ} (U U' : Fin n → ℕ) (s : ℕ) {δ : ℝ} (h0 : 0 < δ) (h1 : δ ≤ 1)
    (hU : ∀ c, U' c ≤ U c) :
    pooled n U' (fun c => min s (U' c)) δ ≤ pooled n U (fun c => min s (U c)) δ := by
  unfold pooled
  apply Real.sSup_le
  · rintro x ⟨f', ⟨hf', hB'⟩, rfl⟩
    have hbdd : BddAbove ((fun f : Fin n → ℝ => ∑ t, f t * U t) ''
        {f | (∀ t, 0 ≤ f t ∧ f t ≤ 1 - ((min s (U t) : ℕ) : ℝ) / U t) ∧
          ∑ t, ((min s (U t) : ℕ) : ℝ) * f t ≤ budget (∑ t, min s (U t)) δ}) := by
      refine ⟨∑ t, (U t : ℝ), ?_⟩
      rintro y ⟨f, ⟨hf, -⟩, rfl⟩
      exact share_sum_le hf
    apply le_csSup hbdd
    refine ⟨fun t => f' t * U' t / U t, ⟨fun t => ?_, ?_⟩, ?_⟩
    · have hf0 := (hf' t).1
      have hft := (hf' t).2
      try dsimp only at hft ⊢
      refine ⟨by positivity, ?_⟩
      rcases Nat.eq_zero_or_pos (U t) with hz | hpos
      · simp [hz]
      have hUR : (0 : ℝ) < U t := by exact_mod_cast hpos
      have hsub : U' t - min s (U' t) ≤ U t - min s (U t) := by have := hU t; omega
      have hsubR : (U' t : ℝ) - (min s (U' t) : ℕ) ≤ (U t : ℝ) - (min s (U t) : ℕ) := by
        have := (Nat.cast_le (α := ℝ)).2 hsub
        rwa [Nat.cast_sub (min_le_right _ _), Nat.cast_sub (min_le_right _ _)] at this
      have hfU : f' t * U' t ≤ (U' t : ℝ) - (min s (U' t) : ℕ) := by
        rcases Nat.eq_zero_or_pos (U' t) with hz' | hpos'
        · simp [hz']
        have hUR' : (0 : ℝ) < U' t := by exact_mod_cast hpos'
        have := mul_le_mul_of_nonneg_right hft hUR'.le
        rwa [sub_mul, one_mul, div_mul_cancel₀ _ hUR'.ne'] at this
      rw [div_le_iff₀ hUR, sub_mul, one_mul, div_mul_cancel₀ _ hUR.ne']
      linarith
    · try dsimp only
      have hd : ∀ t, ((min s (U t) : ℕ) : ℝ) * (f' t * U' t / U t) ≤ ((min s (U' t) : ℕ) : ℝ) * f' t := by
        intro t
        have hf0 := (hf' t).1
        rcases Nat.eq_zero_or_pos (U t) with hz | hpos
        · simp only [hz, Nat.cast_zero, div_zero, mul_zero]
          positivity
        have hUR : (0 : ℝ) < U t := by exact_mod_cast hpos
        have hnat : min s (U t) * U' t ≤ min s (U' t) * U t := by
          have := hU t
          rcases le_total s (U' t) with h | h
          · rw [min_eq_left h, min_eq_left (h.trans this)]
            exact Nat.mul_le_mul_left _ this
          · rw [min_eq_right h, Nat.mul_comm]
            exact Nat.mul_le_mul_left _ (min_le_right _ _)
        have hR : ((min s (U t) : ℕ) : ℝ) * U' t ≤ ((min s (U' t) : ℕ) : ℝ) * U t := by exact_mod_cast hnat
        rw [mul_div_assoc', div_le_iff₀ hUR]
        nlinarith
      have hS : ∑ t, min s (U' t) ≤ ∑ t, min s (U t) :=
        Finset.sum_le_sum fun t _ => min_le_min_left _ (hU t)
      calc ∑ t, ((min s (U t) : ℕ) : ℝ) * (f' t * U' t / U t)
          ≤ ∑ t, ((min s (U' t) : ℕ) : ℝ) * f' t := Finset.sum_le_sum fun t _ => hd t
        _ ≤ budget (∑ t, min s (U' t)) δ := hB'
        _ ≤ budget (∑ t, min s (U t)) δ := budget_mono h0 h1 hS
    · try dsimp only
      apply Finset.sum_congr rfl
      intro t _
      rcases Nat.eq_zero_or_pos (U t) with hz | hpos
      · have : U' t = 0 := by have := hU t; omega
        simp [hz, this]
      have hUR : (0 : ℝ) < U t := by exact_mod_cast hpos
      rw [div_mul_cancel₀ _ hUR.ne']
  · apply Real.sSup_nonneg
    rintro y ⟨f, ⟨hf, -⟩, rfl⟩
    exact Finset.sum_nonneg fun t _ => mul_nonneg (hf t).1 (Nat.cast_nonneg _)

end Properties.Poub.Subprotocol.SecurityProofs
