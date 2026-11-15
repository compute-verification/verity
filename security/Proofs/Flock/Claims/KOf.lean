module

public import Definitions.Claims.Recursion
public import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# `k = ⌈4·N₀/e⌉` in arithmetic on `ℚ`

`Claims.Recursion.kOf` computes `E^rep`'s accepted reruns per wait from `e`'s Taylor sums (`eSum`) alone, and is
`⌈4·N₀/e⌉` at every rational `N₀` (`kOf_eq`), so an outer session's `k` (`Nums.k`) is the ceiling it was
(`Nums.k_eq`). The two steps:

* where the ceilings at `e`'s two Taylor bounds agree, they are `⌈4·N₀/e⌉` (`ceil_eSum_step`);
* at `j + 1` terms with `j ≥ ⌈2·N₀⌉·den(N₀)`, the lower bound's ceiling is `⌈4·N₀/e⌉` (`ceil_eSum_exact`): `j!·eSum (j+1)`
  is an integer and `j!·(e − eSum (j+1)) < 1/j`, so no integer lies between `k·den·j!·eSum (j+1)` and `k·den·j!·e`.
-/

@[expose] public section

namespace Claims.Recursion

open Finset

theorem eSum_cast (m : ℕ) : (eSum m : ℝ) = ∑ i ∈ range m, (1 : ℝ) ^ i / i.factorial := by
  simp [eSum]

theorem eSum_le_exp (m : ℕ) : (eSum m : ℝ) ≤ Real.exp 1 := by
  rw [eSum_cast]; exact Real.sum_le_exp_of_nonneg zero_le_one m

theorem one_le_eSum {m : ℕ} (hm : 1 ≤ m) : (1 : ℚ) ≤ eSum m :=
  calc (1 : ℚ) = ∑ i ∈ range 1, (i.factorial : ℚ)⁻¹ := by simp
    _ ≤ eSum m := sum_le_sum_of_subset_of_nonneg (range_mono hm) fun _ _ _ => by positivity

theorem eSum_pos {m : ℕ} (hm : 1 ≤ m) : (0 : ℝ) < eSum m := by
  have := one_le_eSum hm
  exact_mod_cast (by linarith : (0 : ℚ) < eSum m)

theorem exp_le_eSum {m : ℕ} (hm : 0 < m) :
    Real.exp 1 ≤ (eSum m : ℝ) + ((m : ℝ) + 1) / ((m.factorial : ℝ) * m) := by
  have := Real.exp_bound' zero_le_one le_rfl hm
  rw [eSum_cast]; simpa using this

theorem natCeil_ratCast (r : ℚ) : ⌈(r : ℝ)⌉₊ = ⌈r⌉₊ := by
  rw [← Int.ceil_toNat, ← Int.ceil_toNat, Rat.ceil_cast]

theorem ceil_div_anti {q a b : ℝ} (hq : 0 ≤ q) (ha : 0 < a) (hab : a ≤ b) : ⌈4 * q / b⌉₊ ≤ ⌈4 * q / a⌉₊ :=
  Nat.ceil_mono (div_le_div_of_nonneg_left (by linarith) ha hab)

theorem ceil_div_nonpos {q a : ℝ} (hq : q ≤ 0) (ha : 0 < a) : ⌈4 * q / a⌉₊ = 0 :=
  Nat.ceil_eq_zero.2 (div_nonpos_of_nonpos_of_nonneg (by linarith) ha.le)

theorem factorial_mul_eSum (j : ℕ) :
    (j.factorial : ℚ) * eSum (j + 1) = ((∑ i ∈ range (j + 1), j.factorial / i.factorial : ℕ) : ℚ) := by
  rw [eSum, mul_sum, Nat.cast_sum]
  refine sum_congr rfl fun i hi => ?_
  rw [Nat.cast_div (Nat.factorial_dvd_factorial (Nat.lt_succ_iff.1 (mem_range.1 hi))) (by positivity),
    div_eq_mul_inv]

/-- **`j + 1` terms are enough** at `j ≥ ⌈2·q⌉·den(q)`: `⌈4·q/eSum (j+1)⌉ = ⌈4·q/e⌉`. -/
theorem ceil_eSum_exact {q : ℚ} {j : ℕ} (hj : ⌈2 * q⌉₊ * q.den ≤ j) :
    ⌈4 * q / eSum (j + 1)⌉₊ = ⌈4 * (q : ℝ) / Real.exp 1⌉₊ := by
  have hL0 := eSum_pos (m := j + 1) (by omega)
  have he := Real.exp_pos 1
  rw [← natCeil_ratCast,
    show ((4 * q / eSum (j + 1) : ℚ) : ℝ) = 4 * (q : ℝ) / (eSum (j + 1) : ℝ) by push_cast; ring]
  rcases le_or_gt q 0 with hq | hq
  · have hq' : (q : ℝ) ≤ 0 := by exact_mod_cast hq
    rw [ceil_div_nonpos hq' hL0, ceil_div_nonpos hq' he]
  have hqr : (0 : ℝ) < q := by exact_mod_cast hq
  refine le_antisymm ?_ (ceil_div_anti hqr.le hL0 (eSum_le_exp _))
  set n₀ := ⌈4 * (q : ℝ) / Real.exp 1⌉₊ with hn₀
  set L : ℝ := (eSum (j + 1) : ℝ) with hL
  set F : ℝ := (j.factorial : ℝ) with hF
  have hF0 : 0 < F := by positivity
  have hd : (0 : ℝ) < q.den := by exact_mod_cast q.den_pos
  have hqpd : (q : ℝ) = q.num / q.den := Rat.cast_def q
  obtain ⟨A, hA⟩ : ∃ A : ℕ, F * L = A :=
    ⟨_, by rw [hF, hL]; exact_mod_cast factorial_mul_eSum j⟩
  have hn₂ : n₀ ≤ ⌈2 * q⌉₊ := by
    rw [← natCeil_ratCast, hn₀]
    refine Nat.ceil_mono ?_
    rw [div_le_iff₀ he]
    have := Real.exp_one_gt_d9
    push_cast
    nlinarith
  have hnd : (n₀ : ℝ) * q.den ≤ j := by
    have : n₀ * q.den ≤ j := le_trans (Nat.mul_le_mul_right _ hn₂) hj
    exact_mod_cast this
  have hce : 4 * (q : ℝ) ≤ n₀ * Real.exp 1 := by
    have := Nat.le_ceil (4 * (q : ℝ) / Real.exp 1)
    rw [← hn₀, div_le_iff₀ he] at this
    exact this
  have hU := exp_le_eSum (m := j + 1) (by omega)
  rw [← hL, Nat.factorial_succ] at hU
  push_cast at hU
  have hX : F * (Real.exp 1 - L) * ((j : ℝ) + 1) ^ 2 ≤ (j : ℝ) + 2 := by
    have h1 : Real.exp 1 - L ≤ ((j : ℝ) + 1 + 1) / ((j + 1) * F * (j + 1)) := by rw [hF]; linarith
    have hpos : (0 : ℝ) < (j + 1) * F * (j + 1) := by positivity
    rw [le_div_iff₀ hpos] at h1
    nlinarith
  have hX0 : 0 ≤ F * (Real.exp 1 - L) := mul_nonneg hF0.le (by linarith [eSum_le_exp (j + 1)])
  rw [Nat.ceil_le, div_le_iff₀ hL0]
  refine le_of_not_gt fun hlt => ?_
  have hlt' : ((n₀ * q.den * A : ℕ) : ℝ) < ((4 * q.num * j.factorial : ℤ) : ℝ) := by
    push_cast
    rw [← hA, hqpd] at *
    rw [hqpd] at hlt
    have := mul_lt_mul_of_pos_right hlt (mul_pos hd hF0)
    field_simp at this
    nlinarith
  have hint : ((n₀ * q.den * A : ℕ) : ℤ) + 1 ≤ 4 * q.num * j.factorial := by
    have : ((n₀ * q.den * A : ℕ) : ℤ) < 4 * q.num * j.factorial := by exact_mod_cast hlt'
    omega
  have hint' : (n₀ : ℝ) * q.den * A + 1 ≤ 4 * q.num * F := by
    have : (((n₀ * q.den * A : ℕ) : ℤ) : ℝ) + 1 ≤ ((4 * q.num * j.factorial : ℤ) : ℝ) := by exact_mod_cast hint
    push_cast at this
    rw [hF]; exact this
  have hce' : 4 * (q.num : ℝ) * F ≤ n₀ * q.den * F * Real.exp 1 := by
    rw [hqpd] at hce
    have := mul_le_mul_of_nonneg_right hce (mul_nonneg hd.le hF0.le)
    field_simp at this
    nlinarith
  have h1 : 1 ≤ (n₀ : ℝ) * q.den * (F * (Real.exp 1 - L)) := by
    rw [← hA] at hint'
    nlinarith
  have h2 : 1 ≤ (j : ℝ) * (F * (Real.exp 1 - L)) :=
    h1.trans (mul_le_mul_of_nonneg_right hnd hX0)
  nlinarith

/-- **Where the ceilings at `e`'s two Taylor bounds agree, they are `⌈4·q/e⌉`.** -/
theorem ceil_eSum_step {q : ℚ} {m : ℕ} (hm : 1 ≤ m)
    (h : ⌈4 * q / (eSum m + (m + 1) / (m.factorial * m))⌉₊ = ⌈4 * q / eSum m⌉₊) :
    ⌈4 * q / eSum m⌉₊ = ⌈4 * (q : ℝ) / Real.exp 1⌉₊ := by
  have hL0 := eSum_pos hm
  have he := Real.exp_pos 1
  rcases le_or_gt q 0 with hq | hq
  · have hq' : (q : ℝ) ≤ 0 := by exact_mod_cast hq
    rw [← natCeil_ratCast, show ((4 * q / eSum m : ℚ) : ℝ) = 4 * (q : ℝ) / (eSum m : ℝ) by push_cast; ring,
      ceil_div_nonpos hq' hL0, ceil_div_nonpos hq' he]
  have hqr : (0 : ℝ) < q := by exact_mod_cast hq
  apply le_antisymm
  · rw [← h, ← natCeil_ratCast, show ((4 * q / (eSum m + (m + 1) / (m.factorial * m)) : ℚ) : ℝ) =
      4 * (q : ℝ) / ((eSum m : ℝ) + ((m : ℝ) + 1) / ((m.factorial : ℝ) * m)) by push_cast; ring]
    exact ceil_div_anti hqr.le he (exp_le_eSum (by omega))
  · rw [← natCeil_ratCast, show ((4 * q / eSum m : ℚ) : ℝ) = 4 * (q : ℝ) / (eSum m : ℝ) by push_cast; ring]
    exact ceil_div_anti hqr.le hL0 (eSum_le_exp m)

theorem kOf_go (q : ℚ) : ∀ f m : ℕ, 1 ≤ m → kOf.go q f m = ⌈4 * (q : ℝ) / Real.exp 1⌉₊
  | 0, _, _ => ceil_eSum_exact le_rfl
  | f + 1, m, hm => by
    rw [kOf.go]
    split_ifs with h
    · exact ceil_eSum_step hm h
    · exact kOf_go q f (2 * m) (by omega)

/-- **`kOf` is `⌈4·N₀/e⌉`** at every rational `N₀`. -/
theorem kOf_eq (q : ℚ) : kOf q = ⌈4 * (q : ℝ) / Real.exp 1⌉₊ := kOf_go q 6 8 (by norm_num)

/-- **An outer session's `k` is `⌈4·N₀/e⌉`**, as it was before `Nums.k` computed. -/
theorem Nums.k_eq (n : Nums) : n.k = ⌈4 * (n.N₀ : ℝ) / Real.exp 1⌉₊ := kOf_eq n.N₀

end Claims.Recursion
