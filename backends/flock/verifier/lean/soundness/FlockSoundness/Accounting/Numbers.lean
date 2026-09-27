import FlockSoundness.Accounting.Fast100

/-!
# The numbers

`tableError` for `fast100` run twice on one root, with live coins, for every supported `m` and
every statement shape in `Stmt.InRange`:

* at most `2^-195.5` for `22 ≤ m ≤ 33`;
* at most `2^-195.4` for `m = 34, 35`.

The rational bounds are checked by exact arithmetic in Lean's kernel (`decide +kernel`, not
`native_decide`: no compiled code is trusted).
-/

namespace FlockSoundness.Accounting

theorem fast100_levels_ok : ∀ m s, fast100 m = some s → ∀ l ∈ s.levels, l.Ok := by
  intro m s h
  have hm : ∀ m', fast100 m' ≠ none → 22 ≤ m' ∧ m' ≤ 35 := by
    intro m' hm'
    by_contra hc
    apply hm'
    unfold fast100
    split <;> simp_all
  obtain ⟨h1, h2⟩ := hm m (by simp [h])
  have key : ∀ m', 22 ≤ m' → m' ≤ 35 → ∀ s', fast100 m' = some s' →
      (s'.levels.all fun l => decide l.Ok) = true := by
    intro m' h1' h2' s' hs'
    interval_cases m' <;> (simp only [fast100, Option.some.injEq] at hs'; subst hs'; decide +kernel)
  intro l hl
  have := List.all_eq_true.1 (key m h1 h2 s h) l hl
  simpa using this

/-- The rational check for `22 ≤ m ≤ 33`. -/
theorem check_le_33 : ∀ m, 22 ≤ m → m ≤ 33 → ∀ s, fast100 m = some s →
    linkUp s + repUp s ^ 2 ≤ (1 / 2) ^ 196 * (141421356 / 10 ^ 8) := by
  intro m h1 h2 s hs
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at hs; subst hs; decide +kernel)

/-- The rational check for `m = 34, 35`. -/
theorem check_34_35 : ∀ m, 34 ≤ m → m ≤ 35 → ∀ s, fast100 m = some s →
    linkUp s + repUp s ^ 2 ≤ (1 / 2) ^ 196 * (3 / 2) := by
  intro m h1 h2 s hs
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at hs; subst hs; decide +kernel)

theorem half_pow_196_mul_sqrt2_le :
    ((1 / 2 : ℝ) ^ 196 * (141421356 / 10 ^ 8)) ≤ (2 : ℝ) ^ (-(195.5 : ℝ)) := by
  have h : (2 : ℝ) ^ (-(195.5 : ℝ)) = ((2 : ℝ) ^ (196 : ℕ))⁻¹ * Real.sqrt 2 := by
    rw [Real.sqrt_eq_rpow, ← Real.rpow_natCast, ← Real.rpow_neg (by norm_num),
      ← Real.rpow_add (by norm_num)]
    norm_num
  rw [h]
  have hs : (141421356 / 10 ^ 8 : ℝ) ≤ Real.sqrt 2 :=
    (Real.le_sqrt (by norm_num) (by norm_num)).2 (by norm_num)
  rw [one_div, inv_pow]
  exact mul_le_mul_of_nonneg_left hs (by positivity)

theorem half_pow_196_mul_three_halves_le :
    ((1 / 2 : ℝ) ^ 196 * (3 / 2)) ≤ (2 : ℝ) ^ (-(195.4 : ℝ)) := by
  have h : (2 : ℝ) ^ (-(195.4 : ℝ)) = ((2 : ℝ) ^ (196 : ℕ))⁻¹ * ((8 : ℝ) ^ ((5 : ℕ)⁻¹ : ℝ)) := by
    have h8 : (8 : ℝ) ^ ((5 : ℕ)⁻¹ : ℝ) = (2 : ℝ) ^ ((3 : ℝ) / 5) := by
      rw [show (8 : ℝ) = 2 ^ (3 : ℝ) by norm_num, ← Real.rpow_mul (by norm_num)]
      norm_num
    rw [h8, ← Real.rpow_natCast, ← Real.rpow_neg (by norm_num), ← Real.rpow_add (by norm_num)]
    norm_num
  rw [h]
  have hc : (3 / 2 : ℝ) ≤ (8 : ℝ) ^ ((5 : ℕ)⁻¹ : ℝ) := by
    have e : (3 / 2 : ℝ) = ((3 / 2 : ℝ) ^ (5 : ℕ)) ^ ((5 : ℕ)⁻¹ : ℝ) :=
      (Real.pow_rpow_inv_natCast (by norm_num) (by norm_num)).symm
    rw [e]
    exact Real.rpow_le_rpow (by positivity) (by norm_num) (by positivity)
  rw [one_div, inv_pow]
  exact mul_le_mul_of_nonneg_left hc (by positivity)

/-- **The soundness number for `fast100` run twice on one root, `22 ≤ m ≤ 33`:** at most
`2^-195.5` per table, for every statement shape in range. -/
theorem tableError_le_of_le_33 {m : ℕ} (h1 : 22 ≤ m) (h2 : m ≤ 33) {s : Schedule}
    (hs : fast100 m = some s) (st : Stmt) (hst : st.InRange) :
    tableError s st ≤ (2 : ℝ) ^ (-(195.5 : ℝ)) := by
  refine (tableError_le_up s (fast100_levels_ok m s hs) st hst).trans ?_
  refine le_trans ?_ half_pow_196_mul_sqrt2_le
  have h := Rat.cast_le (K := ℝ) |>.2 (check_le_33 m h1 h2 s hs)
  refine h.trans (le_of_eq ?_)
  push_cast; ring

/-- **The soundness number for `m = 34, 35`:** at most `2^-195.4` per table (a seventh Ligerito
level adds its query term). -/
theorem tableError_le_of_34_35 {m : ℕ} (h1 : 34 ≤ m) (h2 : m ≤ 35) {s : Schedule}
    (hs : fast100 m = some s) (st : Stmt) (hst : st.InRange) :
    tableError s st ≤ (2 : ℝ) ^ (-(195.4 : ℝ)) := by
  refine (tableError_le_up s (fast100_levels_ok m s hs) st hst).trans ?_
  refine le_trans ?_ half_pow_196_mul_three_halves_le
  have h := Rat.cast_le (K := ℝ) |>.2 (check_34_35 m h1 h2 s hs)
  refine h.trans (le_of_eq ?_)
  push_cast; ring

end FlockSoundness.Accounting
