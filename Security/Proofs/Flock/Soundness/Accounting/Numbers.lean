import Proofs.Flock.Soundness.Accounting.Fast100

/-!
# The numbers

`tableError` for `fast100` run twice on one root, with live coins, for every supported `m` and
every statement shape in `Stmt.InRange`:

* at most `2^-205` for every `22 ≤ m ≤ 35`, at the analysis slack `η = 1/200` (`Bound.eta`).

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
    linkUp s + repUp s ^ 2 ≤ (1 / 2) ^ 205 := by
  intro m h1 h2 s hs
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at hs; subst hs; decide +kernel)

/-- The rational check for `m = 34, 35`. -/
theorem check_34_35 : ∀ m, 34 ≤ m → m ≤ 35 → ∀ s, fast100 m = some s →
    linkUp s + repUp s ^ 2 ≤ (1 / 2) ^ 205 := by
  intro m h1 h2 s hs
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at hs; subst hs; decide +kernel)

theorem half_pow_205_le : ((1 / 2 : ℝ) ^ 205) ≤ (2 : ℝ) ^ (-(205 : ℝ)) := by
  rw [Real.rpow_neg (by norm_num), one_div, inv_pow]
  exact le_of_eq (by norm_cast)

/-- **The soundness number for `fast100` run twice on one root, `22 ≤ m ≤ 33`:** at most
`2^-205` per table, for every statement shape in range. -/
theorem tableError_le_of_le_33 {m : ℕ} (h1 : 22 ≤ m) (h2 : m ≤ 33) {s : Schedule}
    (hs : fast100 m = some s) (st : Stmt) (hst : st.InRange) :
    tableError s st ≤ (2 : ℝ) ^ (-(205 : ℝ)) := by
  refine (tableError_le_up s (fast100_levels_ok m s hs) st hst).trans ?_
  refine le_trans ?_ half_pow_205_le
  have h := Rat.cast_le (K := ℝ) |>.2 (check_le_33 m h1 h2 s hs)
  refine h.trans (le_of_eq ?_)
  push_cast; ring

/-- **The soundness number for `m = 34, 35`:** at most `2^-205` per table. -/
theorem tableError_le_of_34_35 {m : ℕ} (h1 : 34 ≤ m) (h2 : m ≤ 35) {s : Schedule}
    (hs : fast100 m = some s) (st : Stmt) (hst : st.InRange) :
    tableError s st ≤ (2 : ℝ) ^ (-(205 : ℝ)) := by
  refine (tableError_le_up s (fast100_levels_ok m s hs) st hst).trans ?_
  refine le_trans ?_ half_pow_205_le
  have h := Rat.cast_le (K := ℝ) |>.2 (check_34_35 m h1 h2 s hs)
  refine h.trans (le_of_eq ?_)
  push_cast; ring

end FlockSoundness.Accounting
