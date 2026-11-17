module

public import Proofs.Flock.Soundness.Refine.Schedule
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
import Mathlib.Algebra.Order.Algebra
import Mathlib.Tactic.Positivity.Finset

/-! The executable's `fast100` schedule and its constants are the model's, checked by the kernel for `m < 40`; both
schedules are `none` above 35. -/

@[expose] public section

namespace FlockSoundness.Refine

open Model Accounting

theorem exec_fast100_none (m : ℕ) (h : 35 < m) : Flock.fast100 m = none := by
  have : (m < 22 || m > 35) = true := by rw [Bool.or_eq_true, decide_eq_true_iff, decide_eq_true_iff]; omega
  unfold Flock.fast100
  rw [ite_eq_left_of_eq_true _ _ (eq_true this)]
  rfl

theorem model_fast100_none (m : ℕ) (h : 35 < m) : Accounting.fast100 m = none := by
  unfold Accounting.fast100
  split <;> first | rfl | omega

theorem fast100_schedOf (m : ℕ) : Option.map schedOf (Flock.fast100 m) = Accounting.fast100 m := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, Option.map schedOf (Flock.fast100 m.val) = Accounting.fast100 m.val := by
      decide +kernel
    exact key ⟨m, hm⟩
  · rw [exec_fast100_none m (by omega), model_fast100_none m (by omega)]; rfl

theorem fast100_level (m : ℕ) (s : Flock.Schedule) (hs : Flock.fast100 m = some s) :
    ∀ lv ∈ s.levels, LevelSeam lv := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ s ∈ Flock.fast100 m.val, (s.levels.all fun lv => decide (LevelSeam lv)) = true := by
      decide +kernel
    intro lv hlv
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hlv
    simpa using Array.all_eq_true.1 (key ⟨m, hm⟩ s hs) i hi
  · rw [exec_fast100_none m (by omega)] at hs; cases hs

theorem fast100_alpha (m : ℕ) (s : Flock.Schedule) (hs : Flock.fast100 m = some s) :
    ∀ lv ∈ s.levels, ceilLog2 lv.q ≠ 1 := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ s ∈ Flock.fast100 m.val,
        (s.levels.all fun lv => decide (ceilLog2 lv.q ≠ 1)) = true := by
      decide +kernel
    intro lv hlv
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hlv
    simpa using Array.all_eq_true.1 (key ⟨m, hm⟩ s hs) i hi
  · rw [exec_fast100_none m (by omega)] at hs; cases hs

theorem fast100_dims (m : ℕ) (s : Flock.Schedule) (hs : Flock.fast100 m = some s) : DimSeam s ∧ s.m = m := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ s ∈ Flock.fast100 m.val, decide (DimSeam s ∧ s.m = m.val) = true := by
      decide +kernel
    simpa using key ⟨m, hm⟩ s hs
  · rw [exec_fast100_none m (by omega)] at hs; cases hs

end FlockSoundness.Refine
