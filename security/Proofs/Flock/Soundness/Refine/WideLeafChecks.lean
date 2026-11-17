module

import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
public import Flock.WideLeaf
public import Proofs.Flock.Soundness.Accounting.PaddedNumbers
public import Proofs.Flock.Soundness.Refine.Seam

/-! The executable's wide-leaf schedule and its constants are the model's, checked by the kernel for `m < 40`; both
schedules are `none` above 35. -/

@[expose] public section

namespace FlockSoundness.Refine

open Model Accounting

theorem exec_fast100Wide_none (m : ℕ) (h : 35 < m) : Flock.fast100Wide m = none := by
  unfold Flock.fast100Wide
  repeat' split
  all_goals first | rfl | omega

theorem model_fast100Wide_none (m : ℕ) (h : 35 < m) : Accounting.fast100Wide m = none := by
  rw [Accounting.fast100Wide_of_gt h, model_fast100_none m h]

theorem fast100Wide_schedOf (m : ℕ) : Option.map schedOf (Flock.fast100Wide m) = Accounting.fast100Wide m := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, Option.map schedOf (Flock.fast100Wide m.val) = Accounting.fast100Wide m.val := by
      decide +kernel
    exact key ⟨m, hm⟩
  · rw [exec_fast100Wide_none m (by omega), model_fast100Wide_none m (by omega)]; rfl

theorem fast100Wide_level (m : ℕ) (s : Flock.Schedule) (hs : Flock.fast100Wide m = some s) :
    ∀ lv ∈ s.levels, LevelSeam lv := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ s ∈ Flock.fast100Wide m.val,
        (s.levels.all fun lv => decide (LevelSeam lv)) = true := by
      decide +kernel
    intro lv hlv
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hlv
    simpa using Array.all_eq_true.1 (key ⟨m, hm⟩ s hs) i hi
  · rw [exec_fast100Wide_none m (by omega)] at hs; cases hs

theorem fast100Wide_alpha (m : ℕ) (s : Flock.Schedule) (hs : Flock.fast100Wide m = some s) :
    ∀ lv ∈ s.levels, ceilLog2 lv.q ≠ 1 := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ s ∈ Flock.fast100Wide m.val,
        (s.levels.all fun lv => decide (ceilLog2 lv.q ≠ 1)) = true := by
      decide +kernel
    intro lv hlv
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hlv
    simpa using Array.all_eq_true.1 (key ⟨m, hm⟩ s hs) i hi
  · rw [exec_fast100Wide_none m (by omega)] at hs; cases hs

theorem fast100Wide_dims (m : ℕ) (s : Flock.Schedule) (hs : Flock.fast100Wide m = some s) : DimSeam s ∧ s.m = m := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ s ∈ Flock.fast100Wide m.val, decide (DimSeam s ∧ s.m = m.val) = true := by
      decide +kernel
    simpa using key ⟨m, hm⟩ s hs
  · rw [exec_fast100Wide_none m (by omega)] at hs; cases hs

theorem exec_fast100At_none (wide : Bool) (m : ℕ) (h : 35 < m) : Flock.fast100At wide m = none := by
  cases wide
  · exact exec_fast100_none m h
  · exact exec_fast100Wide_none m h

theorem fast100At_schedOf (wide : Bool) (m : ℕ) :
    Option.map schedOf (Flock.fast100At wide m) = Accounting.fast100At wide m := by
  cases wide
  · exact fast100_schedOf m
  · exact fast100Wide_schedOf m

variable {wide : Bool}

theorem fast100At_level (m : ℕ) (s : Flock.Schedule) (hs : Flock.fast100At wide m = some s) :
    ∀ lv ∈ s.levels, LevelSeam lv := by
  cases wide
  · exact fast100_level m s hs
  · exact fast100Wide_level m s hs

theorem fast100At_alpha (m : ℕ) (s : Flock.Schedule) (hs : Flock.fast100At wide m = some s) :
    ∀ lv ∈ s.levels, ceilLog2 lv.q ≠ 1 := by
  cases wide
  · exact fast100_alpha m s hs
  · exact fast100Wide_alpha m s hs

theorem fast100At_dims (m : ℕ) (s : Flock.Schedule) (hs : Flock.fast100At wide m = some s) : DimSeam s ∧ s.m = m := by
  cases wide
  · exact fast100_dims m s hs
  · exact fast100Wide_dims m s hs

theorem fast100At_isSome (m : ℕ) : (Flock.fast100At wide m).isSome = (Flock.fast100 m).isSome := by
  cases wide
  · rfl
  · by_cases hm : m < 40
    · have key : ∀ m : Fin 40, (Flock.fast100Wide m.val).isSome = (Flock.fast100 m.val).isSome := by decide +kernel
      exact key ⟨m, hm⟩
    · show (Flock.fast100Wide m).isSome = _
      rw [exec_fast100Wide_none m (by omega), exec_fast100_none m (by omega)]

end FlockSoundness.Refine
