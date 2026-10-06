import Proofs.Flock.Soundness.Refine.Schedule

/-!
# Seams: the executable's constants are the model's

The executable verifier (`Flock.*`) and the model (`Model.*`, `Accounting.*`) each define the `fast100` schedule and the
quantities derived from it. These theorems say the two agree, for every `m`:

* `fast100_schedOf`: the executable's schedule, read as the model's (`schedOf`), is the model's schedule (PROTOCOL.md §15);
* `fast100_level`: at every level, the stratified queries (§13.5), the cap depth and path length (§14), the
  consistency-coin count and the codeword length are the model's (`LevelSeam`);
* `fast100_dims`: the dimensions the executable's Ligerito loop tracks are the ones the model's rounds use (`DimSeam`,
  §13.3);
* `fast100_alpha`: at every level the number of consistency coins, `ceilLog2 q`, is not one, so the model's draw of
  them is a vector draw.

The kernel checks each for `m < 40`, and both schedules are `none` above 35.
-/

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

/-- **The schedule.** The executable's `fast100` schedule is the model's, for every `m`. -/
theorem fast100_schedOf (m : ℕ) : Option.map schedOf (Flock.fast100 m) = Accounting.fast100 m := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, Option.map schedOf (Flock.fast100 m.val) = Accounting.fast100 m.val := by
      decide +kernel
    exact key ⟨m, hm⟩
  · rw [exec_fast100_none m (by omega), model_fast100_none m (by omega)]; rfl

/-- **Every level.** At each level of an executable `fast100` schedule, the strata, cap depth, path length,
consistency-coin count and codeword length are the model's. -/
theorem fast100_level (m : ℕ) (s : Flock.Schedule) (hs : Flock.fast100 m = some s) :
    ∀ lv ∈ s.levels, LevelSeam lv := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ s ∈ Flock.fast100 m.val, (s.levels.all fun lv => decide (LevelSeam lv)) = true := by
      decide +kernel
    intro lv hlv
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hlv
    simpa using Array.all_eq_true.1 (key ⟨m, hm⟩ s hs) i hi
  · rw [exec_fast100_none m (by omega)] at hs; cases hs

/-- **Not one consistency coin.** At each level of an executable `fast100` schedule, `ceilLog2 q ≠ 1`, so the
level's consistency coins are a vector draw of `Fin (ceilLog2 q) → 𝔽`, not a scalar one. -/
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

/-- **The dimensions.** Every executable `fast100` schedule is for its own `m` and satisfies `DimSeam`. -/
theorem fast100_dims (m : ℕ) (s : Flock.Schedule) (hs : Flock.fast100 m = some s) : DimSeam s ∧ s.m = m := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ s ∈ Flock.fast100 m.val, decide (DimSeam s ∧ s.m = m.val) = true := by
      decide +kernel
    simpa using key ⟨m, hm⟩ s hs
  · rw [exec_fast100_none m (by omega)] at hs; cases hs

end FlockSoundness.Refine
