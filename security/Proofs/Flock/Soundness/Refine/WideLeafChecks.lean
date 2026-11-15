module

import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
public import Flock.WideLeaf
public import Proofs.Flock.Soundness.Accounting.WideLeafChecks
public import Proofs.Flock.Soundness.Refine.Seam
import Mathlib.Tactic.IntervalCases

/-!
# Seams of the wide-leaf schedule: the executable's constants are the model's

The executable's wide-leaf schedule (`Flock.fast100Wide`, and under `--zk` `Flock.Zk.scheduleWide`) against the model's
(`Accounting.fast100Wide`, `Accounting.m1ScheduleWide`), as `Refine.Seam` has `fast100`'s, for every `m`:

* `fast100Wide_schedOf`: the executable's schedule, read as the model's (`schedOf`), is the model's schedule;
* `fast100Wide_level`, `fast100Wide_alpha`, `fast100Wide_dims`: at every level the stratified queries, the cap depth and
  path length, the consistency-coin count and the codeword length are the model's (`LevelSeam`), the number of
  consistency coins is not one, and the dimensions are the ones the model's rounds use (`DimSeam`);
* `scheduleWide_m1`: `Zk.scheduleWide m`'s schedule is `m1ScheduleWide m` and its `t_pad` is `m1PadWide m`, at every `m`
  it accepts (25 to 35);
* `padTableError_le_scheduleWide`: so a table at the executable's wide-leaf `--zk` schedule is sound to `2^-205`.

The kernel checks each for `m < 40`, and both schedules are `none` above 35.
-/

@[expose] public section

namespace FlockSoundness.Refine

open Model Accounting

theorem exec_fast100Wide_none (m : ℕ) (h : 35 < m) : Flock.fast100Wide m = none := by
  unfold Flock.fast100Wide
  repeat' split
  all_goals first | rfl | omega

theorem model_fast100Wide_none (m : ℕ) (h : 35 < m) : Accounting.fast100Wide m = none := by
  rw [Accounting.fast100Wide_of_gt h, model_fast100_none m h]

/-- **The schedule.** The executable's wide-leaf schedule is the model's, for every `m`. -/
theorem fast100Wide_schedOf (m : ℕ) : Option.map schedOf (Flock.fast100Wide m) = Accounting.fast100Wide m := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, Option.map schedOf (Flock.fast100Wide m.val) = Accounting.fast100Wide m.val := by
      decide +kernel
    exact key ⟨m, hm⟩
  · rw [exec_fast100Wide_none m (by omega), model_fast100Wide_none m (by omega)]; rfl

/-- **Every level.** At each level of an executable wide-leaf schedule, the strata, cap depth, path length,
consistency-coin count and codeword length are the model's. -/
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

/-- **Not one consistency coin.** At each level of an executable wide-leaf schedule, `ceilLog2 q ≠ 1`. -/
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

/-- **The dimensions.** Every executable wide-leaf schedule is for its own `m` and satisfies `DimSeam`: level 0 has
`k0` lanes and `m − 7 − k0` columns, `2^18` of them at `32 ≤ m ≤ 35`. -/
theorem fast100Wide_dims (m : ℕ) (s : Flock.Schedule) (hs : Flock.fast100Wide m = some s) : DimSeam s ∧ s.m = m := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ s ∈ Flock.fast100Wide m.val, decide (DimSeam s ∧ s.m = m.val) = true := by
      decide +kernel
    simpa using key ⟨m, hm⟩ s hs
  · rw [exec_fast100Wide_none m (by omega)] at hs; cases hs

/-- `Zk.queries0Wide` gives a count only at `25 ≤ m ≤ 35`. -/
theorem queries0Wide_spec {m q : ℕ} (h : Flock.Zk.queries0Wide m = .ok q) : 25 ≤ m ∧ m ≤ 35 := by
  unfold Flock.Zk.queries0Wide Flock.Zk.queries0 at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at *; omega)

/-- `Zk.scheduleWide` accepts only `25 ≤ m ≤ 35`. -/
theorem scheduleWide_spec {m : ℕ} {r : Flock.Schedule × ℕ} (h : Flock.Zk.scheduleWide m = .ok r) :
    25 ≤ m ∧ m ≤ 35 := by
  unfold Flock.Zk.scheduleWide at h
  cases hq : Flock.Zk.queries0Wide m with
  | error e => rw [hq] at h; cases h
  | ok q => exact queries0Wide_spec hq

/-- What the wide rows' numbers need of `Zk.scheduleWide m`'s schedule and `t_pad`, as a `Bool` (`true` where it
refuses `m`). -/
def zkWideFacts (m : ℕ) : Bool :=
  match Flock.Zk.scheduleWide m with
  | .ok (s, t) => decide (schedOf s = m1ScheduleWide m) && decide (t = m1PadWide m)
  | .error _ => true

/-- **The m check**: at every `m` `Zk.scheduleWide` accepts (25 to 35), its schedule and `t_pad` are Flock ZK's on
the wide-leaf schedule, by the kernel's `decide`. -/
theorem zkWideFacts_all : ∀ m, 25 ≤ m → m ≤ 35 → zkWideFacts m = true := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

/-- **`Zk.scheduleWide`'s schedule is `m1ScheduleWide`, its `t_pad` `m1PadWide`.** -/
theorem scheduleWide_m1 {m : ℕ} {sch : Flock.Schedule} {tPad : ℕ} (h : Flock.Zk.scheduleWide m = .ok (sch, tPad)) :
    schedOf sch = m1ScheduleWide m ∧ tPad = m1PadWide m := by
  obtain ⟨h1, h2⟩ := scheduleWide_spec h
  have hf := zkWideFacts_all m h1 h2
  unfold zkWideFacts at hf
  rw [h] at hf
  simpa only [Bool.and_eq_true, decide_eq_true_eq] using hf

/-- **A table at the executable's wide-leaf `--zk` schedule is sound to `2^-205`**, for every statement shape in
range: `padTableError_le_m1Wide` at the schedule and padding `Zk.scheduleWide` gives. -/
theorem padTableError_le_scheduleWide {m : ℕ} {sch : Flock.Schedule} {tPad : ℕ}
    (h : Flock.Zk.scheduleWide m = .ok (sch, tPad)) (st : Accounting.Stmt) (hst : st.InRange) :
    padTableError (schedOf sch) st tPad 2 ≤ (2 : ℝ) ^ (-(205 : ℝ)) := by
  obtain ⟨h1, h2⟩ := scheduleWide_spec h
  obtain ⟨hs, ht⟩ := scheduleWide_m1 h
  rw [hs, ht]
  exact padTableError_le_m1Wide m h1 h2 st hst

end FlockSoundness.Refine
