module

public import Proofs.Flock.Soundness.Accounting.PaddedNumbers
import Mathlib.Tactic.IntervalCases

/-!
# The numbers on the wide-leaf schedule

The wide-leaf schedule (`Flock.fast100Wide`) is `fast100` with level 0's codeword held at `2^19` symbols: for a
witness of `2^m` bits with `32 ≤ m ≤ 35`, level 0 folds `k0 = m − 25` lanes where `fast100` folds 6, and the levels are
those of `m = 31`; up to `m = 31` the two schedules are one. `fast100` stays beside it, unchanged: the two are the
schedules of two statement versions.

This file is that schedule's table as data (`fast100Wide`), Flock ZK's values on it (`m1ScheduleWide`: 219 level-0
queries at `32 ≤ m ≤ 35`, padding `t = 2q₀`), and the soundness number: every table is sound to `2^-205` at
`η = 1/200` for every statement shape in range, at `25 ≤ m ≤ 35` (`padTableError_le_m1Wide`), as on `fast100`
(`padTableError_le_m1`). The rational checks run in Lean's kernel (`decide +kernel`). No guarantee reads the
wide-leaf schedule yet.
-/

@[expose] public section

namespace FlockSoundness.Accounting

/-- The wide-leaf schedule for `m` (`Flock.fast100Wide`), or `none` outside `22 ≤ m ≤ 35`: `fast100 m`, except
that at `32 ≤ m ≤ 35` level 0 folds `k0 = m − 25` lanes over `m = 31`'s levels. -/
def fast100Wide : ℕ → Option Schedule
  | 32 => some (mk 32 7 [18, 15, 12, 9, 6, 3] 3)
  | 33 => some (mk 33 8 [18, 15, 12, 9, 6, 3] 3)
  | 34 => some (mk 34 9 [18, 15, 12, 9, 6, 3] 3)
  | 35 => some (mk 35 10 [18, 15, 12, 9, 6, 3] 3)
  | m => fast100 m

/-- The `d` column at the wide rows, checked against the schedule: level 0's codeword has `2^19` symbols. -/
example : (fast100Wide 32).get! |>.WellShaped [19, 17, 15, 13, 11, 9] := by decide
example : (fast100Wide 33).get! |>.WellShaped [19, 17, 15, 13, 11, 9] := by decide
example : (fast100Wide 34).get! |>.WellShaped [19, 17, 15, 13, 11, 9] := by decide
example : (fast100Wide 35).get! |>.WellShaped [19, 17, 15, 13, 11, 9] := by decide

theorem fast100Wide_of_le {m : ℕ} (h : m ≤ 31) : fast100Wide m = fast100 m := by
  unfold fast100Wide
  split <;> first | rfl | omega

theorem fast100Wide_of_gt {m : ℕ} (h : 35 < m) : fast100Wide m = fast100 m := by
  unfold fast100Wide
  split <;> first | rfl | omega

/-- Flock ZK's level-0 query count on the wide-leaf schedule (`Zk.queries0Wide`): 219 at `32 ≤ m ≤ 35`, where
level 0 is `m = 31`'s, and `m1Queries₀ m` elsewhere. -/
def m1Queries₀Wide : ℕ → ℕ
  | 32 | 33 | 34 | 35 => 219
  | m => m1Queries₀ m

/-- Flock ZK's schedule at `m` on the wide-leaf schedule: `fast100Wide` with level 0's query count for the padded
rate. -/
def m1ScheduleWide (m : ℕ) : Schedule := ((fast100Wide m).getD default).withQueries₀ (m1Queries₀Wide m)

/-- Flock ZK's padding at `m` on the wide-leaf schedule: `t = 2q₀`. -/
def m1PadWide (m : ℕ) : ℕ := 2 * m1Queries₀Wide m

/-- The rational upper bounds on `√ρ'` on the wide-leaf schedule: `m = 31`'s at `32 ≤ m ≤ 35`, whose padded level 0
is `m = 31`'s. -/
def m1SqrtUpWide : ℕ → ℚ
  | 32 | 33 | 34 | 35 => 70769727 / 100000000
  | m => m1SqrtUp m

theorem m1Queries₀Wide_of_le {m : ℕ} (h : m ≤ 31) : m1Queries₀Wide m = m1Queries₀ m := by
  unfold m1Queries₀Wide
  split <;> first | rfl | omega

/-- Up to `m = 31` Flock ZK's schedule on the wide-leaf schedule is its schedule on `fast100`. -/
theorem m1ScheduleWide_of_le {m : ℕ} (h : m ≤ 31) : m1ScheduleWide m = m1Schedule m := by
  unfold m1ScheduleWide m1Schedule
  rw [fast100Wide_of_le h, m1Queries₀Wide_of_le h]

theorem m1PadWide_of_le {m : ℕ} (h : m ≤ 31) : m1PadWide m = m1Pad m := by
  unfold m1PadWide m1Pad
  rw [m1Queries₀Wide_of_le h]

/-- A schedule `s` for `m` has two levels or more, and its level 0 padded by `t` and its levels meet the conditions
of the rational bounds at `s̄` and of the soundness theorem: one check, run in the kernel (`m1Facts`, at any
schedule). -/
def padFacts (m : ℕ) (s : Schedule) (t : ℕ) (sbar : ℚ) : Bool :=
  let l := s.levels.headD default
  decide (2 ≤ s.levels.length) && decide (s.m = m) && decide (1 ≤ t) && decide (l.logLen = l.logCols + 1) &&
    decide (l.logLen ≤ 40) && decide (t ≤ 2 ^ l.logCols) && decide (2 ^ l.logCols + t ≤ 2 ^ l.logLen) &&
    decide (1 < 2 ^ l.logCols + t) && decide (0 ≤ sbar) &&
    decide (((2 ^ l.logCols + t : ℕ) : ℚ) ≤ sbar ^ 2 * 2 ^ l.logLen) && decide (sbar < 49 / 50) &&
    s.levels.all (fun l => decide l.Ok)

theorem padFacts_spec {m : ℕ} {s : Schedule} {t : ℕ} {sbar : ℚ} (h : padFacts m s t sbar = true) :
    2 ≤ s.levels.length ∧ s.m = m ∧ PadNumOK (s.levels.headD default) t sbar ∧
      PadOK (s.levels.headD default) t ∧ ∀ l ∈ s.levels, l.Ok := by
  simp only [padFacts, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, and_assoc] at h
  obtain ⟨h2, hm, ht1, hd, hd40, ht, hk, hk1, h0, hq, hlt, hok⟩ := h
  have hs := padRate_le_sq _ _ _ hq
  have hr := padRadius_pos_of _ _ _ h0 hs hlt
  exact ⟨h2, hm, ⟨ht1, hd, hk, h0, hs, hr⟩, ⟨hd40, ht, hk, hk1, hr⟩, hok⟩

/-- The wide rows meet the conditions of the rational bounds and of the soundness theorem. -/
theorem m1Wide_facts : ∀ m, 32 ≤ m → m ≤ 35 →
    padFacts m (m1ScheduleWide m) (m1PadWide m) (m1SqrtUpWide m) = true := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

/-- The rational check at the wide rows, `32 ≤ m ≤ 35`. -/
theorem m1Wide_check : ∀ m, 32 ≤ m → m ≤ 35 →
    padLinkUp (m1ScheduleWide m) (m1PadWide m) +
        padRepUp (m1ScheduleWide m) (m1PadWide m) 2 (m1SqrtUpWide m) ^ 2 ≤ (1 / 2) ^ 205 := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

/-- **The soundness number with Flock ZK's padded level 0 on the wide-leaf schedule**, `25 ≤ m ≤ 35`: at most
`2^-205` per table, for every statement shape in range. -/
theorem padTableError_le_m1Wide : ∀ m, 25 ≤ m → m ≤ 35 → ∀ st : Stmt, st.InRange →
    padTableError (m1ScheduleWide m) st (m1PadWide m) 2 ≤ (2 : ℝ) ^ (-(205 : ℝ)) := by
  intro m h1 h2 st hst
  rcases Nat.lt_or_ge m 32 with h | h
  · rw [m1ScheduleWide_of_le (by omega), m1PadWide_of_le (by omega)]
    exact padTableError_le_m1 m h1 h2 st hst
  · have hf := padFacts_spec (m1Wide_facts m h h2)
    exact padTableError_le_of_check (m1ScheduleWide m) st hst (m1PadWide m) 2 (m1SqrtUpWide m) hf.1 hf.2.2.1
      hf.2.2.2.2 (m1Wide_check m h h2)

end FlockSoundness.Accounting
