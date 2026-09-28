import Proofs.Flock.Soundness.Discharge.ZkExec.Statements
import Proofs.Flock.Soundness.Discharge.ZkSession.Session

/-!
# `Zk.schedule` at every m it accepts

`Zk.schedule` accepts `m` from 25 to 35 (`schedule_spec`). At each, its schedule has the shape `TabZK` needs, its `m`, and
level 0 padded by `t_pad` meets `PadOK` (`schedule_zk`), from `zkSchedFacts_all`, checked by `decide +kernel` at each `m`.
-/

namespace FlockSoundness.Discharge.ZkLink

open Flock Accounting Refine

/-- What `DrawSetupZK` needs of `Zk.schedule m`'s schedule and `t_pad`, as a `Bool` (`true` where it refuses `m`). -/
def zkSchedFacts (m : ℕ) : Bool :=
  match Zk.schedule m with
  | .ok (s, t) =>
    let a := schedOf s
    let l := a.levels.headD default
    decide (2 ≤ a.levels.length) && decide (l.lanes = a.k0) && decide (a.m - 7 = a.k0 + l.logCols) &&
      levelsChained a.levels a.ext && decide (a.m = m) && decide (l.logLen ≤ 40) && decide (t ≤ 2 ^ l.logCols) &&
      decide (2 ^ l.logCols + t ≤ 2 ^ l.logLen) && decide (1 < 2 ^ l.logCols + t) &&
      decide (100 * (2 ^ l.logCols + t) ≤ 98 * 2 ^ l.logLen)
  | .error _ => true

/-- **The m check**: every `m` `Zk.schedule` accepts (25 to 35) meets `zkSchedFacts`, by the kernel's `decide`. -/
theorem zkSchedFacts_all : ∀ m, 25 ≤ m → m ≤ 35 → zkSchedFacts m = true := by
  intro m h1 h2; interval_cases m <;> decide +kernel

/-- `Zk.schedule` accepts only `25 ≤ m ≤ 35`, and only where `fast100` has a schedule. -/
theorem schedule_spec {m : ℕ} {r : Flock.Schedule × ℕ} (h : Zk.schedule m = .ok r) :
    25 ≤ m ∧ m ≤ 35 ∧ ∃ s, Flock.fast100 m = some s := by
  unfold Zk.schedule at h
  obtain ⟨q0, hq, h⟩ := ebind_ok h
  have hm : 25 ≤ m ∧ m ≤ 35 := by
    unfold Zk.queries0 at hq
    split_ifs at hq <;> first | (simp only [beq_iff_eq] at *; omega)
  refine ⟨hm.1, hm.2, ?_⟩
  revert h
  cases Flock.fast100 m with
  | none => intro h; cases h
  | some s => intro _; exact ⟨s, rfl⟩


/-- **`Zk.schedule`'s schedule has `DrawSetupZK`'s shape, `m` and padding** (`PadOK`, its radius from `ρ' ≤ 0.98`). -/
theorem schedule_zk {m : ℕ} {sch : Flock.Schedule} {tPad : ℕ} (h : Zk.schedule m = .ok (sch, tPad)) :
    (2 ≤ (schedOf sch).levels.length ∧ ((schedOf sch).levels.headD default).lanes = (schedOf sch).k0 ∧
      (schedOf sch).m - 7 = (schedOf sch).k0 + ((schedOf sch).levels.headD default).logCols ∧
      levelsChained (schedOf sch).levels (schedOf sch).ext = true) ∧
    (schedOf sch).m = m ∧ Accounting.PadOK (schedOf sch).level₀ tPad := by
  obtain ⟨h1, h2, -⟩ := schedule_spec h
  have hf := zkSchedFacts_all m h1 h2
  unfold zkSchedFacts at hf
  rw [h] at hf
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hf
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨a1, a2⟩, a3⟩, a4⟩, a5⟩, a6⟩, a7⟩, a8⟩, a9⟩, a10⟩ := hf
  refine ⟨⟨a1, a2, a3, a4⟩, a5, ⟨a6, a7, a8, a9, ?_⟩⟩
  set l := (schedOf sch).levels.headD default
  show 0 < 1 - Real.sqrt (l.padRate tPad) - eta
  have hn : (0 : ℝ) < 2 ^ l.logLen := by positivity
  have hρ : l.padRate tPad ≤ 98 / 100 := by
    unfold Level.padRate
    rw [div_le_iff₀ hn]
    have : ((100 * (2 ^ l.logCols + tPad) : ℕ) : ℝ) ≤ ((98 * 2 ^ l.logLen : ℕ) : ℝ) := by exact_mod_cast a10
    push_cast at this ⊢
    linarith
  have hs : Real.sqrt (l.padRate tPad) < 199 / 200 := by
    rw [Real.sqrt_lt' (by norm_num)]
    linarith
  unfold eta
  linarith

end FlockSoundness.Discharge.ZkLink
