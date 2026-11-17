module

import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
public import Definitions.Flock.Accounting.Padded
public import Flock.Zk
public import Proofs.Flock.Soundness.LigeritoPhase
public import Proofs.Flock.Soundness.Refine.Schedule
public import Proofs.Flock.Soundness.Refine.WideLeafChecks
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Refine.Rep

/-! `Zk.schedule` at every `m` it accepts. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkLink

open Flock Accounting Refine

variable {wideLeaf : Bool}

/-- `true` also where `Zk.schedule` refuses `m`. -/
def zkSchedFacts (wideLeaf : Bool) (m : ℕ) : Bool :=
  match Zk.schedule wideLeaf m with
  | .ok (s, t) =>
    let a := schedOf s
    let l := a.levels.headD default
    decide (2 ≤ a.levels.length) && decide (l.lanes = a.k0) && decide (a.m - 7 = a.k0 + l.logCols) &&
      levelsChained a.levels a.ext && decide (a.m = m) && decide (l.logLen ≤ 40) && decide (t ≤ 2 ^ l.logCols) &&
      decide (2 ^ l.logCols + t ≤ 2 ^ l.logLen) && decide (1 < 2 ^ l.logCols + t) &&
      decide (100 * (2 ^ l.logCols + t) ≤ 98 * 2 ^ l.logLen)
  | .error _ => true

theorem zkSchedFacts_all : ∀ m, 25 ≤ m → m ≤ 35 → zkSchedFacts wideLeaf m = true := by
  intro m h1 h2; cases wideLeaf <;> interval_cases m <;> decide +kernel

theorem schedule_at {m : ℕ} {r : Flock.Schedule × ℕ} (h : Zk.schedule wideLeaf m = .ok r) :
    (25 ≤ m ∧ m ≤ 35) ∧ ∃ s, Flock.fast100At wideLeaf m = some s := by
  unfold Zk.schedule at h
  obtain ⟨q0, hq, h⟩ := ebind_ok h
  have hm : 25 ≤ m ∧ m ≤ 35 := by
    unfold Zk.queries0 at hq
    split_ifs at hq <;> first | (simp only [beq_iff_eq] at *; omega)
  refine ⟨hm, ?_⟩
  revert h
  cases Flock.fast100At wideLeaf m with
  | none => intro h; cases h
  | some s => intro _; exact ⟨s, rfl⟩

theorem schedule_spec {m : ℕ} {r : Flock.Schedule × ℕ} (h : Zk.schedule wideLeaf m = .ok r) :
    25 ≤ m ∧ m ≤ 35 ∧ ∃ s, Flock.fast100 m = some s := by
  obtain ⟨hm, s, hs⟩ := schedule_at h
  have hsome := fast100At_isSome (wide := wideLeaf) m
  rw [hs] at hsome
  exact ⟨hm.1, hm.2, Option.isSome_iff_exists.1 hsome.symm⟩


theorem schedule_zk {m : ℕ} {sch : Flock.Schedule} {tPad : ℕ} (h : Zk.schedule wideLeaf m = .ok (sch, tPad)) :
    (2 ≤ (schedOf sch).levels.length ∧ ((schedOf sch).levels.headD default).lanes = (schedOf sch).k0 ∧
      (schedOf sch).m - 7 = (schedOf sch).k0 + ((schedOf sch).levels.headD default).logCols ∧
      levelsChained (schedOf sch).levels (schedOf sch).ext = true) ∧
    (schedOf sch).m = m ∧ Accounting.PadOK (schedOf sch).level₀ tPad := by
  obtain ⟨h1, h2, -⟩ := schedule_spec h
  have hf := zkSchedFacts_all (wideLeaf := wideLeaf) m h1 h2
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
