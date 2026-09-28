import FlockSoundness.Discharge.Hidden.Defs
import FlockSoundness.Discharge.Exec.Setup
import FlockSoundness.Discharge.Exec.Merkle

/-!
# What `Stmt.setupHidden` checked and produced (`Spec`)

`setupHidden_spec` walks `Stmt.setupHidden` as `setupH_spec` walks `Stmt.setupH`: every return is the statement its
checks built. `Facts.lean` reads off it what [wave2-hidden-exec] reads.
-/

open Flock FlockLevel3
open Lean (Json)

namespace FlockSoundness.Discharge.Hidden

open Refine

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}

theorem klog_of {k m : ℕ} (h : ¬ (decide (k > 27) || decide (m > 35)) = true) : k ≤ 27 ∧ m ≤ 35 := by
  simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at h
  omega

set_option maxHeartbeats 4000000 in
/-- **What `Stmt.setupHidden` accepted** is its `Spec`. -/
theorem setupHidden_spec' (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    Spec tags cf pf coins tables draw st := by
  cases draw <;> unfold Stmt.setupHidden at h <;> dsimp only at h <;> simp only [throw_bind', pure_bind] at h <;>
    repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hk := klog_of ‹¬ (_ || _) = true›
    exact {
      tagsEq := rfl
      leaf := rfl
      parse := by assumption
      merkle := by assumption
      pub := by
        refine ⟨_, ‹HmOut.loadPublic _ _ _ = _›, fun hd => ?_, fun d' hd => ?_⟩
        · first | exact absurd hd (Option.some_ne_none _) | rfl
        · first | exact absurd hd.symm (Option.some_ne_none _) | (cases hd; assumption)
      checkPublic := by assumption
      n := rfl
      blocksReal := rfl
      nbl := rfl
      m := rfl
      kLog := hk.1
      m35 := hk.2
      mPts := rfl
      pin := by assumption
      da := rfl
      db := rfl
      regions := by assumption
      regionWords := by assumption
      inRange := by assumption
      sigma := rfl
      hello := by assumption }

end FlockSoundness.Discharge.Hidden
