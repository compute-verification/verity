/-
A walk of the program's call hierarchy as total functions: one step at a time, in at most a number of steps past which the
verifier refuses. The bound is more steps than any run takes, so a walk the verifier finishes is the one it would finish
without the bound (`Steps.run_mono`).
-/
import Flock.Bytes

namespace Flock.Steps

/-- The steps a walk follows before it refuses. -/
def FUEL : Nat := 2 ^ 62

/-- `step` from `a` until it is done, in at most `fuel` steps, then a refusal naming `what`. -/
@[specialize] def run {α β : Type} (what : String) (step : α → V (β ⊕ α)) : Nat → α → V β
  | 0, _ => throw s!"{what} takes more steps than this verifier follows"
  | n + 1, a => do
    match ← step a with
    | .inl b => pure b
    | .inr a' => run what step n a'

theorem run_succ {α β : Type} {what : String} {step : α → V (β ⊕ α)} :
    ∀ {n : Nat} {a : α} {b : β}, run what step n a = .ok b → run what step (n + 1) a = .ok b
  | 0, _, _, h => by simp [run] at h
  | n + 1, a, b, h => by
    rw [run.eq_2] at h ⊢
    cases hs : step a with
    | error _ => simp [hs, bind, Except.bind] at h
    | ok x =>
      simp only [hs, bind, Except.bind] at h ⊢
      cases x with
      | inl _ => exact h
      | inr _ => exact run_succ h

/-- A walk that finishes in `n` steps finishes, with the same result, in any more. -/
theorem run_mono {α β : Type} {what : String} {step : α → V (β ⊕ α)} {n m : Nat} {a : α} {b : β} (hnm : n ≤ m)
    (h : run what step n a = .ok b) : run what step m a = .ok b := by
  induction hnm with
  | refl => exact h
  | step _ ih => exact run_succ ih

end Flock.Steps
