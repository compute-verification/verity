module

public import Spot.OneStage
import Mathlib.Tactic.DefEqTransformations

/-!
# The verifier program v2's parameters name parses from some bytes (`paramsParsed_program`)

`Spot.OneStage.paramsParsed` binds the verifier program to `Program.parse` of the canonical text of the parameters'
`program` and returns it as the parameters' `verifier.program`. The walk peels each bind into a hypothesis
(`ok_of_bind`), splits only a `match` or `if` at the head of the block, and closes a branch that throws at once
(`throw_ne_ok`, `throw_bind_ne_ok`), so a `match` that only computes a bound value is never split.
-/

@[expose] public section

namespace Verity.Sampled

theorem ok_of_bind {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases hx : x with
  | error e => rw [hx] at h; cases h
  | ok a => rw [hx] at h; exact ⟨a, rfl, h⟩

theorem throw_ne_ok {ε α : Type} {e : ε} {a : α} : (throw e : Except ε α) ≠ .ok a := by
  simp [throw, throwThe, MonadExceptOf.throw]

theorem throw_bind_ne_ok {ε α β : Type} {e : ε} {f : α → Except ε β} {b : β} :
    ((throw e : Except ε α) >>= f) ≠ .ok b := by
  simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind]

set_option maxHeartbeats 1000000 in -- `paramsParsed` is some hundred binds, each peeled once
/-- **v2's parameters' verifier program is one some bytes parse to**: `paramsParsed` parses it from the canonical text
of the parameters' `program`. -/
theorem paramsParsed_program {j : Lean.Json} {p : _root_.Spot.OneStage.Params}
    (h : _root_.Spot.OneStage.paramsParsed j = .ok p) :
    ∃ b, _root_.Spot.Program.parse b = .ok p.verifier.program := by
  unfold _root_.Spot.OneStage.paramsParsed at h
  repeat' (first
    | (with_reducible exact absurd h throw_ne_ok)
    | (with_reducible exact absurd h throw_bind_ne_ok)
    | (with_reducible obtain ⟨_, _, h⟩ := ok_of_bind h; try beta_reduce at h)
    | (split at h; try beta_reduce at h))
  all_goals first
    | (with_reducible exact absurd h throw_ne_ok)
    | (simp only [pure, Except.pure, Except.ok.injEq] at h
       subst h
       exact ⟨_, by with_reducible assumption⟩)

end Verity.Sampled
