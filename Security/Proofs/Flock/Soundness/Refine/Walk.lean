import Proofs.Flock.Soundness.Refine.Rep

/-!
# A tactic that walks a successful executable computation

`walk_step` takes one step through `h : x = .ok v` for a computation `x` in the executable's monad `V`, chosen by `x`'s
head: a bind is split into its halves (`ebind_ok`), an `if` or `dite` is split on its condition, a `match` is split, and a
`throw` closes the goal. It fails at a `pure`, so `repeat' walk_step` reaches every return. Choosing the step by the head,
rather than trying each and backtracking, keeps Lean from unfolding the executable to look for a bind that isn't there.
The kernel checks every proof it builds, as it does every other proof in the package.
-/

namespace FlockSoundness.Refine

open Flock

theorem throw_ne_ok {α : Type} (e : String) (a : α) : (throw e : V α) ≠ .ok a := fun h => by cases h

open Lean Elab Tactic Meta in
/-- One step through `h : x = .ok v`, by `x`'s head (see the module docstring). -/
elab "walk_step" : tactic => do
  let g ← getMainGoal
  g.withContext do
    let some d := (← getLCtx).findFromUserName? `h | throwError "walk_step: no h"
    let ty ← instantiateMVars d.type
    let some (_, lhs, _) := ty.eq? | throwError "walk_step: h is not an equation"
    let lhs := lhs.headBeta.consumeMData
    let h := mkIdent `h
    if lhs.isAppOfArity ``Bind.bind 6 then
      evalTactic (← `(tactic| obtain ⟨_, _, $h:ident⟩ := ebind_ok $h))
    else if lhs.isAppOfArity ``ite 5 then
      let cs ← Term.exprToSyntax (lhs.getArg! 1)
      evalTactic (← `(tactic| by_cases hc : $cs <;>
        first | rw [ite_eq_left hc] at $h:ident | rw [ite_eq_right hc] at $h:ident))
    else if lhs.isAppOfArity ``dite 5 then
      let cs ← Term.exprToSyntax (lhs.getArg! 1)
      evalTactic (← `(tactic| by_cases hc : $cs <;>
        first | rw [dite_eq_left hc] at $h:ident | rw [dite_eq_right hc] at $h:ident))
    else if lhs.isAppOfArity ``MonadExcept.throw 5 || lhs.isAppOfArity ``throwThe 5 then
      evalTactic (← `(tactic| exact absurd $h (throw_ne_ok _ _)))
    else if (← Lean.Meta.isMatcherApp lhs) then
      evalTactic (← `(tactic| split at $h:ident))
    else throwError "walk_step: stop"

end FlockSoundness.Refine
