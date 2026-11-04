import Lean.Elab.Tactic.Basic

/-!
# A tactic that walks the rows audit

`rows_step` takes one step through `h : x = .ok v` for a computation `x` in `Except String`, the certifier
program's `NetworkCertifier.Rows.audit` among them, chosen by `x`'s head as Flock's `walk_step` is: a bind is split
into its halves (`ebind_ok`), the value and its equation named `v` and `h_v` after the binder when the program named
it; an `if` or `dite` is split on its condition `hc`; a `match` is split; and a `throw` closes the goal. It fails at a
`pure`. It runs only inside the elaborator, and the kernel checks every proof it builds.
-/

namespace Properties.NetworkCertifier.RowsAudit

theorem ebind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem pure_ok {α : Type} {a b : α} (h : (pure a : Except String α) = .ok b) : a = b := by
  simp only [pure, Except.pure, Except.ok.injEq] at h
  exact h

theorem throw_ne_ok {α : Type} (e : String) (a : α) : (throw e : Except String α) ≠ .ok a := fun h => by cases h

theorem throw_bind' {α β : Type} (e : String) (f : α → Except String β) :
    ((throw e : Except String α) >>= f) = throw e := rfl

open Lean Elab Tactic Meta in
/-- One step through `h : x = .ok v`, by `x`'s head (see the module docstring). -/
elab "rows_step" : tactic => do
  let g ← getMainGoal
  g.withContext do
    let some d := (← getLCtx).findFromUserName? `h | throwError "rows_step: no h"
    let ty ← instantiateMVars d.type
    let some (_, lhs, _) := ty.eq? | throwError "rows_step: h is not an equation"
    let lhs := lhs.headBeta.consumeMData
    let h := mkIdent `h
    let hc := mkIdent `hc
    if lhs.isAppOfArity ``Bind.bind 6 then
      let named := match (lhs.getArg! 5).consumeMData with
        | .lam n .. => if n.isAnonymous || n.hasMacroScopes || n.isInternal then none else some n
        | _ => none
      match named with
      | some n =>
        let x := mkIdent n
        let hx := mkIdent (n.appendBefore "h_")
        evalTactic (← `(tactic| obtain ⟨$x:ident, $hx:ident, $h:ident⟩ := ebind_ok $h))
      | none => evalTactic (← `(tactic| obtain ⟨_, _, $h:ident⟩ := ebind_ok $h))
    else if lhs.isAppOfArity ``ite 5 then
      let cs ← Term.exprToSyntax (lhs.getArg! 1)
      evalTactic (← `(tactic| by_cases $hc:ident : $cs <;>
        first | rw [ite_eq_left $hc:ident] at $h:ident | rw [ite_eq_right $hc:ident] at $h:ident))
    else if lhs.isAppOfArity ``dite 5 then
      let cs ← Term.exprToSyntax (lhs.getArg! 1)
      evalTactic (← `(tactic| by_cases $hc:ident : $cs <;>
        first | rw [dite_eq_left $hc:ident] at $h:ident | rw [dite_eq_right $hc:ident] at $h:ident))
    else if lhs.isAppOfArity ``MonadExcept.throw 5 || lhs.isAppOfArity ``throwThe 5 then
      evalTactic (← `(tactic| exact absurd $h (throw_ne_ok _ _)))
    else if (← Lean.Meta.isMatcherApp lhs) then
      evalTactic (← `(tactic| split at $h:ident))
    else throwError "rows_step: stop"

end Properties.NetworkCertifier.RowsAudit
