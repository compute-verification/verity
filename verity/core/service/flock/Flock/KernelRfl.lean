/-
`kernel_rfl`: an equation left to the kernel. An `elab` is code run while compiling, so it is alone here, the module
`compile_time` lists, apart from the proofs that use it (`Flock.ProvedScope`'s parses that read no identity).
-/
import Lean.Elab.Tactic.Basic
import Lean.Meta.AppBuilder

namespace Flock

open Lean Elab Tactic Meta in
/-- `kernel_rfl` closes `a = b` with `Eq.refl a` and leaves `a ≡ b` to the kernel, which checks it when the declaration
is added (a false one fails there). The elaborator's `rfl` unfolds an executable function's join points into both arms;
the kernel compares two calls of it whose arguments differ only in fields it never reads in milliseconds. -/
elab "kernel_rfl" : tactic => do
  let g ← getMainGoal
  let some (_, a, _) := (← instantiateMVars (← g.getType)).eq? | throwError "kernel_rfl: the goal is not an equation"
  g.assign (← mkEqRefl a)
  replaceMainGoal []

end Flock
