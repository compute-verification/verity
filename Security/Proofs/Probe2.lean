import Proofs.Flock.Level3.NovelBasis
open Flock FlockLevel3
set_option maxRecDepth 20000

open Lean Elab Tactic Meta in
elab "kernel_rfl" : tactic => do
  let g ← getMainGoal
  let some (_, a, _) := (← instantiateMVars (← g.getType)).eq? | throwError "kernel_rfl: the goal is not an equation"
  g.assign (← mkEqRefl a)
  replaceMainGoal []

theorem q30 : sIter svC 30 (F128.monomial 30) = svC[30]! := by kernel_rfl
theorem q31 : sIter svC 31 (F128.monomial 31) = svC[31]! := by kernel_rfl
theorem q32 : sIter svC 32 (F128.monomial 32) = svC[32]! := by kernel_rfl
theorem q33 : sIter svC 33 (F128.monomial 33) = svC[33]! := by kernel_rfl
theorem q34 : sIter svC 34 (F128.monomial 34) = svC[34]! := by kernel_rfl
theorem q35 : sIter svC 35 (F128.monomial 35) = svC[35]! := by kernel_rfl
theorem q36 : sIter svC 36 (F128.monomial 36) = svC[36]! := by kernel_rfl
theorem q37 : sIter svC 37 (F128.monomial 37) = svC[37]! := by kernel_rfl
theorem q38 : sIter svC 38 (F128.monomial 38) = svC[38]! := by kernel_rfl
theorem q39 : sIter svC 39 (F128.monomial 39) = svC[39]! := by kernel_rfl
