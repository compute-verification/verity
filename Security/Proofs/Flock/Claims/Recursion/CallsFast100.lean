module

public import Proofs.Flock.Claims.Recursion.CallsRep
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic

/-!
# The fast100 schedules' Ligerito calls

The kernel runs `fast100` at each size that has a schedule, and `fast100`'s loops are core's (`Std.Legacy.Range.forIn'`,
`Array`'s), so this module imports those in full. `CallsTop` reads the result from here: with core's definitions in full
there, its walks over a verdict's runs exceed their budget.
-/

@[expose] public section

namespace FlockDeploy.KS

open Flock

/-- **Every fast100 schedule's Ligerito makes at most `2^16` calls**, `CallsTop`'s `ligC` spelled out. -/
theorem fast100_calls : ∀ m < 36, (fast100 m).all (fun s => decide ((1 + (s.levels.size - 1)) * qAll s ≤ 2 ^ 16)) = true := by
  decide +kernel

end FlockDeploy.KS
