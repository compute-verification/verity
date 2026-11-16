module

public import Proofs.Flock.SpotFlock.InstanceStmt.RecordFits

@[expose] public section
open Flock RecDeploy.VStarNets
open FlockComplete (jsz)
open Lean (Json)

namespace SpotFlock.InstanceStmt

example (r : Run) : ((tabAt r).1.unitSample).isSome = true := by decide +kernel
example (r : Run) : jsz (((tabAt r).1.unitSample).getD Json.null) ≤ 100000 := by decide +kernel
example (r : Run) : ((tabAt r).1.unitSample).getD Json.null = sampleJ := by rfl
example (r : Run) : (tabAt r).1.unitSample = some sampleJ := by kernel_rfl'
example : jsz sampleJ ≤ 100000 := by decide +kernel

end SpotFlock.InstanceStmt
