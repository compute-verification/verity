import Proofs.Flock.Level3.ArithFacts
open Flock
#eval Flock.pinned.toList.map fun a => (a.lo.toNat, a.hi.toNat)
set_option profiler true
set_option profiler.threshold 50
example : Flock.pinned[3]! ≠ 1 := by decide +kernel
example : Flock.pinned[6]! ≠ 1 := by decide +kernel
