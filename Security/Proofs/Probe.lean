import Proofs.Flock.Level3.NovelBasis
open Flock FlockLevel3
set_option maxRecDepth 20000
set_option trace.profiler true
set_option trace.profiler.threshold 100

def chainP : Nat → F128 → F128
  | 0, y => y
  | n + 1, y => let z := chainP n y; z * z + z

theorem p1 : (sIter svC 39 (F128.monomial 39)).lo ≠ 7 := by decide +kernel
theorem p2 : (svC[39]!).lo ≠ 7 := by decide +kernel
theorem p3 : (chainP 39 (F128.monomial 39)).lo ≠ 7 := by decide +kernel
theorem p4 : (svC[38]! * svC[37]!).lo ≠ 7 := by decide +kernel
theorem p5 : (svC[38]! * svC[37]! * svC[36]! * svC[35]! * svC[34]! * svC[33]! * svC[32]! * svC[31]! * svC[30]! * svC[29]!).lo ≠ 7 := by decide +kernel
theorem p6 : (sIter svC 20 (F128.monomial 20)).lo ≠ 7 := by decide +kernel
