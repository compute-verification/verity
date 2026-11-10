/-
`NovelBasis`'s kernel checks of `svC`'s entries 20–23.
-/
module

public import Proofs.Flock.Level3.NovelBasis.Svc
public import Flock.Ligerito

@[expose] public section

open Flock

set_option maxRecDepth 20000

namespace FlockLevel3

theorem svC_step_20 : sIter svC 20 (F128.monomial 20) = svC[20]! := by decide +kernel
theorem svC_step_21 : sIter svC 21 (F128.monomial 21) = svC[21]! := by decide +kernel
theorem svC_step_22 : sIter svC 22 (F128.monomial 22) = svC[22]! := by decide +kernel
theorem svC_step_23 : sIter svC 23 (F128.monomial 23) = svC[23]! := by decide +kernel

end FlockLevel3
