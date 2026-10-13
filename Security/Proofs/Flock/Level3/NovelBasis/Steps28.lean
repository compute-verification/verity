/-
`NovelBasis`'s kernel checks of `svC`'s entries 28–30.
-/
import Proofs.Flock.Level3.NovelBasis.Svc

open Flock

set_option maxRecDepth 20000

namespace FlockLevel3

theorem svC_step_28 : sIter svC 28 (F128.monomial 28) = svC[28]! := by decide +kernel
theorem svC_step_29 : sIter svC 29 (F128.monomial 29) = svC[29]! := by decide +kernel
theorem svC_step_30 : sIter svC 30 (F128.monomial 30) = svC[30]! := by decide +kernel

end FlockLevel3
