/-
`NovelBasis`'s kernel checks of `svC`'s entries 34–36.
-/
import Proofs.Flock.Level3.NovelBasis.Svc

open Flock

set_option maxRecDepth 20000

namespace FlockLevel3

theorem svC_step_34 : sIter svC 34 (F128.monomial 34) = svC[34]! := by decide +kernel
theorem svC_step_35 : sIter svC 35 (F128.monomial 35) = svC[35]! := by decide +kernel
theorem svC_step_36 : sIter svC 36 (F128.monomial 36) = svC[36]! := by decide +kernel

end FlockLevel3
