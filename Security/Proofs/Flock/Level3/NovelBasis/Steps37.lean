/-
`NovelBasis`'s kernel checks of `svC`'s entries 37–39.
-/
import Proofs.Flock.Level3.NovelBasis.Svc

open Flock

set_option maxRecDepth 20000

namespace FlockLevel3

theorem svC_step_37 : sIter svC 37 (F128.monomial 37) = svC[37]! := by decide +kernel
theorem svC_step_38 : sIter svC 38 (F128.monomial 38) = svC[38]! := by decide +kernel
theorem svC_step_39 : sIter svC 39 (F128.monomial 39) = svC[39]! := by decide +kernel

end FlockLevel3
