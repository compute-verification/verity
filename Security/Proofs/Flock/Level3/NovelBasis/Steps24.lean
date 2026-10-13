/-
`NovelBasis`'s kernel checks of `svC`'s entries 24–27.
-/
import Proofs.Flock.Level3.NovelBasis.Svc

open Flock

set_option maxRecDepth 20000

namespace FlockLevel3

theorem svC_step_24 : sIter svC 24 (F128.monomial 24) = svC[24]! := by decide +kernel
theorem svC_step_25 : sIter svC 25 (F128.monomial 25) = svC[25]! := by decide +kernel
theorem svC_step_26 : sIter svC 26 (F128.monomial 26) = svC[26]! := by decide +kernel
theorem svC_step_27 : sIter svC 27 (F128.monomial 27) = svC[27]! := by decide +kernel

end FlockLevel3
