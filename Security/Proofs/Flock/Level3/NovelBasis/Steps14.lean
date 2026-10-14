/-
`NovelBasis`'s kernel checks of `svC`'s entries 14–19.
-/
import Proofs.Flock.Level3.NovelBasis.Svc

open Flock

set_option maxRecDepth 20000

namespace FlockLevel3

theorem svC_step_14 : sIter svC 14 (F128.monomial 14) = svC[14]! := by decide +kernel
theorem svC_step_15 : sIter svC 15 (F128.monomial 15) = svC[15]! := by decide +kernel
theorem svC_step_16 : sIter svC 16 (F128.monomial 16) = svC[16]! := by decide +kernel
theorem svC_step_17 : sIter svC 17 (F128.monomial 17) = svC[17]! := by decide +kernel
theorem svC_step_18 : sIter svC 18 (F128.monomial 18) = svC[18]! := by decide +kernel
theorem svC_step_19 : sIter svC 19 (F128.monomial 19) = svC[19]! := by decide +kernel

end FlockLevel3
