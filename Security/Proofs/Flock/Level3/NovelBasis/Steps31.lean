/-
`NovelBasis`'s kernel checks of `svC`'s entries 31–33.
-/
import Proofs.Flock.Level3.NovelBasis.Svc

open Flock

set_option maxRecDepth 20000

namespace FlockLevel3

theorem svC_step_31 : sIter svC 31 (F128.monomial 31) = svC[31]! := by decide +kernel
theorem svC_step_32 : sIter svC 32 (F128.monomial 32) = svC[32]! := by decide +kernel
theorem svC_step_33 : sIter svC 33 (F128.monomial 33) = svC[33]! := by decide +kernel

end FlockLevel3
