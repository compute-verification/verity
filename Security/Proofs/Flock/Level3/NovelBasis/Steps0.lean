/-
`NovelBasis`'s kernel checks of `svC`'s entries 0–13.
-/
import Proofs.Flock.Level3.NovelBasis.Svc

open Flock

set_option maxRecDepth 20000

namespace FlockLevel3

theorem svC_step_0 : sIter svC 0 (F128.monomial 0) = svC[0]! := by decide +kernel
theorem svC_step_1 : sIter svC 1 (F128.monomial 1) = svC[1]! := by decide +kernel
theorem svC_step_2 : sIter svC 2 (F128.monomial 2) = svC[2]! := by decide +kernel
theorem svC_step_3 : sIter svC 3 (F128.monomial 3) = svC[3]! := by decide +kernel
theorem svC_step_4 : sIter svC 4 (F128.monomial 4) = svC[4]! := by decide +kernel
theorem svC_step_5 : sIter svC 5 (F128.monomial 5) = svC[5]! := by decide +kernel
theorem svC_step_6 : sIter svC 6 (F128.monomial 6) = svC[6]! := by decide +kernel
theorem svC_step_7 : sIter svC 7 (F128.monomial 7) = svC[7]! := by decide +kernel
theorem svC_step_8 : sIter svC 8 (F128.monomial 8) = svC[8]! := by decide +kernel
theorem svC_step_9 : sIter svC 9 (F128.monomial 9) = svC[9]! := by decide +kernel
theorem svC_step_10 : sIter svC 10 (F128.monomial 10) = svC[10]! := by decide +kernel
theorem svC_step_11 : sIter svC 11 (F128.monomial 11) = svC[11]! := by decide +kernel
theorem svC_step_12 : sIter svC 12 (F128.monomial 12) = svC[12]! := by decide +kernel
theorem svC_step_13 : sIter svC 13 (F128.monomial 13) = svC[13]! := by decide +kernel

end FlockLevel3
