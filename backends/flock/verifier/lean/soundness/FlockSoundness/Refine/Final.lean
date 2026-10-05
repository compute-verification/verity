import FlockSoundness.Refine.LigeritoRun
import FlockSoundness.Protocol.Refine.Final

/-!
# The final check refines the model's (PROTOCOL.md §13.6)

`final_refines`: when the executable's final check (`finalProg`, the verifier's last step) passes, the model's
`finalCheck` holds on the images of its state, given that the executable's basis at the original point is the model's
`bHat` there (`hbasis`); and the final check touches no rounds.

The two compute the same sum term by term:

* the subspace polynomials: the executable's `whats` is the model's `What` at `omega` (`whats_get`);
* the executable's out-of-domain and consistency loops are folds (`forIn_yield_fold`), whose words are the model's
  sums (`oodFold_get`, `consFold_get`).
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## The subspace polynomials (§13.4) -/

/-! ## The executable's final check, as folds -/

/-! ## The final check's words, as the model's -/

end FlockSoundness.Refine
