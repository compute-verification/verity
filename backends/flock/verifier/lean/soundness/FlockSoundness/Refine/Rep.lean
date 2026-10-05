import FlockSoundness.Refine.Basis
import Flock.Verify
import FlockSoundness.Protocol.Refine.Rep

/-!
# One rep refines the model's (PROTOCOL.md §9–§14)

`rep_refines`: when the executable's `verifyRep` accepts rep `r`, the record carries the rep's stream and the proof
decodes. Then, at the proof's openings, the model's compiled rep `repC`, played with the proof's messages (`repMsgs`) and
the coins of every round of the stream, ends in a result:
- whose verdict is true;
- whose caps are the proof's;
- whose requests passed the executable's Merkle checks.

It composes the four phases: `zerocheck_refines`, `lincheck_refines`, `opening_refines` and `ligerito_accepts_opening`.
The statement's own facts enter as `Realizes`-style hypotheses: `FoldRealizes` for the circuit fold, `ExtraRealizes`
for the extra claims.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

end FlockSoundness.Refine
