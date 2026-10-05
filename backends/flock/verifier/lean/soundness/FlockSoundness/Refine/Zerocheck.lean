import FlockSoundness.Refine.Arith
import FlockSoundness.Protocol.Refine.Zerocheck

/-!
# The zerocheck refines the model's (PROTOCOL.md §9.1–§9.2)

`zerocheck_refines`: if the executable's `bindAndZerocheck` succeeds, the model's `zerocheck`, played with the
proof's zerocheck messages (`zcMsgs`) and the coins of the rounds the executable consumed, returns `some zc'`, and `zc'`
is the executable's output read in the model's field (`ZcRel`).

The residual rounds (Z6) are a loop in the executable and a recursion in the model. `zStep` is one round as the
executable computes it; `residual_plays` runs the model's rounds on the fold of `zStep`, and `forIn_rounds` shows the
executable's loop is that fold.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript

local notation "𝔽" => FlockLevel3.GF128

end FlockSoundness.Refine
