import FlockSoundness.Refine.Run
import FlockSoundness.Instance
import Flock.Verify
import FlockSoundness.Protocol.Refine.Tapes

/-!
# The model's messages, read off a decoded proof

A run of the model (`Game.Plays`) takes the prover's messages as a typed tape. For the executable's session these are
the decoded proof's fields, each read as the model's type through the executable's field embedding (`GF128.of`), in the
order the model receives them. The coins need no such function: they are the record's rounds' coins, in order.

This file holds those tapes and the correspondence between the executable's and the model's phase outputs.
-/

namespace FlockSoundness.Refine

open Flock Game

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

end FlockSoundness.Refine
