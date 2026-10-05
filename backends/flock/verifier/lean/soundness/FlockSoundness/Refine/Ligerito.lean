import FlockSoundness.Refine.ArithK
import FlockSoundness.Refine.ExecLM
import FlockSoundness.Refine.Schedule
import FlockSoundness.Protocol.Refine.Ligerito
import FlockSoundness.Model.Compiled

/-!
# Ligerito's rounds refine the model's (PROTOCOL.md §13.3)

The executable's Ligerito verifier carries the running claim `s` and the round polynomial `q` as values; the compiled
model carries them as functions of the openings (`LigStateC`), since the consistency values read the opened rows, which
arrive after the last coin. At the openings the proof carries, the two agree.

Each phase is a pair: the executable's `LM` program, and the model's game, run on the proof's messages and the
consumed rounds' coins.

* `fold_refines`: the folding rounds (`foldRoundsC`, the executable's `challenge` loop).
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {D : Type}

/-! ## Query positions (§13.5) -/

/-! ## The consistency value (§13.3 step 10) -/

/-! ## Queries and batching (§13.3 steps 7–11) -/

/-! ## The recursive rounds (§13.3) -/

end FlockSoundness.Refine
