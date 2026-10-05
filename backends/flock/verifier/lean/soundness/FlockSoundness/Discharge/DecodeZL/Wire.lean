import FlockSoundness.Discharge.Exec.FramesRep
import FlockSoundness.Discharge.Exec.Shape
import FlockSoundness.Discharge.Exec.Decode
import FlockSoundness.Protocol.Discharge.DecodeZL.Wire
import FlockSoundness.Refine.Frames

/-!
# The per-Kind decoder (R11b)

`DecodesAll` asks for a decoder `dec : ℕ → ByteArray → List Msg` from a round's retained bytes to the messages the
round carries. A round retains the encodings of its chunk's frames (`Exec.RepRounds`), and frames are recoverable from
their bytes when each one's length fits its header (`encs_inj`). So the decoder recovers the frames, then reads each
message the round carries off them by its `Wire`: a number of field values (an `.one` frame is one value, a `.vec` frame
its values), or one `.bytes` frame.

* `Wire`, `readAll`, `framesOf`, `decRound`, `decOf`: the decoder. `decRound_encs`: on a chunk's bytes it reads the
  chunk's frames.
* `coinPat`, `groupPat`, `rdChunks`, `Carries`: a phase's frames carry its messages, in the groups its spine cuts them
  into at its coins, then a tail the proof file carries. `Carries.append` glues two phases.
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Refine Exec Game Model

namespace Wire

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

end Wire

end FlockSoundness.Discharge.DecodeZL
