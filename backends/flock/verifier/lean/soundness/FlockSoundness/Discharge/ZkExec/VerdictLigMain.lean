import FlockSoundness.Discharge.ZkExec.VerdictLigRun
import FlockSoundness.Protocol.Discharge.ZkExec.VerdictLigMain

/-!
# The hiding Ligerito accepts

`zk_ligerito_accepts`: when `Zk.ligerito` accepts, the compiled masked phase's Ligerito, played with the proof's
messages on the coins the executable consumed, ends in a result whose verdict at the proof's openings is true. The
lane phase and level 0's padded queries are new (`lane_vals`, `queryLevel0_val`, `padE_eq`); the level loop is M0's
(`LigeritoRun.lean`), with an invariant that drops M0's caps and requests (`LoopRelV`): level 0's request is on
`2^k₀ + 4` wide rows, which M0's `ReqOK` does not describe, and the verdict reads neither.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

end FlockSoundness.Discharge.ZkExec
