import FlockSoundness.Discharge.ZkExec.PlaysStatements
import FlockSoundness.Discharge.ZkExec.RunsLig
import FlockSoundness.Protocol.Discharge.ZkExec.PlaysLig

/-!
# The hiding Ligerito's runs

`zk_ligerito_plays`: `Zk.ligerito` refines the model's masked level 0 with Ligerito's levels compiled. The model's
games branch only on the schedule, so each of them has a run at any claim, polynomial and state, given its messages and
coins of the right sizes (`foldRoundsC_any`, `foldRoundsZK_any`, `drawKs_any`, `twoOodC_any`,
`queryAndBatchPadC_any`, `ligTailC_any`, `ligeritoPadZC_any`); and its Ligerito verdict is not a reject, since the last
recursive round records a query. What the executable gives is the coins: every round it consumes issued the number of
coins the model draws there (`fold_coins`, `lane_coins`, `queryLevel0_bind`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {D α β : Type}

/-! ## Runs at any values -/

/-! ## The model's phases at any values -/

/-! ## The executable's coins -/

/-! ## Ligerito under `--zk` -/

end FlockSoundness.Discharge.ZkExec
