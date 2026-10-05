import FlockSoundness.Discharge.ZkExec.VerdictLigModel
import FlockSoundness.Discharge.ZkExec.VerdictLigPad
import FlockSoundness.Discharge.ZkExec.VerdictLigFinal
import FlockSoundness.Protocol.Discharge.ZkExec.VerdictLigRun

/-!
# The hiding Ligerito's run, with its values

`zk_ligerito_plays` with the values: the run of the compiled masked phase's Ligerito that the executable's coins give has,
at the openings, the executable's claim, polynomial, out-of-domain and consistency contexts from level 0's queries on,
so its verdict is the executable's final check (`final_refinesZ`). The lane phase is masked, so only its coins, `ρ`
and the blinded claim `T'` carry over (`lane_vals`); level 0's enforced sum is `consE0` (`queryLevel0_val`), and with
the extra lanes' terms it is the model's padded column (`consPad_eq`). Levels ≥ 1 are M0's (`LigeritoRun.lean`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {β : Type}

end FlockSoundness.Discharge.ZkExec
