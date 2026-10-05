import FlockSoundness.Discharge.ZkExec.PlaysLig
import FlockSoundness.Protocol.Discharge.ZkExec.VerdictLigModel

/-!
# The model's masked level 0, with its values

`PlaysLig.lean`'s runs exist at any values; the verdict needs them at the executable's. The lane rounds end in the
lane coins (`foldRoundsZK_val`), `ρ` is its two coins (`drawKs_two`), level 0's padded queries end in `qbPadOut`
(`queryAndBatchPadC_plays`), and `ligeritoPadZC_model` composes the masked level 0 from runs of its parts, as M0's
`ligeritoC_model`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {D : Type}

end FlockSoundness.Discharge.ZkExec
