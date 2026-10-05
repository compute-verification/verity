import FlockSoundness.Discharge.DecodeZL.Statements
import FlockSoundness.Protocol.Discharge.DecodeZL.ZL

/-!
# The zerocheck's and lincheck's rounds carry their messages (R11b)

`zl_carries`: from the domain frame through `z_partial`'s round, the rounds of `zcFrames ++ lcFrames`, read by `wZL`,
are `zcMsgs ++ lcMsgs`, cut at `zcShape ++ lcShape`'s coins. The frames split into rounds that each close at their last
frame (`carries_one`), the residual rounds by induction (`carries_pairs`), and the rounds glue (`Carries.append`).
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Refine Exec Game Model

local notation "𝔽" => FlockLevel3.GF128

end FlockSoundness.Discharge.DecodeZL
