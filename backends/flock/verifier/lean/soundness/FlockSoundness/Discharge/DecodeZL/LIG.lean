import FlockSoundness.Discharge.DecodeZL.ZL
import FlockSoundness.Discharge.DecodeZL.Sizes
import FlockSoundness.Protocol.Discharge.DecodeZL.LIG

/-!
# Ligerito's rounds carry its messages (R11b)

`lig_carries`: Ligerito's frames `ligFrames`, read by `wLIG`, are `ligMsgs`, cut at `ligShape`'s coins. `Carries` reads
the spine only through its coin pattern, so the rounds are proved at a pattern (`CarriesP`) and glued with
`CarriesP.append`: the start, the first out-of-domain value (`carriesP_f`), the `k0` folding rounds (`carriesP_fold`),
the last one with the level-1 cap (`carriesP_cap`), level 0's out-of-domain samples and queries (`carriesP_mid`), and
the level loop (`carriesP_levels`, by recursion on the levels left, ending at `yr` with `carriesP_yr`). `fast100`'s
schedules fold 4 lanes above level 0, so `ligShape`'s pattern is the rounds' (`isCoin_rec`), and their caps and query
counts fit the frames' bounds (`fast100_small`).
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Refine Exec Game Model

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## Carrying at a coin pattern -/

/-! ## Ligerito's rounds -/

/-! ## The coin pattern of Ligerito's spine -/

end FlockSoundness.Discharge.DecodeZL
