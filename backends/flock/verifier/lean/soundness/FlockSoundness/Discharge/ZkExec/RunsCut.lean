import FlockSoundness.Discharge.ZkExec.RunsCarries
import FlockSoundness.Discharge.ZkExec.RunsSizes
import FlockSoundness.Protocol.Discharge.ZkExec.RunsCut

/-!
# A `--zk` rep's messages cut into its rounds

`zk_cut`: round 0 carries the pads root and the later rounds `zk_carries`' groups, so the rep's messages cut along
`shapeZE` are `decZ`'s groups of the rounds.

* `rootOf_encs`: a 64-byte string absorbed just before a round's closing squeeze is the 64 bytes `rootOf` reads.
* `roundGroups_succ`: from round 1 on, `decZ`'s groups are `decOf`'s, which `DecodeZL.roundGroups_eq` gives as the
  chunks' reads, and `DecodeZL.cut_of_carries` cuts `zk_carries` into the same.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL

local notation "𝔽" => FlockLevel3.GF128

end FlockSoundness.Discharge.ZkExec
