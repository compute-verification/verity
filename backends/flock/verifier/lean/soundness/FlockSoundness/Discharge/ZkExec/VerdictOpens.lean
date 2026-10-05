import FlockSoundness.Discharge.ZkExec.VerdictStatements
import FlockSoundness.Discharge.ZkExec.PlaysRep
import FlockSoundness.Discharge.ZkExec.PlaysDet
import FlockSoundness.Protocol.Discharge.ZkExec.VerdictOpens

/-!
# A `--zk` rep's requested rows verify

`zk_opens_ok`: the third conclusion of `ZkRepAccepts`, the `--zk` analogue of M0's `opensOKL_of`. Every row a compiled
`--zk` rep requests verifies against its level's cap (`OpensOKZ`), at openings that are the proof's rows and salts.

The run `zk_rep_plays` builds records its requests; `zk_ligerito_reqs` builds the same run as `zk_ligerito_plays` and
tracks, along Ligerito's level loop, that its caps are the proof's and that each request is one whose Merkle check the
executable passed (`ReqOKZ`): level 0's on the `2^k₀ + 4`-lane rows against the commitment's cap (`Zk.queryLevel0`), a
later level's as M0's `ReqOK`. `zk_rep_reqs` carries that to the rep, and a run has one result (`plays_det`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {D β : Type}

/-! ## Requests that passed the executable's Merkle checks -/

/-! ## The model's phases, with their requests -/

/-! ## The executable's level-0 Merkle check -/

/-! ## Ligerito under `--zk`, with its requests -/

/-! ## The rep, with its requests -/

/-! ## `ZkRepAccepts`'s openings -/

end FlockSoundness.Discharge.ZkExec
