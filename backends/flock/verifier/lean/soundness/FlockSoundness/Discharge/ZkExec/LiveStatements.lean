import FlockSoundness.Discharge.ZkExec.VerdictStatements
import FlockSoundness.Discharge.ZkLink.TablesAt
import FlockSoundness.Protocol.Discharge.ZkExec.LiveStatements

/-!
# `RunsZC` at the live `--zk` tables, from `TablesAtZKLive` (statement)

`RunsZC` at [wave2-zk-link]'s live tables `tablesAtZK` (`ZkLink/TablesAt.lean`): on every live outcome flock-verify
`--zk` accepts, each rep has a `RepRun` on its rounds along `repShapeZ` and the compiled table accepts at the two runs'
ends. Its pieces are proved or frozen apart: the per-rep runs (`zk_rep_plays`, `zk_rep_run`), the verdict
(`ZkTableVerdict`) and the table being the executable's (`TablesAtZKLive`, taken here as a hypothesis).

* `nXZ`, `decAtZ`, `opnAtZ`: M0's `nX`, `decAt` and `opnAt` at `--zk`: the table's number of extra claims, the decoder
  `decZ` at the table's dimensions, and the openings `opnZ` at the statement setupH sets up at the record's draw under
  the `--zk` tags and its `--zk` schedule, on the first two proof files.
* `RunsZCLive`: given `TablesAtZKLive`, the tags (`TagsOk`), `wr` the coin server's record and its draw at each of the
  law's draws, `RunsZC` holds at the law's draws (`RunsOnZC`) at `tablesAtZK` with those, no tail and `capOfRoot`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL ZkSession

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

end FlockSoundness.Discharge.ZkExec
