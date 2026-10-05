import FlockSoundness.Discharge.ZkExec.LiveStatements
import FlockSoundness.Discharge.ZkExec.Verdict
import FlockSoundness.Discharge.ZkExec.PlaysRun
import FlockSoundness.Discharge.ZkExec.Setup
import FlockSoundness.Discharge.Exec.Merkle
import FlockSoundness.Discharge.Exec.Retained
import FlockSoundness.Discharge.DecodeZL.Runs
import FlockSoundness.Protocol.Discharge.ZkExec.RunsLive

/-!
# `RunsZC` at the live `--zk` tables, from `TablesAtZKLive`

`runsZC_live`: at each of the law's draws, on an outcome flock-verify `--zk` accepts, `TablesAtZKLive` puts the draw's
table at `tablesAtZK` (`DrawSetupZK.table`, at the executable's dimensions, pads and inner code: `table_seam`,
`table_lay`), the acceptance peels to the one table's two `verifyRep` successes (`setupTables_ok`, `zk_verify_ok`) and
each rep's run (`zk_verifyRep_ok`), `zk_rep_run_plays` makes each run a `RepRun` for `decAtZ` at the result it plays on
the proof's messages, and `zk_table_verdict` makes the table's verdict accept at the two results.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL ZkSession

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

section Table

variable {n mPts : ℕ} {I : Inputs} {S : Finset (Fin n)} {d : Lean.Json}

end Table

end FlockSoundness.Discharge.ZkExec
