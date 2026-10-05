import FlockSoundness.Discharge.DecodeZL.Runs
import FlockSoundness.Discharge.DecodeZL.Sizes
import FlockSoundness.Protocol.Discharge.DecodeZL.Assemble

/-!
# The decoder is right at the headline's draws

`decodesOn_exec`: `DecodesOn` at #828's tables, from plays-coin's `decodesOnOfRuns`. At an outcome flock-verify
accepts:

* `accepts_reps`: the one table's statement `st` (setupH at the record's draw), and for each rep its stream, its decoded
  proof, its run on the stream's coins with the proof's messages (`accepts_verdict_hm96`), the stream's rounds retaining
  `repFrames`' chunks (`accepts_rep_rounds`), and the salted verdict on the two runs.
* `tableAt_of_accepts` (#828) puts `stmtOf st` at the headline's table, `tableAt_ptLocal` its link layout at
  `PT_LOCAL`, and `fast100_schedOf` its schedule at `fast100`'s.
* `repRun_at` (`Runs.lean`) makes each rep's run a `RepRun` for `decAt`; `RecordsLive` says the outcome's rounds, link
  points, root and proof files are the record's.
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.Integrate

local notation "𝔽" => FlockLevel3.GF128

end FlockSoundness.Discharge.DecodeZL
