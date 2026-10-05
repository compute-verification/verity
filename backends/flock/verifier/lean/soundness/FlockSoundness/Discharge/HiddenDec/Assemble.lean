import FlockSoundness.Discharge.HiddenDec.Statements
import FlockSoundness.Discharge.DecodeZL.Runs
import FlockSoundness.Discharge.HiddenExec.Accepted
import FlockSoundness.Protocol.Discharge.HiddenDec.Assemble
import FlockSoundness.Discharge.HiddenExec.Accepts

/-!
# The decoder is right at the hidden-output headline's draws

`decodesOn_execH`: DecodeZL's `decodesOn_exec` at the hidden tables, from plays-coin's `decodesOnOfRuns`. At an
outcome flock-verify accepts under hidden outputs:

* `accepts_repsH`: DecodeZL's `accepts_reps` under hidden outputs. The one table's statement `st` is setupHidden's at
  the record's draw, and the per-rep facts come from the acceptance lemmas without `hh` (`accepts_verdict_hm96Any`,
  `accepts_rep_roundsAny`, `setupTables_okAny`, `setupTables_pointsAny`, `Discharge/HiddenExec`).
* `tableAtH_of_accepts` puts `stmtOf st` at the hidden table, `tableAtH_ptLocal` its link layout at `PT_LOCAL`, and
  `fast100_schedOf` its schedule at `fast100`'s.
* DecodeZL's `repRun_at`, at its wires' chunk lemmas `op_carries` and `lig_carries` (and `zl_carries`, which `repRun_at`
  reads), makes each rep's run a `RepRun` for `decAt`; `RecordsLive` says the outcome's rounds, link points, root and
  proof files are the record's.
-/

namespace FlockSoundness.Discharge.HiddenDec

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.DecodeZL
open FlockSoundness.Discharge.HiddenExec

local notation "𝔽" => FlockLevel3.GF128

end FlockSoundness.Discharge.HiddenDec
