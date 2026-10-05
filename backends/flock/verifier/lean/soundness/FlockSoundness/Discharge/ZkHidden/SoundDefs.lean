import FlockSoundness.Discharge.ZkHidden.Statements
import FlockSoundness.Discharge.ZkExec.LiveStatements
import FlockSoundness.Discharge.HiddenDec.Statements
import FlockSoundness.Protocol.Discharge.ZkHidden.SoundDefs

/-!
# `RunsZC` at the hidden-output `--zk` tables (statement)

ZkExec's `RunsZCLive` (`ZkExec/LiveStatements.lean`) for the sessions production `--zk` writes, whose tables
`Stmt.setupTables` sets up with `Stmt.setupHidden` (hidden outputs, the default `--statement`). Its `opnAtZ` reads the
statement `Stmt.setupH` sets up, so the hidden case is a different `RunsZC` instance:

* `opnAtZH`: `opnAtZ` at the statement `Stmt.setupHidden` sets up at the record's draw under the `--zk` tags and coin
  tree.
* `RunsZCLiveH`: `RunsZCLive` with `tablesAtZKH`, `opnAtZH`, `TablesAtZKHLive`, `TagsOkH` and `DrawOkZKH` for
  `tablesAtZK`, `opnAtZ`, `TablesAtZKLive`, `TagsOk` and `DrawOkZK`, and `TablesAtZKHLive`'s hypotheses on the circuit
  (`HmOut.parse` accepts it at the `--zk` tags; its unit returns the outputs). The decoder `decAtZ`, the empty tail and
  `capOfRoot` are the public case's.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock Refine Exec Game Model DecodeZL ZkSession ZkExec

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

end FlockSoundness.Discharge.ZkHidden
