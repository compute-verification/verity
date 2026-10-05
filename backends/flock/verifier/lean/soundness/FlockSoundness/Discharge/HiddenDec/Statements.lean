import FlockSoundness.Discharge.DecodeZL.OP
import FlockSoundness.Discharge.DecodeZL.LIG
import FlockSoundness.Discharge.HiddenExec.Tables
import FlockSoundness.Protocol.Discharge.HiddenDec.Statements

/-!
# The decoder at the hidden-output headline's draws: statements

`flock_headline_exec_hidden`'s open guarantee is `hdec : DecodesOn …` at the hidden tables `tablesAtH`. #862's
`DecodesOnExec` (`Discharge/DecodeZL`) is the same guarantee at #828's tables, under `Stmt.setupH`; the decoder `decAt`,
the number of extra claims `nX`, `RecordsLive` and the chunk lemmas (`zl_carries`, `op_carries` at `wOP`, `lig_carries`
at `wLIG`) are generic in the table or the setup and are DecodeZL's. What changes is where the draw is set up:

* `TagsOkH`: DecodeZL's `TagsOk` with hidden outputs (`hiddenOutputs = true`).
* `DrawOkH`: DecodeZL's `DrawOk` with `Stmt.setupHidden` for `Stmt.setupH`.
* `opnAtH`: DecodeZL's `opnAt` at `Stmt.setupHidden`'s statement.
* `DecodesOnExecH`: `DecodesOn` at the hidden tables, the decoder `decAt` at DecodeZL's wires `wOP` and `wLIG`,
  `one := true`, no final messages, `capOfRoot`.
-/

namespace FlockSoundness.Discharge.HiddenDec

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.DecodeZL
open FlockSoundness.Discharge.HiddenExec

local notation "𝔽" => FlockLevel3.GF128

variable {n mPts : ℕ} {Reg : Type}

end FlockSoundness.Discharge.HiddenDec
