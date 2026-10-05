import FlockSoundness.Discharge.ZkLink.TablesAt
import FlockSoundness.Discharge.HiddenExec.Tables
import FlockSoundness.Protocol.Discharge.ZkHidden.Defs

/-!
# The `--zk` tables of a hidden-output session

ZkLink's `DrawSetupZK`, `tableAtZK`, `tablesAtZK` and `DrawOkZK` (`ZkLink/TablesAt.lean`) for `Stmt.setupHidden`, which
`Stmt.setupTables` calls under a tag with hidden outputs (`HiddenExec.setupOne`): the draw's setup is HiddenExec's
`DrawSetupH` at `zkInputs I` in place of Integrate's `DrawSetup`, and the table is built from it as `DrawSetupZK.tabAt`
builds it, field for field.

* `DrawSetupZKH I mPts S d`: a `DrawSetupH` at `zkInputs I`, `Zk.schedule`'s schedule and `t_pad` at its `m`, and
  `DrawSetupZK`'s facts of them.
* `DrawSetupZKH.table`, `.refused`, `tableAtZKH`, `tablesAtZKH`: as `DrawSetupZK`'s.
* `planAtZKH`, `tabAtZKH`: the one-table plan `zk_flock_countC_execOS_cr` takes.
* `DrawOkZKH`: `DrawOkZK` with `setupHidden` for `setupH`.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock FlockLevel3 Game Model Refine Exec DecodeZL ZkSession ZkExec ZkLink HiddenExec
open scoped ENNReal NNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} (I : Inputs) (mPts : ℕ)

variable {I mPts}

namespace DrawSetupZKH

variable {S : Finset (Fin n)} {d : Lean.Json} (y : DrawSetupZKH I mPts S d)

end DrawSetupZKH

variable {Reg : Type} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))

end FlockSoundness.Discharge.ZkHidden
