import FlockSoundness.Discharge.Composed.DefsJ
import FlockSoundness.Discharge.ZkHidden.SoundBind
import FlockSoundness.Discharge.ZkHidden.SoundRuns
import FlockSoundness.Protocol.Discharge.ZkHidden.DefsJ
import FlockSoundness.Discharge.ZkHidden.SoundDefs
import FlockSoundness.Discharge.ZkHidden.Theorems

set_option Elab.async false

/-!
# One hidden-output `--zk` sampled session of `J` tables: the pieces `zk_session_soundHJ` reads

`Composed/DefsJ.lean`'s pieces at `J` tables, under hidden outputs (`Stmt.setupHidden`, the default `--statement`), each
the one-table hidden-output piece (`ZkHidden/Defs.lean`, `SoundBind.lean`, `SoundRuns.lean`) at table `j`:

* `setupTables_partsH`: `Tables.setupTables_parts` under hidden outputs. At `count > 1`, table `j` is `setupHidden`'s
  statement under `Tags.tableIdentity count j` at the subset draw of part `j`; `setupOf_tableJH` is `setupOf_tableJ`'s
  counterpart.
* `parseH_tagsAt`, `loadPublicH_tagsAt`: `HmOut.parse` (untyped) and `HmOut.loadPublic` read no identity.
* `DrawSetupZHJ`, `tableZHJ`, `planZHJ`, `tabZHJ`: `DrawSetupZJ`, `tableZJ`, `planZJ`, `tabZJ` with a `DrawSetupH` at
  table `j`'s inputs (`inputsJ I j`), and the refused table of the reference setup `y₀ : DrawSetupZHJ` at table 0's
  inputs.
* `tableShHJ`, `planShHJ`, `tabShHJ`, `sitesShHJ`: the public hidden-output tables (`HiddenExec.TableH`) at the same
  draws, each at the inputs it is set up on (table `j`'s, or table 0's, `y₀`'s, at a refused table), and their sites.
* `placeZHJ`, `layZHJ`, `rsZHJ`, `rsZHJ_computes`: `keyPlaceH`, `keyLayoutH` and `keyRowsH` at those sites, moved to the
  `--zk` tables (`SoundBind.lean`'s `placeZH`, `layZH`, `rsZH` at `J` tables).
* `TagsOkZHJ`, `DrawOkZKHJ`, `opnAtZHJ`, `runsZCJH_live`: `TagsOkZJ` with hidden outputs, `DrawOkZKJ` and `opnAtZJ` with
  `setupHidden`, and `runsZCJ_live` under hidden outputs.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## flock-verify's tables at `count > 1` under hidden outputs -/

section Parts

open Lean (Json)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Flock.Stmt}

end Parts

/-! ## Table `j`'s tags under hidden outputs -/

variable {n mPts : ℕ} {Reg : Type}

/-! ## The tables -/

namespace DrawSetupZHJ

variable {I' : Inputs} {S : Finset (Fin n)} {d : Lean.Json} (y : DrawSetupZHJ I' mPts S d)

end DrawSetupZHJ

section Shadow

variable {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)

/-! ## The public shadow, the sites, the layout and the rows -/

variable (ht : I.tags.typed = false)

variable {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)

variable (hscope : Layout.scopeOk c = true)

variable (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hpt : pub.tables = none)

end Shadow

/-! ## The tags, the coin server's draws, and the openings -/

/-! ## The message half at the live hidden-output tables -/

end FlockSoundness.Discharge.ZkHidden
