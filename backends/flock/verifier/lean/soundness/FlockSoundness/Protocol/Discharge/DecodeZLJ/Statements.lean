import FlockSoundness.Protocol.Discharge.DecodeZL.OP
import FlockSoundness.Protocol.Discharge.DecodeZL.LIG
import FlockSoundness.Protocol.Discharge.Tables.Statements

/-!
# The decoder at `J` tables: statements

`headline_tables_subset`'s open guarantee is `hdec : DecodesOnJ …`: at every draw, registration and table `j` of the
plan, on every live outcome whose session directory flock-verify accepts, table `j`'s reps run on the tapes decoded from
its streams `(j, 0)`, `(j, 1)`, and the salted table's verdict accepts. This file states it at the tables flock-verify
sets up at `count > 1` (`accepts_tables`), with dec-zl's one-table decoder run on each table's two streams:

* `partJson d count j`: part `j` of the draw `d`'s units, the subset draw `partDraw` that `Stmt.setupTables` sets table
  `j` up at.
* `SetupJ I mPts j d`: under table `j`'s tags (`tagsAt`), setupH accepts the draw `d`, at a `fast100` schedule of at
  least two levels and `mPts` link points. `SetupJ.spec` is its table, built as #828's `DrawSetup.specAt` builds one.
* `tableJ`, `planJ`: table `j` at the draw's JSON is `SetupJ.spec` at part `j` where flock-verify sets that part up, and
  the fallback `T₀` elsewhere; the plan has `I.count` tables.
* `decJ`, `opnJ`: table `j`'s decoder (`wZL ++ (wOP ++ wLIG)` at its dimensions, claims and schedule) and openings
  (`opnOf` at setupH's statement for part `j`, on proof files `2j` and `2j + 1`).
* `TagsOkJ`, `RecordsLiveJ`, `DrawOkJ`: `TagsOk`, `RecordsLive` and `DrawOk` at `J` tables.
* `DecodesOnJExec`: `DecodesOnJ` at `planJ`, for every fallback `T₀`, with the decoder `decJ`, single coins as field
  elements, no final messages, table `j`'s cap from `root_B[j]` and its openings `opnJ`.
-/

namespace FlockSoundness.Discharge.DecodeZLJ

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.Tables FlockSoundness.Discharge.DecodeZL
open Lean (Json)

/-- **Part `j` of the draw `d`**: `partDraw` at `d`'s population and units, the subset draw `Stmt.setupTables` sets
table `j` of `count` up at, when `d` decodes. -/
def partJson (d : Json) (count j : ℕ) : Json :=
  match Draw.UnitDraw.ofJson d with
  | .ok u => partDraw u.population u.units count j
  | .error _ => d

variable {n : ℕ} (I : Inputs) (mPts : ℕ)

variable {I mPts} (ht : I.tags.typed = false)

variable {Reg : Type}

end FlockSoundness.Discharge.DecodeZLJ
