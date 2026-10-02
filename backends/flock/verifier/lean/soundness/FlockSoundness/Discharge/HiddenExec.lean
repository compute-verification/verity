import FlockSoundness.Discharge.HiddenExec.Defs
import FlockSoundness.Discharge.HiddenExec.Tables
import FlockSoundness.Discharge.HiddenExec.Statements
import FlockSoundness.Discharge.HiddenExec.Setup
import FlockSoundness.Discharge.HiddenExec.Flat
import FlockSoundness.Discharge.HiddenExec.Accepts
import FlockSoundness.Discharge.HiddenExec.Accepted
import FlockSoundness.Discharge.HiddenExec.Headline

/-!
# flock-verify's acceptance under hidden outputs, and the headline over hidden-output sessions

Since #757 main's prover writes the hidden-output statement (`verity/flock-circuit`, `Tags.hiddenOutputs = true`), which
`Stmt.setupTables` sets up with `Stmt.setupHidden`. e2e-exec's acceptance lemmas (`Discharge/Exec`) and #828's headline
(`Discharge/Integrate`) are about `Stmt.setupH`'s statements (`hh : I.tags.hiddenOutputs = false`). This area:

* `Defs`: `setupOne`, the setup `setupTables` calls for a tag.
* `Statements`, `Setup`, `Accepts`: e2e-exec's acceptance lemmas without `hh` (`SetupTablesOkAny`, `AcceptsSetupOne`,
  `AcceptsOneAny`, `AcceptsRetainedAny`, `AcceptsAllRetainedAny`, `AcceptsRepRoundsAny`, `AcceptsTablesHm96Any`,
  `AcceptsTablesHm96OfLeafAny`, `AcceptsVerdictHm96Any`), and what `setupHidden` returns (`SetupHiddenWF`,
  `SetupHiddenPublic`, `SetupHiddenParse`, `SetupHiddenDrawn`).
* `Tables`, `Accepted`: the one-table plan of a hidden-output session (`tableAtH`, `planAtH`, `tablesAtH`), and
  `tableAtH_of_accepts` (`TableAtHOfAccepts`).
* `Headline`: `flock_headline_hidden`, `Prog.flock_headline_live_on` at the hidden tables.
-/
