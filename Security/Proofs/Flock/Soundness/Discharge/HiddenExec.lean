import Proofs.Flock.Soundness.Discharge.HiddenExec.Defs
import Proofs.Flock.Soundness.Discharge.HiddenExec.Tables
import Proofs.Flock.Soundness.Discharge.HiddenExec.Statements
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
import Proofs.Flock.Soundness.Discharge.HiddenExec.Flat
import Proofs.Flock.Soundness.Discharge.HiddenExec.Copies
import Proofs.Flock.Soundness.Discharge.HiddenExec.Placed
import Proofs.Flock.Soundness.Discharge.HiddenExec.Sites
import Proofs.Flock.Soundness.Discharge.HiddenExec.Ret
import Proofs.Flock.Soundness.Discharge.HiddenExec.SiteH
import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyH
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepted
import Proofs.Flock.Soundness.Discharge.HiddenExec.Headline
import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineExec

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
* `Flat`, `Copies`, `Placed`, `Sites`: the copy, constant and zero facts at `keyProg`'s places in the hidden tables
  (`ppH`, `hConstH`, `hZeroH`, `sitesAtH`).
* `Ret`, `SiteH`: where the circuit computes each output word (the unit copy and port `HmOut.outCol` reads), and the
  site of a drawn unit (`BcSiteH`, every row's registered `b ‖ c` through `Hidden.bc_drawnW`).
* `KeyH`: `lay` and `rs` at `keyProg`'s places (`keyLayoutH`, `keyRowsH`, `keyRowsH_computes`): every row a commit
  string, a unit's computed committed gates bits of its output rows.
* `HeadlineExec`: `flock_headline_exec_hidden`, #828's `flock_headline_exec` for hidden-output sessions, `hdec` open.
-/
