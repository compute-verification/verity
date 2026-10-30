import Proofs.Flock.Soundness.Discharge.HiddenExec.Defs
import Proofs.Flock.Soundness.Discharge.HiddenExec.Tables
import Proofs.Flock.Soundness.Discharge.HiddenExec.Statements
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
import Proofs.Flock.Soundness.Discharge.HiddenExec.Flat
import Proofs.Flock.Soundness.Discharge.HiddenExec.Copies
import Proofs.Flock.Soundness.Discharge.HiddenExec.CopiesP
import Proofs.Flock.Soundness.Discharge.HiddenExec.Placed
import Proofs.Flock.Soundness.Discharge.HiddenExec.Sites
import Proofs.Flock.Soundness.Discharge.HiddenExec.Ret
import Proofs.Flock.Soundness.Discharge.HiddenExec.SiteH
import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyH
import Proofs.Flock.Soundness.Discharge.HiddenExec.Inst
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.HiddenExec.AcceptsP
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepted
import Proofs.Flock.Soundness.Discharge.HiddenExec.Headline
import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineExec
import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineInst

/-!
# flock-verify's acceptance under hidden outputs, and the headline over hidden-output sessions

Main's prover writes the hidden-output statement (`verity/flock-circuit`, `Tags.hiddenOutputs = true`), which
`Stmt.setupTables` sets up with `Stmt.setupHidden`. e2e-exec's acceptance lemmas (`Discharge/Exec`) and the plain
headline (`Discharge/Integrate`) are about `Stmt.setupH`'s statements (`hh : I.tags.hiddenOutputs = false`). This area:

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
* `CopiesP`: `Copies` at public inputs: kept rows read the flat twin's, a public leaf's bits its instance's public word.
* `Ret`, `SiteH`: where the circuit computes each output word (the unit copy and port `HmOut.outCol` reads), and the
  site of a drawn unit (`BcSiteH`, every row's registered `b ‖ c` through `Hidden.bc_drawnW`).
* `KeyH`: `lay` and `rs` at `keyProg`'s places (`keyLayoutH`, `keyRowsH`, `keyRowsH_computes`): every row a commit
  string, a unit's computed committed gates bits of its output rows.
* `Inst`: the same at `instProg`'s places (`instLayoutH`, `instRowsH`, `instRowsH_computes`), from `InstKeysRow`.
* `HeadlineExec`: `flock_headline_exec_hidden`, `flock_headline_exec` for hidden-output sessions, `hdec` open.
* `HeadlineInst`: `flock_headline_exec_hidden_inst`, the same at `instProg`, with no condition on the unit's input
  groups or on where the outputs are returned.
-/
