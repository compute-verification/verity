module

public import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepted
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
public import Proofs.Flock.Soundness.Discharge.HiddenExec.AcceptsP
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Copies
public import Proofs.Flock.Soundness.Discharge.HiddenExec.CopiesP
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Defs
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Flat
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Headline
public import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineExec
public import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineExecP
public import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineInst
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Inst
public import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyH
public import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyHP
public import Proofs.Flock.Soundness.Discharge.HiddenExec.LiveOnP
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Placed
public import Proofs.Flock.Soundness.Discharge.HiddenExec.PlacedP
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Ret
public import Proofs.Flock.Soundness.Discharge.HiddenExec.RowsAtHP
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
public import Proofs.Flock.Soundness.Discharge.HiddenExec.SetupP
public import Proofs.Flock.Soundness.Discharge.HiddenExec.SiteH
public import Proofs.Flock.Soundness.Discharge.HiddenExec.SiteHP
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Sites
public import Proofs.Flock.Soundness.Discharge.HiddenExec.SitesP
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Statements
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Tables
public import Proofs.Flock.Soundness.Discharge.HiddenExec.TablesP
public import Proofs.Flock.Soundness.Discharge.ZkExec.Event
public import Proofs.Flock.Soundness.Discharge.ZkHidden.ParseStmt

/-!
# flock-verify's acceptance under hidden outputs, and the headline over hidden-output sessions

Main's prover writes the hidden-output statement (`verity/flock-circuit`, `Tags.hiddenOutputs = true`), which
`Stmt.setupTables` sets up with `Stmt.setupHidden`. `Discharge/Exec`'s acceptance lemmas and the plain
headline (`Discharge/Integrate`) are about `Stmt.setupH`'s statements (`hh : I.tags.hiddenOutputs = false`). This area:

* `Defs`: `setupOne`, the setup `setupTables` calls for a tag.
* `Statements`, `Setup`, `Accepts`: `Discharge/Exec`'s acceptance lemmas without `hh` (`SetupTablesOkAny`, `AcceptsSetupOne`,
  `AcceptsOneAny`, `AcceptsRetainedAny`, `AcceptsAllRetainedAny`, `AcceptsRepRoundsAny`, `AcceptsTablesHm96Any`,
  `AcceptsTablesHm96OfLeafAny`, `AcceptsVerdictHm96Any`), and what `setupHidden` returns (`SetupHiddenWF`,
  `SetupHiddenPublic`, `SetupHiddenParse`, `SetupHiddenDrawn`).
* `Tables`, `Accepted`: the one-table plan of a hidden-output session (`tableAtH`, `planAtH`, `tablesAtH`), and
  `tableAtH_of_accepts` (`TableAtHOfAccepts`).
* `Headline`: `flock_headline_hidden`, `Prog.flock_headline_live_on` at the hidden tables.
* `Flat`, `Copies`, `Placed`, `Sites`: the copy, constant and zero facts at `keyProg`'s places in the hidden tables
  (`ppH`, `hConstH`, `hZeroH`, `sitesAtH`).
* `CopiesP`: `Copies` at public inputs: kept rows read the flat twin's, a public leaf's bits its instance's public word.
* `PlacedP`: `Placed` at public inputs: `AcceptedHP`, `SitesHP` and `ProgPlaces` for `Placed.keyProgP`.
* `Ret`, `SiteH`: where the circuit computes each output word (the unit copy and port `HmOut.outCol` reads), and the
  site of a sampled proof block (`BcSiteH`, every row's registered `b ‖ c` through `Hidden.bc_drawnW`).
* `SiteHP`: `SiteH` at public inputs, through the flat twin off the unit input rows (`Hidden.Any.sat_flatFP`): copies
  off a public leaf, returned outputs and the rows' `b ‖ c`.
* `KeyH`: `lay` and `rs` at `keyProg`'s places (`keyLayoutH`, `keyRowsH`, `keyRowsH_computes`): every row a commit
  string, a unit's computed committed gates bits of its output rows.
* `KeyHP`: `KeyH` at `Placed.keyProgP`'s places at public inputs (`keyLayoutHP`, `keyRowsHP`, `keyRowsHP_computes`): a
  unit's public gates hold its public input words' bits in the population's public file (`pub_siteHP`).
* `Inst`: the same at `instProg`'s places (`instLayoutH`, `instRowsH`, `instRowsH_computes`), from `InstKeysRow`.
* `HeadlineExec`: `flock_headline_exec_hidden`, `flock_headline_exec` for hidden-output sessions, `hdec` open.
* `HeadlineInst`: `flock_headline_exec_hidden_inst`, the same at `instProg`, with no condition on the unit's input
  groups or on where the outputs are returned.
* `TablesP`, `SitesP`: the one-table plan at the verifier's public inputs `own` (`tableAtHP`, `planAtHP`, `tablesAtHP`)
  and its sites (`sitesAtHP`, `ppHP`, `hConstHP`, `hZeroHP`) for `Placed.keyProgP`.
* `RowsAtHP`: `lay` and `rs` there (`layAtHP`, `rsAtHP`, `rsAtHP_computes`), `BcSiteHP` at every satisfiable site.
* `LiveOnP`: the execution refinement for one-table sessions at `own` (`AcceptsP`, `DecodesOnP`,
  `Prog.flock_headline_live_onXP`).
* `HeadlineExecP`: `flock_headline_exec_hiddenP`, `flock_headline_exec_hidden` at `own` and `Placed.keyProgP`.
* `SetupP`: what a setup at `own` establishes (its tags, leaf scheme, well-formedness and sampled public file, the
  session's rounds and link points), and `drawSetupHP_of_setupHiddenI`.
-/

@[expose] public section
