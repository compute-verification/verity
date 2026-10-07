import Proofs.Flock.Soundness.Discharge.HiddenDec.Statements
import Proofs.Flock.Soundness.Discharge.HiddenDec.Assemble

/-!
# R11b's message half at the hidden-output headline: `DecodesOn` at the hidden tables

#862's `decodesOn_exec` (`Discharge/DecodeZL`) for the statement main's prover writes since #757
(`hiddenOutputs = true`, `Stmt.setupHidden`), at `flock_headline_exec_hidden`'s tables `tablesAtH`
(`Discharge/HiddenExec`).

* `Statements.lean`: the frozen statements (`TagsOkH`, `DrawOkH`, `opnAtH`, `DecodesOnExecH`).
* `Assemble.lean`: `decodesOn_execH`.
-/
