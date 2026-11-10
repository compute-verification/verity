module

public import Proofs.Flock.Soundness.Discharge.HiddenDec.Statements
public import Proofs.Flock.Soundness.Discharge.HiddenDec.Assemble

/-!
# The decoder premise's message half at the hidden-output headline: `DecodesOn` at the hidden tables

`decodesOn_exec` (`Discharge/DecodeZL`) for the statement main's prover writes
(`hiddenOutputs = true`, `Stmt.setupHidden`), at `flock_headline_exec_hidden`'s tables `tablesAtH`
(`Discharge/HiddenExec`).

* `Statements.lean`: the frozen statements (`TagsOkH`, `DrawOkH`, `opnAtH`, `DecodesOnExecH`).
* `Assemble.lean`: `decodesOn_execH`.
-/

@[expose] public section
