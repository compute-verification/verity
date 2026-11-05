module

public import Proofs.Flock.Soundness.Discharge.HiddenDec.Statements
public import Proofs.Flock.Soundness.Discharge.HiddenDec.Assemble
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

/-!
# R11b's message half at the hidden-output headline: `DecodesOn` at the hidden tables

`decodesOn_exec` (`Discharge/DecodeZL`) for the statement main's prover writes
(`hiddenOutputs = true`, `Stmt.setupHidden`), at `flock_headline_exec_hidden`'s tables `tablesAtH`
(`Discharge/HiddenExec`).

* `Statements.lean`: the frozen statements (`TagsOkH`, `DrawOkH`, `opnAtH`, `DecodesOnExecH`).
* `Assemble.lean`: `decodesOn_execH`.
-/

@[expose] public section
