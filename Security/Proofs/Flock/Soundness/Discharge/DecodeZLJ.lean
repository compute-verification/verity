module

public import Proofs.Flock.Soundness.Discharge.DecodeZLJ.Statements
public import Proofs.Flock.Soundness.Discharge.DecodeZLJ.Reps
public import Proofs.Flock.Soundness.Discharge.DecodeZLJ.Assemble
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
# R11b at `J` tables: the one-table decoder on each table's streams

* `Statements.lean`: the frozen statement `DecodesOnJExec` (`DecodesOnJ` at the tables flock-verify sets up at
  `count > 1`), and the plan, decoder and openings it reads.
* `Reps.lean`: `accepts_reps_at`, flock-verify's acceptance at table `j`, rep by rep; `planJ_get`, `tableJ_of`.
* `Assemble.lean`: `decodesOnJ_exec`.
-/

@[expose] public section
