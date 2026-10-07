import Proofs.Flock.Soundness.Discharge.DecodeZLJ.Statements
import Proofs.Flock.Soundness.Discharge.DecodeZLJ.Reps
import Proofs.Flock.Soundness.Discharge.DecodeZLJ.Assemble

/-!
# R11b at `J` tables: the one-table decoder on each table's streams

* `Statements.lean`: the frozen statement `DecodesOnJExec` (`DecodesOnJ` at the tables flock-verify sets up at
  `count > 1`), and the plan, decoder and openings it reads.
* `Reps.lean`: `accepts_reps_at`, flock-verify's acceptance at table `j`, rep by rep; `planJ_get`, `tableJ_of`.
* `Assemble.lean`: `decodesOnJ_exec`.
-/
