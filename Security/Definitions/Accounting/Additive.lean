import Definitions.Core.Accel.Machine

/-!
# A device's period on the machine model (trusted)

What compute additivity on the machine model 𝓜 (`Verity.Accel.Machine`) is stated over
(`Accounting.Guarantees.AccelRate`, `AccelAdditive`):

* `AccelRun m`: one run of 𝓜, a program, the state it starts from, a schedule and its seconds `T` (the run meets the
  model when it `Fits` at `T`). A run receives no data once it starts, so a device's period is a list of runs back to
  back: each datum it receives (a verifier's coins, a request) starts the next run from the state it is written into.
* `stream es`: the instructions the runs `es` issue, in order; `periodSeconds es`, their seconds summed.
* `issued iss`: the instructions of `iss`, each an occurrence `k`, the `k`th instruction; `occUse iss r k`, the units
  of resource `r` occurrence `k` uses. A subprotocol's instructions are a set of occurrences, so subprotocols that share
  no instruction are disjoint sets.

𝓜 has no time inside a run, only the run's seconds, so a window it measures is a block of a period's runs back to back,
which is a period itself.
-/

open scoped NNRat

namespace Accounting

open Finset Verity.Accel

/-- **A run of 𝓜**: a program over threads `ℕ`, the state it starts from, its schedule, and its seconds. -/
structure AccelRun (m : Machine) where
  P : m.Prog ℕ
  σ : m.State ℕ
  sched : List ℕ
  T : ℚ≥0

variable {m : Machine}

/-- The instructions the run issues, in order. -/
def AccelRun.instrs (e : AccelRun m) : List m.Instr := (Machine.run e.P e.σ e.sched).2

/-- **The period's instructions**: the runs `es` back to back. -/
def stream (es : List (AccelRun m)) : List m.Instr := (es.map AccelRun.instrs).flatten

/-- **The period's seconds**: its runs' seconds summed. -/
def periodSeconds (es : List (AccelRun m)) : ℚ≥0 := (es.map AccelRun.T).sum

/-- **The instructions issued**: the occurrences `k`, the `k`th instruction of `iss`. -/
def issued (iss : List m.Instr) : Finset ℕ := range iss.length

/-- The units of resource `r` occurrence `k` uses (none for one past the end). -/
def occUse (iss : List m.Instr) (r : Res) (k : ℕ) : ℚ≥0 := ((iss[k]?).map fun i => i.use r).getD 0

end Accounting
