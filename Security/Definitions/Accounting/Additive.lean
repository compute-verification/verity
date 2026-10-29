import Definitions.Core.Accel.Machine

/-!
# A device's period on the machine model (trusted)

What compute additivity on the machine model 𝓜 (`Verity.Accel.Machine`) is stated over
(`Accounting.Guarantees.AccelRate`, `AccelAdditive`):

* `AccelRun m`: one run of 𝓜, a program, the state it starts from and a schedule. A run receives no data once it
  starts, so a device's period is a list of runs back to back: each datum it receives (a verifier's coins, a request)
  starts the next run from the state it is written into (`Verity.Accel.Machine`, "The schedule costs the adversary
  nothing").
* `stream es`: the instructions the runs `es` issue, per tick of the period.
* `issued iss w`: the instructions issued during the window `w`, each an occurrence `(t, k)`, the `k`th instruction of
  tick `t`; `occUse iss r p`, the units of resource `r` occurrence `p` uses. A subprotocol's instructions are a set of
  occurrences, so subprotocols that share no instruction are disjoint sets.
-/

open scoped NNRat

namespace Accounting

open Finset Verity.Accel

variable {ι : Type}

/-- **A run of 𝓜**: a program over threads `ℕ`, the state it starts from, and its schedule. -/
structure AccelRun (m : Machine ι) where
  P : m.Prog ℕ
  σ : m.State ℕ
  sched : List (List ℕ)

variable {m : Machine ι}

/-- The instructions the run issues, per tick. -/
def AccelRun.instrs (e : AccelRun m) : List (List m.Instr) := (Machine.run e.P e.σ e.sched).2

/-- **The period's instructions** per tick: the runs `es` back to back. -/
def stream (es : List (AccelRun m)) : List (List m.Instr) := (es.map AccelRun.instrs).flatten

/-- **The instructions issued during `w`**: the occurrences `(t, k)`, the `k`th instruction of tick `t`. -/
def issued (iss : List (List m.Instr)) (w : Window) : Finset (ℕ × ℕ) :=
  w.ticks.biUnion fun t => (range (iss.getD t []).length).image (Prod.mk t)

/-- The units of resource `r` occurrence `p` uses (none for one past its tick's instructions). -/
def occUse (iss : List (List m.Instr)) (r : m.Res) (p : ℕ × ℕ) : ℚ≥0 :=
  (((iss.getD p.1 [])[p.2]?).map fun i => i.use r).getD 0

end Accounting
