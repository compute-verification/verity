module

public import Definitions.Core.Accel.Machine

/-! A device's period on the machine model 𝓜: runs back to back, since a run receives no data once it starts. -/

@[expose] public section

open scoped NNRat

namespace ResourceAccounting

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

end ResourceAccounting
