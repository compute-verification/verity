import Definitions.Core.Accel.Machine

/-!
# A device's period on the machine model (trusted)

What compute additivity on the machine model 𝓜 (`Verity.Accel.Machine`) is stated over
(`Accounting.Guarantees.AccelRate`, `AccelAdditive`, `GatesAdditive`):

* `AccelRun m`: one run of 𝓜, a program, the state it starts from and a schedule. A run receives no data once it
  starts, so a device's period is a list of runs back to back: each datum it receives (a verifier's coins, a request)
  starts the next run from the state it is written into (`Verity.Accel.Machine`, "The schedule costs the adversary
  nothing").
* `stream es`: the instructions the runs `es` issue, per tick of the period.
* `issued iss w`: the instructions issued during the window `w`, each an occurrence `(t, k)`, the `k`th instruction of
  tick `t`; `occUse iss r p`, the units of resource `r` occurrence `p` uses. A subprotocol's instructions are a set of
  occurrences, so subprotocols that share no instruction are disjoint sets.
* `PeriodRate iss b`: the rate `AccelAdditive` takes, named, for a period's instructions `iss` at a backlog of `b`
  ticks. 𝓜 states its rate (G2) run by run, so a period of `n` runs meets it at `b = n · slack` (`AccelRate`). At
  `b = slack` it is `Machine.Rate`: the period meets 𝓜's rate as one run does.
* `supply m R κ`: `Φ = Σ_{r ∈ R} κ_r ρ_r`, the work a tick the resources `R` pay for at `κ_r` work a unit
  (`Accounting.Assumptions.PaidBy`). The work is a pricing's, so `Φ` is gates a tick at a pricing that counts gates.
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

/-- **𝓜's rate over a period at a backlog of `b` ticks**: over every window, the instructions `iss` use at most `ρ_r`
units of each resource `r` a tick, plus `b` ticks' worth. -/
def PeriodRate (iss : List (List m.Instr)) (b : ℕ) : Prop :=
  ∀ (w : Window) (r : m.Res), ∑ t ∈ w.ticks, Machine.usage iss t r ≤ m.ρ r * ((w.len + b : ℕ) : ℚ≥0)

/-- **The supply** `Φ = Σ_{r ∈ R} κ_r ρ_r`: the work a tick the resources `R` pay for at `κ_r` work a unit. -/
def supply (m : Machine ι) (R : Finset m.Res) (κ : m.Res → ℚ≥0) : ℚ≥0 := ∑ r ∈ R, κ r * m.ρ r

end Accounting
