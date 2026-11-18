module

public import Definitions.Core.Challenger
public import Definitions.Core.Site
public import Mathlib.Basic.Real.Basic

/-!
# The clock's premise

`ClockWithin S k l real`, the one named assumption about time: the clock of the auditor's device registered under key
`k` runs at the true rate within the drift the Site registers for it (`Site.Loc.drift`, on the auditor's host the
device is in). Between two of that device's readings in a challenger's log (`Challenger.Reading.elapsed`), its clock
counts within that share of the real time between them, `real` giving the real time in nanoseconds at which each entry
of the log was written. It is two-sided, and each consumer leans on one side:

* `NotSlow`: the clock counts no less than `1 - q` of real time. A deadline and a salt's age lean on it: what reads as
  in time on the device was in time.
* `NotFast`: the clock counts no more than `1 + q` of real time. Buckets a week, a capacity share and a monitoring
  period's length lean on it: ticks a registered period apart on the device are at least `period / (1 + q)` apart in
  real time.

A device at the site, or in a host that registers no drift, times nothing: the premise says nothing of its readings.
It is a premise about the auditor's device on a host the prover doesn't control (`trusted-host`), which nothing
attests.
-/

@[expose] public section

namespace Verity.Assumptions

open Verity.Challenger

/-- **Not slow**: over `d` nanoseconds of real time, a clock within drift `q` counts `e`, at least `(1 - q) · d`. -/
def NotSlow (q : ℚ≥0) (d e : ℝ) : Prop := (1 - (q : ℝ)) * d ≤ e

/-- **Not fast**: over `d` nanoseconds of real time, a clock within drift `q` counts `e`, at most `(1 + q) · d`. -/
def NotFast (q : ℚ≥0) (d e : ℝ) : Prop := e ≤ (1 + (q : ℝ)) * d

/-- **`clock-within-drift`**: the auditor's device registered under key `k` in the Site `S`, in a host that registers
the drift `q`, has a clock that runs at the true rate within `q`: for its readings `a` at entry `i` and `b` at entry
`j ≥ i` of the log `l`, written at the real times `real i` and `real j` in nanoseconds, `a.elapsed b` is not slow and
not fast against `real j - real i`. It is trust in the auditor's device and its host (`trusted-host`). -/
def ClockWithin {R C : Type} (S : Site) (k : String) (l : Log R C) (real : ℕ → ℝ) : Prop :=
  ∀ x ∈ S.locs, x.kind.auditor = true → x.key = k → ∀ p ∈ x.parent, ∀ h ∈ S.loc? p, ∀ q ∈ h.drift,
    ∀ (i j : ℕ) (a b : Reading k), i ≤ j → l[i]? = some (.read k a) → l[j]? = some (.read k b) →
      NotSlow q (real j - real i) (a.elapsed b) ∧ NotFast q (real j - real i) (a.elapsed b)

end Verity.Assumptions
