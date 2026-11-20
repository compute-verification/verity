module

public import Definitions.Core.Challenger
public import Definitions.Core.Site
public import Mathlib.Basic.Real.Basic

/-!
# The clock's premise

`ClockWithin S k real`, an assumption about one of the auditor's devices, beside `trusted-device`, and not a part of the
model: the clock of the device registered under key `k` runs at the true rate within the drift the Site registers for
it (`Site.Loc.drift`, on the auditor's host the device is in). For two of that device's entries
(`Challenger.Attested k`) that carry readings, `real` giving the real time in nanoseconds at which it made each, the
readings' difference (`Challenger.Reading.elapsed`) is within that share of the real time between them. It is
two-sided, and each consumer leans on one side:

* `NotSlow`: the clock counts no less than `1 - q` of real time. A deadline and a salt's age lean on it: what reads as
  in time on the device was in time.
* `NotFast`: the clock counts no more than `1 + q` of real time. Buckets a week, a capacity share and a monitoring
  period's length lean on it: ticks a registered period apart on the device are at least `period / (1 + q)` apart in
  real time. Ticks no faster than the period, alone, is this side, the weaker form.

A device at the site, or in a host that registers no drift, times nothing: the premise says nothing of its readings.
A registered drift of 1 or more makes `NotSlow` say nothing, and a site with no device of key `k` or no drift on its
host makes the premise hold emptily, so a consumer's check refuses each by name. A consumer takes no order of real
times as its own hypothesis: it checks `0 ≤ ra.elapsed rb` and derives `real a ≤ real b` (`ClockWithin.real_le`,
`Proofs.Core.Challenger`). Useful work's
`honest-clock` and the bandwidth audit's `Paced` are restated over this premise, not kept beside it.
How late the device acts after it reads its clock is a term of a consumer's statement, not of this premise. It is a
premise about the auditor's device on a host the prover doesn't control (`trusted-device`), which nothing attests.
-/

@[expose] public section

namespace Verity.Assumptions

open Verity.Challenger

/-- **Not slow**: over `d` nanoseconds of real time, a clock within drift `q` counts `e`, at least `(1 - q) · d`. -/
def NotSlow (q : ℚ≥0) (d e : ℝ) : Prop := (1 - (q : ℝ)) * d ≤ e

/-- **Not fast**: over `d` nanoseconds of real time, a clock within drift `q` counts `e`, at most `(1 + q) · d`. -/
def NotFast (q : ℚ≥0) (d e : ℝ) : Prop := e ≤ (1 + (q : ℝ)) * d

/-- **`clock-within-drift`**: the auditor's device registered under key `k` in the Site `S`, in a host that registers
the drift `q`, has a clock that runs at the true rate within `q`: for two of its entries `a` and `b`, with readings
`ra` and `rb`, at the real times `real a ≤ real b` in nanoseconds, `ra.elapsed rb` is not slow and not fast against
`real b - real a`. It is trust in the auditor's device and its host (`trusted-device`). -/
def ClockWithin (S : Site) (k : String) (real : Attested k → ℝ) : Prop :=
  ∀ x ∈ S.locs, x.kind.auditor = true → x.key = k → ∀ p ∈ x.parent, ∀ h ∈ S.loc? p, ∀ q ∈ h.drift,
    ∀ (a b : Attested k), ∀ ra ∈ a.reading, ∀ rb ∈ b.reading, real a ≤ real b →
      NotSlow q (real b - real a) (ra.elapsed rb) ∧ NotFast q (real b - real a) (ra.elapsed rb)

end Verity.Assumptions
