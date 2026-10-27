import Mathlib.Basic.Real.Basic

/-!
# The monitoring period (trusted)

The **monitoring period** is the accounting unit: the stretch of time a result is about. Accounting is per monitoring
period, and it pools the execution periods the monitoring period holds. Every result is presented for monitoring periods
of a week, a month and a year (`MonitoringPeriod.week`, `month`, `year`, in seconds).

The periods nest: a monitoring period holds execution periods, each the span one commitment covers; an execution period
holds sampling periods, each one sampling block's span; a sampling period holds ticks, each one step of a device's
clock.

* **A monitoring period** (`MonitoringPeriod E n`): its length, in seconds, and the `n` execution periods it pools, each
  an `E`, in order. A protocol picks `E`, what it reads of one execution period, and reads the monitoring period through
  its own projection of this one.
* **The execution periods a monitoring period pools** (`MonitoringPeriod.pools`): of a given length, the whole ones
  that fit in it.

A bound over a monitoring period has two readings. *Per sampling block*, every sampling block's bound holds at once, at
their errors summed. *Pooled*, the monitoring period's total holds at one error, because each sampling block draws its
own live coins.

**The execution periods are private.** A monitoring period's length and the number of execution periods it pools are
public; what each execution period runs is the prover's value, as a partition's units are.
-/

namespace Verity

universe u

/-- **A monitoring period**, the accounting unit: `len` seconds, pooling `n` execution periods, each an `E`, in order.
`len` and `n` are public; the execution periods are the prover's. -/
structure MonitoringPeriod (E : Type u) (n : ℕ) where
  /-- its length, in seconds -/
  len : ℝ
  /-- the execution periods it pools, in order -/
  period : Fin n → E

namespace MonitoringPeriod

/-- **A week**, in seconds: the shortest of the three monitoring periods every result is presented at. -/
def week : ℕ := 7 * 86400

/-- **A month** of 30 days, in seconds. -/
def month : ℕ := 30 * 86400

/-- **A year** of 365 days, in seconds. -/
def year : ℕ := 365 * 86400

/-- **The execution periods a monitoring period of `len` seconds pools**, each `s > 0` seconds long: the whole ones
that fit in it, `⌊len / s⌋`. -/
def pools (len s : ℕ) : ℕ := len / s

end MonitoringPeriod

end Verity
