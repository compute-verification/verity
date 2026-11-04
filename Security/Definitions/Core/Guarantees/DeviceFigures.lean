import Definitions.Core.Assumptions.DeviceFigures

/-!
# What a registered device can do, against the auditor's estimate

`CapacityWithinEstimate`: under `DeviceWithinRegisteredFigures`, a device registration accepted can do at most the
tolerance more than the auditor's estimate, clock and peak rate alike. So the capacity an accounting bound charges a
registered device is the public estimate's up to its public tolerance, whatever figures the prover registered.
-/

namespace Verity.Guarantees

open Verity Verity.Assumptions

/-- **A registered device's capacity is within its estimate's tolerance.** For every estimate `est`, registered figures
`d` and device figures `actual`: if registration accepts `d` against `est` and the device performs within its
registered figures (`DeviceWithinRegisteredFigures`), its tensor clock is at most `(1 + τ)` times the estimate's and so
is its peak rate for each operand class, at the estimate's tolerance `τ`. -/
def CapacityWithinEstimate : Prop :=
  ∀ (est : DeviceEstimate) (d actual : DeviceFigures), d.Accepted est → DeviceWithinRegisteredFigures est d actual →
    actual.clock ≤ (1 + est.tolerance.capacity) * est.figures.clock ∧
      ∀ c, actual.peakRate c ≤ (1 + est.tolerance.capacity) * est.figures.peakRate c

end Verity.Guarantees
