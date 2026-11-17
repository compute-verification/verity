module

public import Definitions.Core.DeviceFigures

/-! The premise that a device performs within its registered figures (`DeviceWithinRegisteredFigures`), which nothing
checks. -/

@[expose] public section

namespace Verity.Assumptions

open Verity

/-- **`device-within-registered-figures(device)`**: a device whose figures `d` registration accepted against the
auditor's estimate `est` can do no more than the figures the bound reads (`DeviceFigures.bounding est d`, the larger of
the registered figures and the estimate's, clock and peak rate alike). `actual` is what the device can do whatever
firmware, driver and clocks the prover picks: its highest tensor clock, and for each operand class its peak rate of
tensor-core MACs. `Accepted` requires every registered peak rate to be positive, so no registration with a zero rate
satisfies the premise vacuously. Nothing the device reports about itself enters: the figures come from the estimate's
source, the data sheet now and a physical calibration later, and the premise is that the hardware stays inside them. -/
def DeviceWithinRegisteredFigures (est : DeviceEstimate) (d actual : DeviceFigures) : Prop :=
  d.Accepted est → actual.CapacityLE (d.bounding est)

end Verity.Assumptions
