import Definitions.Core.DeviceFigures

/-!
# A device performs within its accepted declaration

The premise that carries a device's registered figures (`Definitions/Core/DeviceFigures`) to the device itself
(`DeviceWithinDeclaration`). Registration checks a declaration against the auditor's estimate
(`DeviceFigures.Accepted`); nothing checks the device against either, and the device and its vendor are untrusted, so
what the device can do is assumed, not attested.
-/

namespace Verity.Assumptions

open Verity

/-- **`device-within-declaration(device)`**: a device whose declaration `d` registration accepted against the
auditor's estimate `est` can do no more than the figures the bound reads (`DeviceFigures.bounding est d`, the larger
capacity of the declaration and the estimate, figure by figure). `actual` is what the device can do whatever firmware,
driver and clocks the prover picks: its highest tensor clock, and for each operand class its peak rate of tensor-core
multiply-adds. Nothing the device reports about itself enters: the figures come from the estimate's source, the data
sheet now and a physical calibration later, and the premise is that the hardware stays inside them. -/
def DeviceWithinDeclaration (est : DeviceEstimate) (d actual : DeviceFigures) : Prop :=
  d.Accepted est → actual.CapacityLE (d.bounding est)

end Verity.Assumptions
