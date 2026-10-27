import Definitions.Core.Device

/-!
# The device model, as resource accounting reads it

The device model is core's (`Definitions/Core/Device.lean`, `Verity.Device`). Resource accounting reads it under its
own namespace through these exports, until its readers name core's directly.
-/

namespace Accounting

export Verity (OpClass Device Device.priced Step Step.Fits Schedule Schedule.Fits Schedule.ops Schedule.work
  Schedule.read Schedule.side Schedule.held Schedule.chipHeld Window Window.ticks Price)

end Accounting
