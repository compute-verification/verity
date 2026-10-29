import Definitions.Core.Guarantees.DeviceFigures
import Proofs.Core.DeviceFigures

/-!
# A registered device's capacity is within its estimate's tolerance

`CapacityWithinEstimate`: the device's figures are at most the bound's (`DeviceWithinRegisteredFigures`), and the
bound's are themselves accepted against the estimate (`DeviceFigures.Accepted.bounding`), so within its tolerance from
above.
-/

namespace Verity.SecurityProofs

open Verity

theorem CapacityWithinEstimate : Guarantees.CapacityWithinEstimate := by
  intro est d actual hd hw
  have hb := hd.bounding
  have ha := hw hd
  exact ⟨ha.1.trans hb.2.1.2, fun c => (ha.2 c).trans (hb.2.2.1 c).2⟩

end Verity.SecurityProofs
