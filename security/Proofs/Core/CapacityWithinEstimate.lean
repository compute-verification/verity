module

public import Definitions.Core.Guarantees.DeviceFigures
public import Proofs.Core.DeviceFigures

/-! A registered device's capacity is within its estimate's tolerance. -/

@[expose] public section

namespace Verity.SecurityProofs

open Verity

theorem CapacityWithinEstimate : Guarantees.CapacityWithinEstimate := by
  intro est d actual hd hw
  have hb := hd.bounding
  have ha := hw hd
  exact ⟨ha.1.trans hb.2.1.2, fun c => (ha.2 c).trans (hb.2.2.1 c).2⟩

end Verity.SecurityProofs
