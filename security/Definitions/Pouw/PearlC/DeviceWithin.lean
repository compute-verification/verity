module

public import Definitions.Pouw.PearlC.DeviceKernelWref

/-! What a statement over every device record asks of the record: `KernelWithin`, its one hypothesis about it. -/

@[expose] public section

namespace Pouw.PearlC

/-- Pearl-C's shape domain at a device record, restricted to the shapes `Ss` holds of. -/
def pearlCDomainDevOn (d : PearlCDevice) (Ss : Shape → Prop) (L : Layout) : Prop :=
  pearlCDomainDev d L ∧ ∀ u < L.N, Ss (L.shape u)

/-- **What a gap statement over every device record asks of the record**: `ω` is positive and, at every shape `Ss`
holds of, the honest kernel's `W_ref` under rev1 with `c` per activation element (`wrefDevRev1K d c`) is non-negative
and at most `ω` times the credit left after the inert-work share `ρ`, `(1 − ρ)·creditDevRev1 d`. -/
def KernelWithin (d : PearlCDevice) (ρ c : ℚ) (Ss : Shape → Prop) (ω : ℚ) : Prop :=
  0 < ω ∧ ∀ s, Ss s → 0 ≤ wrefDevRev1K d c s ∧ wrefDevRev1K d c s ≤ ω * ((1 - ρ) * creditDevRev1 d s)

end Pouw.PearlC
