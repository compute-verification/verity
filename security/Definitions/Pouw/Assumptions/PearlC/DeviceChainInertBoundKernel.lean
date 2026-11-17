module

public import Definitions.Pouw.Assumptions.PearlC.DeviceChainInertBound
public import Definitions.Pouw.PearlC.DeviceKernelWref

/-! The chain-share cap at a device record with `W_ref` at the honest kernel's `wrefDevK d c`. -/

@[expose] public section

namespace Pouw.PearlC

variable {Q R S : Type}

noncomputable def pearlCProtocolDevChainInertBoundK (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) : Protocol Q R S :=
  { pearlCProtocolDevChainInertBound d sem ρ with Wref := fun L u => (wrefDevK d c (L.shape u) : ℝ) }

noncomputable def pearlCTilesDevChainInertBoundK (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) : TileRules Q R S :=
  { pearlCTilesDevChainInertBound d sem ρ with
    Wref := fun L g =>
      let tl := auditTiling 64 64 L
      let sh := L.shape (tl.unit g)
      ((wrefDevK d c sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ)
    Wcred := fun L g act =>
      let tl := auditTiling 64 64 L
      let sh := L.shape (tl.unit g)
      ((wrefDevK d c sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ) }

end Pouw.PearlC
