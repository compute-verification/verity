module

public import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeed
public import Definitions.Pouw.PearlC.DeviceKernelWref

/-! (C2)'s per-row cap protocol and tiles with `W_ref` at the honest kernel's `wrefDevRev1K d c`, with nothing assumed. -/

@[expose] public section

namespace Pouw.PearlC

variable {Q R S : Type}

noncomputable def pearlCProtocolDevRev1KRowInertBound (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) :
    Protocol Q R S :=
  { pearlCProtocolDevRev1RowInertBound d sem ρ with Wref := fun L u => (wrefDevRev1K d c (L.shape u) : ℝ) }

noncomputable def pearlCTilesDevRev1KRowInertBound (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) :
    TileRules Q R S :=
  { pearlCTilesDevRev1RowInertBound d sem ρ with
    Wref := fun L g =>
      let tl := auditTiling 64 64 L
      let sh := L.shape (tl.unit g)
      ((wrefDevRev1K d c sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ)
    Wcred := fun L g act =>
      let tl := auditTiling 64 64 L
      let sh := L.shape (tl.unit g)
      ((wrefDevRev1K d c sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ) }

noncomputable def pearlCProtocolDevRev1KRowInertBoundAt (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr c : ℚ) :
    Protocol Q R S :=
  { pearlCProtocolDevRev1RowInertBoundAt d sem ρ ρr with Wref := fun L u => (wrefDevRev1K d c (L.shape u) : ℝ) }

noncomputable def pearlCTilesDevRev1KRowInertBoundAt (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr c : ℚ) :
    TileRules Q R S :=
  { pearlCTilesDevRev1RowInertBoundAt d sem ρ ρr with
    Wref := fun L g =>
      let tl := auditTiling 64 64 L
      let sh := L.shape (tl.unit g)
      ((wrefDevRev1K d c sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ)
    Wcred := fun L g act =>
      let tl := auditTiling 64 64 L
      let sh := L.shape (tl.unit g)
      ((wrefDevRev1K d c sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ) }

end Pouw.PearlC
