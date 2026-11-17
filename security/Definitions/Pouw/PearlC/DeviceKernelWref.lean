module

public import Definitions.Pouw.PearlC.DeviceRev1

/-! Pearl-C at a device record with the honest kernel's `W_ref`, `c` more per activation element than the record prices, staged. -/

@[expose] public section

namespace Pouw.PearlC

variable {Q R S : Type}

def wrefDevK (d : PearlCDevice) (c : ℚ) (s : Shape) : ℚ := wrefDev d s + c * s.m * s.k

/-- **The honest kernel's `W_ref` under rev1**: `wrefDevRev1` plus `c` per activation element. -/
def wrefDevRev1K (d : PearlCDevice) (c : ℚ) (s : Shape) : ℚ := wrefDevRev1 d s + c * s.m * s.k

noncomputable def pearlCProtocolDevK (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) : Protocol Q R S :=
  { pearlCProtocolDev d sem ρ with Wref := fun L u => (wrefDevK d c (L.shape u) : ℝ) }

noncomputable def pearlCTilesDevK (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) : TileRules Q R S :=
  { pearlCTilesDev d sem ρ with
    Wref := fun L g =>
      let tl := auditTiling 64 64 L
      let sh := L.shape (tl.unit g)
      ((wrefDevK d c sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ)
    Wcred := fun L g act =>
      let tl := auditTiling 64 64 L
      let sh := L.shape (tl.unit g)
      ((wrefDevK d c sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ) }

/-- `pearlCProtocolDevRev1` with `W_ref` at `wrefDevRev1K d c`. -/
noncomputable def pearlCProtocolDevRev1K (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) : Protocol Q R S :=
  { pearlCProtocolDevRev1 d sem ρ with Wref := fun L u => (wrefDevRev1K d c (L.shape u) : ℝ) }

/-- `pearlCTilesDevRev1` with `W_ref` and the credited rows' `W_ref` at `wrefDevRev1K d c`. -/
noncomputable def pearlCTilesDevRev1K (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) : TileRules Q R S :=
  { pearlCTilesDevRev1 d sem ρ with
    Wref := fun L g =>
      let tl := auditTiling 64 64 L
      let sh := L.shape (tl.unit g)
      ((wrefDevRev1K d c sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ)
    Wcred := fun L g act =>
      let tl := auditTiling 64 64 L
      let sh := L.shape (tl.unit g)
      ((wrefDevRev1K d c sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ) }

end Pouw.PearlC
