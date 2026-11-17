module

public import Definitions.Pouw.PearlC.DeviceKernelWref

/-! The chain-only reading at a device record, staged: a unit's credit without its salt-dependent forming. -/

@[expose] public section

namespace Pouw.PearlC

variable {Q R S : Type}

def creditDevChainOnly (d : PearlCDevice) (s : Shape) : ℚ := creditDev d s - (d.prices.costs.fs : ℚ) * s.m * s.k

def creditDevRev1ChainOnly (d : PearlCDevice) (s : Shape) : ℚ :=
  creditDevRev1 d s - (d.prices.costs.fs : ℚ) * s.m * s.k

noncomputable def pearlCProtocolDevKChainOnly (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) :
    Protocol Q R S :=
  { pearlCProtocolDevK d sem ρ c with
    credit := fun H s U u act =>
      ((creditDevChainOnly d (U.layout.shape u) - inertWorkDev d (sem.unitInertWorkDev d H s U u act) : ℚ) : ℝ) }

noncomputable def pearlCTilesDevKChainOnly (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) : TileRules Q R S :=
  { pearlCTilesDevK d sem ρ c with
    credit := fun _ _ U g act =>
      let tl := auditTiling 64 64 U.layout
      let sh := U.layout.shape (tl.unit g)
      (((creditDevChainOnly d sh - ρ * creditDev d sh) * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) :
        ℚ) : ℝ) }

noncomputable def pearlCProtocolDevRev1KChainOnly (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) :
    Protocol Q R S :=
  { pearlCProtocolDevRev1K d sem ρ c with
    credit := fun H s U u act =>
      ((creditDevRev1ChainOnly d (U.layout.shape u) - sem.unitInertWorkRev1 d H s U u act : ℚ) : ℝ) }

noncomputable def pearlCTilesDevRev1KChainOnly (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) :
    TileRules Q R S :=
  { pearlCTilesDevRev1K d sem ρ c with
    credit := fun _ _ U g act =>
      let tl := auditTiling 64 64 U.layout
      let sh := U.layout.shape (tl.unit g)
      (((creditDevRev1ChainOnly d sh - ρ * creditDevRev1 d sh) *
        tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ) }

end Pouw.PearlC
