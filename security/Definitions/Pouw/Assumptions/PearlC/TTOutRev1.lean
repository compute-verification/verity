module

public import Definitions.Pouw.Assumptions.PearlC.TTOut
public import Definitions.Pouw.PearlC.DeviceRev1

/-! The named assumption TT_OUT rev1, Pearl-C's TT_OUT(1/400) with every promotion into a +0 total out of the credit; the falsified granted rows imply it, which is no evidence for it. -/

@[expose] public section

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- **TT_OUT(1/400) for Pearl-C at a device record `d`, rev1**: `TTOutPearlCDev` with every promotion into a +0 total
left out of the credit. `CostModel` is abstract, so that `CM` prices a tensor-core instruction by its whole fragment,
never per output word, is part of which `CM` it is stated at, not a property Lean checks. -/
def TTOutPearlCDevRev1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (d : PearlCDevice)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOut CM (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDev d) (1 / 400) δPearlC

/-- `TTOutTilePearlCDev` with rev1's credit, cap and `W_ref`. -/
def TTOutTilePearlCDevRev1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTile CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDev d) (1 / 400) δPearlC

end Pouw.PearlC.Assumptions
