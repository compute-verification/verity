module

public import Definitions.Pouw.Assumptions.PearlC.TTOut
public import Definitions.Pouw.PearlC.DeviceChainOnly

/-! TT_OUT's chain-only forms at a device record, each implied by the record's own TT_OUT, so none is a new assumption. -/

@[expose] public section

namespace Pouw.PearlC.Assumptions

open Pouw.PearlC

variable {Q R S : Type}

/-- Its credit leaves out the forming and does not read `W_ref`, so `0` is no choice. It is weaker than `TTOutPearlCDev`
(`ttOutPearlCDevChainOnly_of_ttOut`). -/
def TTOutPearlCDevChainOnly [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOut CM (pearlCProtocolDevKChainOnly d sem ρ 0) (pearlCDomainDev d) (1 / 400) δPearlC

def TTOutTilePearlCDevChainOnly [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTile CM (pearlCProtocolDevKChainOnly d sem ρ 0) (pearlCTilesDevKChainOnly d sem ρ 0) (pearlCDomainDev d)
    (1 / 400) δPearlC

def TTOutPearlCDevRev1ChainOnly [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOut CM (pearlCProtocolDevRev1KChainOnly d sem ρ 0) (pearlCDomainDev d) (1 / 400) δPearlC

def TTOutTilePearlCDevRev1ChainOnly [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTile CM (pearlCProtocolDevRev1KChainOnly d sem ρ 0) (pearlCTilesDevRev1KChainOnly d sem ρ 0)
    (pearlCDomainDev d) (1 / 400) δPearlC

end Pouw.PearlC.Assumptions
