module

public import Definitions.Pouw.Assumptions.PearlC.TTOutW1
public import Definitions.Pouw.PearlC.DeviceChainOnly

/-! TT_OUT's chain-only forms at a device record with the cost model `W1.w1Priced dv salt`, as named assumptions. -/

@[expose] public section

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

def TTOutPearlCDevChainOnlyW1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device)
    (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutW1 dv salt (pearlCProtocolDevKChainOnly d sem ρ 0) (pearlCDomainDev d) (1 / 400) δPearlC

def TTOutPearlCDevRev1ChainOnlyW1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device)
    (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutW1 dv salt (pearlCProtocolDevRev1KChainOnly d sem ρ 0) (pearlCDomainDev d) (1 / 400) δPearlC

end Pouw.PearlC.Assumptions
