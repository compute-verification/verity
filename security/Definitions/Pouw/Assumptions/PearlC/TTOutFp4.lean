module

public import Definitions.Pouw.Assumptions.PearlC.TTOut
public import Definitions.Pouw.PearlC.DeviceFp4

/-! TT_OUT for Pearl-C4 on sm_120, the `Prop`s the proofs of its `W1` forms reduce to. -/

@[expose] public section

namespace Pouw.PearlC.Assumptions

open Finset

variable {Q R S : Type}

/-- **TT_OUT(1/400) at an FP4 record** `d`: `TTOutPearlCDev`'s statement on `Fp4Device`'s maps. Its domain includes
D-SK, so at a `sem` whose `B̃` is not the one SHA-512 keys by this job's salt it claims nothing. `ρ_D` rests on
`CodeProofs.PinnedScales`, which is argued, not proved. -/
def TTOutFp4 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (d : Fp4Device)
    (sem : Fp4Sem Q R S) (ρ : ℚ) : Prop :=
  TTOut CM (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4 sem) (1 / 400) δPearlC

def TTOutTileFp4 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (d : Fp4Device)
    (sem : Fp4Sem Q R S) (ρ : ℚ) : Prop :=
  TTOutTile CM (pearlCProtocolFp4 d sem ρ) (pearlCTilesFp4 d sem ρ) (pearlCDomainFp4 sem) (1 / 400) δPearlC

end Pouw.PearlC.Assumptions
