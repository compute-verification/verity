module

public import Definitions.Pouw.Assumptions.PearlC.TTOutW1

/-! TT_OUT-FP4 with its cost model the term `W1.w1Priced dv salt`, as named assumptions believed only at a `rootA` that reads the activations. -/

@[expose] public section

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- **TT_OUT(1/400) at an FP4 record `d`, at W1 on `dv`**: `TTOutFp4` with its cost model the term
`W1.w1Priced dv salt`. -/
def TTOutFp4W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (d : Fp4Device) (sem : Fp4Sem Q R S) (ρ : ℚ) : Prop :=
  TTOutW1 dv salt (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4 sem) (1 / 400) δPearlC

/-- **TT_OUT(1/400) per audit tile at an FP4 record `d`, at W1 on `dv`**: `TTOutTileFp4` at `W1.w1Priced dv salt`. -/
def TTOutTileFp4W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (d : Fp4Device) (sem : Fp4Sem Q R S) (ρ : ℚ) : Prop :=
  TTOutTileW1 dv salt (pearlCProtocolFp4 d sem ρ) (pearlCTilesFp4 d sem ρ) (pearlCDomainFp4 sem) (1 / 400) δPearlC

end Pouw.PearlC.Assumptions
