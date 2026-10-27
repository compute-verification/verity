import Definitions.Pouw.Assumptions.PearlC.TTOutW1
import Definitions.Pouw.PearlC.DeviceChainOnly

/-!
# TT_OUT under the chain-only reading at W1 (trusted; named assumptions, `Prop`s taken as hypotheses, never `axiom`s)

The chain-only forms of `TTOutChainOnly` (`Definitions.Pouw.Assumptions.PearlC.TTOutChainOnly`) with their cost model the term
`W1.w1Priced dv salt`, as `TTOutW1` states the device forms: `TTOutW1`'s game at the chain-only protocols
(`DeviceChainOnly`, whose credit leaves out the forming). What W1 charges, the device `dv` paired with the record `d`,
`sha`, `fetch` and the noise are `TTOutW1`'s. `ttOutPearlCDev{,Rev1}ChainOnly_of_w1`
(`Proofs.Pouw.PearlC.TTOutW1Variants`) gives the abstract form at `CM := W1.w1Priced dv salt`.
-/

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- **TT_OUT(1/400) at a device record, chain-only, at W1 on `dv`**: `TTOutPearlCDevChainOnly` at
`W1.w1Priced dv salt`. -/
def TTOutPearlCDevChainOnlyW1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device)
    (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutW1 dv salt (pearlCProtocolDevKChainOnly d sem ρ 0) (pearlCDomainDev d) (1 / 400) εPearlC

/-- **TT_OUT(1/400) rev1 at a device record, chain-only, at W1 on `dv`**: `TTOutPearlCDevRev1ChainOnly` at
`W1.w1Priced dv salt`. -/
def TTOutPearlCDevRev1ChainOnlyW1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device)
    (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutW1 dv salt (pearlCProtocolDevRev1KChainOnly d sem ρ 0) (pearlCDomainDev d) (1 / 400) εPearlC

end Pouw.PearlC.Assumptions
