module

public import Proofs.Pouw.PearlC.Records
public import Definitions.Pouw.Assumptions.PearlC.TTOutUOnly
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.UOnlyGap

/-! U-only binding at sm_120's in-loop prices, staged; every g here is an upper bound on the in-loop g, because
`Prices.sm120Loop` rounds the A-only forming up to 2. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

theorem pearlCGapUOnlySm120LoopRev1_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4)
    (hp : d.prices = Prices.sm120Loop) (hTT : TTOutPearlCDevUOnly CM d sem (1 / 400)) :
    GGapU CM (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCDomainDevAt d sh8192) ((310284613 / 59477920000 : ℚ) : ℝ)
      δPearlC :=
  pearlCGapDevUOnlyAt CM d sem _ _ (148694800 / 148289813) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapUOnlyUnpromotedInertShare1000Loop_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0)
    (hp : d.prices = Prices.sm120Loop) (hTT : TTOutPearlCDevUOnly CM d sem (1 / 1000)) :
    GGapU CM (pearlCProtocolDevRev1 d sem (1 / 1000)) (pearlCDomainDevAt d sh8192) ((522517 / 139900000 : ℚ) : ℝ)
      δPearlC :=
  pearlCGapDevUOnlyAt CM d sem _ _ (349750 / 349317) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

end Pouw.PearlC
