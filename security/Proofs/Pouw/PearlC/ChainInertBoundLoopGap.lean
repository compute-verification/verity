module

public import Proofs.Pouw.PearlC.DeviceChainInertBoundGap
public import Proofs.Pouw.PearlC.Records
public import Definitions.Pouw.Assumptions.PearlC.DeviceChainInertBound
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.DeviceChainInertBoundGap

/-! v2 at the chain cap 1/1,000 at sm_120's in-loop prices (staged); each g is an upper bound on the in-loop g,
because `Prices.sm120Loop` rounds the A-only forming up to 2. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

theorem pearlCGapUnpromotedChainInertShare1000Loop_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0)
    (hp : d.prices = Prices.sm120Loop) (hTT : TTOutPearlCDevChainInertBound CM d sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevChainInertBound d sem (1 / 1000)) (pearlCDomainDevAt d sh8192) ((64899 / 17487500 : ℚ) : ℝ)
      δPearlC :=
  pearlCGapDevChainInertBoundAt CM d sem _ _ (524625 / 523988) _ (by norm_num)
    (by rw [wrefDev, creditDev, chainCreditDev, hG, hp]; norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapUnpromotedChainInertShare1000Loop_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0)
    (hp : d.prices = Prices.sm120Loop) (hTT : TTOutPearlCDevChainInertBound CM d sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevChainInertBound d sem (1 / 1000)) (pearlCDomainDevAt d sh16384) ((373769 / 103662500 : ℚ) : ℝ)
      δPearlC :=
  pearlCGapDevChainInertBoundAt CM d sem _ _ (1036625 / 1035476) _ (by norm_num)
    (by rw [wrefDev, creditDev, chainCreditDev, hG, hp]; norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledUnpromotedChainInertShare1000Loop_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0)
    (hp : d.prices = Prices.sm120Loop) (hTT : TTOutTilePearlCDevChainInertBound CM d sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevChainInertBound d sem (1 / 1000)) (pearlCTilesDevChainInertBound d sem (1 / 1000))
      (pearlCDomainDevAt d sh8192) ((522517 / 139900000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevChainInertBoundAt CM d sem _ _ (349750 / 349317) _ (by norm_num)
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledUnpromotedChainInertShare1000Loop_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0)
    (hp : d.prices = Prices.sm120Loop) (hTT : TTOutTilePearlCDevChainInertBound CM d sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevChainInertBound d sem (1 / 1000)) (pearlCTilesDevChainInertBound d sem (1 / 1000))
      (pearlCDomainDevAt d sh16384) ((3000127 / 829300000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevChainInertBoundAt CM d sem _ _ (2073250 / 2070927) _ (by norm_num)
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

end Pouw.PearlC
