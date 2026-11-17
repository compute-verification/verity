module

public import Proofs.Pouw.PearlC.DeviceInertBoundGap
public import Proofs.Pouw.PearlC.Records
public import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.DeviceGap
import Proofs.Pouw.PearlC.DeviceRev1Gap

/-! v2 at the cap 1/1,000 and v1 under rev1 at sm_120's in-loop prices, staged; every g here is an upper bound on
the in-loop g. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

theorem pearlCGapUnpromotedInertShare1000Loop_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0)
    (hp : d.prices = Prices.sm120Loop) (hTT : TTOutPearlCDev CM d sem (1 / 1000)) :
    GGap CM (pearlCProtocolDev d sem (1 / 1000)) (pearlCDomainDevAt d sh8192) ((522517 / 139900000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevAt CM d sem _ _ (349750 / 349317) _ (by norm_num)
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapUnpromotedInertShare1000Loop_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0)
    (hp : d.prices = Prices.sm120Loop) (hTT : TTOutPearlCDev CM d sem (1 / 1000)) :
    GGap CM (pearlCProtocolDev d sem (1 / 1000)) (pearlCDomainDevAt d sh16384) ((3000127 / 829300000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevAt CM d sem _ _ (2073250 / 2070927) _ (by norm_num)
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledUnpromotedInertShare1000Loop_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0)
    (hp : d.prices = Prices.sm120Loop) (hTT : TTOutTilePearlCDev CM d sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDev d sem (1 / 1000)) (pearlCTilesDev d sem (1 / 1000)) (pearlCDomainDevAt d sh8192)
      ((522517 / 139900000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevAt CM d sem _ _ (349750 / 349317) _ (by norm_num)
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledUnpromotedInertShare1000Loop_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0)
    (hp : d.prices = Prices.sm120Loop) (hTT : TTOutTilePearlCDev CM d sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDev d sem (1 / 1000)) (pearlCTilesDev d sem (1 / 1000)) (pearlCDomainDevAt d sh16384)
      ((3000127 / 829300000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevAt CM d sem _ _ (2073250 / 2070927) _ (by norm_num)
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120LoopRev1_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4)
    (hp : d.prices = Prices.sm120Loop) (hTT : TTOutPearlCDevRev1 CM d sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCDomainDevAt d sh8192)
      ((310284613 / 59477920000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1At CM d sem _ _ (148694800 / 148289813) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120LoopRev1_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4)
    (hp : d.prices = Prices.sm120Loop) (hTT : TTOutPearlCDevRev1 CM d sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCDomainDevAt d sh16384)
      ((1802569231 / 352995040000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1At CM d sem _ _ (882487600 / 880181631) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120LoopRev1_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4)
    (hp : d.prices = Prices.sm120Loop) (hTT : TTOutTilePearlCDevRev1 CM d sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCTilesDevRev1 d sem (1 / 400))
      (pearlCDomainDevAt d sh8192)
      ((310284613 / 59477920000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1At CM d sem _ _ (148694800 / 148289813) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120LoopRev1_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4)
    (hp : d.prices = Prices.sm120Loop) (hTT : TTOutTilePearlCDevRev1 CM d sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCTilesDevRev1 d sem (1 / 400))
      (pearlCDomainDevAt d sh16384)
      ((1802569231 / 352995040000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1At CM d sem _ _ (882487600 / 880181631) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

end Pouw.PearlC
