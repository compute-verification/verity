module

public import Definitions.Pouw.Assumptions.PearlC.DeviceChainInertBound
public import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
public import Definitions.Pouw.Assumptions.PearlC.TTOutUOnly
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.InertBoundLoopGap
import Proofs.Pouw.PearlC.ChainInertBoundLoopGap
import Proofs.Pouw.PearlC.UOnlyLoopGap
public import Proofs.Pouw.PearlC.DeviceSm120Gap
public import Proofs.Pouw.PearlC.InertBoundLoopGap
public import Proofs.Pouw.PearlC.ChainInertBoundLoopGap
public import Proofs.Pouw.PearlC.UOnlyLoopGap
public import Proofs.Pouw.PearlC.Records

/-! g at the RTX PRO 6000 (sm_120) records at the in-loop prices (staged); each g is an upper bound on the
in-loop g, because `Prices.sm120Loop` rounds the A-only forming up to 2. -/

@[expose] public section

namespace Pouw.PearlC

open Pouw.Fp8Atom Pouw.PearlC.Assumptions

theorem devSm120v1Loop_domain (s : Shape) (hs : s = sh8192 ∨ s = sh16384) :
    pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) s ⟨1, fun _ => s, fun _ => 0⟩ := by
  refine ⟨fun u _ => ?_, fun _ _ => rfl⟩
  rcases hs with rfl | rfl <;> dsimp only <;>
    exact ⟨by decide, by decide, by decide, by decide, Or.inr (by decide), by decide⟩

theorem devSm120v2Loop_domain (s : Shape) (hs : s = sh8192 ∨ s = sh16384) :
    pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) s ⟨1, fun _ => s, fun _ => 0⟩ := by
  refine ⟨fun u _ => ?_, fun _ _ => rfl⟩
  rcases hs with rfl | rfl <;> dsimp only <;> exact ⟨by decide, by decide, by decide, by decide, Or.inl rfl, by decide⟩

section Gap

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
  (sem : PearlCSem Q R S)

theorem pearlCGapSm120v1LoopRev1_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120Loop) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((310284613 / 59477920000 : ℚ) : ℝ) δPearlC :=
  pearlCGapSm120LoopRev1_8192 CM (devSm120v1 Prices.sm120Loop) sem rfl rfl hTT

theorem pearlCGapSm120v1LoopRev1_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120Loop) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((1802569231 / 352995040000 : ℚ) : ℝ) δPearlC :=
  pearlCGapSm120LoopRev1_16384 CM (devSm120v1 Prices.sm120Loop) sem rfl rfl hTT

theorem pearlCSampledSm120v1LoopRev1_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120Loop) sem (1 / 400))
      (pearlCTilesDevRev1 (devSm120v1 Prices.sm120Loop) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((310284613 / 59477920000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledSm120LoopRev1_8192 CM (devSm120v1 Prices.sm120Loop) sem rfl rfl hTT

theorem pearlCSampledSm120v1LoopRev1_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120Loop) sem (1 / 400))
      (pearlCTilesDevRev1 (devSm120v1 Prices.sm120Loop) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((1802569231 / 352995040000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledSm120LoopRev1_16384 CM (devSm120v1 Prices.sm120Loop) sem rfl rfl hTT

theorem pearlCGapUOnlySm120v1LoopRev1_8192
    (hTT : TTOutPearlCDevUOnly CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapU CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120Loop) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((310284613 / 59477920000 : ℚ) : ℝ) δPearlC :=
  pearlCGapUOnlySm120LoopRev1_8192 CM (devSm120v1 Prices.sm120Loop) sem rfl rfl hTT

theorem pearlCGapSm120v2LoopInertShare1000_8192
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDev (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((522517 / 139900000 : ℚ) : ℝ) δPearlC :=
  pearlCGapUnpromotedInertShare1000Loop_8192 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

theorem pearlCGapSm120v2LoopInertShare1000_16384
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDev (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((3000127 / 829300000 : ℚ) : ℝ) δPearlC :=
  pearlCGapUnpromotedInertShare1000Loop_16384 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

theorem pearlCSampledSm120v2LoopInertShare1000_8192
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDev (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCTilesDev (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((522517 / 139900000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledUnpromotedInertShare1000Loop_8192 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

theorem pearlCSampledSm120v2LoopInertShare1000_16384
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDev (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCTilesDev (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((3000127 / 829300000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledUnpromotedInertShare1000Loop_16384 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

theorem pearlCGapUOnlySm120v2LoopInertShare1000_8192
    (hTT : TTOutPearlCDevUOnly CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapU CM (pearlCProtocolDevRev1 (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((522517 / 139900000 : ℚ) : ℝ) δPearlC :=
  pearlCGapUOnlyUnpromotedInertShare1000Loop_8192 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

theorem pearlCGapSm120v2LoopChainInertShare1000_8192
    (hTT : TTOutPearlCDevChainInertBound CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevChainInertBound (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((64899 / 17487500 : ℚ) : ℝ) δPearlC :=
  pearlCGapUnpromotedChainInertShare1000Loop_8192 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

theorem pearlCGapSm120v2LoopChainInertShare1000_16384
    (hTT : TTOutPearlCDevChainInertBound CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevChainInertBound (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((373769 / 103662500 : ℚ) : ℝ) δPearlC :=
  pearlCGapUnpromotedChainInertShare1000Loop_16384 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

theorem pearlCSampledSm120v2LoopChainInertShare1000_8192
    (hTT : TTOutTilePearlCDevChainInertBound CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevChainInertBound (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCTilesDevChainInertBound (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((522517 / 139900000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledUnpromotedChainInertShare1000Loop_8192 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

theorem pearlCSampledSm120v2LoopChainInertShare1000_16384
    (hTT : TTOutTilePearlCDevChainInertBound CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevChainInertBound (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCTilesDevChainInertBound (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((3000127 / 829300000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledUnpromotedChainInertShare1000Loop_16384 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

end Gap

end Pouw.PearlC
