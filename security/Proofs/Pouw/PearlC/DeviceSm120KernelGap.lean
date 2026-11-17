module

public import Definitions.Pouw.Assumptions.PearlC.TTOutChainOnly
public import Proofs.Pouw.PearlC.ChainOnlyGap
public import Proofs.Pouw.Lemmas.PearlCRecords
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.KernelWrefGap
import Proofs.Pouw.PearlC.TTOutW1
import Proofs.Pouw.PearlC.TTOutW1Variants
public import Proofs.Pouw.PearlC.Records

/-! g at the RTX PRO 6000 (sm_120) records against the kernel as it runs (staged), one theorem per cast price,
with `W_ref` the kernel's at each price. -/

@[expose] public section

namespace Pouw.PearlC

open Pouw.Fp8Atom Pouw.PearlC.Assumptions

section Gap

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
  (sem : PearlCSem Q R S)

theorem pearlCGapSm120v1Cast32Rev1_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (32 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((173561 / 22302500 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (223025 / 221844) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v1Cast32Rev1_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (32 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (32 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((173561 / 22302500 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (223025 / 221844) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v1Cast32Rev1_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (32 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((3763 / 587500 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (5875 / 5852) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v1Cast32Rev1_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (32 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (32 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((3763 / 587500 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (5875 / 5852) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v2Cast32InertShare1000_8192
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (32 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((143029 / 22150000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (1052125 / 1047951) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v2Cast32InertShare1000_8192
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (32 - 8))
      (pearlCTilesDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (32 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((143029 / 22150000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (1052125 / 1047951) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v2Cast32InertShare1000_16384
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (32 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((4150127 / 830450000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (2076125 / 2070927) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v2Cast32InertShare1000_16384
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (32 - 8))
      (pearlCTilesDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (32 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((4150127 / 830450000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (2076125 / 2070927) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v1Cast32p06Rev1_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((173711 / 22302650 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (446053 / 443688) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v1Cast32p06Rev1_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((173711 / 22302650 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (446053 / 443688) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v1Cast32p06Rev1_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((1255 / 195834 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (293751 / 292600) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v1Cast32p06Rev1_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((1255 / 195834 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (293751 / 292600) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v2Cast32p06InertShare1000_8192
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (1603/50 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((2720551 / 420853000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (2104265 / 2095902) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v2Cast32p06InertShare1000_8192
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (1603/50 - 8))
      (pearlCTilesDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (1603/50 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((2720551 / 420853000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (2104265 / 2095902) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v2Cast32p06InertShare1000_16384
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (1603/50 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((4153127 / 830453000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (4152265 / 4141854) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v2Cast32p06InertShare1000_16384
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (1603/50 - 8))
      (pearlCTilesDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (1603/50 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((4153127 / 830453000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (4152265 / 4141854) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v1Cast32p06ChainOnlyRev1_8192
    (hTT : TTOutPearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((273461 / 22302650 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KChainOnlyAt CM _ sem _ _ _ (446053 / 441688) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v1Cast32p06ChainOnlyRev1_8192
    (hTT : TTOutTilePearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((273461 / 22302650 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KChainOnlyAt CM _ sem _ _ _ (446053 / 441688) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v1Cast32p06ChainOnlyRev1_16384
    (hTT : TTOutPearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((5095 / 587502 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KChainOnlyAt CM _ sem _ _ _ (881253 / 875800) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v1Cast32p06ChainOnlyRev1_16384
    (hTT : TTOutTilePearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((5095 / 587502 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KChainOnlyAt CM _ sem _ _ _ (881253 / 875800) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v2Cast32p06ChainOnlyInertShare1000_8192
    (hTT : TTOutPearlCDevChainOnly CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (1603/50 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((4715551 / 420853000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKChainOnlyAt CM _ sem _ _ _ (2104265 / 2085902) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v2Cast32p06ChainOnlyInertShare1000_8192
    (hTT : TTOutTilePearlCDevChainOnly CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (1603/50 - 8))
      (pearlCTilesDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (1603/50 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((4715551 / 420853000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKChainOnlyAt CM _ sem _ _ _ (2104265 / 2085902) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v2Cast32p06ChainOnlyInertShare1000_16384
    (hTT : TTOutPearlCDevChainOnly CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (1603/50 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((6148127 / 830453000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKChainOnlyAt CM _ sem _ _ _ (4152265 / 4131854) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v2Cast32p06ChainOnlyInertShare1000_16384
    (hTT : TTOutTilePearlCDevChainOnly CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (1603/50 - 8))
      (pearlCTilesDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (1603/50 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((6148127 / 830453000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKChainOnlyAt CM _ sem _ _ _ (4152265 / 4131854) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v1Cast16Rev1_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (16 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((133561 / 22262500 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (222625 / 221844) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v1Cast16Rev1_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (16 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (16 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((133561 / 22262500 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (222625 / 221844) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v1Cast16Rev1_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (16 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((9689 / 1760900 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (17609 / 17556) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v1Cast16Rev1_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (16 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (16 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((9689 / 1760900 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (17609 / 17556) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v2Cast16InertShare1000_8192
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (16 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((1917551 / 420050000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (1050125 / 1047951) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v2Cast16InertShare1000_8192
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (16 - 8))
      (pearlCTilesDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (16 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((1917551 / 420050000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (1050125 / 1047951) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v2Cast16InertShare1000_16384
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (16 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((1116709 / 276550000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (691375 / 690309) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v2Cast16InertShare1000_16384
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (16 - 8))
      (pearlCTilesDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (16 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((1116709 / 276550000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (691375 / 690309) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v1Cast16ChainOnlyRev1_8192
    (hTT : TTOutPearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (16 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((131 / 12500 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KChainOnlyAt CM _ sem _ _ _ (125 / 124) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v1Cast16ChainOnlyRev1_8192
    (hTT : TTOutTilePearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (16 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (16 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((131 / 12500 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KChainOnlyAt CM _ sem _ _ _ (125 / 124) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v1Cast16ChainOnlyRev1_16384
    (hTT : TTOutPearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (16 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((13679 / 1760900 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KChainOnlyAt CM _ sem _ _ _ (17609 / 17516) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v1Cast16ChainOnlyRev1_16384
    (hTT : TTOutTilePearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (16 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (16 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((13679 / 1760900 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KChainOnlyAt CM _ sem _ _ _ (17609 / 17516) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v2Cast16ChainOnlyInertShare1000_8192
    (hTT : TTOutPearlCDevChainOnly CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (16 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((3912551 / 420050000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKChainOnlyAt CM _ sem _ _ _ (1050125 / 1042951) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v2Cast16ChainOnlyInertShare1000_8192
    (hTT : TTOutTilePearlCDevChainOnly CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (16 - 8))
      (pearlCTilesDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (16 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((3912551 / 420050000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKChainOnlyAt CM _ sem _ _ _ (1050125 / 1042951) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v2Cast16ChainOnlyInertShare1000_16384
    (hTT : TTOutPearlCDevChainOnly CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (16 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((1781709 / 276550000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKChainOnlyAt CM _ sem _ _ _ (2074125 / 2065927) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v2Cast16ChainOnlyInertShare1000_16384
    (hTT : TTOutTilePearlCDevChainOnly CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (16 - 8))
      (pearlCTilesDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (16 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((1781709 / 276550000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKChainOnlyAt CM _ sem _ _ _ (2074125 / 2065927) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v1Cast8p72Rev1_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((115361 / 22244300 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (222443 / 221844) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v1Cast8p72Rev1_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((115361 / 22244300 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (222443 / 221844) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v1Cast8p72Rev1_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2987 / 586724 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (146681 / 146300) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v1Cast8p72Rev1_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2987 / 586724 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (146681 / 146300) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v2Cast8p72InertShare1000_8192
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((1553551 / 419686000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (1049215 / 1047951) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v2Cast8p72InertShare1000_8192
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
      (pearlCTilesDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((1553551 / 419686000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (1049215 / 1047951) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v2Cast8p72InertShare1000_16384
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((2986127 / 829286000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (2073215 / 2070927) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v2Cast8p72InertShare1000_16384
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
      (pearlCTilesDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((2986127 / 829286000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (2073215 / 2070927) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v1Cast8p72ChainOnlyRev1_8192
    (hTT : TTOutPearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16547 / 1711100 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KChainOnlyAt CM _ sem _ _ _ (17111 / 16988) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v1Cast8p72ChainOnlyRev1_8192
    (hTT : TTOutTilePearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16547 / 1711100 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KChainOnlyAt CM _ sem _ _ _ (17111 / 16988) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v1Cast8p72ChainOnlyRev1_16384
    (hTT : TTOutPearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((4317 / 586724 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KChainOnlyAt CM _ sem _ _ _ (440043 / 437900) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v1Cast8p72ChainOnlyRev1_16384
    (hTT : TTOutTilePearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((4317 / 586724 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KChainOnlyAt CM _ sem _ _ _ (440043 / 437900) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120).G = 4 from rfl,
          show (devSm120v1 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v2Cast8p72ChainOnlyInertShare1000_8192
    (hTT : TTOutPearlCDevChainOnly CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((3548551 / 419686000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKChainOnlyAt CM _ sem _ _ _ (1049215 / 1042951) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v2Cast8p72ChainOnlyInertShare1000_8192
    (hTT : TTOutTilePearlCDevChainOnly CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
      (pearlCTilesDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((3548551 / 419686000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKChainOnlyAt CM _ sem _ _ _ (1049215 / 1042951) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v2Cast8p72ChainOnlyInertShare1000_16384
    (hTT : TTOutPearlCDevChainOnly CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((4981127 / 829286000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKChainOnlyAt CM _ sem _ _ _ (2073215 / 2065927) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v2Cast8p72ChainOnlyInertShare1000_16384
    (hTT : TTOutTilePearlCDevChainOnly CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
      (pearlCTilesDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((4981127 / 829286000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKChainOnlyAt CM _ sem _ _ _ (2073215 / 2065927) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120).G = 0 from rfl,
          show (devSm120v2 Prices.sm120).prices = Prices.sm120 from rfl]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast32Rev1_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (32 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((1391793839 / 178894700000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (447236750 / 444869439) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast32Rev1_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (32 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (32 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((1391793839 / 178894700000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (447236750 / 444869439) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast32Rev1_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (32 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((754503077 / 117818660000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (294546650 / 293393877) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast32Rev1_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (32 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (32 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((754503077 / 117818660000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (294546650 / 293393877) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v2LoopCast32InertShare1000_8192
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (32 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((2719901 / 420852350 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (8417047 / 8383608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast32InertShare1000_8192
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (32 - 8953 / 1000))
      (pearlCTilesDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (32 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((2719901 / 420852350 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (8417047 / 8383608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v2LoopCast32InertShare1000_16384
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (32 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((197737 / 39545350 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (5536349 / 5522472) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast32InertShare1000_16384
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (32 - 8953 / 1000))
      (pearlCTilesDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (32 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((197737 / 39545350 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (5536349 / 5522472) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast32p06Rev1_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((1392993839 / 178895900000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (447239750 / 444869439) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast32p06Rev1_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((1392993839 / 178895900000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (447239750 / 444869439) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast32p06Rev1_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((251634359 / 39273020000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (98182550 / 97797959) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast32p06Rev1_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((251634359 / 39273020000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (98182550 / 97797959) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v2LoopCast32p06InertShare1000_8192
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (1603/50 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((2722901 / 420855350 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (8417107 / 8383608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast32p06InertShare1000_8192
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (1603/50 - 8953 / 1000))
      (pearlCTilesDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (1603/50 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((2722901 / 420855350 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (8417107 / 8383608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v2LoopCast32p06InertShare1000_16384
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (1603/50 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((1385159 / 276818450 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (5536369 / 5522472) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast32p06InertShare1000_16384
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (1603/50 - 8953 / 1000))
      (pearlCTilesDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (1603/50 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((1385159 / 276818450 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (5536369 / 5522472) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast32p06ChainOnlyRev1_8192
    (hTT : TTOutPearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((2190993839 / 178895900000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KChainOnlyAt CM _ sem _ _ _ (447239750 / 442869439) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast32p06ChainOnlyRev1_8192
    (hTT : TTOutTilePearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((2190993839 / 178895900000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KChainOnlyAt CM _ sem _ _ _ (447239750 / 442869439) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast32p06ChainOnlyRev1_16384
    (hTT : TTOutPearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((1020903077 / 117819060000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KChainOnlyAt CM _ sem _ _ _ (883642950 / 878181631) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast32p06ChainOnlyRev1_16384
    (hTT : TTOutTilePearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((1020903077 / 117819060000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KChainOnlyAt CM _ sem _ _ _ (883642950 / 878181631) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v2LoopCast32p06ChainOnlyInertShare1000_8192
    (hTT : TTOutPearlCDevChainOnly CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (1603/50 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((4717901 / 420855350 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKChainOnlyAt CM _ sem _ _ _ (8417107 / 8343608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast32p06ChainOnlyInertShare1000_8192
    (hTT : TTOutTilePearlCDevChainOnly CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (1603/50 - 8953 / 1000))
      (pearlCTilesDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (1603/50 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((4717901 / 420855350 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKChainOnlyAt CM _ sem _ _ _ (8417107 / 8343608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v2LoopCast32p06ChainOnlyInertShare1000_16384
    (hTT : TTOutPearlCDevChainOnly CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (1603/50 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((2050159 / 276818450 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKChainOnlyAt CM _ sem _ _ _ (16609107 / 16527416) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast32p06ChainOnlyInertShare1000_16384
    (hTT : TTOutTilePearlCDevChainOnly CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (1603/50 - 8953 / 1000))
      (pearlCTilesDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (1603/50 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((2050159 / 276818450 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKChainOnlyAt CM _ sem _ _ _ (16609107 / 16527416) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast16Rev1_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (16 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((3277657 / 546100000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (1365250 / 1360457) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast16Rev1_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (16 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (16 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((3277657 / 546100000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (1365250 / 1360457) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast16Rev1_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (16 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((1943509231 / 353135980000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (882839950 / 880181631) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast16Rev1_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (16 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (16 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((1943509231 / 353135980000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (882839950 / 880181631) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v2LoopCast16InertShare1000_8192
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (16 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((639967 / 140017450 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (2800349 / 2794536) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast16InertShare1000_8192
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (16 - 8953 / 1000))
      (pearlCTilesDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (16 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((639967 / 140017450 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (2800349 / 2794536) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v2LoopCast16InertShare1000_16384
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (16 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((3352477 / 829652350 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (16593047 / 16567416) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast16InertShare1000_16384
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (16 - 8953 / 1000))
      (pearlCTilesDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (16 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((3352477 / 829652350 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (16593047 / 16567416) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast16ChainOnlyRev1_8192
    (hTT : TTOutPearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (16 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((623264613 / 59524900000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KChainOnlyAt CM _ sem _ _ _ (446436750 / 442869439) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast16ChainOnlyRev1_8192
    (hTT : TTOutTilePearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (16 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (16 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((623264613 / 59524900000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KChainOnlyAt CM _ sem _ _ _ (446436750 / 442869439) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast16ChainOnlyRev1_16384
    (hTT : TTOutPearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (16 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((2741509231 / 353135980000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KChainOnlyAt CM _ sem _ _ _ (882839950 / 878181631) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast16ChainOnlyRev1_16384
    (hTT : TTOutTilePearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (16 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (16 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((2741509231 / 353135980000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KChainOnlyAt CM _ sem _ _ _ (882839950 / 878181631) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v2LoopCast16ChainOnlyInertShare1000_8192
    (hTT : TTOutPearlCDevChainOnly CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (16 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((1304967 / 140017450 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKChainOnlyAt CM _ sem _ _ _ (8401047 / 8343608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast16ChainOnlyInertShare1000_8192
    (hTT : TTOutTilePearlCDevChainOnly CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (16 - 8953 / 1000))
      (pearlCTilesDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (16 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((1304967 / 140017450 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKChainOnlyAt CM _ sem _ _ _ (8401047 / 8343608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v2LoopCast16ChainOnlyInertShare1000_16384
    (hTT : TTOutPearlCDevChainOnly CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (16 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((5347477 / 829652350 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKChainOnlyAt CM _ sem _ _ _ (16593047 / 16527416) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast16ChainOnlyInertShare1000_16384
    (hTT : TTOutTilePearlCDevChainOnly CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (16 - 8953 / 1000))
      (pearlCTilesDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (16 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((5347477 / 829652350 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKChainOnlyAt CM _ sem _ _ _ (16593047 / 16527416) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast8p72Rev1_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((926193839 / 178429100000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (446072750 / 444869439) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast8p72Rev1_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((926193839 / 178429100000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (446072750 / 444869439) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast8p72Rev1_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((599303077 / 117663460000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (294158650 / 293393877) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast8p72Rev1_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((599303077 / 117663460000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (294158650 / 293393877) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v2LoopCast8p72InertShare1000_8192
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((1555901 / 419688350 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (8393767 / 8383608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast8p72InertShare1000_8192
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
      (pearlCTilesDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((1555901 / 419688350 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (8393767 / 8383608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v2LoopCast8p72InertShare1000_16384
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((332053 / 92143150 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (1842863 / 1840824) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast8p72InertShare1000_16384
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
      (pearlCTilesDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((332053 / 92143150 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (1842863 / 1840824) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast8p72ChainOnlyRev1_8192
    (hTT : TTOutPearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((1724193839 / 178429100000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KChainOnlyAt CM _ sem _ _ _ (446072750 / 442869439) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast8p72ChainOnlyRev1_8192
    (hTT : TTOutTilePearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((1724193839 / 178429100000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KChainOnlyAt CM _ sem _ _ _ (446072750 / 442869439) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast8p72ChainOnlyRev1_16384
    (hTT : TTOutPearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((50900181 / 6921380000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KChainOnlyAt CM _ sem _ _ _ (51910350 / 51657743) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast8p72ChainOnlyRev1_16384
    (hTT : TTOutTilePearlCDevRev1ChainOnly CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((50900181 / 6921380000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KChainOnlyAt CM _ sem _ _ _ (51910350 / 51657743) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1ChainOnly, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v2LoopCast8p72ChainOnlyInertShare1000_8192
    (hTT : TTOutPearlCDevChainOnly CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((3550901 / 419688350 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKChainOnlyAt CM _ sem _ _ _ (8393767 / 8343608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast8p72ChainOnlyInertShare1000_8192
    (hTT : TTOutTilePearlCDevChainOnly CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
      (pearlCTilesDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((3550901 / 419688350 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKChainOnlyAt CM _ sem _ _ _ (8393767 / 8343608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v2LoopCast8p72ChainOnlyInertShare1000_16384
    (hTT : TTOutPearlCDevChainOnly CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((1661159 / 276429450 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKChainOnlyAt CM _ sem _ _ _ (16585767 / 16527416) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast8p72ChainOnlyInertShare1000_16384
    (hTT : TTOutTilePearlCDevChainOnly CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
      (pearlCTilesDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((1661159 / 276429450 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKChainOnlyAt CM _ sem _ _ _ (16585767 / 16527416) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevK, wrefDev, creditDevChainOnly, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast8Rev1_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (8 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((911793839 / 178414700000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (446036750 / 444869439) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast8Rev1_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (8 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (8 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((911793839 / 178414700000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (446036750 / 444869439) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v1LoopCast8Rev1_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (8 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((84929011 / 16808380000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KAt CM _ sem _ _ _ (42020950 / 41913411) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast8Rev1_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (8 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (8 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((84929011 / 16808380000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KAt CM _ sem _ _ _ (42020950 / 41913411) _ (by norm_num)
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
          show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
          show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCGapSm120v2LoopCast8InertShare1000_8192
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((1519901 / 419652350 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (8393047 / 8383608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast8InertShare1000_8192
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCTilesDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((1519901 / 419652350 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (8393047 / 8383608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v2LoopCast8InertShare1000_16384
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((109351 / 30713050 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevKAt CM _ sem _ _ _ (614261 / 613608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast8InertShare1000_16384
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCTilesDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((109351 / 30713050 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevKAt CM _ sem _ _ _ (614261 / 613608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

end Gap

end Pouw.PearlC

theorem Pouw.SecurityProofs.PearlC.GapSm120v1Cast16Rev1_8192 : Pouw.Guarantees.PearlC.GapSm120v1Cast16Rev1_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1Cast16Rev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1Cast32p06ChainOnlyRev1_8192 : Pouw.Guarantees.PearlC.GapSm120v1Cast32p06ChainOnlyRev1_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1Cast32p06ChainOnlyRev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1ChainOnly_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1Cast32p06Rev1_8192 : Pouw.Guarantees.PearlC.GapSm120v1Cast32p06Rev1_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1Cast32p06Rev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1Cast8p72ChainOnlyRev1_8192 : Pouw.Guarantees.PearlC.GapSm120v1Cast8p72ChainOnlyRev1_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1Cast8p72ChainOnlyRev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1ChainOnly_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1Cast8p72Rev1_16384 : Pouw.Guarantees.PearlC.GapSm120v1Cast8p72Rev1_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1Cast8p72Rev1_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1Cast8p72Rev1_8192 : Pouw.Guarantees.PearlC.GapSm120v1Cast8p72Rev1_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1Cast8p72Rev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1LoopCast16Rev1_8192 : Pouw.Guarantees.PearlC.GapSm120v1LoopCast16Rev1_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1LoopCast16Rev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1LoopCast32p06ChainOnlyRev1_8192 : Pouw.Guarantees.PearlC.GapSm120v1LoopCast32p06ChainOnlyRev1_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1LoopCast32p06ChainOnlyRev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1ChainOnly_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1LoopCast32p06Rev1_8192 : Pouw.Guarantees.PearlC.GapSm120v1LoopCast32p06Rev1_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1LoopCast32p06Rev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1LoopCast8Rev1_8192 : Pouw.Guarantees.PearlC.GapSm120v1LoopCast8Rev1_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1LoopCast8Rev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1LoopCast8p72ChainOnlyRev1_8192 : Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72ChainOnlyRev1_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1LoopCast8p72ChainOnlyRev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1ChainOnly_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1LoopCast8p72Rev1_16384 : Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1LoopCast8p72Rev1_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1LoopCast8p72Rev1_8192 : Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1LoopCast8p72Rev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v2Cast16InertShare1000_8192 : Pouw.Guarantees.PearlC.GapSm120v2Cast16InertShare1000_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v2Cast16InertShare1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDev_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v2Cast8p72InertShare1000_8192 : Pouw.Guarantees.PearlC.GapSm120v2Cast8p72InertShare1000_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v2Cast8p72InertShare1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDev_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v2Cast8p72ChainOnlyInertShare1000_8192 : Pouw.Guarantees.PearlC.GapSm120v2Cast8p72ChainOnlyInertShare1000_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v2Cast8p72ChainOnlyInertShare1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevChainOnly_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v2LoopCast16InertShare1000_8192 : Pouw.Guarantees.PearlC.GapSm120v2LoopCast16InertShare1000_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v2LoopCast16InertShare1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDev_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v2LoopCast8InertShare1000_8192 : Pouw.Guarantees.PearlC.GapSm120v2LoopCast8InertShare1000_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v2LoopCast8InertShare1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDev_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v2LoopCast8p72InertShare1000_8192 : Pouw.Guarantees.PearlC.GapSm120v2LoopCast8p72InertShare1000_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v2LoopCast8p72InertShare1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDev_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v2LoopCast8p72ChainOnlyInertShare1000_8192 : Pouw.Guarantees.PearlC.GapSm120v2LoopCast8p72ChainOnlyInertShare1000_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v2LoopCast8p72ChainOnlyInertShare1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevChainOnly_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Cast8p72Rev1_16384 : Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1Cast8p72Rev1_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Cast8p72Rev1_8192 : Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1Cast8p72Rev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1_16384 : Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1LoopCast8p72Rev1_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1_8192 : Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1LoopCast8p72Rev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)
