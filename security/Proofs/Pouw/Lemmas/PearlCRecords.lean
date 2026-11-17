module

public import Definitions.Pouw.Advantage
public import Definitions.Pouw.PearlC.DeviceKernelWref
public import Definitions.Pouw.Assumptions.PearlC.TTOutW1
public import Definitions.Pouw.Assumptions.PearlC.TTOutFp4W1
public import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
public import Definitions.Pouw.PearlC.Device
public import Definitions.Pouw.PearlC.DeviceFp4
public import Definitions.Pouw.PearlC.DeviceFp4Issue
public import Definitions.Pouw.PearlC.DeviceRev1
public import Definitions.Pouw.PearlC.Game
public import Definitions.Pouw.PearlC.Instance
public import Definitions.Pouw.PearlC.Tile
public import Definitions.Pouw.Assumptions.PearlC.RowSeedAssumptions
public import Definitions.Pouw.Assumptions.PearlC.TTOut
public import Definitions.Pouw.Assumptions.PearlC.TTOutFp4
public import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeed
public import Definitions.Pouw.Assumptions.PearlC.TTOutUOnly
public import Definitions.Pouw.PearlC.DeviceChainOnly
public import Definitions.Pouw.PearlC.DeviceFp4ChainOnly
public import Proofs.Pouw.PearlC.Records
public import Definitions.Pouw.Assumptions.PearlC.DeviceChainInertBound
public import Definitions.Pouw.Assumptions.PearlC.TTOutChainOnly
public import Definitions.Pouw.Assumptions.PearlC.TTOutChainOnlyW1
public import Definitions.Pouw.Guarantees.PearlC4

/-! Pearl-C's gap statements at named device records, as Props that no record pins. -/

@[expose] public section

namespace Pouw.Guarantees.PearlC

open Pouw.PearlC Pouw.PearlC.Assumptions

def GapSm120v1LoopCast8p72Rev1InertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
      ((1648180439 / 446072750000 : ℚ) : ℝ) δPearlC

def EpsSm120v1LoopCast8p72Rev1InertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.Gε (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
      ((1648180439 / 444424569561 : ℚ) : ℝ) δPearlC

end Pouw.Guarantees.PearlC

namespace Pouw.PearlC
open Pouw.PearlC.Assumptions

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.PearlC.Assumptions

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1InertShare1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
    ((1058166877 / 294158650000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

def _root_.Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1InertShare1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
    ((1058166877 / 294158650000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

def _root_.Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1InertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
    ((1648180439 / 446072750000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.PearlC.Assumptions

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.PearlC.Assumptions

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

def _root_.Pouw.Guarantees.PearlC.GapH100Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCH100Rev1W1 sha fetch salt sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.h100 sha fetch) salt) (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCDomainDevAt devH100 sh16384)
    ((89359 / 17230000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

def _root_.Pouw.Guarantees.PearlC.GapH100Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCH100Rev1W1 sha fetch salt sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.h100 sha fetch) salt) (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCDomainDevAt devH100 sh8192)
    ((140237 / 26090000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v1Cast16Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (16 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((133561 / 22262500 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v1Cast32p06ChainOnlyRev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1ChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((273461 / 22302650 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v1Cast32p06Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((173711 / 22302650 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v1Cast8p72ChainOnlyRev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1ChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16547 / 1711100 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v1Cast8p72Rev1InertShare1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384)
    ((5273 / 1466810 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v1Cast8p72Rev1InertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192)
    ((205211 / 55610750 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v1Cast8p72Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2987 / 586724 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v1Cast8p72Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((115361 / 22244300 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast16Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (16 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((3277657 / 546100000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast32p06ChainOnlyRev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1ChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((2190993839 / 178895900000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast32p06Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((1392993839 / 178895900000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast8Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (8 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((911793839 / 178414700000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72ChainOnlyRev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1ChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((1724193839 / 178429100000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1InertShare1000_16384OfInertShare400 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
    ((1058166877 / 294158650000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1InertShare1000_8192OfInertShare400 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
    ((1648180439 / 446072750000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((599303077 / 117663460000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((926193839 / 178429100000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v1Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2963 / 586700 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v1Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16223 / 3177500 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v2InertShare1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((2950127 / 829250000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v2InertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v2Cast16InertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (16 - 8))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((1917551 / 420050000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v2Cast8p72InertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((1553551 / 419686000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v2Cast8p72ChainOnlyInertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((3548551 / 419686000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v2ChainInertShare1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevChainInertBoundW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevChainInertBound (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((367519 / 103656250 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v2ChainInertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevChainInertBoundW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevChainInertBound (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((26921 / 7493750 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v2LoopCast16InertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v2 Prices.sm120Loop) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (16 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((639967 / 140017450 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v2LoopCast8InertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v2 Prices.sm120Loop) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((1519901 / 419652350 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v2LoopCast8p72InertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v2 Prices.sm120Loop) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((1555901 / 419688350 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapSm120v2LoopCast8p72ChainOnlyInertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v2 Prices.sm120Loop) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((3550901 / 419688350 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

def _root_.Pouw.Guarantees.PearlC.GapUOnlySm120Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4) (hp : d.prices = Prices.sm120) (hTT : TTOutPearlCDevUOnlyW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt d sem (1 / 400)),
    GGapU (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCDomainDevAt d sh8192) ((16223 / 3177500 : ℚ) : ℝ)
    δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

def _root_.Pouw.Guarantees.PearlC.GapUOnlyUnpromotedInertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120) (hTT : TTOutPearlCDevUOnlyW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt d sem (1 / 1000)),
    GGapU (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 d sem (1 / 1000)) (pearlCDomainDevAt d sh8192) ((216793 / 59950000 : ℚ) : ℝ)
    δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

def _root_.Pouw.Guarantees.PearlC.GapUnpromotedInertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt d sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDev d sem (1 / 1000)) (pearlCDomainDevAt d sh8192) ((216793 / 59950000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

def _root_.Pouw.Guarantees.PearlC.GapUnpromotedChainInertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120) (hTT : TTOutPearlCDevChainInertBoundW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt d sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevChainInertBound d sem (1 / 1000)) (pearlCDomainDevAt d sh8192) ((26921 / 7493750 : ℚ) : ℝ)
    δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

def _root_.Pouw.Guarantees.PearlC.SampledH100Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCH100Rev1W1 sha fetch salt sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.h100 sha fetch) salt) (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCTilesDevRev1 devH100 sem (1 / 400))
    (pearlCDomainDevAt devH100 sh16384) ((89359 / 17230000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

def _root_.Pouw.Guarantees.PearlC.SampledH100Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCH100Rev1W1 sha fetch salt sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.h100 sha fetch) salt) (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCTilesDevRev1 devH100 sem (1 / 400))
    (pearlCDomainDevAt devH100 sh8192) ((140237 / 26090000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1InertShare1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384)
    ((5273 / 1466810 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1InertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192)
    ((205211 / 55610750 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2987 / 586724 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((115361 / 22244300 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((599303077 / 117663460000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((926193839 / 178429100000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCTilesDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2963 / 586700 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCTilesDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16223 / 3177500 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.SampledSm120v2InertShare1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCTilesDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((2950127 / 829250000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.SampledSm120v2InertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCTilesDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.SampledSm120v2ChainInertShare1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevChainInertBoundW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevChainInertBound (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCTilesDevChainInertBound (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((2950127 / 829250000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.SampledSm120v2ChainInertShare1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevChainInertBoundW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevChainInertBound (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCTilesDevChainInertBound (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.Guarantees.PearlC

open Pouw.PearlC Pouw.PearlC.Assumptions

def GapSm120v1LoopCast8p72Rev1InertShare1000_n6144k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 6144 4096)
      ((851373343 / 226641700000 : ℚ) : ℝ) δPearlC

def GapSm120v1LoopCast8p72Rev1InertShare1000_n4096k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 4096)
      ((295006781 / 75957050000 : ℚ) : ℝ) δPearlC

def GapSm120v1LoopCast8p72Rev1InertShare1000_n28672k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 28672 4096)
      ((1863165467 / 524322650000 : ℚ) : ℝ) δPearlC

def GapSm120v1LoopCast8p72Rev1InertShare1000_n4096k14336 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 14336)
      ((3045273083 / 782596025000 : ℚ) : ℝ) δPearlC

def GapSm120v1LoopCast8p72Rev1InertShare1000_llama31_8b : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120Loop) llama31_8bLinears)
      ((3045273083 / 782596025000 : ℚ) : ℝ) δPearlC

def SampledSm120v1LoopCast8p72Rev1InertShare1000_n6144k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 6144 4096)
      ((851373343 / 226641700000 : ℚ) : ℝ) δPearlC

def SampledSm120v1LoopCast8p72Rev1InertShare1000_n4096k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 4096)
      ((295006781 / 75957050000 : ℚ) : ℝ) δPearlC

def SampledSm120v1LoopCast8p72Rev1InertShare1000_n28672k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 28672 4096)
      ((1863165467 / 524322650000 : ℚ) : ℝ) δPearlC

def SampledSm120v1LoopCast8p72Rev1InertShare1000_n4096k14336 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 14336)
      ((3045273083 / 782596025000 : ℚ) : ℝ) δPearlC

def SampledSm120v1LoopCast8p72Rev1InertShare1000_llama31_8b : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120Loop) llama31_8bLinears)
      ((3045273083 / 782596025000 : ℚ) : ℝ) δPearlC

def GapSm120v1Cast8p72Rev1InertShare1000_n6144k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 6144 4096)
      ((508661 / 135634400 : ℚ) : ℝ) δPearlC

def GapSm120v1Cast8p72Rev1InertShare1000_n4096k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 4096 4096)
      ((1359 / 350750 : ℚ) : ℝ) δPearlC

def GapSm120v1Cast8p72Rev1InertShare1000_n28672k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 28672 4096)
      ((928813 / 261481000 : ℚ) : ℝ) δPearlC

def GapSm120v1Cast8p72Rev1InertShare1000_n4096k14336 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 4096 14336)
      ((106307 / 27386000 : ℚ) : ℝ) δPearlC

def GapSm120v1Cast8p72Rev1InertShare1000_llama31_8b : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120) llama31_8bLinears)
      ((106307 / 27386000 : ℚ) : ℝ) δPearlC

def SampledSm120v1Cast8p72Rev1InertShare1000_n6144k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 6144 4096)
      ((508661 / 135634400 : ℚ) : ℝ) δPearlC

def SampledSm120v1Cast8p72Rev1InertShare1000_n4096k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 4096 4096)
      ((1359 / 350750 : ℚ) : ℝ) δPearlC

def SampledSm120v1Cast8p72Rev1InertShare1000_n28672k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 28672 4096)
      ((928813 / 261481000 : ℚ) : ℝ) δPearlC

def SampledSm120v1Cast8p72Rev1InertShare1000_n4096k14336 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 4096 14336)
      ((106307 / 27386000 : ℚ) : ℝ) δPearlC

def SampledSm120v1Cast8p72Rev1InertShare1000_llama31_8b : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120) llama31_8bLinears)
      ((106307 / 27386000 : ℚ) : ℝ) δPearlC

end Pouw.Guarantees.PearlC
