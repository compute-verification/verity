module

public import Definitions.Pouw.Advantage
public import Definitions.Pouw.PearlC.DeviceKernelWref
public import Definitions.Pouw.PearlC.DevicePricesLoop
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
public import Definitions.Pouw.PearlC.DevicePrices
public import Definitions.Pouw.Assumptions.PearlC.DeviceChainCap
public import Definitions.Pouw.Assumptions.PearlC.TTOutChainOnly
public import Definitions.Pouw.Assumptions.PearlC.TTOutChainOnlyW1

/-!
# Pearl-C's statements at device records: lemmas

Pearl-C's gap statements at a named device record (`devSm120v1`, `devSm120v2`, `devH100`, `devFp4Sm120`) and its prices,
each a `def G : Prop` under the guarantees' namespace, proved by the theorem `Pouw.SecurityProofs.PearlC.G` and audited
with the proofs: no record pins it. The listed statements are the same forms over every device record
(`Definitions.Pouw.Guarantees.PearlCDev`); a statement here is one of them at a record's numbers, or an earlier revision
or comparison (v2 without promotion, other casts, chain-only, chain-cap, U-only, Pearl-C4, the H100).

**At W1.** A statement that rests on a device TT_OUT form (`TTOutPearlCDev`, `TTOutPearlCDevRev1`, their tile forms and
the H100's; the chain-only, chain-cap and U-only forms) is stated at the cost model `W1.w1Priced dv salt`, the W1
machine on the record's device `dv` (`W1Device.sm120`, `.sm120Loop` or `.h100` at prices `sha` and `fetch`) with the
salt's bytes `salt`, and rests on that form at W1 (`Definitions.Pouw.Assumptions.PearlC.TTOutW1`,
`Definitions.Pouw.Assumptions.PearlC.TTOutChainOnlyW1`, and the chain-cap and U-only forms' `…W1` beside them).
Pearl-C4's rest on `TTOutFp4Sm120W1` or `TTOutTileFp4Sm120W1` and conclude at
`W1.w1Priced (W1Device.fp4Sm120 sha fetch) salt` (`Definitions.Pouw.Assumptions.PearlC.TTOutFp4W1`). The rest (the
row-seed readers) are still stated over any `CostModel`. `EpsFp4Sm120_8192` is Pearl-C4's gap at 8192³ in ε form.
The hidden-audit form at the served record is `Proofs.Pouw.Lemmas.PearlCFlockTile`'s.
-/

@[expose] public section

namespace Pouw.Guarantees.PearlC

open Pouw.PearlC Pouw.PearlC.Assumptions

/-- **v1 under rev1 at the cap 1/1,000, 8192³**, FP32 at 8.376, the honest cast at 8.72:
`g = 1648180439/446072750000` (0.36949%). -/
def GapSm120v1LoopCast8p72Rev1Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
      ((1648180439 / 446072750000 : ℚ) : ℝ) δPearlC

end Pouw.Guarantees.PearlC

namespace Pouw.PearlC
open Pouw.PearlC.Assumptions

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.PearlC.Assumptions

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120, 16384³**, worst case under the cap `1/400`: `g = 5473473967/895794400000` (0.61102%).
It holds only for noise the domain's rule allows (D-SK, `Fp4Sem.SaltKeyedB`, in `pearlCDomainFp4`), and says nothing yet
about the deployed NVFP4 noise, whose lines are SHA-512 of the salt: there the domain is empty
(`Definitions.Pouw.Assumptions.PearlC.TTOutFp4W1`). -/
def _root_.Pouw.Guarantees.PearlC.GapFp4Sm120_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : Fp4Sem Q R S) (hTT : TTOutFp4Sm120W1 sha fetch salt sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCDomainFp4At sem sh16384)
    ((5473473967 / 895794400000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120, 8192³**, worst case under the cap `1/400`: `g = 9875013901/1376663200000` (0.71732%).
It holds only for noise the domain's rule allows (D-SK, `Fp4Sem.SaltKeyedB`, in `pearlCDomainFp4`), and says nothing yet
about the deployed NVFP4 noise, whose lines are SHA-512 of the salt: there the domain is empty
(`Definitions.Pouw.Assumptions.PearlC.TTOutFp4W1`). -/
def _root_.Pouw.Guarantees.PearlC.GapFp4Sm120_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : Fp4Sem Q R S) (hTT : TTOutFp4Sm120W1 sha fetch salt sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCDomainFp4At sem sh8192)
    ((9875013901 / 1376663200000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v1 under rev1 at the cap 1/1,000, 16384³**, FP32 at 8.376, the honest cast at 8.72:
`g = 1058166877/294158650000` (0.35973%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
    ((1058166877 / 294158650000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120 per audit tile, 16384³**: `GGapSampled` at `g = 5473473967/895794400000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledFp4Sm120_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : Fp4Sem Q R S) (hTT : TTOutTileFp4Sm120W1 sha fetch salt sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCTilesFp4 devFp4Sm120 sem (1 / 400))
    (pearlCDomainFp4At sem sh16384) ((5473473967 / 895794400000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120 per audit tile, 8192³**: `GGapSampled` at `g = 9875013901/1376663200000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledFp4Sm120_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : Fp4Sem Q R S) (hTT : TTOutTileFp4Sm120W1 sha fetch salt sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCTilesFp4 devFp4Sm120 sem (1 / 400))
    (pearlCDomainFp4At sem sh8192) ((9875013901 / 1376663200000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v1 under rev1 per audit tile at the cap 1/1,000, 16384³**, FP32 at 8.376, the honest cast at 8.72:
`g = 1058166877/294158650000` (0.35973%). -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
    ((1058166877 / 294158650000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v1 under rev1 per audit tile at the cap 1/1,000, 8192³**, FP32 at 8.376, the honest cast at 8.72:
`g = 1648180439/446072750000` (0.36949%). -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_8192 : Prop :=
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

/-- **The H100 headline under rev1, 16384³**, worst case under the cap `1/400`: `g = 89359/17230000` (0.51862%). -/
def _root_.Pouw.Guarantees.PearlC.GapH100Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCH100Rev1W1 sha fetch salt sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.h100 sha fetch) salt) (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCDomainDevAt devH100 sh16384)
    ((89359 / 17230000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **The H100 headline under rev1, 8192³**, worst case under the cap `1/400`: `g = 140237/26090000` (0.53751%). -/
def _root_.Pouw.Guarantees.PearlC.GapH100Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCH100Rev1W1 sha fetch salt sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.h100 sha fetch) salt) (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCDomainDevAt devH100 sh8192)
    ((140237 / 26090000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 8192³**, FP32 at 8.00, the honest cast at 16: `g = 133561/22262500` (0.59994%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1Cast16Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (16 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((133561 / 22262500 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, chain-only, 8192³**, FP32 at 8.00, the honest cast at 32.06: `g = 273461/22302650` (1.22614%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1Cast32p06ChainOnlyRev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1ChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((273461 / 22302650 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 8192³**, FP32 at 8.00, the honest cast at 32.06: `g = 173711/22302650` (0.77888%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1Cast32p06Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((173711 / 22302650 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, chain-only, 8192³**, FP32 at 8.00, the honest cast at 8.72: `g = 16547/1711100` (0.96704%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1Cast8p72ChainOnlyRev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1ChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16547 / 1711100 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1 at the cap 1/1,000, 16384³**, FP32 at 8.00, the honest cast at 8.72:
`g = 5273/1466810` (0.35949%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1Cast8p72Rev1Cap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384)
    ((5273 / 1466810 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1 at the cap 1/1,000, 8192³**, FP32 at 8.00, the honest cast at 8.72:
`g = 205211/55610750` (0.36901%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1Cast8p72Rev1Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192)
    ((205211 / 55610750 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 16384³**, FP32 at 8.00, the honest cast at 8.72: `g = 2987/586724` (0.50910%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1Cast8p72Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2987 / 586724 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 8192³**, FP32 at 8.00, the honest cast at 8.72: `g = 115361/22244300` (0.51861%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1Cast8p72Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((115361 / 22244300 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 8192³**, FP32 at 8.376, the honest cast at 16: `g = 3277657/546100000` (0.60019%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast16Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (16 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((3277657 / 546100000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, chain-only, 8192³**, FP32 at 8.376, the honest cast at 32.06: `g = 2190993839/178895900000`
(1.22473%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast32p06ChainOnlyRev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1ChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((2190993839 / 178895900000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 8192³**, FP32 at 8.376, the honest cast at 32.06: `g = 1392993839/178895900000` (0.77866%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast32p06Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((1392993839 / 178895900000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 8192³, the exact in-loop value**: FP32 at 8.376, the statement's cast at 8.0 and the A-only
forming at its measured 1.047, `g = 911793839/178414700000` (0.51105%). `pearlCGapSm120v1LoopRev1_8192` bounds it
from above (`qa` rounded up to 2). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast8Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (8 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((911793839 / 178414700000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, chain-only, 8192³**, FP32 at 8.376, the honest cast at 8.72: `g = 1724193839/178429100000`
(0.96632%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72ChainOnlyRev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1ChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((1724193839 / 178429100000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v1 under rev1 at the cap 1/1,000 from rev1 at 1/400, 16384³**, FP32 at 8.376, the honest cast at 8.72:
`g = 1058166877/294158650000` (0.35973%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_16384OfCap400 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
    ((1058166877 / 294158650000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v1 under rev1 at the cap 1/1,000 from rev1 at 1/400, 8192³**, FP32 at 8.376, the honest cast at 8.72:
`g = 1648180439/446072750000` (0.36949%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_8192OfCap400 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
    ((1648180439 / 446072750000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 16384³**, FP32 at 8.376, the honest cast at 8.72: `g = 599303077/117663460000` (0.50934%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((599303077 / 117663460000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 8192³**, FP32 at 8.376, the honest cast at 8.72: `g = 926193839/178429100000` (0.51908%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((926193839 / 178429100000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1 at 16384³**: `g = 2963/586700`. -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2963 / 586700 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1 at 8192³**: `g = 16223/3177500`. -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v1Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16223 / 3177500 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000 at 16384³**: `g = 2950127/829250000`. -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v2Cap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((2950127 / 829250000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000 at 8192³**: `g = 216793/59950000`. -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v2Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000, 8192³**, FP32 at 8.00, the honest cast at 16: `g = 1917551/420050000` (0.45651%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v2Cast16Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (16 - 8))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((1917551 / 420050000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000, 8192³**, FP32 at 8.00, the honest cast at 8.72: `g = 1553551/419686000` (0.37017%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v2Cast8p72Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((1553551 / 419686000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000, chain-only, 8192³**, FP32 at 8.00, the honest cast at 8.72: `g = 3548551/419686000`
(0.84553%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v2Cast8p72ChainOnlyCap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((3548551 / 419686000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the chain cap 1/1,000 at 16384³**, per unit: `g = 367519/103656250`. -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v2ChainCap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevChainCapW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((367519 / 103656250 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the chain cap 1/1,000 at 8192³**, per unit: `g = 26921/7493750`. -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v2ChainCap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevChainCapW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((26921 / 7493750 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000, 8192³**, FP32 at 8.376, the honest cast at 16: `g = 639967/140017450` (0.45706%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v2LoopCast16Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v2 Prices.sm120Loop) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (16 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((639967 / 140017450 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000, 8192³, the exact in-loop value**: FP32 at 8.376, the statement's cast at 8.0 and the
A-only forming at its measured 1.047, `g = 1519901/419652350` (0.36218%). `pearlCGapSm120v2LoopCap1000_8192` bounds
it from above (`qa` rounded up to 2). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v2LoopCast8Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v2 Prices.sm120Loop) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((1519901 / 419652350 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000, 8192³**, FP32 at 8.376, the honest cast at 8.72: `g = 1555901/419688350` (0.37073%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v2LoopCast8p72Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v2 Prices.sm120Loop) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((1555901 / 419688350 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000, chain-only, 8192³**, FP32 at 8.376, the honest cast at 8.72: `g = 3550901/419688350`
(0.84608%). -/
def _root_.Pouw.Guarantees.PearlC.GapSm120v2LoopCast8p72ChainOnlyCap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v2 Prices.sm120Loop) sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((3550901 / 419688350 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v1 under rev1 with U-only binding, 8192³**, at sm_120's prices, cap 1/400: `g = 16223/3177500`. -/
def _root_.Pouw.Guarantees.PearlC.GapUOnlySm120Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4) (hp : d.prices = Prices.sm120) (hTT : TTOutPearlCDevUOnlyW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt d sem (1 / 400)),
    GGapU (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCDomainDevAt d sh8192) ((16223 / 3177500 : ℚ) : ℝ)
    δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v2 at the cap 1/1,000 with U-only binding, 8192³**, at sm_120's prices: `g = 216793/59950000`. -/
def _root_.Pouw.Guarantees.PearlC.GapUOnlyUnpromotedCap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120) (hTT : TTOutPearlCDevUOnlyW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt d sem (1 / 1000)),
    GGapU (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 d sem (1 / 1000)) (pearlCDomainDevAt d sh8192) ((216793 / 59950000 : ℚ) : ℝ)
    δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v2 at the cap 1/1,000, 8192³**, at a record with no promotion and sm_120's prices: `g = 216793/59950000`. -/
def _root_.Pouw.Guarantees.PearlC.GapUnpromotedCap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt d sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDev d sem (1 / 1000)) (pearlCDomainDevAt d sh8192) ((216793 / 59950000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v2 at the chain cap 1/1,000, 8192³**, per unit, at a record with no promotion and sm_120's prices:
`g = 26921/7493750`. -/
def _root_.Pouw.Guarantees.PearlC.GapUnpromotedChainCap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120) (hTT : TTOutPearlCDevChainCapW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt d sem (1 / 1000)),
    GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevChainCap d sem (1 / 1000)) (pearlCDomainDevAt d sh8192) ((26921 / 7493750 : ℚ) : ℝ)
    δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **The H100 headline under rev1 per audit tile, 16384³**: `GGapSampled` at `g = 89359/17230000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledH100Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCH100Rev1W1 sha fetch salt sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.h100 sha fetch) salt) (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCTilesDevRev1 devH100 sem (1 / 400))
    (pearlCDomainDevAt devH100 sh16384) ((89359 / 17230000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **The H100 headline under rev1 per audit tile, 8192³**: `GGapSampled` at `g = 140237/26090000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledH100Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCH100Rev1W1 sha fetch salt sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.h100 sha fetch) salt) (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCTilesDevRev1 devH100 sem (1 / 400))
    (pearlCDomainDevAt devH100 sh8192) ((140237 / 26090000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1 per audit tile at the cap 1/1,000, 16384³**, FP32 at 8.00, the honest cast at 8.72:
`g = 5273/1466810` (0.35949%). -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384)
    ((5273 / 1466810 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1 per audit tile at the cap 1/1,000, 8192³**, FP32 at 8.00, the honest cast at 8.72:
`g = 205211/55610750` (0.36901%). -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192)
    ((205211 / 55610750 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, per audit tile, 16384³**, FP32 at 8.00, the honest cast at 8.72: `g = 2987/586724` (0.50910%). -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2987 / 586724 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, per audit tile, 8192³**, FP32 at 8.00, the honest cast at 8.72: `g = 115361/22244300` (0.51861%).
-/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((115361 / 22244300 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, per audit tile, 16384³**, FP32 at 8.376, the honest cast at 8.72: `g = 599303077/117663460000`
(0.50934%). -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((599303077 / 117663460000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, per audit tile, 8192³**, FP32 at 8.376, the honest cast at 8.72: `g = 926193839/178429100000`
(0.51908%). -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((926193839 / 178429100000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1 per audit tile at 16384³**: `g = 2963/586700`. -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCTilesDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2963 / 586700 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1 per audit tile at 8192³**: `g = 16223/3177500`. -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCTilesDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16223 / 3177500 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000 per audit tile at 16384³**: `g = 2950127/829250000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v2Cap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCTilesDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((2950127 / 829250000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000 per audit tile at 8192³**: `g = 216793/59950000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v2Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCTilesDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the chain cap 1/1,000 per audit tile at 16384³**: `g = 2950127/829250000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v2ChainCap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevChainCapW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCTilesDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((2950127 / 829250000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the chain cap 1/1,000 per audit tile at 8192³**: `g = 216793/59950000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v2ChainCap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevChainCapW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCTilesDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) δPearlC

end Pouw.PearlC

namespace Pouw.Guarantees.PearlC

open Pouw.PearlC Pouw.PearlC.Assumptions

/-! ## Llama-3.1-8B's linears at any batch size

The gap at v1's in-loop record under rev1 at the cap 1/1,000 (`Proofs.Pouw.PearlC.ServedRev1Gap`, `ServedMixedRev1Gap`),
per linear and on the served layout, whose g is the max of the four. -/

/-- **v1 under rev1 at the cap 1/1,000, `m × 6144 × 4096` at any `m` (Llama-3.1-8B's qkv)**, FP32 at 8.376, the honest
cast at 8.72: `g = 851373343/226641700000` (0.37565%). -/
def GapSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 6144 4096)
      ((851373343 / 226641700000 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 at the cap 1/1,000, `m × 4096 × 4096` at any `m` (Llama-3.1-8B's o)**, FP32 at 8.376, the honest
cast at 8.72: `g = 295006781/75957050000` (0.38839%). -/
def GapSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 4096)
      ((295006781 / 75957050000 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 at the cap 1/1,000, `m × 28672 × 4096` at any `m` (Llama-3.1-8B's gate_up)**, FP32 at 8.376, the
honest cast at 8.72: `g = 1863165467/524322650000` (0.35535%). -/
def GapSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 28672 4096)
      ((1863165467 / 524322650000 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 at the cap 1/1,000, `m × 4096 × 14336` at any `m` (Llama-3.1-8B's down)**, FP32 at 8.376, the honest
cast at 8.72: `g = 3045273083/782596025000` (0.38912%). -/
def GapSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 14336)
      ((3045273083 / 782596025000 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 at the cap 1/1,000 on Llama-3.1-8B's served layout**, its four linears at any `m` in one layout,
FP32 at 8.376, the honest cast at 8.72: `g = 3045273083/782596025000` (0.38912%), the down projection's, the max of the
four. -/
def GapSm120v1LoopCast8p72Rev1Cap1000_llama31_8b : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120Loop) llama31_8bLinears)
      ((3045273083 / 782596025000 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 6144 × 4096` at any `m` (Llama-3.1-8B's qkv)**, FP32 at
8.376, the honest cast at 8.72: `g = 851373343/226641700000` (0.37565%). -/
def SampledSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 6144 4096)
      ((851373343 / 226641700000 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 4096 × 4096` at any `m` (Llama-3.1-8B's o)**, FP32 at 8.376,
the honest cast at 8.72: `g = 295006781/75957050000` (0.38839%). -/
def SampledSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 4096)
      ((295006781 / 75957050000 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 28672 × 4096` at any `m` (Llama-3.1-8B's gate_up)**, FP32 at
8.376, the honest cast at 8.72: `g = 1863165467/524322650000` (0.35535%). -/
def SampledSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 28672 4096)
      ((1863165467 / 524322650000 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 4096 × 14336` at any `m` (Llama-3.1-8B's down)**, FP32 at
8.376, the honest cast at 8.72: `g = 3045273083/782596025000` (0.38912%). -/
def SampledSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 14336)
      ((3045273083 / 782596025000 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000 on Llama-3.1-8B's served layout**, its four linears at any `m` in
one layout, FP32 at 8.376, the honest cast at 8.72: `g = 3045273083/782596025000` (0.38912%), the down projection's, the
max of the four. -/
def SampledSm120v1LoopCast8p72Rev1Cap1000_llama31_8b : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120Loop) llama31_8bLinears)
      ((3045273083 / 782596025000 : ℚ) : ℝ) δPearlC

/-! ## Llama-3.1-8B's linears with FP32 at 8

The same g at v1's record with the cheapest bit-exact FP32 add, 8 (`Proofs.Pouw.PearlC.ServedFadd8Rev1Gap`), the
twin the ledger reads beside the in-loop one. Each is below the in-loop g at the same linear. -/

/-- **v1 under rev1 at the cap 1/1,000, `m × 6144 × 4096` at any `m` (Llama-3.1-8B's qkv)**, FP32 at 8, the honest cast at 8.72:
`g = 508661/135634400` (0.37502%). -/
def GapSm120v1Cast8p72Rev1Cap1000_n6144k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 6144 4096)
      ((508661 / 135634400 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 at the cap 1/1,000, `m × 4096 × 4096` at any `m` (Llama-3.1-8B's o)**, FP32 at 8, the honest cast at 8.72:
`g = 1359/350750` (0.38746%). -/
def GapSm120v1Cast8p72Rev1Cap1000_n4096k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 4096 4096)
      ((1359 / 350750 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 at the cap 1/1,000, `m × 28672 × 4096` at any `m` (Llama-3.1-8B's gate_up)**, FP32 at 8, the honest cast at 8.72:
`g = 928813/261481000` (0.35521%). -/
def GapSm120v1Cast8p72Rev1Cap1000_n28672k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 28672 4096)
      ((928813 / 261481000 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 at the cap 1/1,000, `m × 4096 × 14336` at any `m` (Llama-3.1-8B's down)**, FP32 at 8, the honest cast at 8.72:
`g = 106307/27386000` (0.38818%). -/
def GapSm120v1Cast8p72Rev1Cap1000_n4096k14336 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 4096 14336)
      ((106307 / 27386000 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 at the cap 1/1,000, Llama-3.1-8B's served layout, its four linears at any `m` in one layout**, FP32 at 8, the honest cast at 8.72:
`g = 106307/27386000` (0.38818%), the down projection's, the max of the four. -/
def GapSm120v1Cast8p72Rev1Cap1000_llama31_8b : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120) llama31_8bLinears)
      ((106307 / 27386000 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 6144 × 4096` at any `m` (Llama-3.1-8B's qkv)**, FP32 at 8, the honest cast at 8.72:
`g = 508661/135634400` (0.37502%). -/
def SampledSm120v1Cast8p72Rev1Cap1000_n6144k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 6144 4096)
      ((508661 / 135634400 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 4096 × 4096` at any `m` (Llama-3.1-8B's o)**, FP32 at 8, the honest cast at 8.72:
`g = 1359/350750` (0.38746%). -/
def SampledSm120v1Cast8p72Rev1Cap1000_n4096k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 4096 4096)
      ((1359 / 350750 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 28672 × 4096` at any `m` (Llama-3.1-8B's gate_up)**, FP32 at 8, the honest cast at 8.72:
`g = 928813/261481000` (0.35521%). -/
def SampledSm120v1Cast8p72Rev1Cap1000_n28672k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 28672 4096)
      ((928813 / 261481000 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 4096 × 14336` at any `m` (Llama-3.1-8B's down)**, FP32 at 8, the honest cast at 8.72:
`g = 106307/27386000` (0.38818%). -/
def SampledSm120v1Cast8p72Rev1Cap1000_n4096k14336 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 4096 14336)
      ((106307 / 27386000 : ℚ) : ℝ) δPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, Llama-3.1-8B's served layout, its four linears at any `m` in one layout**, FP32 at 8, the honest cast at 8.72:
`g = 106307/27386000` (0.38818%), the down projection's, the max of the four. -/
def SampledSm120v1Cast8p72Rev1Cap1000_llama31_8b : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120) llama31_8bLinears)
      ((106307 / 27386000 : ℚ) : ℝ) δPearlC

end Pouw.Guarantees.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120, 8192³, in ε form**: `GapFp4Sm120_8192` at the advantage
`ε = g/(1 − g) = 9875013901/1366788186099` (0.72250%). Like `GapFp4Sm120_8192`, it holds only for noise the domain's
rule allows (D-SK, `Fp4Sem.SaltKeyedB`) and says nothing yet about the deployed NVFP4 noise, at which the domain is
empty. -/
def _root_.Pouw.Guarantees.PearlC.EpsFp4Sm120_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : Fp4Sem Q R S),
    TTOutFp4Sm120W1 sha fetch salt sem (1 / 400) →
    Pouw.PearlC.Gε (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCDomainFp4At sem sh8192)
    ((9875013901 / 1366788186099 : ℚ) : ℝ) δPearlC

end Pouw.PearlC
