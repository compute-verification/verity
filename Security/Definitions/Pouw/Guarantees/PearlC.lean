import Definitions.Pouw.PearlC.DeviceKernelWref
import Definitions.Pouw.PearlC.DevicePricesLoop
import Definitions.Pouw.Assumptions.PearlC.TTOutW1
import Definitions.Pouw.Assumptions.PearlC.TTOutFp4W1
import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
import Definitions.Pouw.PearlC.Device
import Definitions.Pouw.PearlC.DeviceFp4
import Definitions.Pouw.PearlC.DeviceFp4Issue
import Definitions.Pouw.PearlC.DeviceRev1
import Definitions.Pouw.PearlC.Game
import Definitions.Pouw.PearlC.Instance
import Definitions.Pouw.PearlC.Tile
import Definitions.Pouw.Assumptions.PearlC.RowSeedAssumptions
import Definitions.Pouw.Assumptions.PearlC.TTOut
import Definitions.Pouw.Assumptions.PearlC.TTOutFp4
import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeed
import Definitions.Pouw.Assumptions.PearlC.TTOutUOnly
import Definitions.Pouw.PearlC.DeviceChainOnly
import Definitions.Pouw.PearlC.DeviceFp4ChainOnly
import Definitions.Pouw.PearlC.DevicePrices
import Definitions.Pouw.Assumptions.PearlC.DeviceChainCap
import Definitions.Pouw.Assumptions.PearlC.TTOutChainOnly
import Definitions.Pouw.Assumptions.PearlC.TTOutChainOnlyW1

/-!
# Pearl-C's guarantees (trusted)

Each guarantee is a `def G : Prop`, stated over `Definitions.Pouw`'s definitions with the named assumptions it rests on as
hypotheses (`Pouw.Assumptions`). Its proof is the theorem `Pouw.SecurityProofs.PearlC.G : Pouw.Guarantees.PearlC.G`.
γ at the RTX PRO 6000's v1 record under rev1 at the cap 1/1,000 (`Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_8192`
in `Pouw.SecurityProofs.PearlC.DeviceCapRev1Gamma`), its binders written as a `∀`. `Pouw.CostModel` and `Pouw.Gγ` are
the game's, so Pearl-C's are named in full.

**At W1.** A guarantee that rests on a device TT_OUT form (`TTOutPearlCDev`, `TTOutPearlCDevRev1`, their tile forms and
the H100's; the chain-only, chain-cap and U-only forms) is stated at the cost model `W1.w1Priced dv salt`, the W1
machine on the record's device `dv` (`W1Device.sm120`, `.sm120Loop` or `.h100` at prices `sha` and `fetch`) with the
salt's bytes `salt`, and rests on that form at W1 (`Definitions.Pouw.Assumptions.PearlC.TTOutW1`,
`Definitions.Pouw.Assumptions.PearlC.TTOutChainOnlyW1`, and the chain-cap and U-only forms' `…W1` beside them).
Pearl-C4's headline readers rest on `TTOutFp4Sm120W1` or `TTOutTileFp4Sm120W1` and conclude at
`W1.w1Priced (W1Device.fp4Sm120 sha fetch) salt` (`Definitions.Pouw.Assumptions.PearlC.TTOutFp4W1`). The rest (the
row-seed readers) are still stated over any `CostModel`.
-/

namespace Pouw.Guarantees.PearlC

open Pouw.PearlC Pouw.PearlC.Assumptions

/-- **v1 under rev1 at the cap 1/1,000, 8192³**, FP32 at 8.376, the honest cast at 8.72:
`γ = 1648180439/446072750000` (0.36949%). -/
def GammaSm120v1LoopCast8p72Rev1Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
      ((1648180439 / 446072750000 : ℚ) : ℝ) εPearlC

end Pouw.Guarantees.PearlC

namespace Pouw.PearlC
open Pouw.PearlC.Assumptions

/-- Pearl-C4's γ at FP32 `1047/125` = 8.376, the `lut256` scale path, 16384³, cap `1/400`: `5473473967/895794400000`
(0.61102%). -/
def _root_.Pouw.Guarantees.PearlC.GammaFp4Sm120LoopLut256_16384 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256),
    gammaFp4 d (1 / 400) sh16384 = 5473473967 / 895794400000

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.PearlC.Assumptions

/-- Pearl-C4's γ at FP32 `1047/125` = 8.376, the `lut256` scale path, 8192³, cap `1/400`: `9875013901/1376663200000`
(0.71732%). -/
def _root_.Pouw.Guarantees.PearlC.GammaFp4Sm120LoopLut256_8192 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256),
    gammaFp4 d (1 / 400) sh8192 = 9875013901 / 1376663200000

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120, 16384³**, worst case under the cap `1/400`: `γ = 5473473967/895794400000` (0.61102%). -/
def _root_.Pouw.Guarantees.PearlC.GammaFp4Sm120_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : Fp4Sem Q R S) (hTT : TTOutFp4Sm120W1 sha fetch salt sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCDomainFp4At sem sh16384)
    ((5473473967 / 895794400000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120, 8192³**, worst case under the cap `1/400`: `γ = 9875013901/1376663200000` (0.71732%). -/
def _root_.Pouw.Guarantees.PearlC.GammaFp4Sm120_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : Fp4Sem Q R S) (hTT : TTOutFp4Sm120W1 sha fetch salt sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCDomainFp4At sem sh8192)
    ((9875013901 / 1376663200000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v1 under rev1 at the cap 1/1,000, 16384³**, FP32 at 8.376, the honest cast at 8.72:
`γ = 1058166877/294158650000` (0.35973%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
    ((1058166877 / 294158650000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120 per audit tile, 16384³**: `GγSampled` at `γ = 5473473967/895794400000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledFp4Sm120_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : Fp4Sem Q R S) (hTT : TTOutTileFp4Sm120W1 sha fetch salt sem (1 / 400)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCTilesFp4 devFp4Sm120 sem (1 / 400))
    (pearlCDomainFp4At sem sh16384) ((5473473967 / 895794400000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120 per audit tile, 8192³**: `GγSampled` at `γ = 9875013901/1376663200000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledFp4Sm120_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : Fp4Sem Q R S) (hTT : TTOutTileFp4Sm120W1 sha fetch salt sem (1 / 400)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCTilesFp4 devFp4Sm120 sem (1 / 400))
    (pearlCDomainFp4At sem sh8192) ((9875013901 / 1376663200000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v1 under rev1 per audit tile at the cap 1/1,000, 16384³**, FP32 at 8.376, the honest cast at 8.72:
`γ = 1058166877/294158650000` (0.35973%). -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
    ((1058166877 / 294158650000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v1 under rev1 per audit tile at the cap 1/1,000, 8192³**, FP32 at 8.376, the honest cast at 8.72:
`γ = 1648180439/446072750000` (0.36949%). -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
    ((1648180439 / 446072750000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **Row additivity for P2 (a theorem).** rev1's TT_OUT at `sem` bounds the row-seeded game at `sem'`, with the cap
checked per row: the row-seeded adversary is the split game's, relabelled row by row (C3), each row of a correct unit is
a correct one-row unit whose credit is the row's (C0, C1), and the error is the split layout's, at `R = Σ_u m_u` units.
One-row units are in the domain (`pearlCDomainDev_split`). -/
def _root_.Pouw.Guarantees.PearlC.TTOutRowSeedOfTTOut : Prop :=
  ∀ {Q R S : Type} {sem sem' : PearlCSem Q R S} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S} {d : PearlCDevice} {ρ : ℚ} (hTT : TTOutPearlCDevRev1 CM d sem ρ) (hN : RowSeedNoise sem sem') (hCM : CM.SplitClosed) (hd : d.RowLocal) (hρ : ρ ≤ 1) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16),
    TTOutRowSeed CM (pearlCProtocolDevRev1RowCap d sem' ρ) (pearlCDomainDev d) (1 / 400)

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions

/-- Pearl-C4's chain-only γ at FP32 8.00, the `lut256` scale path, 8192³, cap `1/400`:
`17611477/917600000` (1.91930%), with the chain-only worst case positive. -/
def _root_.Pouw.Guarantees.PearlC.GammaFp4ChainOnlySm120IssueLut256_8192 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256),
    0 < creditFp4ChainOnly d sh8192 - 1 / 400 * creditFp4 d sh8192 ∧
    gammaFp4ChainOnly d (1 / 400) sh8192 = 17611477 / 917600000

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions

/-- Pearl-C4's chain-only γ at FP32 `1047/125` = 8.376, the `lut256` scale path, 8192³, cap `1/400`:
`26680734301/1376663200000` (1.93807%), with the chain-only worst case positive. -/
def _root_.Pouw.Guarantees.PearlC.GammaFp4ChainOnlySm120LoopLut256_8192 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256),
    0 < creditFp4ChainOnly d sh8192 - 1 / 400 * creditFp4 d sh8192 ∧
    gammaFp4ChainOnly d (1 / 400) sh8192 = 26680734301 / 1376663200000

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.PearlC.Assumptions

/-- Pearl-C4's γ at FP32 8.00, the `lut256` scale path, 16384³, cap `1/400`: `32568847/5374240000` (0.60602%). -/
def _root_.Pouw.Guarantees.PearlC.GammaFp4Sm120IssueLut256_16384 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256),
    gammaFp4 d (1 / 400) sh16384 = 32568847 / 5374240000

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.PearlC.Assumptions

/-- Pearl-C4's γ at FP32 8.00, the `lut256` scale path, 8192³, cap `1/400`: `6492677/917600000` (0.70757%). -/
def _root_.Pouw.Guarantees.PearlC.GammaFp4Sm120IssueLut256_8192 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256),
    gammaFp4 d (1 / 400) sh8192 = 6492677 / 917600000

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **The H100 headline under rev1, 16384³**, worst case under the cap `1/400`: `γ = 89359/17230000` (0.51862%). -/
def _root_.Pouw.Guarantees.PearlC.GammaH100Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCH100Rev1W1 sha fetch salt sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.h100 sha fetch) salt) (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCDomainDevAt devH100 sh16384)
    ((89359 / 17230000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **The H100 headline under rev1, 8192³**, worst case under the cap `1/400`: `γ = 140237/26090000` (0.53751%). -/
def _root_.Pouw.Guarantees.PearlC.GammaH100Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCH100Rev1W1 sha fetch salt sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.h100 sha fetch) salt) (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCDomainDevAt devH100 sh8192)
    ((140237 / 26090000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 8192³**, FP32 at 8.00, the honest cast at 16: `γ = 133561/22262500` (0.59994%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1Cast16Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (16 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((133561 / 22262500 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, chain-only, 8192³**, FP32 at 8.00, the honest cast at 32.06: `γ = 273461/22302650` (1.22614%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1Cast32p06ChainOnlyRev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1ChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((273461 / 22302650 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 8192³**, FP32 at 8.00, the honest cast at 32.06: `γ = 173711/22302650` (0.77888%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1Cast32p06Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (1603/50 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((173711 / 22302650 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, chain-only, 8192³**, FP32 at 8.00, the honest cast at 8.72: `γ = 16547/1711100` (0.96704%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72ChainOnlyRev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1ChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16547 / 1711100 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1 at the cap 1/1,000, 16384³**, FP32 at 8.00, the honest cast at 8.72:
`γ = 5273/1466810` (0.35949%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384)
    ((5273 / 1466810 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1 at the cap 1/1,000, 8192³**, FP32 at 8.00, the honest cast at 8.72:
`γ = 205211/55610750` (0.36901%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192)
    ((205211 / 55610750 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 16384³**, FP32 at 8.00, the honest cast at 8.72: `γ = 2987/586724` (0.50910%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2987 / 586724 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 8192³**, FP32 at 8.00, the honest cast at 8.72: `γ = 115361/22244300` (0.51861%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((115361 / 22244300 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 8192³**, FP32 at 8.376, the honest cast at 16: `γ = 3277657/546100000` (0.60019%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1LoopCast16Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (16 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((3277657 / 546100000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, chain-only, 8192³**, FP32 at 8.376, the honest cast at 32.06: `γ = 2190993839/178895900000`
(1.22473%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1LoopCast32p06ChainOnlyRev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1ChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((2190993839 / 178895900000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 8192³**, FP32 at 8.376, the honest cast at 32.06: `γ = 1392993839/178895900000` (0.77866%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1LoopCast32p06Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (1603/50 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((1392993839 / 178895900000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 8192³, the exact in-loop value**: FP32 at 8.376, the statement's cast at 8.0 and the A-only
forming at its measured 1.047, `γ = 911793839/178414700000` (0.51105%). `pearlCGammaSm120v1LoopRev1_8192` bounds it
from above (`qa` rounded up to 2). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (8 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((911793839 / 178414700000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, chain-only, 8192³**, FP32 at 8.376, the honest cast at 8.72: `γ = 1724193839/178429100000`
(0.96632%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72ChainOnlyRev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1ChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((1724193839 / 178429100000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v1 under rev1 at the cap 1/1,000 from rev1 at 1/400, 16384³**, FP32 at 8.376, the honest cast at 8.72:
`γ = 1058166877/294158650000` (0.35973%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_16384OfCap400 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
    ((1058166877 / 294158650000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v1 under rev1 at the cap 1/1,000 from rev1 at 1/400, 8192³**, FP32 at 8.376, the honest cast at 8.72:
`γ = 1648180439/446072750000` (0.36949%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_8192OfCap400 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
    ((1648180439 / 446072750000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 16384³**, FP32 at 8.376, the honest cast at 8.72: `γ = 599303077/117663460000` (0.50934%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((599303077 / 117663460000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, 8192³**, FP32 at 8.376, the honest cast at 8.72: `γ = 926193839/178429100000` (0.51908%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((926193839 / 178429100000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1 at 16384³**: `γ = 2963/586700`. -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2963 / 586700 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1 at 8192³**: `γ = 16223/3177500`. -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16223 / 3177500 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000 at 16384³**: `γ = 2950127/829250000`. -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v2Cap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((2950127 / 829250000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000 at 8192³**: `γ = 216793/59950000`. -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v2Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000, 8192³**, FP32 at 8.00, the honest cast at 16: `γ = 1917551/420050000` (0.45651%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v2Cast16Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (16 - 8))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((1917551 / 420050000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000, 8192³**, FP32 at 8.00, the honest cast at 8.72: `γ = 1553551/419686000` (0.37017%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v2Cast8p72Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((1553551 / 419686000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000, chain-only, 8192³**, FP32 at 8.00, the honest cast at 8.72: `γ = 3548551/419686000`
(0.84553%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v2Cast8p72ChainOnlyCap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120) sem (1 / 1000) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((3548551 / 419686000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the chain cap 1/1,000 at 16384³**, per unit: `γ = 367519/103656250`. -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v2ChainCap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevChainCapW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((367519 / 103656250 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the chain cap 1/1,000 at 8192³**, per unit: `γ = 26921/7493750`. -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v2ChainCap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevChainCapW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((26921 / 7493750 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000, 8192³**, FP32 at 8.376, the honest cast at 16: `γ = 639967/140017450` (0.45706%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v2LoopCast16Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v2 Prices.sm120Loop) sem (1 / 1000)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (16 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((639967 / 140017450 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000, 8192³, the exact in-loop value**: FP32 at 8.376, the statement's cast at 8.0 and the
A-only forming at its measured 1.047, `γ = 1519901/419652350` (0.36218%). `pearlCGammaSm120v2LoopCap1000_8192` bounds
it from above (`qa` rounded up to 2). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v2LoopCast8Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v2 Prices.sm120Loop) sem (1 / 1000)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((1519901 / 419652350 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000, 8192³**, FP32 at 8.376, the honest cast at 8.72: `γ = 1555901/419688350` (0.37073%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v2LoopCast8p72Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v2 Prices.sm120Loop) sem (1 / 1000)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((1555901 / 419688350 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000, chain-only, 8192³**, FP32 at 8.376, the honest cast at 8.72: `γ = 3550901/419688350`
(0.84608%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v2LoopCast8p72ChainOnlyCap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevChainOnlyW1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v2 Prices.sm120Loop) sem (1 / 1000)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((3550901 / 419688350 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v1 under rev1 with U-only binding, 8192³**, at sm_120's prices, cap 1/400: `γ = 16223/3177500`. -/
def _root_.Pouw.Guarantees.PearlC.GammaUOnlySm120Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4) (hp : d.prices = Prices.sm120) (hTT : TTOutPearlCDevUOnlyW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt d sem (1 / 400)),
    GγU (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCDomainDevAt d sh8192) ((16223 / 3177500 : ℚ) : ℝ)
    εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v2 at the cap 1/1,000 with U-only binding, 8192³**, at sm_120's prices: `γ = 216793/59950000`. -/
def _root_.Pouw.Guarantees.PearlC.GammaUOnlyUnpromotedCap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120) (hTT : TTOutPearlCDevUOnlyW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt d sem (1 / 1000)),
    GγU (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 d sem (1 / 1000)) (pearlCDomainDevAt d sh8192) ((216793 / 59950000 : ℚ) : ℝ)
    εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v2 at the cap 1/1,000, 8192³**, at a record with no promotion and sm_120's prices: `γ = 216793/59950000`. -/
def _root_.Pouw.Guarantees.PearlC.GammaUnpromotedCap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120) (hTT : TTOutPearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt d sem (1 / 1000)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDev d sem (1 / 1000)) (pearlCDomainDevAt d sh8192) ((216793 / 59950000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v2 at the chain cap 1/1,000, 8192³**, per unit, at a record with no promotion and sm_120's prices:
`γ = 26921/7493750`. -/
def _root_.Pouw.Guarantees.PearlC.GammaUnpromotedChainCap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120) (hTT : TTOutPearlCDevChainCapW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt d sem (1 / 1000)),
    Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevChainCap d sem (1 / 1000)) (pearlCDomainDevAt d sh8192) ((26921 / 7493750 : ℚ) : ℝ)
    εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **The H100 headline under rev1 per audit tile, 16384³**: `GγSampled` at `γ = 89359/17230000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledH100Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCH100Rev1W1 sha fetch salt sem (1 / 400)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.h100 sha fetch) salt) (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCTilesDevRev1 devH100 sem (1 / 400))
    (pearlCDomainDevAt devH100 sh16384) ((89359 / 17230000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **The H100 headline under rev1 per audit tile, 8192³**: `GγSampled` at `γ = 140237/26090000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledH100Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCH100Rev1W1 sha fetch salt sem (1 / 400)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.h100 sha fetch) salt) (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCTilesDevRev1 devH100 sem (1 / 400))
    (pearlCDomainDevAt devH100 sh8192) ((140237 / 26090000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1 per audit tile at the cap 1/1,000, 16384³**, FP32 at 8.00, the honest cast at 8.72:
`γ = 5273/1466810` (0.35949%). -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384)
    ((5273 / 1466810 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1 per audit tile at the cap 1/1,000, 8192³**, FP32 at 8.00, the honest cast at 8.72:
`γ = 205211/55610750` (0.36901%). -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192)
    ((205211 / 55610750 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, per audit tile, 16384³**, FP32 at 8.00, the honest cast at 8.72: `γ = 2987/586724` (0.50910%). -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2987 / 586724 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, per audit tile, 8192³**, FP32 at 8.00, the honest cast at 8.72: `γ = 115361/22244300` (0.51861%).
-/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 400) (218/25 - 8))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((115361 / 22244300 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, per audit tile, 16384³**, FP32 at 8.376, the honest cast at 8.72: `γ = 599303077/117663460000`
(0.50934%). -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((599303077 / 117663460000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1, per audit tile, 8192³**, FP32 at 8.376, the honest cast at 8.72: `γ = 926193839/178429100000`
(0.51908%). -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 400)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 400) (218/25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((926193839 / 178429100000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1 per audit tile at 16384³**: `γ = 2963/586700`. -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Rev1_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCTilesDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2963 / 586700 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v1 under rev1 per audit tile at 8192³**: `γ = 16223/3177500`. -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1Rev1_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 400)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCTilesDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16223 / 3177500 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000 per audit tile at 16384³**: `γ = 2950127/829250000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v2Cap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCTilesDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((2950127 / 829250000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the cap 1/1,000 per audit tile at 8192³**: `γ = 216793/59950000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v2Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCTilesDev (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the chain cap 1/1,000 per audit tile at 16384³**: `γ = 2950127/829250000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v2ChainCap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevChainCapW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCTilesDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((2950127 / 829250000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- **v2 at the chain cap 1/1,000 per audit tile at 8192³**: `γ = 216793/59950000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v2ChainCap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevChainCapW1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v2 Prices.sm120) sem (1 / 1000)),
    GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCTilesDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
    (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.Guarantees.PearlC

open Pouw.PearlC Pouw.PearlC.Assumptions

/-! ## Llama-3.1-8B's linears at any batch size

γ at v1's in-loop record under rev1 at the cap 1/1,000 (`Proofs.Pouw.PearlC.ServedRev1Gamma`, `ServedMixedRev1Gamma`),
per linear and on the served layout, whose γ is the max of the four. -/

/-- **v1 under rev1 at the cap 1/1,000, `m × 6144 × 4096` at any `m` (Llama-3.1-8B's qkv)**, FP32 at 8.376, the honest
cast at 8.72: `γ = 851373343/226641700000` (0.37565%). -/
def GammaSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 6144 4096)
      ((851373343 / 226641700000 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 at the cap 1/1,000, `m × 4096 × 4096` at any `m` (Llama-3.1-8B's o)**, FP32 at 8.376, the honest
cast at 8.72: `γ = 295006781/75957050000` (0.38839%). -/
def GammaSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 4096)
      ((295006781 / 75957050000 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 at the cap 1/1,000, `m × 28672 × 4096` at any `m` (Llama-3.1-8B's gate_up)**, FP32 at 8.376, the
honest cast at 8.72: `γ = 1863165467/524322650000` (0.35535%). -/
def GammaSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 28672 4096)
      ((1863165467 / 524322650000 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 at the cap 1/1,000, `m × 4096 × 14336` at any `m` (Llama-3.1-8B's down)**, FP32 at 8.376, the honest
cast at 8.72: `γ = 3045273083/782596025000` (0.38912%). -/
def GammaSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 14336)
      ((3045273083 / 782596025000 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 at the cap 1/1,000 on Llama-3.1-8B's served layout**, its four linears at any `m` in one layout,
FP32 at 8.376, the honest cast at 8.72: `γ = 3045273083/782596025000` (0.38912%), the down projection's, the max of the
four. -/
def GammaSm120v1LoopCast8p72Rev1Cap1000_llama31_8b : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120Loop) llama31_8bLinears)
      ((3045273083 / 782596025000 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 6144 × 4096` at any `m` (Llama-3.1-8B's qkv)**, FP32 at
8.376, the honest cast at 8.72: `γ = 851373343/226641700000` (0.37565%). -/
def SampledSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 6144 4096)
      ((851373343 / 226641700000 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 4096 × 4096` at any `m` (Llama-3.1-8B's o)**, FP32 at 8.376,
the honest cast at 8.72: `γ = 295006781/75957050000` (0.38839%). -/
def SampledSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 4096)
      ((295006781 / 75957050000 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 28672 × 4096` at any `m` (Llama-3.1-8B's gate_up)**, FP32 at
8.376, the honest cast at 8.72: `γ = 1863165467/524322650000` (0.35535%). -/
def SampledSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 28672 4096)
      ((1863165467 / 524322650000 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 4096 × 14336` at any `m` (Llama-3.1-8B's down)**, FP32 at
8.376, the honest cast at 8.72: `γ = 3045273083/782596025000` (0.38912%). -/
def SampledSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 14336)
      ((3045273083 / 782596025000 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000 on Llama-3.1-8B's served layout**, its four linears at any `m` in
one layout, FP32 at 8.376, the honest cast at 8.72: `γ = 3045273083/782596025000` (0.38912%), the down projection's, the
max of the four. -/
def SampledSm120v1LoopCast8p72Rev1Cap1000_llama31_8b : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120Loop) llama31_8bLinears)
      ((3045273083 / 782596025000 : ℚ) : ℝ) εPearlC

/-! ## Llama-3.1-8B's linears with FP32 at 8

The same γ at v1's record with the cheapest bit-exact FP32 add, 8 (`Proofs.Pouw.PearlC.ServedFadd8Rev1Gamma`), the
twin the ledger reads beside the in-loop one. Each is below the in-loop γ at the same linear. -/

/-- **v1 under rev1 at the cap 1/1,000, `m × 6144 × 4096` at any `m` (Llama-3.1-8B's qkv)**, FP32 at 8, the honest cast at 8.72:
`γ = 508661/135634400` (0.37502%). -/
def GammaSm120v1Cast8p72Rev1Cap1000_n6144k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 6144 4096)
      ((508661 / 135634400 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 at the cap 1/1,000, `m × 4096 × 4096` at any `m` (Llama-3.1-8B's o)**, FP32 at 8, the honest cast at 8.72:
`γ = 1359/350750` (0.38746%). -/
def GammaSm120v1Cast8p72Rev1Cap1000_n4096k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 4096 4096)
      ((1359 / 350750 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 at the cap 1/1,000, `m × 28672 × 4096` at any `m` (Llama-3.1-8B's gate_up)**, FP32 at 8, the honest cast at 8.72:
`γ = 928813/261481000` (0.35521%). -/
def GammaSm120v1Cast8p72Rev1Cap1000_n28672k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 28672 4096)
      ((928813 / 261481000 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 at the cap 1/1,000, `m × 4096 × 14336` at any `m` (Llama-3.1-8B's down)**, FP32 at 8, the honest cast at 8.72:
`γ = 106307/27386000` (0.38818%). -/
def GammaSm120v1Cast8p72Rev1Cap1000_n4096k14336 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 4096 14336)
      ((106307 / 27386000 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 at the cap 1/1,000, Llama-3.1-8B's served layout, its four linears at any `m` in one layout**, FP32 at 8, the honest cast at 8.72:
`γ = 106307/27386000` (0.38818%), the down projection's, the max of the four. -/
def GammaSm120v1Cast8p72Rev1Cap1000_llama31_8b : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.Gγ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120) llama31_8bLinears)
      ((106307 / 27386000 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 6144 × 4096` at any `m` (Llama-3.1-8B's qkv)**, FP32 at 8, the honest cast at 8.72:
`γ = 508661/135634400` (0.37502%). -/
def SampledSm120v1Cast8p72Rev1Cap1000_n6144k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 6144 4096)
      ((508661 / 135634400 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 4096 × 4096` at any `m` (Llama-3.1-8B's o)**, FP32 at 8, the honest cast at 8.72:
`γ = 1359/350750` (0.38746%). -/
def SampledSm120v1Cast8p72Rev1Cap1000_n4096k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 4096 4096)
      ((1359 / 350750 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 28672 × 4096` at any `m` (Llama-3.1-8B's gate_up)**, FP32 at 8, the honest cast at 8.72:
`γ = 928813/261481000` (0.35521%). -/
def SampledSm120v1Cast8p72Rev1Cap1000_n28672k4096 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 28672 4096)
      ((928813 / 261481000 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 4096 × 14336` at any `m` (Llama-3.1-8B's down)**, FP32 at 8, the honest cast at 8.72:
`γ = 106307/27386000` (0.38818%). -/
def SampledSm120v1Cast8p72Rev1Cap1000_n4096k14336 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120) 4096 14336)
      ((106307 / 27386000 : ℚ) : ℝ) εPearlC

/-- **v1 under rev1 per audit tile at the cap 1/1,000, Llama-3.1-8B's served layout, its four linears at any `m` in one layout**, FP32 at 8, the honest cast at 8.72:
`γ = 106307/27386000` (0.38818%), the down projection's, the max of the four. -/
def SampledSm120v1Cast8p72Rev1Cap1000_llama31_8b : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt (devSm120v1 Prices.sm120) sem (1 / 1000) →
    Pouw.PearlC.GγSampled (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120) sem (1 / 1000) (218 / 25 - 8))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120) llama31_8bLinears)
      ((106307 / 27386000 : ℚ) : ℝ) εPearlC

end Pouw.Guarantees.PearlC
