import Definitions.Pouw.PearlC.DeviceKernelWref
import Definitions.Pouw.PearlC.DevicePricesLoop
import Specs.Pouw.Assumptions.PearlC.TTOutRev1
import Definitions.Pouw.Basic.Prob
import Definitions.Pouw.Basic.QTree
import Definitions.Pouw.Fp8Atom.Atom
import Definitions.Pouw.Fp8Atom.E4M3
import Definitions.Pouw.Fp8Atom.Fp32
import Definitions.Pouw.Game.Defs
import Definitions.Pouw.PearlC.Accum
import Definitions.Pouw.PearlC.Defs
import Definitions.Pouw.PearlC.Device
import Definitions.Pouw.PearlC.DeviceFp4
import Definitions.Pouw.PearlC.DeviceFp4Issue
import Definitions.Pouw.PearlC.DeviceRev1
import Definitions.Pouw.PearlC.FormingPDefs
import Definitions.Pouw.PearlC.Fp4
import Definitions.Pouw.PearlC.Fp4Chain
import Definitions.Pouw.PearlC.Fp4Skip
import Definitions.Pouw.PearlC.Game
import Definitions.Pouw.PearlC.Instance
import Definitions.Pouw.PearlC.PeelExact
import Definitions.Pouw.PearlC.PeelExactP
import Definitions.Pouw.PearlC.SaltDead
import Definitions.Pouw.PearlC.Skip
import Definitions.Pouw.PearlC.SkipP
import Definitions.Pouw.PearlC.Sm120
import Definitions.Pouw.PearlC.Tile
import Specs.Pouw.Assumptions.PearlC.RowSeedAssumptions
import Specs.Pouw.Assumptions.PearlC.TTOut
import Specs.Pouw.Assumptions.PearlC.TTOutFp4
import Specs.Pouw.Assumptions.PearlC.TTOutRowSeed
import Specs.Pouw.Assumptions.PearlC.TTOutUOnly

/-!
# Pearl-C's guarantees (trusted)

Each guarantee is a `def G : Prop`, stated over `Pouw.Protocol`'s definitions with the named assumptions it rests on as
hypotheses (`Pouw.Assumptions`). Its proof is the theorem `Pouw.SecurityProofs.PearlC.G : Pouw.Guarantees.PearlC.G`.
γ at the RTX PRO 6000's v1 record under rev1 at the cap 1/1,000 (`Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_8192`
in `Pouw.SecurityProofs.PearlC.DeviceCapRev1Gamma`), its binders written as a `∀`. `Pouw.CostModel` and `Pouw.Gγ` are
the game's, so Pearl-C's are named in full.
-/

namespace Pouw.Guarantees.PearlC

open Pouw.PearlC Pouw.PearlC.Assumptions

/-- **v1 under rev1 at the cap 1/1,000, 8192³**, FP32 at 8.376, the honest cast at 8.72:
`γ = 1648180439/446072750000` (0.36949%). -/
def GammaSm120v1LoopCast8p72Rev1Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : Pouw.PearlC.CostModel Q R S)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
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
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- `G_γ^U` gives `G_γ` when `W_ref` is non-negative. -/
def _root_.Pouw.Guarantees.PearlC.GammaUImpliesGamma : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ) (hW : ∀ L u, 0 ≤ P.Wref L u) (h : GγU CM P D γ η),
    Gγ CM P D γ η

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120, 16384³**, worst case under the cap `1/400`: `γ = 5473473967/895794400000` (0.61102%). -/
def _root_.Pouw.Guarantees.PearlC.GammaFp4Sm120_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (sem : Fp4Sem Q R S) (hTT : TTOutFp4Sm120 CM sem (1 / 400)),
    Gγ CM (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCDomainFp4At sem sh16384)
    ((5473473967 / 895794400000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120, 8192³**, worst case under the cap `1/400`: `γ = 9875013901/1376663200000` (0.71732%). -/
def _root_.Pouw.Guarantees.PearlC.GammaFp4Sm120_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (sem : Fp4Sem Q R S) (hTT : TTOutFp4Sm120 CM sem (1 / 400)),
    Gγ CM (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCDomainFp4At sem sh8192)
    ((9875013901 / 1376663200000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v1 under rev1 at the cap 1/1,000, 16384³**, FP32 at 8.376, the honest cast at 8.72:
`γ = 1058166877/294158650000` (0.35973%). -/
def _root_.Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (sem : PearlCSem Q R S) (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)),
    Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
    ((1058166877 / 294158650000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120 per audit tile, 16384³**: `GγSampled` at `γ = 5473473967/895794400000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledFp4Sm120_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (sem : Fp4Sem Q R S) (hTT : TTOutTileFp4Sm120 CM sem (1 / 400)),
    GγSampled CM (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCTilesFp4 devFp4Sm120 sem (1 / 400))
    (pearlCDomainFp4At sem sh16384) ((5473473967 / 895794400000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- **Pearl-C4's headline on sm_120 per audit tile, 8192³**: `GγSampled` at `γ = 9875013901/1376663200000`. -/
def _root_.Pouw.Guarantees.PearlC.SampledFp4Sm120_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (sem : Fp4Sem Q R S) (hTT : TTOutTileFp4Sm120 CM sem (1 / 400)),
    GγSampled CM (pearlCProtocolFp4 devFp4Sm120 sem (1 / 400)) (pearlCTilesFp4 devFp4Sm120 sem (1 / 400))
    (pearlCDomainFp4At sem sh8192) ((9875013901 / 1376663200000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v1 under rev1 per audit tile at the cap 1/1,000, 16384³**, FP32 at 8.376, the honest cast at 8.72:
`γ = 1058166877/294158650000` (0.35973%). -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)),
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
    (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
    ((1058166877 / 294158650000 : ℚ) : ℝ) εPearlC

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **v1 under rev1 per audit tile at the cap 1/1,000, 8192³**, FP32 at 8.376, the honest cast at 8.72:
`γ = 1648180439/446072750000` (0.36949%). -/
def _root_.Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)),
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
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
open Finset Pouw.Fp8Atom

/-- **The fragment corollary** (the 16:05Z ruling's item 5), a supporting line: P2 rests on `ttOutRowSeed_of_ttOut`
alone, and this takes no TT_OUT. Under the named assumption `FragDraw` at `Λ = 100`, (C4) and (C6): for every
adversary program `T` with at most `q` oracle calls whose result is a skip program on unit `u`, the probability over
the oracle that the skip program is in the skip class and saves more than `(4096/11)·100` units per row of the unit,
against the honest program on some activation input and state, is at most `(q + m)/2^128`, `m` the unit's rows. -/
def _root_.Pouw.Guarantees.PearlC.TTOutRowSeedSkipClass : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] {CM : CostModel Q R S} {sem : PearlCSem Q R S} {p : Pipeline} {G : ℕ} {s : S} {U : Workload} {u : ℕ} {xJ : Q} {key : ℕ → (Fin (U.layout.shape u).k → ℕ) → Q} (e : SkipEmbed CM sem p G U u) (hD : sem.RowDrawn s U u xJ key) (hF : Assumptions.FragDraw sem p G s U u key 100) (hC4 : CM.FragmentPriced e) (T : QTree Q R SkipProgram) (q : ℕ) (hq : ∀ H, T.calls H ≤ q),
    pr (fun H : Q → R => (T.eval H).InClass sem p G s U u H ∧ ∃ (act : ℕ → Codes) (σ : State),
      4096 / 11 * 100 * ((U.layout.shape u).m : ℝ) <
        CM.cost e.honest H U act σ s - CM.cost (e.embed (T.eval H)) H U act σ s)
    ≤ (q + (U.layout.shape u).m) / 2 ^ 128

end Pouw.PearlC

namespace Pouw.PearlC
open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- **Row additivity for P2 per audit tile (a theorem).** rev1's TT_OUT per tile at `sem` bounds the row-seeded game
per tile at `sem'`, with the cap checked per credited row: each credited row of a good tile is a good one-row tile of
the split layout carrying that row's credit, and the error is the split layout's, at `R = Σ_u m_u` units. -/
def _root_.Pouw.Guarantees.PearlC.TTOutTileRowSeedOfTTOutTile : Prop :=
  ∀ {Q R S : Type} {sem sem' : PearlCSem Q R S} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S} {d : PearlCDevice} {ρ : ℚ} (hTT : TTOutTilePearlCDevRev1 CM d sem ρ) (hN : RowSeedNoise sem sem') (hCM : CM.SplitClosed) (hd : d.RowLocal) (hρ : ρ ≤ 1) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16),
    TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCap d sem' ρ) (pearlCTilesDevRev1RowCap d sem' ρ)
    (pearlCDomainDev d) (1 / 400)

end Pouw.PearlC
