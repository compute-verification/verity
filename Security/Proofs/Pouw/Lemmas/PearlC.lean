module

public import Definitions.Pouw.Guarantees.PearlC

/-!
# Pearl-C's guarantees: lemmas

The statements of `Definitions.Pouw.Guarantees.PearlC` that are steps of its guarantees' proofs rather than guarantees.
Each is a `def G : Prop` under the guarantees' namespace, proved by the `SecurityProofs` theorem of the same name, and
audited with the proofs: no record pins it.
-/

@[expose] public section

namespace Pouw.PearlC

open Pouw.PearlC.Assumptions

/-- Pearl-C4's gap at FP32 `1047/125` = 8.376, the `lut256` scale path, 16384³, cap `1/400`: `5473473967/895794400000`
(0.61102%). -/
def _root_.Pouw.Guarantees.PearlC.GapFp4Sm120LoopLut256_16384 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256),
    gapFp4 d (1 / 400) sh16384 = 5473473967 / 895794400000

/-- Pearl-C4's gap at FP32 `1047/125` = 8.376, the `lut256` scale path, 8192³, cap `1/400`: `9875013901/1376663200000`
(0.71732%). -/
def _root_.Pouw.Guarantees.PearlC.GapFp4Sm120LoopLut256_8192 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256),
    gapFp4 d (1 / 400) sh8192 = 9875013901 / 1376663200000

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

/-- Pearl-C4's chain-only gap at FP32 8.00, the `lut256` scale path, 8192³, cap `1/400`:
`17611477/917600000` (1.91930%), with the chain-only worst case positive. -/
def _root_.Pouw.Guarantees.PearlC.GapFp4ChainOnlySm120IssueLut256_8192 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256),
    0 < creditFp4ChainOnly d sh8192 - 1 / 400 * creditFp4 d sh8192 ∧
    gapFp4ChainOnly d (1 / 400) sh8192 = 17611477 / 917600000

/-- Pearl-C4's chain-only gap at FP32 `1047/125` = 8.376, the `lut256` scale path, 8192³, cap `1/400`:
`26680734301/1376663200000` (1.93807%), with the chain-only worst case positive. -/
def _root_.Pouw.Guarantees.PearlC.GapFp4ChainOnlySm120LoopLut256_8192 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256),
    0 < creditFp4ChainOnly d sh8192 - 1 / 400 * creditFp4 d sh8192 ∧
    gapFp4ChainOnly d (1 / 400) sh8192 = 26680734301 / 1376663200000

end Pouw.PearlC

namespace Pouw.PearlC

open Pouw.PearlC.Assumptions

/-- Pearl-C4's gap at FP32 8.00, the `lut256` scale path, 16384³, cap `1/400`: `32568847/5374240000` (0.60602%). -/
def _root_.Pouw.Guarantees.PearlC.GapFp4Sm120IssueLut256_16384 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256),
    gapFp4 d (1 / 400) sh16384 = 32568847 / 5374240000

/-- Pearl-C4's gap at FP32 8.00, the `lut256` scale path, 8192³, cap `1/400`: `6492677/917600000` (0.70757%). -/
def _root_.Pouw.Guarantees.PearlC.GapFp4Sm120IssueLut256_8192 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256),
    gapFp4 d (1 / 400) sh8192 = 6492677 / 917600000

end Pouw.PearlC
