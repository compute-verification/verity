module

public import Proofs.Pouw.Lemmas.PearlCRecords

/-! Steps of Pearl-C's guarantees' proofs, stated as Props that no record pins. -/

@[expose] public section

namespace Pouw.PearlC

open Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapFp4Sm120LoopLut256_16384 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256),
    gapFp4 d (1 / 400) sh16384 = 5473473967 / 895794400000

def _root_.Pouw.Guarantees.PearlC.GapFp4Sm120LoopLut256_8192 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256),
    gapFp4 d (1 / 400) sh8192 = 9875013901 / 1376663200000

end Pouw.PearlC

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

def _root_.Pouw.Guarantees.PearlC.TTOutRowSeedOfTTOut : Prop :=
  ∀ {Q R S : Type} {sem sem' : PearlCSem Q R S} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S} {d : PearlCDevice} {ρ : ℚ} (hTT : TTOutPearlCDevRev1 CM d sem ρ) (hN : RowSeedNoise sem sem') (hCM : CM.SplitClosed) (hd : d.RowLocal) (hρ : ρ ≤ 1) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16),
    TTOutRowSeed CM (pearlCProtocolDevRev1RowInertBound d sem' ρ) (pearlCDomainDev d) (1 / 400)

end Pouw.PearlC

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapFp4ChainOnlySm120IssueLut256_8192 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256),
    0 < creditFp4ChainOnly d sh8192 - 1 / 400 * creditFp4 d sh8192 ∧
    gapFp4ChainOnly d (1 / 400) sh8192 = 17611477 / 917600000

def _root_.Pouw.Guarantees.PearlC.GapFp4ChainOnlySm120LoopLut256_8192 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256),
    0 < creditFp4ChainOnly d sh8192 - 1 / 400 * creditFp4 d sh8192 ∧
    gapFp4ChainOnly d (1 / 400) sh8192 = 26680734301 / 1376663200000

end Pouw.PearlC

namespace Pouw.PearlC

open Pouw.PearlC.Assumptions

def _root_.Pouw.Guarantees.PearlC.GapFp4Sm120IssueLut256_16384 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256),
    gapFp4 d (1 / 400) sh16384 = 32568847 / 5374240000

def _root_.Pouw.Guarantees.PearlC.GapFp4Sm120IssueLut256_8192 : Prop :=
  ∀ (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256),
    gapFp4 d (1 / 400) sh8192 = 6492677 / 917600000

end Pouw.PearlC
