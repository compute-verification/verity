import Definitions.Pouw.Guarantees.PearlC
import Proofs.Pouw.PearlC.DeviceCapRev1Gamma
import Proofs.Pouw.PearlC.DeviceFp4Gamma
import Proofs.Pouw.PearlC.Fp4IssueGamma
import Proofs.Pouw.PearlC.RowSeedGamma
import Proofs.Pouw.PearlC.ServedFadd8Rev1Gamma
import Proofs.Pouw.PearlC.ServedMixedRev1Gamma
import Proofs.Pouw.PearlC.UOnlyGamma
import Proofs.Pouw.PearlC.TTOutW1
import Proofs.Pouw.PearlC.TTOutFp4W1
import Proofs.Pouw.PearlC.TTOutW1Item1
import Proofs.Pouw.QTree

/-!
# Proofs of Pearl-C's guarantees

One theorem `Pouw.SecurityProofs.PearlC.G : Pouw.Guarantees.PearlC.G` per guarantee, which is the theorem in
`Pouw.PearlC` (`DeviceCapRev1Gamma`) that proves the same statement under its old name, or, for a guarantee stated at
W1, that theorem at the cost model `W1.w1Priced` (`Proofs.Pouw.PearlC.TTOutW1`).
-/

namespace Pouw.SecurityProofs.PearlC

theorem GammaSm120v1LoopCast8p72Rev1Cap1000_8192 : Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

end Pouw.SecurityProofs.PearlC

theorem Pouw.SecurityProofs.PearlC.GammaFp4Sm120LoopLut256_16384 : Pouw.Guarantees.PearlC.GammaFp4Sm120LoopLut256_16384 := @Pouw.PearlC.gammaFp4_sm120Loop_lut256_16384

theorem Pouw.SecurityProofs.PearlC.GammaFp4Sm120LoopLut256_8192 : Pouw.Guarantees.PearlC.GammaFp4Sm120LoopLut256_8192 := @Pouw.PearlC.gammaFp4_sm120Loop_lut256_8192

theorem Pouw.SecurityProofs.PearlC.GammaFp4Sm120_16384 : Pouw.Guarantees.PearlC.GammaFp4Sm120_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGammaFp4Sm120_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutFp4W1.ttOutFp4Sm120_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GammaFp4Sm120_8192 : Pouw.Guarantees.PearlC.GammaFp4Sm120_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGammaFp4Sm120_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutFp4W1.ttOutFp4Sm120_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_16384 : Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledFp4Sm120_16384 : Pouw.Guarantees.PearlC.SampledFp4Sm120_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledFp4Sm120_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutFp4W1.ttOutTileFp4Sm120_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledFp4Sm120_8192 : Pouw.Guarantees.PearlC.SampledFp4Sm120_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledFp4Sm120_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutFp4W1.ttOutTileFp4Sm120_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_16384 : Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_8192 : Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.TTOutRowSeedOfTTOut : Pouw.Guarantees.PearlC.TTOutRowSeedOfTTOut := @Pouw.PearlC.ttOutRowSeed_of_ttOut

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 :
    Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 :
    Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 :
    Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 :
    Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_llama31_8b :
    Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_llama31_8b :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_llama31_8b _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 :
    Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 :
    Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 :
    Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 :
    Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_llama31_8b :
    Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_llama31_8b :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_llama31_8b _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_n6144k4096 :
    Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_n6144k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGammaSm120v1Cast8p72Rev1Cap1000_n6144k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_n4096k4096 :
    Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_n4096k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGammaSm120v1Cast8p72Rev1Cap1000_n4096k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_n28672k4096 :
    Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_n28672k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGammaSm120v1Cast8p72Rev1Cap1000_n28672k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_n4096k14336 :
    Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_n4096k14336 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGammaSm120v1Cast8p72Rev1Cap1000_n4096k14336 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_llama31_8b :
    Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_llama31_8b :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGammaSm120v1Cast8p72Rev1Cap1000_llama31_8b _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_n6144k4096 :
    Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_n6144k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1Cast8p72Rev1Cap1000_n6144k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_n4096k4096 :
    Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_n4096k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1Cast8p72Rev1Cap1000_n4096k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_n28672k4096 :
    Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_n28672k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1Cast8p72Rev1Cap1000_n28672k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_n4096k14336 :
    Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_n4096k14336 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1Cast8p72Rev1Cap1000_n4096k14336 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_llama31_8b :
    Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_llama31_8b :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1Cast8p72Rev1Cap1000_llama31_8b _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)
