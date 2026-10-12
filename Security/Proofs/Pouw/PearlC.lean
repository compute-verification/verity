import Models.Pouw.Guarantees.PearlC
import Proofs.Pouw.PearlC.DeviceCapRev1Gamma
import Proofs.Pouw.PearlC.DeviceFp4Gamma
import Proofs.Pouw.PearlC.Fp4IssueGamma
import Proofs.Pouw.PearlC.RowSeedGamma
import Proofs.Pouw.PearlC.RowSeedSkip
import Proofs.Pouw.PearlC.ServedFadd8Rev1Gamma
import Proofs.Pouw.PearlC.ServedMixedRev1Gamma
import Proofs.Pouw.PearlC.UOnlyGamma

/-!
# Proofs of Pearl-C's guarantees

One theorem `Pouw.SecurityProofs.PearlC.G : Pouw.Guarantees.PearlC.G` per guarantee, which is the theorem in
`Pouw.PearlC` (`DeviceCapRev1Gamma`) that proves the same statement under its old name.
-/

namespace Pouw.SecurityProofs.PearlC

theorem GammaSm120v1LoopCast8p72Rev1Cap1000_8192 : Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_8192 :=
  fun CM sem hTT => Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_8192 CM sem hTT

end Pouw.SecurityProofs.PearlC

theorem Pouw.SecurityProofs.PearlC.GammaFp4Sm120LoopLut256_16384 : Pouw.Guarantees.PearlC.GammaFp4Sm120LoopLut256_16384 := @Pouw.PearlC.gammaFp4_sm120Loop_lut256_16384

theorem Pouw.SecurityProofs.PearlC.GammaFp4Sm120LoopLut256_8192 : Pouw.Guarantees.PearlC.GammaFp4Sm120LoopLut256_8192 := @Pouw.PearlC.gammaFp4_sm120Loop_lut256_8192

theorem Pouw.SecurityProofs.PearlC.GammaFp4Sm120_16384 : Pouw.Guarantees.PearlC.GammaFp4Sm120_16384 := @Pouw.PearlC.pearlCGammaFp4Sm120_16384

theorem Pouw.SecurityProofs.PearlC.GammaFp4Sm120_8192 : Pouw.Guarantees.PearlC.GammaFp4Sm120_8192 := @Pouw.PearlC.pearlCGammaFp4Sm120_8192

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_16384 : Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_16384 := @Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_16384

theorem Pouw.SecurityProofs.PearlC.SampledFp4Sm120_16384 : Pouw.Guarantees.PearlC.SampledFp4Sm120_16384 := @Pouw.PearlC.pearlCSampledFp4Sm120_16384

theorem Pouw.SecurityProofs.PearlC.SampledFp4Sm120_8192 : Pouw.Guarantees.PearlC.SampledFp4Sm120_8192 := @Pouw.PearlC.pearlCSampledFp4Sm120_8192

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_16384 : Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_16384 := @Pouw.PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_16384

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_8192 : Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_8192 := @Pouw.PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_8192

theorem Pouw.SecurityProofs.PearlC.TTOutRowSeedOfTTOut : Pouw.Guarantees.PearlC.TTOutRowSeedOfTTOut := @Pouw.PearlC.ttOutRowSeed_of_ttOut

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 :
    Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 :=
  @Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n6144k4096

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 :
    Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 :=
  @Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n4096k4096

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 :
    Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 :=
  @Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n28672k4096

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 :
    Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 :=
  @Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n4096k14336

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_llama31_8b :
    Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_llama31_8b :=
  @Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_llama31_8b

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 :
    Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 :=
  @Pouw.PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n6144k4096

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 :
    Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 :=
  @Pouw.PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n4096k4096

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 :
    Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 :=
  @Pouw.PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n28672k4096

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 :
    Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 :=
  @Pouw.PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n4096k14336

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_llama31_8b :
    Pouw.Guarantees.PearlC.SampledSm120v1LoopCast8p72Rev1Cap1000_llama31_8b :=
  @Pouw.PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_llama31_8b

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_n6144k4096 :
    Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_n6144k4096 :=
  @Pouw.PearlC.pearlCGammaSm120v1Cast8p72Rev1Cap1000_n6144k4096

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_n4096k4096 :
    Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_n4096k4096 :=
  @Pouw.PearlC.pearlCGammaSm120v1Cast8p72Rev1Cap1000_n4096k4096

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_n28672k4096 :
    Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_n28672k4096 :=
  @Pouw.PearlC.pearlCGammaSm120v1Cast8p72Rev1Cap1000_n28672k4096

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_n4096k14336 :
    Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_n4096k14336 :=
  @Pouw.PearlC.pearlCGammaSm120v1Cast8p72Rev1Cap1000_n4096k14336

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_llama31_8b :
    Pouw.Guarantees.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_llama31_8b :=
  @Pouw.PearlC.pearlCGammaSm120v1Cast8p72Rev1Cap1000_llama31_8b

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_n6144k4096 :
    Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_n6144k4096 :=
  @Pouw.PearlC.pearlCSampledSm120v1Cast8p72Rev1Cap1000_n6144k4096

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_n4096k4096 :
    Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_n4096k4096 :=
  @Pouw.PearlC.pearlCSampledSm120v1Cast8p72Rev1Cap1000_n4096k4096

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_n28672k4096 :
    Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_n28672k4096 :=
  @Pouw.PearlC.pearlCSampledSm120v1Cast8p72Rev1Cap1000_n28672k4096

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_n4096k14336 :
    Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_n4096k14336 :=
  @Pouw.PearlC.pearlCSampledSm120v1Cast8p72Rev1Cap1000_n4096k14336

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_llama31_8b :
    Pouw.Guarantees.PearlC.SampledSm120v1Cast8p72Rev1Cap1000_llama31_8b :=
  @Pouw.PearlC.pearlCSampledSm120v1Cast8p72Rev1Cap1000_llama31_8b
