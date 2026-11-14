module

public import Proofs.Pouw.Lemmas.PearlCRecords
public import Proofs.Pouw.PearlC.DeviceCapRev1Gap
public import Proofs.Pouw.PearlC.DeviceFp4Gap
public import Proofs.Pouw.PearlC.Fp4IssueGap
public import Proofs.Pouw.PearlC.RowSeedGap
public import Proofs.Pouw.PearlC.ServedFadd8Rev1Gap
public import Proofs.Pouw.PearlC.ServedMixedRev1Gap
public import Proofs.Pouw.PearlC.UOnlyGap
public import Proofs.Pouw.PearlC.TTOutW1
public import Proofs.Pouw.PearlC.TTOutFp4W1
public import Proofs.Pouw.PearlC.TTOutW1Item1
public import Proofs.Pouw.Lemmas.PearlC
public import Proofs.Pouw.QTree
public import Proofs.Pouw.PearlC.RowSeedSkip

/-!
# Proofs of Pearl-C's guarantees

One theorem `Pouw.SecurityProofs.PearlC.G : Pouw.Guarantees.PearlC.G` per guarantee, which is the theorem in
`Pouw.PearlC` (`DeviceCapRev1Gap`) that proves the same statement under its old name, or, for a guarantee stated at
W1, that theorem at the cost model `W1.w1Priced` (`Proofs.Pouw.PearlC.TTOutW1`).
-/

@[expose] public section

namespace Pouw.SecurityProofs.PearlC

theorem GapSm120v1LoopCast8p72Rev1Cap1000_8192 : Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1LoopCast8p72Rev1Cap1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

end Pouw.SecurityProofs.PearlC

theorem Pouw.SecurityProofs.PearlC.GapFp4Sm120LoopLut256_16384 : Pouw.Guarantees.PearlC.GapFp4Sm120LoopLut256_16384 := @Pouw.PearlC.gapFp4_sm120Loop_lut256_16384

theorem Pouw.SecurityProofs.PearlC.GapFp4Sm120LoopLut256_8192 : Pouw.Guarantees.PearlC.GapFp4Sm120LoopLut256_8192 := @Pouw.PearlC.gapFp4_sm120Loop_lut256_8192

theorem Pouw.SecurityProofs.PearlC.GapFp4Sm120_16384 : Pouw.Guarantees.PearlC.GapFp4Sm120_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapFp4Sm120_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutFp4W1.ttOutFp4Sm120_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapFp4Sm120_8192 : Pouw.Guarantees.PearlC.GapFp4Sm120_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapFp4Sm120_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.fp4Sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutFp4W1.ttOutFp4Sm120_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_16384 : Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1LoopCast8p72Rev1Cap1000_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
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

theorem Pouw.SecurityProofs.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 :
    Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 :
    Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 :
    Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 :
    Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_llama31_8b :
    Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_llama31_8b :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1LoopCast8p72Rev1Cap1000_llama31_8b _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) sem
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

theorem Pouw.SecurityProofs.PearlC.GapSm120v1Cast8p72Rev1Cap1000_n6144k4096 :
    Pouw.Guarantees.PearlC.GapSm120v1Cast8p72Rev1Cap1000_n6144k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1Cast8p72Rev1Cap1000_n6144k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1Cast8p72Rev1Cap1000_n4096k4096 :
    Pouw.Guarantees.PearlC.GapSm120v1Cast8p72Rev1Cap1000_n4096k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1Cast8p72Rev1Cap1000_n4096k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1Cast8p72Rev1Cap1000_n28672k4096 :
    Pouw.Guarantees.PearlC.GapSm120v1Cast8p72Rev1Cap1000_n28672k4096 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1Cast8p72Rev1Cap1000_n28672k4096 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1Cast8p72Rev1Cap1000_n4096k14336 :
    Pouw.Guarantees.PearlC.GapSm120v1Cast8p72Rev1Cap1000_n4096k14336 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1Cast8p72Rev1Cap1000_n4096k14336 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1Cast8p72Rev1Cap1000_llama31_8b :
    Pouw.Guarantees.PearlC.GapSm120v1Cast8p72Rev1Cap1000_llama31_8b :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1Cast8p72Rev1Cap1000_llama31_8b _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
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
