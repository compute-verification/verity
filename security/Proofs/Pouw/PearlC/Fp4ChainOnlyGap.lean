module

public import Proofs.Pouw.PearlC.Fp4IssueGap
public import Proofs.Pouw.Lemmas.PearlCRecords
public import Proofs.Pouw.Lemmas.PearlC

/-! Pearl-C4's chain-only gap, at any FP32 price and scale path (staged). -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions

theorem gapFp4ChainOnly_sm120Issue_rcpApprox_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.rcpApprox) :
    0 < creditFp4ChainOnly d sh8192 - 1 / 400 * creditFp4 d sh8192 ∧
      gapFp4ChainOnly d (1 / 400) sh8192 = 451194057 / 22950880000 := by
  constructor
  · rw [creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh8192]
  · rw [gapFp4ChainOnly, wrefFp4, creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh8192]

theorem gapFp4ChainOnly_sm120Issue_rcpApprox_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.rcpApprox) :
    0 < creditFp4ChainOnly d sh16384 - 1 / 400 * creditFp4 d sh16384 ∧
      gapFp4ChainOnly d (1 / 400) sh16384 = 1680852571 / 134388640000 := by
  constructor
  · rw [creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh16384]
  · rw [gapFp4ChainOnly, wrefFp4, creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh16384]

theorem gapFp4ChainOnly_sm120Issue_lut256_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256) :
    0 < creditFp4ChainOnly d sh8192 - 1 / 400 * creditFp4 d sh8192 ∧
      gapFp4ChainOnly d (1 / 400) sh8192 = 17611477 / 917600000 := by
  constructor
  · rw [creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh8192]
  · rw [gapFp4ChainOnly, wrefFp4, creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh8192]

theorem gapFp4ChainOnly_sm120Issue_lut256_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256) :
    0 < creditFp4ChainOnly d sh16384 - 1 / 400 * creditFp4 d sh16384 ∧
      gapFp4ChainOnly d (1 / 400) sh16384 = 65925247 / 5374240000 := by
  constructor
  · rw [creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh16384]
  · rw [gapFp4ChainOnly, wrefFp4, creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh16384]

theorem gapFp4ChainOnly_sm120Loop_rcpApprox_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.rcpApprox) :
    0 < creditFp4ChainOnly d sh8192 - 1 / 400 * creditFp4 d sh8192 ∧
      gapFp4ChainOnly d (1 / 400) sh8192 = 4557116829 / 229553920000 := by
  constructor
  · rw [creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh8192]
  · rw [gapFp4ChainOnly, wrefFp4, creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh8192]

theorem gapFp4ChainOnly_sm120Loop_rcpApprox_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.rcpApprox) :
    0 < creditFp4ChainOnly d sh16384 - 1 / 400 * creditFp4 d sh16384 ∧
      gapFp4ChainOnly d (1 / 400) sh16384 = 16944054487 / 1344021760000 := by
  constructor
  · rw [creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh16384]
  · rw [gapFp4ChainOnly, wrefFp4, creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh16384]

theorem gapFp4ChainOnly_sm120Loop_lut256_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256) :
    0 < creditFp4ChainOnly d sh8192 - 1 / 400 * creditFp4 d sh8192 ∧
      gapFp4ChainOnly d (1 / 400) sh8192 = 26680734301 / 1376663200000 := by
  constructor
  · rw [creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh8192]
  · rw [gapFp4ChainOnly, wrefFp4, creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh8192]

theorem gapFp4ChainOnly_sm120Loop_lut256_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256) :
    0 < creditFp4ChainOnly d sh16384 - 1 / 400 * creditFp4 d sh16384 ∧
      gapFp4ChainOnly d (1 / 400) sh16384 = 11075380767 / 895794400000 := by
  constructor
  · rw [creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh16384]
  · rw [gapFp4ChainOnly, wrefFp4, creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh16384]

end Pouw.PearlC

theorem Pouw.SecurityProofs.PearlC.GapFp4ChainOnlySm120IssueLut256_8192 : Pouw.Guarantees.PearlC.GapFp4ChainOnlySm120IssueLut256_8192 := @Pouw.PearlC.gapFp4ChainOnly_sm120Issue_lut256_8192

theorem Pouw.SecurityProofs.PearlC.GapFp4ChainOnlySm120LoopLut256_8192 : Pouw.Guarantees.PearlC.GapFp4ChainOnlySm120LoopLut256_8192 := @Pouw.PearlC.gapFp4ChainOnly_sm120Loop_lut256_8192
