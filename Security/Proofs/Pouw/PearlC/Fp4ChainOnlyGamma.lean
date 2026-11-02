import Proofs.Pouw.PearlC.Fp4IssueGamma
import Definitions.Pouw.Guarantees.PearlC

/-!
# Pearl-C4's chain-only γ, at any FP32 price and scale path (staged)

**The values** (`gammaFp4ChainOnly_sm120{Issue,Loop}_{rcpApprox,lut256}_…`): `gammaFp4ChainOnly` at the cap `1/400`, at
FADD 8.00 and `1047/125` = 8.376, on both scale paths, each with the chain-only worst case
`creditFp4ChainOnly − ρ·creditFp4` positive.

Held, as the forming-credited twins are: `tt-out/fp4-sm120` is rated D until the base-split fix.
-/

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions

/-! ## The values: both FP32 prices, both scale paths -/

/-- Pearl-C4's chain-only γ at FP32 8.00, the `rcpApprox` scale path, 8192³, cap `1/400`:
`451194057/22950880000` (1.96591%), with the chain-only worst case positive. -/
theorem gammaFp4ChainOnly_sm120Issue_rcpApprox_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.rcpApprox) :
    0 < creditFp4ChainOnly d sh8192 - 1 / 400 * creditFp4 d sh8192 ∧
      gammaFp4ChainOnly d (1 / 400) sh8192 = 451194057 / 22950880000 := by
  constructor
  · rw [creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh8192]
  · rw [gammaFp4ChainOnly, wrefFp4, creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh8192]

/-- Pearl-C4's chain-only γ at FP32 8.00, the `rcpApprox` scale path, 16384³, cap `1/400`:
`1680852571/134388640000` (1.25074%), with the chain-only worst case positive. -/
theorem gammaFp4ChainOnly_sm120Issue_rcpApprox_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.rcpApprox) :
    0 < creditFp4ChainOnly d sh16384 - 1 / 400 * creditFp4 d sh16384 ∧
      gammaFp4ChainOnly d (1 / 400) sh16384 = 1680852571 / 134388640000 := by
  constructor
  · rw [creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh16384]
  · rw [gammaFp4ChainOnly, wrefFp4, creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh16384]

/-- Pearl-C4's chain-only γ at FP32 8.00, the `lut256` scale path, 8192³, cap `1/400`:
`17611477/917600000` (1.91930%), with the chain-only worst case positive. -/
theorem gammaFp4ChainOnly_sm120Issue_lut256_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256) :
    0 < creditFp4ChainOnly d sh8192 - 1 / 400 * creditFp4 d sh8192 ∧
      gammaFp4ChainOnly d (1 / 400) sh8192 = 17611477 / 917600000 := by
  constructor
  · rw [creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh8192]
  · rw [gammaFp4ChainOnly, wrefFp4, creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh8192]

/-- Pearl-C4's chain-only γ at FP32 8.00, the `lut256` scale path, 16384³, cap `1/400`:
`65925247/5374240000` (1.22669%), with the chain-only worst case positive. -/
theorem gammaFp4ChainOnly_sm120Issue_lut256_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256) :
    0 < creditFp4ChainOnly d sh16384 - 1 / 400 * creditFp4 d sh16384 ∧
      gammaFp4ChainOnly d (1 / 400) sh16384 = 65925247 / 5374240000 := by
  constructor
  · rw [creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh16384]
  · rw [gammaFp4ChainOnly, wrefFp4, creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh16384]

/-- Pearl-C4's chain-only γ at FP32 `1047/125` = 8.376, the `rcpApprox` scale path, 8192³, cap `1/400`:
`4557116829/229553920000` (1.98521%), with the chain-only worst case positive. -/
theorem gammaFp4ChainOnly_sm120Loop_rcpApprox_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.rcpApprox) :
    0 < creditFp4ChainOnly d sh8192 - 1 / 400 * creditFp4 d sh8192 ∧
      gammaFp4ChainOnly d (1 / 400) sh8192 = 4557116829 / 229553920000 := by
  constructor
  · rw [creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh8192]
  · rw [gammaFp4ChainOnly, wrefFp4, creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh8192]

/-- Pearl-C4's chain-only γ at FP32 `1047/125` = 8.376, the `rcpApprox` scale path, 16384³, cap `1/400`:
`16944054487/1344021760000` (1.26070%), with the chain-only worst case positive. -/
theorem gammaFp4ChainOnly_sm120Loop_rcpApprox_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.rcpApprox) :
    0 < creditFp4ChainOnly d sh16384 - 1 / 400 * creditFp4 d sh16384 ∧
      gammaFp4ChainOnly d (1 / 400) sh16384 = 16944054487 / 1344021760000 := by
  constructor
  · rw [creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh16384]
  · rw [gammaFp4ChainOnly, wrefFp4, creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh16384]

/-- Pearl-C4's chain-only γ at FP32 `1047/125` = 8.376, the `lut256` scale path, 8192³, cap `1/400`:
`26680734301/1376663200000` (1.93807%), with the chain-only worst case positive. -/
theorem gammaFp4ChainOnly_sm120Loop_lut256_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256) :
    0 < creditFp4ChainOnly d sh8192 - 1 / 400 * creditFp4 d sh8192 ∧
      gammaFp4ChainOnly d (1 / 400) sh8192 = 26680734301 / 1376663200000 := by
  constructor
  · rw [creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh8192]
  · rw [gammaFp4ChainOnly, wrefFp4, creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh8192]

/-- Pearl-C4's chain-only γ at FP32 `1047/125` = 8.376, the `lut256` scale path, 16384³, cap `1/400`:
`11075380767/895794400000` (1.23638%), with the chain-only worst case positive. -/
theorem gammaFp4ChainOnly_sm120Loop_lut256_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256) :
    0 < creditFp4ChainOnly d sh16384 - 1 / 400 * creditFp4 d sh16384 ∧
      gammaFp4ChainOnly d (1 / 400) sh16384 = 11075380767 / 895794400000 := by
  constructor
  · rw [creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh16384]
  · rw [gammaFp4ChainOnly, wrefFp4, creditFp4ChainOnly, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh16384]

end Pouw.PearlC

theorem Pouw.SecurityProofs.PearlC.GammaFp4ChainOnlySm120IssueLut256_8192 : Pouw.Guarantees.PearlC.GammaFp4ChainOnlySm120IssueLut256_8192 := @Pouw.PearlC.gammaFp4ChainOnly_sm120Issue_lut256_8192

theorem Pouw.SecurityProofs.PearlC.GammaFp4ChainOnlySm120LoopLut256_8192 : Pouw.Guarantees.PearlC.GammaFp4ChainOnlySm120LoopLut256_8192 := @Pouw.PearlC.gammaFp4ChainOnly_sm120Loop_lut256_8192
