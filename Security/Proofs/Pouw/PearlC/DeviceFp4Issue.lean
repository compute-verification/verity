import Definitions.Pouw.PearlC.DeviceFp4Issue

/-! The proof-only part of `Definitions.Pouw.PearlC.DeviceFp4Issue`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC

/-- **sm_120's prices in FP4 units at the issue-bound FP32 price**: every FP32 operation at 8.00. -/
def Fp4Prices.sm120Issue : Fp4Prices := ⟨4, 5429 / 50, 18⟩

/-- **`rcp.approx`**: the UE4M3 encode alone (8.0), the `F2FP` decode (12.74) and the `MUFU` reciprocal (63.9) per
block, `2·84.64/16 = 10.58` per element, and two FP32 ops per block, the FMUL `amax·1/6` and the decode's `HADD2.F32`
(`theory-pearl-c4-domain.md` §8a). -/
def Fp4ScalePath.rcpApprox : Fp4ScalePath := ⟨529 / 50, 2, by norm_num, by norm_num⟩

end Pouw.PearlC
