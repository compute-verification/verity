module

public import Definitions.Pouw.PearlC.DeviceFp4

/-! Pearl-C4's prices on sm_120 as a function of the FP32 price and the block-scale path, staged. -/

@[expose] public section

namespace Pouw.PearlC

/-- Per element in FP4 units: `fixed` is the work no FP32 price reprices, and `fp32Ops` the FP32 operations per block of
16, each at the FADD price. -/
structure Fp4ScalePath where
  fixed : ℚ
  fp32Ops : ℚ
  fixed_nonneg : 0 ≤ fixed
  fp32Ops_nonneg : 0 ≤ fp32Ops

/-- The UE4M3 encode, the `F2FP` decode and the `MUFU` reciprocal per block, and two FP32 ops per block: the FMUL
`amax·1/6` and the decode's `HADD2.F32`. -/
def Fp4ScalePath.rcpApprox : Fp4ScalePath := ⟨529 / 50, 2, by norm_num, by norm_num⟩

/-- Half an `F2FP`, 1.25 `PRMT` and one `LDS` per block at W1's per-instruction prices, and one FP32 op per block, the FMUL
`amax·1/6`: the table replaces the decode and its widening. -/
def Fp4ScalePath.lut256 : Fp4ScalePath := ⟨15 / 2, 1, by norm_num, by norm_num⟩

/-- In FP4 units; at `fadd = 8` it is the issue-bound record, and at `1047/125` the in-loop one. -/
def Fp4Prices.sm120At (fadd : ℚ) (p : Fp4ScalePath) : Fp4Prices :=
  ⟨4, 64 + p.fixed + p.fp32Ops * fadd / 8 + 2 * fadd + 16, 2 * fadd + fadd / 4⟩

def gapFp4 (d : Fp4Device) (ρ : ℚ) (s : Shape) : ℚ :=
  1 - (1 - 1 / 400) / (wrefFp4 d s / ((1 - ρ) * creditFp4 d s))

end Pouw.PearlC
