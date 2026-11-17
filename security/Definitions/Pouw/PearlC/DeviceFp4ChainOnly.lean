module

public import Definitions.Pouw.PearlC.DeviceFp4

/-! Pearl-C4's chain-only reading at an FP4 record, staged. -/

@[expose] public section

namespace Pouw.PearlC

def creditFp4ChainOnly (d : Fp4Device) (s : Shape) : ℚ := creditFp4 d s - d.prices.fs * s.m * s.k

def gapFp4ChainOnly (d : Fp4Device) (ρ : ℚ) (s : Shape) : ℚ :=
  1 - (1 - 1 / 400) / (wrefFp4 d s / (creditFp4ChainOnly d s - ρ * creditFp4 d s))

end Pouw.PearlC
