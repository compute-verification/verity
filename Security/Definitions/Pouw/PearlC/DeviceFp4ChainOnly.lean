import Definitions.Pouw.PearlC.DeviceFp4Issue

/-!
# Pearl-C4's chain-only reading at an FP4 record (definitions only, staged)

The forming-credit reading credits the salt-dependent forming (`fs` per activation element). The chain-only reading
credits the unit without it, as `creditChainOnly` does at the H100 and `DeviceChainOnly` does for FP8: `creditFp4` less
`fs·m·k`. The cap stays on the full credit, so a unit within it keeps at least `creditFp4 − fs·m·k − ρ·creditFp4`.
-/

namespace Pouw.PearlC

/-- **The chain-only credit at an FP4 record**: `creditFp4` less the credited forming `fs·m·k`. -/
def creditFp4ChainOnly (d : Fp4Device) (s : Shape) : ℚ := creditFp4 d s - d.prices.fs * s.m * s.k

/-- **Pearl-C4's chain-only γ** at the cap `ρ`, which stays on the full credit: `1 − (399/400)/ω` with
`ω = wrefFp4/(creditFp4ChainOnly − ρ·creditFp4)`. -/
def gammaFp4ChainOnly (d : Fp4Device) (ρ : ℚ) (s : Shape) : ℚ :=
  1 - (1 - 1 / 400) / (wrefFp4 d s / (creditFp4ChainOnly d s - ρ * creditFp4 d s))

end Pouw.PearlC
