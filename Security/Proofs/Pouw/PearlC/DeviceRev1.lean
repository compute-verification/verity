import Definitions.Pouw.PearlC.DeviceRev1

/-! The proof-only part of `Definitions.Pouw.PearlC.DeviceRev1`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

/-- The first promotion add of every word at H100 prices (an FP32 add costs 32), as `creditOf` prices its adds. -/
def firstAddOf (p : Params) (s : Shape) : ℚ := if p.G = 0 then 0 else 32 * s.m * s.n

/-- **rev1's credited work of one unit** at `Params` and `Costs` (H100 prices): `creditOf` less the first add. -/
def creditOfRev1 (p : Params) (c : Costs) (s : Shape) : ℚ := creditOf p c s - firstAddOf p s

/-- rev1's honest reference at `Params` and `Costs`. -/
def wrefOfRev1 (p : Params) (c : Costs) (s : Shape) : ℚ := creditOfRev1 p c s + (c.qa : ℚ) * s.m * s.k

variable {Q R S : Type}

end Pouw.PearlC
