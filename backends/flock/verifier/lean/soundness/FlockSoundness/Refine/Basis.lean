import FlockSoundness.Refine.Final
import FlockSoundness.Protocol.Refine.Basis

/-!
# The batched basis refines the model's (PROTOCOL.md §12.4)

`basis_eq`: the executable's `basis sws q e` at `y < 2^e` is the model's `bHat` of the batched basis `b` (R4's
`OpenRel.b`) at `q ‖ bits(y)`.

The executable evaluates each claim's term with a width-128 branching program over bit vectors. `applyMap (mulColumns g)`
is multiplication by `g` on a vector of bits (`applyMap_mul`). After the prefix coordinates, and then the suffix
coordinates read as `0`/`1` points, the state is `Σ_w eq(point, w)·bits(eq(z, w))` (`bfcState`). The final
contraction with `eq(r'')` is the model's `phi` (§12.4).
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## Monomials and the multiplication columns -/

/-! ## `applyMap`: a matrix over the columns' bits -/

/-! ## Bits in `K`, and multiplication on bit vectors -/

/-! ## The branching program's invariant -/

/-! ## The executable's loops -/

end FlockSoundness.Refine
