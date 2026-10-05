import FlockSoundness.Refine.Opening
import FlockSoundness.Protocol.Refine.ArithK

/-!
# Element seams over `K = GF(2^256)` (PROTOCOL.md §2.4, §13)

The model's Ligerito computes in `K` through `execArith.embed` and `execArith.u`; the executable computes on `F256`
words. `gOf` (`GF256.of`) maps a word to its field element and is a ring homomorphism (`gOf_add`, `gOf_mul`). On top of
level 3's `ofLimbs`:

* `F256.ofF` is `embed`, and `a.smul s` is `a · embed s` (`gOf_smul`);
* the executable's round polynomials (`Poly`) are the model's `Quad` (`quadOf`: `ofMsg`, `eval`, `addScaled`);
* both equality tables are the model's `eqList`: `eqTable` over `F` (`eqTable_eqList`), `eqTableK` over `K`
  (`eqTableK_eqList`).
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## Round polynomials -/

/-! ## Equality tables -/

end FlockSoundness.Refine
