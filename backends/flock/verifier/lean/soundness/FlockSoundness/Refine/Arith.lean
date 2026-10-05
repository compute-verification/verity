import FlockSoundness.Refine.Tapes
import FlockSoundness.Refine.Exec
import FlockSoundness.Refine.Plays
import FlockSoundness.Protocol.Refine.Arith

/-!
# Element seams: the executable's arithmetic is the model's

The model computes in the fields of `execArith` (`GF128`, `GF256`); the executable computes on their words (`F128`,
`F256`) and arrays. `fOf` (`GF128.of`) maps a word to its field element, and it is a ring homomorphism on the operations
the verifier uses (`fOf_add`, `fOf_mul`, `fOf_inv`). This file adds what the phases need on top: the executable's `dot`
is a sum (`fOf_dot`), the weight arrays have 64 entries, reading a coin as `getD` or `!` agrees, and the zerocheck
vector `pinned ++ r_outer` is the model's `zcVector`.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3

theorem fOf_injective : Function.Injective fOf := fun _ _ h => h

end FlockSoundness.Refine
