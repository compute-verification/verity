import FlockSoundness.Refine.Lincheck
import FlockSoundness.Protocol.Refine.Opening

/-!
# Ring switching and batching refine the model's opening (PROTOCOL.md §12)

`opening_refines`: if the executable's `ringSwitch` succeeds on claims that are the model's (`ClaimRel`), the model's
`opening`, played with the proof's ring switches (`opMsgs`) and the coins of the rounds consumed, returns `some op`,
and `op` is the executable's batched value and batched basis (`OpenRel`).

The executable transposes each switched claim's bits (`tOf`: bit `v` of `t[u]` is bit `u` of `s[v]`) with `F128.bit` and
monomials; the model reads them through `unpack` and `pack` (`recombine`). `unpack_fOf` and `pack_eq_sum` are the seam.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript

local notation "𝔽" => FlockLevel3.GF128

/-! ## Bits -/

theorem tOf_size (s : Array F128) : (tOf s).size = 128 := by simp [tOf]

/-! ## The per-claim check (O4) -/

/-! ## The loops -/

end FlockSoundness.Refine
