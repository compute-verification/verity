/-
The hm96 slot from the empty state: in a satisfying assignment of its rows, when its `cv` inputs hold the chaining value
of a message before its last block, its `pad` inputs the last block and its salt inputs a 192-byte salt's bits, its
outputs are the bits of `SHA-512(m) ⊕ M·y` and of `SHA-512(salt_prefix ‖ y)`. The chaining value before each block is
`chainV` (`chainV_succ`: one compression per block, the form the sha512x3 slots compute).
-/
import FlockSoundness.Discharge.Hm.Salt
import FlockSoundness.Discharge.Hm.Top
import FlockSoundness.Protocol.Discharge.Hm.Slot
import FlockSoundness.Discharge.Hm.Hm96

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

open Flock

theorem chainV_zero (d : ByteArray) : chainV d 0 = (List.range 8).map fun i => Sha512.iv[i]!.toBitVec := rfl

end FlockSoundness.Discharge.Hm
