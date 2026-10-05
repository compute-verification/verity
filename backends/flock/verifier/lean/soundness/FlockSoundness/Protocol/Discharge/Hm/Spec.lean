/-
SHA-512's compression (FIPS 180-4 §6.4.2) on 64-bit vectors, in the builder's shape (`Flock.HmNets`): the schedule one word at
a time, the rounds from constant `t`, the feed-forward. `Compress` proves the builder's rows force it.
-/
import Flock.HmNets

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

/-- Bit `p` of a list of words in word order: bit `p % 64` of word `p / 64`. -/
def wbit (ws : List (BitVec 64)) (p : Nat) : Bool := (ws.getD (p / 64) 0).getLsbD (p % 64)

end FlockSoundness.Discharge.Hm
