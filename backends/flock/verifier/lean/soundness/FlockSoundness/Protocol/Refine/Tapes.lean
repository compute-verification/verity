import Flock.Verify
import FlockSoundness.Protocol.Refine.Run
import FlockSoundness.Protocol.Instance

/-!
# The model's messages, read off a decoded proof

A run of the model (`Game.Plays`) takes the prover's messages as a typed tape. For the executable's session these are
the decoded proof's fields, each read as the model's type through the executable's field embedding (`GF128.of`), in the
order the model receives them. The coins need no such function: they are the record's rounds' coins, in order.

This file holds those tapes and the correspondence between the executable's and the model's phase outputs.
-/

namespace FlockSoundness.Refine

open Flock Game

/-- A word of the proof as a model field element. -/
abbrev fOf (a : F128) : 𝔽 := FlockLevel3.GF128.of a

/-- An array of words as a model vector of length `n`. -/
def vecOf (n : ℕ) (a : Array F128) : Fin n → 𝔽 := fun i => fOf a[i.val]!

/-- A pair of words as a pair of model field elements. -/
def pairOf (p : F128 × F128) : 𝔽 × 𝔽 := (fOf p.1, fOf p.2)

end FlockSoundness.Refine
