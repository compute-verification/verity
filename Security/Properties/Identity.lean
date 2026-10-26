import Identity.Tagged

/-!
# Core's identity hash is C-Flock's SHA-512

`Identity.Sha512` (package `core_identity`) streams SHA-512 over C-Flock's compression: a state absorbs pieces
(`State.update`) and pads once (`State.finalize`). `Flock.Sha512.hash` pads a whole message and compresses it. These say
that the two compute one function, so that a statement about what SHA-512 hashes, such as a collision among the strings a
run hashes, reads `Flock.Sha512.hash` whichever of the two the code calls:

* `StreamIsFlock`: a state that absorbed pieces `ms` finalizes to `Flock.Sha512.hash` of their concatenation;
* `TaggedIsFlock`: `Identity.taggedSha512 tag payload` is `Flock.Sha512.hash` of `taggedInput tag payload`, the string
  `tagged_sha512` hashes, and both refuse together.

`Proofs.Properties.Identity` proves both.
-/

namespace Properties.Identity

open Identity.Sha512

/-- What `tagged_sha512(tag, payload)` hashes: its header (`Identity.taggedHeader`) and the payload. It refuses what the
header refuses: an empty tag, a tag of `2 ^ 32` bytes or more, a payload of `2 ^ 64` bytes or more. -/
def taggedInput (tag : String) (payload : ByteArray) : Flock.V ByteArray := do
  return (← Identity.taggedHeader tag payload.size) ++ payload

/-- **Streaming SHA-512 is C-Flock's.** Absorbing `ms` piece by piece and finalizing gives `Flock.Sha512.hash` of the
pieces' concatenation. -/
def StreamIsFlock : Prop :=
  ∀ ms : List ByteArray, (ms.foldl State.update init).finalize = Flock.Sha512.hash (ms.foldl (· ++ ·) .empty)

/-- **The tagged SHA-512 is C-Flock's SHA-512 of its input.** `Identity.taggedSha512 tag payload` refuses exactly when
`taggedInput tag payload` does, and is otherwise `Flock.Sha512.hash` of it. -/
def TaggedIsFlock : Prop :=
  ∀ (tag : String) (payload : ByteArray),
    Identity.taggedSha512 tag payload = Flock.Sha512.hash <$> taggedInput tag payload

end Properties.Identity
