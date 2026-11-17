module

public import Identity.Tagged

/-! Core's identity hash is Flock's SHA-512. -/

@[expose] public section

namespace Properties.Identity

open Identity.Sha512

/-- What `tagged_sha512(tag, payload)` hashes: its header (`Identity.taggedHeader`) and the payload. It refuses what the
header refuses: an empty tag, a tag of `2 ^ 32` bytes or more, a payload of `2 ^ 64` bytes or more. -/
def taggedInput (tag : String) (payload : ByteArray) : Flock.V ByteArray := do
  return (← Identity.taggedHeader tag payload.size) ++ payload

/-- **Streaming SHA-512 is Flock's.** Absorbing `ms` piece by piece and finalizing gives `Flock.Sha512.hash` of the
pieces' concatenation. -/
def StreamIsFlock : Prop :=
  ∀ ms : List ByteArray, (ms.foldl State.update init).finalize = Flock.Sha512.hash (ms.foldl (· ++ ·) .empty)

/-- **The tagged SHA-512 is Flock's SHA-512 of its input.** `Identity.taggedSha512 tag payload` refuses exactly when
`taggedInput tag payload` does, and is otherwise `Flock.Sha512.hash` of it. -/
def TaggedIsFlock : Prop :=
  ∀ (tag : String) (payload : ByteArray),
    Identity.taggedSha512 tag payload = Flock.Sha512.hash <$> taggedInput tag payload

end Properties.Identity
