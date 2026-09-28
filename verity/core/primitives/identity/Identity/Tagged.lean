import Flock.Canon
import Identity.Sha512

/-!
The tagged SHA-512 identity hash (`verity/ml/commitments/identity.py`'s `tagged_sha512` and `identity_digest_sha512`):

~~~text
tagged_sha512(tag, x) = SHA512("verity/tagged-sha512/v1\0" ‖ u32be(len tag) ‖ tag ‖ u64be(len x) ‖ x)
identity_digest_sha512(tag, manifest) = tagged_sha512(tag, canonical_json_bytes(manifest))
~~~

`tag` is UTF-8 and nonempty, and the lengths fit their fields; Python refuses anything else (`TypeError`, `OverflowError`),
and so does `taggedState`. The manifest's canonical bytes are `Flock.canon`'s, which writes what `canonical_json_bytes`
writes on every value both accept (`Flock.canon` refuses a non-integer number, as Python refuses a float). `taggedState` is
the hash after the header, which a long payload of known length is streamed into.
-/

namespace Identity

open Lean (Json)

def TAGGED_SHA512 : ByteArray := "verity/tagged-sha512/v1\x00".toUTF8

/-- The tagged hash's header: everything before the payload's bytes. -/
def taggedHeader (tag : String) (payloadLen : Nat) : Flock.V ByteArray := do
  let t := tag.toUTF8
  if t.size == 0 then throw "hash tag must be a nonempty string"
  if t.size ≥ 2 ^ 32 then throw "hash tag is too long for its u32 length"
  if payloadLen ≥ 2 ^ 64 then throw "tagged hash payload is too long for its u64 length"
  return TAGGED_SHA512 ++ Flock.be32 t.size.toUInt32 ++ t ++ Flock.be64 payloadLen.toUInt64

/-- The hash after the header of a `payloadLen`-byte payload under `tag`. -/
def taggedState (tag : String) (payloadLen : Nat) : Flock.V Sha512.State := do
  return Sha512.init.update (← taggedHeader tag payloadLen)

def taggedSha512 (tag : String) (payload : ByteArray) : Flock.V ByteArray := do
  return ((← taggedState tag payload.size).update payload).finalize

def identityDigestSha512 (tag : String) (manifest : Json) : Flock.V ByteArray := do
  taggedSha512 tag (← Flock.canon manifest).toUTF8

#guard (taggedSha512 "" .empty).toOption.isNone
-- Python's `tagged_sha512("t", b"x")` and `identity_digest_sha512("verity/x", {"b": [1, "é"], "a": None})`
#guard (taggedSha512 "t" "x".toUTF8).toOption.map Flock.hexOf == some
  "92796af09f80b798f659913c883d82ba24d9829b8eb441b7e3fdd5d3b0b0862d87dd5f55932bf52c3fd4afb87e80271f15227594ae6fbd0e20e25ffbf31bb30b"
#guard (identityDigestSha512 "verity/x" (Json.mkObj [("b", Json.arr #[1, "é"]), ("a", .null)])).toOption.map Flock.hexOf
  == some
  "2c80a026ed11d96f4ede3b80f60fafd3d610371faca1f7e012ca42bb51a41dc2d4bf0b94f5c777dac8fc969e5f4b813026a1a17d84f7a200e8317c98d43d48f4"

end Identity
