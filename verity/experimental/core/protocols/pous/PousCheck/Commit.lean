import Flock.Canon
import PousCheck.P2
import PousCheck.Prim.Sha512

/-!
The commitment to `W` (`setup.commit`, `p2/PROTOCOL.md` §4 "Setup commitment, salt and verification root"):

~~~text
manifest      canonical_json_bytes(params.manifest())
payload       u64be(len(manifest)) ‖ manifest ‖ W
w_commitment  tagged_sha512("verity-pous/w-commitment/v2", payload)
tagged_sha512(tag, x) = SHA512("verity/tagged-sha512/v1\0" ‖ u32be(len tag) ‖ tag ‖ u64be(len x) ‖ x)
~~~

`verity.ml.commitments.identity.tagged_sha512` is `taggedSha512`. `commitState` is the hash before the first byte of `W`
(`setup._commitment_hash`), which setup streams the decoded payload into, one segment at a time. The manifest is
`P2Params.manifest()` of a v4 codec: `scheme`, `w`, `c0`, `profile`, `blocks_per_segment`, `segments` and `expander`,
through Flock's canonical JSON (sorted keys, compact; its integers and ASCII strings are what Python's
`canonical_json_bytes` writes).
-/

namespace PousCheck.Commit

open Lean (Json)
open PousCheck.Prim

def TAGGED_SHA512 : ByteArray := "verity/tagged-sha512/v1\x00".toUTF8
def COMMIT_TAG : String := "verity-pous/w-commitment/v2"

/-- The tagged hash's header before the payload's bytes: everything but the payload. -/
def taggedHeader (tag : String) (payloadLen : Nat) : ByteArray :=
  let t := tag.toUTF8
  TAGGED_SHA512 ++ Flock.be32 t.size.toUInt32 ++ t ++ Flock.be64 payloadLen.toUInt64

def taggedSha512 (tag : String) (payload : ByteArray) : ByteArray :=
  (Sha512.init.update (taggedHeader tag payload.size) |>.update payload).finalize

def manifestJson (q : P2.Params) : Json :=
  Json.mkObj [("scheme", .str "p2"), ("w", .num q.w), ("c0", .num q.c0), ("profile", .str q.profile),
    ("blocks_per_segment", .num q.blocksPerSegment), ("segments", .num q.segments), ("expander", .str P2.EXPANDER)]

def manifest (q : P2.Params) : Except String ByteArray := do
  return (← Flock.canon (manifestJson q)).toUTF8

/-- The commitment's running hash before `W`: header, `u64be(len manifest)`, manifest. -/
def commitState (q : P2.Params) : Except String Sha512.State := do
  let m ← manifest q
  return Sha512.init.update (taggedHeader COMMIT_TAG (8 + m.size + q.wBytes) ++ Flock.be64 m.size.toUInt64 ++ m)

/-- `setup.commit(params, W)` for `W` whole, of `wBytes` bytes. -/
def commit (q : P2.Params) (w : ByteArray) : Except String ByteArray := do
  if w.size != q.wBytes then throw s!"W has {w.size} bytes, the parameters say {q.wBytes}"
  return ((← commitState q).update w).finalize

end PousCheck.Commit
