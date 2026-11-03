import Flock.Canon
import PousCheck.P2
import Identity.Tagged

/-!
The commitment to `W` (`setup.commit`):

~~~text
manifest      canonical_json_bytes(params.manifest())
payload       u64be(len(manifest)) ‖ manifest ‖ W
w_commitment  tagged_sha512("verity-pous/w-commitment/v2", payload)
tagged_sha512(tag, x) = SHA512("verity/tagged-sha512/v1\0" ‖ u32be(len tag) ‖ tag ‖ u64be(len x) ‖ x)
~~~

`verity.ml.commitments.identity.tagged_sha512` is `Identity.taggedSha512`. `commitState` is the hash before the first byte of `W`
(`setup._commitment_hash`), which setup streams the decoded payload into, one segment at a time. The manifest is
`P2Params.manifest()` of a v4 codec: `scheme`, `w`, `c0`, `profile`, `blocks_per_segment`, `segments` and `expander`,
through Flock's canonical JSON (sorted keys, compact; its integers and ASCII strings are what Python's
`canonical_json_bytes` writes).
-/

namespace PousCheck.Commit

open Lean (Json)
open Identity

def COMMIT_TAG : String := "verity-pous/w-commitment/v2"

def manifestJson (q : P2.Params) : Json :=
  Json.mkObj [("scheme", .str "p2"), ("w", .num q.w), ("c0", .num q.c0), ("profile", .str q.profile),
    ("blocks_per_segment", .num q.blocksPerSegment), ("segments", .num q.segments), ("expander", .str P2.EXPANDER)]

def manifest (q : P2.Params) : Except String ByteArray := do
  return (← Flock.canonUtf8 (manifestJson q)).toUTF8

/-- The commitment's running hash before `W`: header, `u64be(len manifest)`, manifest. -/
def commitState (q : P2.Params) : Except String Sha512.State := do
  let m ← manifest q
  return (← Identity.taggedState COMMIT_TAG (8 + m.size + q.wBytes)).update (Flock.be64 m.size.toUInt64 ++ m)

/-- `setup.commit(params, W)` for `W` whole, of `wBytes` bytes. -/
def commit (q : P2.Params) (w : ByteArray) : Except String ByteArray := do
  if w.size != q.wBytes then throw s!"W has {w.size} bytes, the parameters say {q.wBytes}"
  return ((← commitState q).update w).finalize

end PousCheck.Commit
