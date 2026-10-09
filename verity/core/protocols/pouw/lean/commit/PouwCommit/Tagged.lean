import Randomness

/-!
# `tagged`: PoUW's domain-separated hash (`verity.core.protocols.pouw.protocol.tagged`)

`tagged tag parts = SHA-512(tag ‖ 0x00 ‖ frame(part)…)`, all 64 bytes. A part is framed as core's randomness frames it
(`verity.ml.randomness.frame`, here `FlockSoundness.Randomness.frame`): a tag byte (`b` bytes, `s` a UTF-8 string, `i` a
natural as minimal big-endian bytes, one zero byte for 0), the body's length as 8 bytes big-endian, then the body.
-/

namespace PouwCommit

open Flock

export FlockSoundness.Randomness (Part frame be8 natBytes)

/-- SHA-512 over `tag ‖ 0x00 ‖ frame(part)…`, all 64 bytes (`protocol.tagged`). -/
def tagged (tag : String) (parts : List Part) : ByteArray :=
  Sha512.hash (parts.foldl (fun acc p => acc ++ frame p) (tag.toUTF8.push 0))

end PouwCommit
