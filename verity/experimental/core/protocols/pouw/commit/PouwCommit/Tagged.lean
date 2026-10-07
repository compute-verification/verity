import Flock.Hash

/-!
# `tagged`: PoUW's domain-separated hash (`verity.experimental.protocols.pouw.protocol.tagged`)

`tagged tag parts = SHA-512(tag ‖ 0x00 ‖ frame(part)…)`, all 64 bytes. A part is framed as core's randomness frames it
(`verity.ml.randomness.frame`): a tag byte (`b` bytes, `s` a UTF-8 string, `i` a natural as minimal big-endian bytes, one
zero byte for 0), the body's length as 8 bytes big-endian, then the body.

`frame` is also `FlockSoundness.Randomness.frame`, which code may not import (it is in `Security/Proofs`). When core's
randomness moves into a code package, this module imports its `frame` and keeps only `tagged`.
-/

namespace PouwCommit

open Flock

/-- A framed part (`verity.ml.randomness.Part`). -/
inductive Part where
  | bytes (b : ByteArray)
  | str (s : String)
  | nat (n : Nat)

/-- `n` as 8 bytes big-endian (`n` below 2^64). -/
def be8 (n : Nat) : ByteArray := ⟨(Array.range 8).map fun i => (n >>> (8 * (7 - i))).toUInt8⟩

/-- `n` as minimal big-endian bytes, one zero byte for 0 (`int.to_bytes(max(1, ceil(bit_length / 8)), "big")`). -/
def natBytes (n : Nat) : ByteArray :=
  let w := (Nat.log2 n + 8) / 8
  ⟨(Array.range w).map fun i => (n >>> (8 * (w - 1 - i))).toUInt8⟩

/-- `frame(part)`: tag ‖ len(body) as 8 bytes big-endian ‖ body. -/
def frame : Part → ByteArray
  | .bytes b => "b".toUTF8 ++ be8 b.size ++ b
  | .str s => "s".toUTF8 ++ be8 s.toUTF8.size ++ s.toUTF8
  | .nat n => "i".toUTF8 ++ be8 (natBytes n).size ++ natBytes n

/-- SHA-512 over `tag ‖ 0x00 ‖ frame(part)…`, all 64 bytes (`protocol.tagged`). -/
def tagged (tag : String) (parts : List Part) : ByteArray :=
  Sha512.hash (parts.foldl (fun acc p => acc ++ frame p) (tag.toUTF8.push 0))

end PouwCommit
