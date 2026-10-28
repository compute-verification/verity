import Flock.Hash
import Flock.Bytes
import Identity.Tagged
import PouwCommit.Tagged

/-!
# frame-v3 trees (`verity.ml.commitments.merkle`) over range positions

A frame hashes `FRAME ‖ be32 |tag| ‖ tag ‖ Σ (be64 |part| ‖ part)` under the frame's hash: `sha256` (32-byte digests) or
`sha512` (64). A domain over positions `0 … count-1` is `binding` (32 or 64 bytes), `owner ≥ -2`, and its id
`H("domain", binding, uint(owner + 2), identity, uint(count))`, `identity` the positions' range identity under the frame's
tagged hash. A tree pads its leaves to a power of two with `H("pad", id, uint rank)`, nodes are
`H("node", id, uint level, uint index, l, r)`, and an empty tree's root is `H("empty", id)`. Integers are minimal big-endian,
any size (`natBytes`).
-/

namespace PouwCommit

open Flock

/-- A frame's hash: `sha256` or `sha512`. -/
inductive HashKind where
  | sha256
  | sha512
  deriving BEq, Repr, Inhabited

namespace HashKind

def run : HashKind → ByteArray → ByteArray
  | .sha256 => Sha256.hash
  | .sha512 => Sha512.hash

def bytes : HashKind → Nat
  | .sha256 => 32
  | .sha512 => 64

/-- `identity.tagged_sha256` / `tagged_sha512`: the hash of `prefix ‖ be32 |tag| ‖ tag ‖ be64 |payload| ‖ payload`. The
SHA-512 one is core's `Identity.taggedSha512`, which refuses an empty tag and a payload of `2 ^ 64` bytes or more, as Python
does; this hash is taken only of a range's manifest under a nonempty constant tag, which it never refuses short of a count of
`2 ^ 64` decimal digits, so a refusal is the empty string and `tagged` stays total. -/
def tagged (h : HashKind) (tag : String) (payload : ByteArray) : ByteArray :=
  match h with
  | .sha256 => Sha256.hash ("veritor/tagged-sha256/v1".toUTF8.push 0 ++ be32 tag.toUTF8.size.toUInt32 ++ tag.toUTF8 ++
      be8 payload.size ++ payload)
  | .sha512 => (Identity.taggedSha512 tag payload).toOption.getD .empty

end HashKind

def FRAME : ByteArray := "veritor/protocol/merkle/frame/v3".toUTF8.push 0

/-- The message a frame hashes: `FRAME ‖ be32 |tag| ‖ tag ‖ Σ (be64 |part| ‖ part)`. -/
def frameBytes (tag : String) (parts : List ByteArray) : ByteArray :=
  parts.foldl (fun acc p => acc ++ be8 p.size ++ p) (FRAME ++ be32 tag.toUTF8.size.toUInt32 ++ tag.toUTF8)

def frameHash (h : HashKind) (tag : String) (parts : List ByteArray) : ByteArray := h.run (frameBytes tag parts)

/-- `merkle._frame`'s hash, refusing what it refuses: a tag of `2 ^ 32` bytes or more, or a part of `2 ^ 64` or more, whose
length overflows its field (Python's `int.to_bytes` raises `OverflowError`, where `frameBytes` wraps it). The trees' frames
(`Domain`) are of constant tags and of digests, minimal integers and values under `max_proof_bytes`, which it never
refuses. -/
def frameHash? (h : HashKind) (tag : String) (parts : List ByteArray) : Except String ByteArray :=
  if 2 ^ 32 ≤ tag.toUTF8.size then throw "frame tag must be shorter than 2^32 bytes"
  else if parts.any (2 ^ 64 ≤ ·.size) then throw "frame part must be shorter than 2^64 bytes"
  else pure (frameHash h tag parts)

/-- `RangeIndexedDomain(count)`'s identity: its canonical manifest `{"start":0,"step":1,"stop":count}` under the frame's
tagged hash, tag `veritor/indexed-domain/range/v1`. -/
def rangeIdentity (h : HashKind) (count : Nat) : ByteArray :=
  h.tagged "veritor/indexed-domain/range/v1" s!"\{\"start\":0,\"step\":1,\"stop\":{count}}".toUTF8

/-- A frame-v3 domain over the positions `0 … count-1`, with its id. -/
structure Domain where
  hash : HashKind
  count : Nat
  id : ByteArray

/-- `CommitmentDomain(binding, owner, RangeIndexedDomain(count), hash)`, refusing what its constructor refuses. -/
def Domain.mk? (h : HashKind) (binding : ByteArray) (owner : Int) (count : Nat) : Except String Domain := do
  if binding.size != 32 && binding.size != 64 then
    throw "binding must be 64 bytes (a SHA-512 identity) or 32 (a SHA-256 one)"
  if owner < -2 then throw "owner must be -2 (weights), -1 (boundary) or a replay unit"
  pure ⟨h, count, frameHash h "domain" [binding, natBytes (owner + 2).toNat, rangeIdentity h count, natBytes count]⟩

namespace Domain

/-- The frame leaf over `value` at `rank` (its position too: positions are ranks), schema `schema`. -/
def leaf (d : Domain) (rank : Nat) (schema : String) (value : ByteArray) : ByteArray :=
  frameHash d.hash "leaf" [d.id, natBytes rank, natBytes rank, schema.toUTF8, value]

def node (d : Domain) (level index : Nat) (l r : ByteArray) : ByteArray :=
  frameHash d.hash "node" [d.id, natBytes level, natBytes index, l, r]

def pad (d : Domain) (rank : Nat) : ByteArray := frameHash d.hash "pad" [d.id, natBytes rank]

def empty (d : Domain) : ByteArray := frameHash d.hash "empty" [d.id]

end Domain

/-- `merkle_depth`. -/
def depth (count : Nat) : Nat := if count ≤ 1 then 0 else Nat.log2 (count - 1) + 1

/-- The root over leaf digests in rank order (`MerkleTree.commitment.root`); `none` unless there are `count` of them. -/
def Domain.root (d : Domain) (leaves : Array ByteArray) : Option ByteArray := Id.run do
  if leaves.size != d.count then return none
  if leaves.isEmpty then return some d.empty
  let mut level := leaves ++ (Array.range (2 ^ depth d.count - d.count)).map fun k => d.pad (d.count + k)
  for lv in [0:depth d.count] do
    level := (Array.range (level.size / 2)).map fun i => d.node lv i level[2 * i]! level[2 * i + 1]!
  return level[0]?

/-- `Commitment`'s constructor's refusal: a root of neither 32 nor 64 bytes. -/
def checkRoot (root : ByteArray) : Except String Unit :=
  if root.size != 32 && root.size != 64 then
    throw "commitment root must be a 32-byte (SHA-256 frame) or 64-byte (SHA-512 frame) digest"
  else pure ()

/-- `Opening`'s constructor's refusal: a path item of neither 32 nor 64 bytes. -/
def checkPath (path : List ByteArray) : Except String Unit :=
  if path.any fun p => p.size != 32 && p.size != 64 then throw "opening path must be a tuple of 32- or 64-byte digests"
  else pure ()

/-- `merkle.verify_opening` of `value` at `rank` against `root` (the commitment over `d.count` positions) under the default
limits, `leafOf` the domain's leaf at that rank (`Domain.leaf`, or a domain's own). It throws where Python raises: an opening
of more than `max_proof_bytes` (64 MiB) of value and path, or a value `leafOf` refuses. -/
def Domain.verify (d : Domain) (root : ByteArray) (rank : Nat) (value : ByteArray) (path : List ByteArray)
    (leafOf : ByteArray → Except String ByteArray) : Except String Bool := do
  if root.size != d.hash.bytes || (d.count == 0 && root != d.empty) then return false
  if rank ≥ d.count then return false
  if path.length != depth d.count then return false
  if path.any (·.size != d.hash.bytes) then return false
  if value.size + d.hash.bytes * path.length > 64 <<< 20 then throw "max_proof_bytes exceeded"
  let leaf ← leafOf value
  let (digest, _) := path.zipIdx.foldl (fun (acc : ByteArray × Nat) (sib, lv) =>
    let (dg, cur) := acc
    (if cur % 2 == 0 then d.node lv (cur / 2) dg sib else d.node lv (cur / 2) sib dg, cur / 2)) (leaf, rank)
  return digest == root

end PouwCommit
