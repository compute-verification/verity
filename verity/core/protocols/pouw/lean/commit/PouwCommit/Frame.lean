import Flock.Hash
import Flock.Bytes
import Flock.FV3
import PouwCommit.Tagged
import Identity.Domains

/-!
# frame-v3 trees (`verity.ml.commitments.merkle`) over range positions

C-Flock's frame-v3 (`Flock.FV3`) under the frame's hash: `sha256` (`Flock.FV3.Sha256`, 32-byte digests) or `sha512`
(`Flock.FV3.Sha512`, 64). A domain over positions `0 … count-1` is `binding` (32 or 64 bytes), `owner ≥ -2`, and its id
`H("domain", binding, uint(owner + 2), identity, uint(count))`, `identity` the positions' range identity under the frame's
tagged hash; its leaves, nodes, pads and empty root are `Flock.FV3.Domain`'s. What this module adds is what the Python
refuses: a binding or owner `CommitmentDomain` refuses (`Domain.mk?`), a frame `merkle._frame` refuses (`frameHash?`), and
`verify_opening`'s checks and limits (`Domain.verify`).
-/

namespace PouwCommit

open Flock

/-- A frame's hash: `sha256` or `sha512`. -/
inductive Hash where
  | sha256
  | sha512
  deriving BEq, Repr, Inhabited

namespace Hash

/-- The hash's frame-v3 scheme: its digest, its digests' size and its tagged hash. -/
def scheme : Hash → FV3.Scheme
  | .sha256 => FV3.Sha256.scheme
  | .sha512 => FV3.Sha512.scheme

def run (h : Hash) : ByteArray → ByteArray := h.scheme.H

def bytes (h : Hash) : Nat := h.scheme.bytes

end Hash

/-- The message a frame hashes: `FRAME ‖ be32 |tag| ‖ tag ‖ Σ (be64 |part| ‖ part)`. -/
def frameBytes (tag : String) (parts : List ByteArray) : ByteArray := FV3.input tag parts

def frameHash (h : Hash) (tag : String) (parts : List ByteArray) : ByteArray := h.scheme.hash tag parts

/-- `merkle._frame`'s hash, refusing what it refuses: a tag of `2 ^ 32` bytes or more, or a part of `2 ^ 64` or more, whose
length overflows its field (Python's `int.to_bytes` raises `OverflowError`, where `frameBytes` wraps it). The trees' frames
(`Domain`) are of constant tags and of digests, minimal integers and values under `max_proof_bytes`, which it never
refuses. -/
def frameHash? (h : Hash) (tag : String) (parts : List ByteArray) : Except String ByteArray :=
  if 2 ^ 32 ≤ tag.toUTF8.size then throw "frame tag must be shorter than 2^32 bytes"
  else if parts.any (2 ^ 64 ≤ ·.size) then throw "frame part must be shorter than 2^64 bytes"
  else pure (frameHash h tag parts)

/-- `RangeIndexedDomain(count)`'s identity: its canonical manifest `{"start":0,"step":1,"stop":count}` under the frame's
tagged hash, tag `veritor/indexed-domain/range/v1`. -/
def rangeIdentity (h : Hash) (count : Nat) : ByteArray := h.scheme.rangeIdentity 0 count

/-- A frame-v3 domain over the positions `0 … count-1`, with its id. -/
structure Domain where
  hash : Hash
  count : Nat
  id : ByteArray

/-- `CommitmentDomain(binding, owner, RangeIndexedDomain(count), hash)`, refusing what its constructor refuses. -/
def Domain.mk? (h : Hash) (binding : ByteArray) (owner : Int) (count : Nat) : Except String Domain := do
  if binding.size != 32 && binding.size != 64 then
    throw "binding must be 64 bytes (a SHA-512 identity) or 32 (a SHA-256 one)"
  if owner < -2 then throw "owner must be -2 (weights), -1 (boundary) or a replay unit"
  pure ⟨h, count, frameHash h "domain" [binding, FV3.uint (owner + 2).toNat, rangeIdentity h count, FV3.uint count]⟩

namespace Domain

/-- The domain as C-Flock's: positions start at 0. -/
def fv3 (d : Domain) : FV3.Domain d.hash.scheme := { id := d.id, count := d.count }

/-- The frame leaf over `value` at `rank` (its position too: positions are ranks), schema `schema`. -/
def leaf (d : Domain) (rank : Nat) (schema : String) (value : ByteArray) : ByteArray := d.fv3.leaf rank rank schema value

def node (d : Domain) (level index : Nat) (l r : ByteArray) : ByteArray := d.fv3.node level index l r

def pad (d : Domain) (rank : Nat) : ByteArray := d.fv3.pad rank

def empty (d : Domain) : ByteArray := d.fv3.emptyRoot

end Domain

/-- `merkle_depth`. -/
def depth (count : Nat) : Nat := if count ≤ 1 then 0 else Nat.log2 (count - 1) + 1

/-- The root over leaf digests in rank order (`MerkleTree.commitment.root`); `none` unless there are `count` of them. -/
def Domain.root (d : Domain) (leaves : Array ByteArray) : Option ByteArray :=
  if leaves.size != d.count then none else some (d.fv3.root leaves)

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

/-! Each separator below keeps its literal while a lock reads its definition, whose record hashes the value as
written: tied here to its entry of the table (`Identity.Domains`), it reads the entry at its next record change. -/
example : ("veritor/tagged-sha256/v1" : String) = Domains.TAGGED_SHA256 := rfl
example : FV3.FRAME = Domains.FRAME_V3.toUTF8.push 0 := rfl
example : ("veritor/indexed-domain/range/v1" : String) = Domains.INDEXED_RANGE := rfl

end PouwCommit
