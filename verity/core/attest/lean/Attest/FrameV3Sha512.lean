import Flock.Public
import Randomness

/-!
# frame-v3-sha512 over a range of positions

`verity.ml.commitments.merkle.CommitmentDomain` with `hash = "sha512"` and `positions = RangeIndexedDomain(0, count)`, and
`tree_levels`, `verify_opening` over it. Every hash is C-Flock's frame (`Flock.FV3.hash`) through SHA-512. Integers are
`_uint`'s minimal big-endian bytes (`FlockSoundness.Randomness.natBytes`), of any size: `Flock.FV3.uint` is 8 bytes at most.

* domain id: `H(frame "domain" [binding, uint(owner + 2), identity_sha512(range), uint count])`, the range's identity
  `tagged_sha512("veritor/indexed-domain/range/v1", {"start":0,"step":1,"stop":count})`;
* leaf `H(frame "leaf" [id, uint rank, uint position, schema, value])`, node `H(frame "node" [id, uint level, uint index,
  left, right])`, pad `H(frame "pad" [id, uint rank])`, empty root `H(frame "empty" [id])`;
* the tree pads its leaves to a power of two, `2^depth` with `depth = bit_length(count − 1)` (0 for one leaf).

The memory challenger's session records and every auditor party's period tree (`Attest.Period`) are trees of this
frame; a later shared package of core's commitments takes it as it is.
-/

namespace FrameV3Sha512

open Flock

def uint (n : Nat) : ByteArray := FlockSoundness.Randomness.natBytes n

def hash (tag : String) (parts : List ByteArray) : ByteArray := FV3.hash tag parts Sha512.hash

def TAGGED : ByteArray := "verity/tagged-sha512/v1".toUTF8.push 0

/-- `identity.tagged_sha512`. -/
def taggedSha512 (tag : String) (payload : ByteArray) : ByteArray :=
  let t := tag.toUTF8
  Sha512.hash (TAGGED ++ be32 t.size.toUInt32 ++ t ++ be64 payload.size.toUInt64 ++ payload)

/-- `RangeIndexedDomain(start, stop, step).identity_sha512()`. -/
def rangeIdentity (start stop step : Nat) : ByteArray :=
  taggedSha512 "veritor/indexed-domain/range/v1" s!"\{\"start\":{start},\"step\":{step},\"stop\":{stop}}".toUTF8

/-- A domain over the positions `0 … count − 1`, each its own rank. -/
structure Domain where
  id : ByteArray
  count : Nat

/-- `CommitmentDomain(binding, owner, RangeIndexedDomain(0, count), hash="sha512")`; the binding is 32 or 64 bytes and
`owner ≥ −2`, which the callers here fix. -/
def Domain.make (binding : ByteArray) (owner : Int) (count : Nat) : Domain :=
  ⟨hash "domain" [binding, uint (owner + 2).toNat, rangeIdentity 0 count 1, uint count], count⟩

def Domain.leaf (d : Domain) (rank position : Nat) (schema : String) (value : ByteArray) : ByteArray :=
  hash "leaf" [d.id, uint rank, uint position, schema.toUTF8, value]

def Domain.node (d : Domain) (level index : Nat) (left right : ByteArray) : ByteArray :=
  hash "node" [d.id, uint level, uint index, left, right]

def Domain.pad (d : Domain) (rank : Nat) : ByteArray := hash "pad" [d.id, uint rank]

def Domain.emptyRoot (d : Domain) : ByteArray := hash "empty" [d.id]

/-- `merkle_depth`. -/
def depth (count : Nat) : Nat := if count ≤ 1 then 0 else Nat.log2 (count - 1) + 1

/-- `tree_levels`: the leaves padded to `2^depth`, then each level of nodes up to the root; none for no leaves. -/
def levels (d : Domain) (leaves : Array ByteArray) : Array (Array ByteArray) := Id.run do
  if leaves.isEmpty then return #[]
  let width := 2 ^ depth leaves.size
  let mut level := leaves ++ (Array.range (width - leaves.size)).map fun k => d.pad (leaves.size + k)
  let mut out := #[level]
  for lvl in [0:depth leaves.size] do
    level := (Array.range (level.size / 2)).map fun i => d.node lvl i level[2 * i]! level[2 * i + 1]!
    out := out.push level
  return out

/-- The root over the leaves, the empty root for none. -/
def root (d : Domain) (leaves : Array ByteArray) : ByteArray :=
  match (levels d leaves).back? with
  | some top => top[0]!
  | none => d.emptyRoot

/-- The siblings of leaf `rank`, leaf side first. -/
def path (ls : Array (Array ByteArray)) (rank : Nat) : Array ByteArray := Id.run do
  let mut out := #[]
  let mut r := rank
  for level in ls.pop do
    out := out.push level[r ^^^ 1]!
    r := r >>> 1
  return out

/-- The root a value at `position` climbs to along `path`. -/
def climb (d : Domain) (position : Nat) (schema : String) (value : ByteArray) (path : Array ByteArray) : ByteArray := Id.run do
  let mut digest := d.leaf position position schema value
  let mut cursor := position
  for h : level in [0:path.size] do
    let sib := path[level]
    digest := if cursor % 2 = 0 then d.node level (cursor >>> 1) digest sib else d.node level (cursor >>> 1) sib digest
    cursor := cursor >>> 1
  return digest

/-- `verify_opening(domain, Commitment(root, count), Opening(position, value, path), schema, limits)`: the commitment's
count is the domain's and its root 64 bytes, the position in range, the path `depth count` siblings of 64 bytes, and the
leaf climbs to the root. (A root or sibling of another width is refused by `Commitment` or `Opening` in the Python,
which a caller there reads as no opening.) -/
def verifyOpening (d : Domain) (root : ByteArray) (count position : Nat) (value : ByteArray) (schema : String)
    (path : Array ByteArray) : Bool :=
  count == d.count && root.size == 64 && position < d.count && path.size == depth d.count &&
    path.all (·.size == 64) && climb d position schema value path == root

end FrameV3Sha512
