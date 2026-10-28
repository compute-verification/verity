import Flock.Hash
import Randomness

/-!
# Bytes, SHA-512 and frame-v3-sha512

The byte encodings the remote audit's formats use (`int.to_bytes`, `int.from_bytes`), C-Flock's SHA-512, and
`verity.ml.commitments.merkle.verify_opening` over a `CommitmentDomain` with `hash = "sha512"` and positions
`RangeIndexedDomain(start, stop)`: a receipt tree and the payload tree start at 0, an append's tree at its first index.

* domain id: `H(frame "domain" [binding, uint(owner + 2), identity_sha512(range), uint count])`, the range's identity
  `tagged_sha512("veritor/indexed-domain/range/v1", {"start":start,"step":1,"stop":stop})`;
* leaf `H(frame "leaf" [id, uint rank, uint position, schema, value])` with `rank = position − start`, node
  `H(frame "node" [id, uint level, uint index, left, right])`;
* a path is `merkle_depth(count)` siblings, `depth = bit_length(count − 1)` (0 for one leaf).

`frame` is the merkle frame v3, `FRAME ‖ u32be |tag| ‖ tag ‖ (u64be |part| ‖ part)*`, and `uint` `_uint`'s minimal
big-endian bytes (`FlockSoundness.Randomness.natBytes`).
-/

namespace RemoteAudit

def sha512 (b : ByteArray) : ByteArray := Flock.Sha512.hash b

/-- `x.to_bytes(n, "big")`, for `x < 256^n`. -/
def be (n x : Nat) : ByteArray := ⟨(Array.range n).map fun i => (x >>> (8 * (n - 1 - i))).toUInt8⟩

/-- `x.to_bytes(n, "little")`, for `x < 256^n`. -/
def le (n x : Nat) : ByteArray := ⟨(Array.range n).map fun i => (x >>> (8 * i)).toUInt8⟩

/-- `int.from_bytes(b, "little")`. -/
def leNat (b : ByteArray) : Nat := b.toList.foldr (fun x acc => acc * 256 + x.toNat) 0

/-- A domain string with its terminating zero byte. -/
def tag0 (s : String) : ByteArray := s.toUTF8.push 0

def zeros (n : Nat) : ByteArray := ⟨Array.replicate n 0⟩

/-- `verity.ml.randomness.frame` of bytes and of a string. -/
def frameBytes (b : ByteArray) : ByteArray := FlockSoundness.Randomness.frame (.bytes b)
def frameStr (s : String) : ByteArray := FlockSoundness.Randomness.frame (.str s)

/-- `body + bytes(-len(body) % 128)`. -/
def pad128 (body : ByteArray) : ByteArray := body ++ zeros ((128 - body.size % 128) % 128)

/-- The first index `r` with `ends[r] > x` (`bisect.bisect_right(ends, x)`), for `ends` nondecreasing. -/
def bisectRight (ends : Array Nat) (x : Nat) : Nat := Id.run do
  let mut lo := 0
  let mut hi := ends.size
  for _ in [0:ends.size + 1] do
    if lo < hi then
      let mid := (lo + hi) / 2
      if x < ends[mid]! then hi := mid else lo := mid + 1
  return lo

namespace FrameV3

def FRAME : ByteArray := tag0 "veritor/protocol/merkle/frame/v3"

def hash (tag : String) (parts : List ByteArray) : ByteArray :=
  let t := tag.toUTF8
  sha512 (parts.foldl (fun acc p => acc ++ be 8 p.size ++ p) (FRAME ++ be 4 t.size ++ t))

def uint (n : Nat) : ByteArray := FlockSoundness.Randomness.natBytes n

/-- `identity.tagged_sha512`. -/
def taggedSha512 (tag : String) (payload : ByteArray) : ByteArray :=
  let t := tag.toUTF8
  sha512 (tag0 "verity/tagged-sha512/v1" ++ be 4 t.size ++ t ++ be 8 payload.size ++ payload)

/-- `RangeIndexedDomain(start, stop).identity_sha512()`. -/
def rangeIdentity (start stop : Nat) : ByteArray :=
  taggedSha512 "veritor/indexed-domain/range/v1" s!"\{\"start\":{start},\"step\":1,\"stop\":{stop}}".toUTF8

/-- A domain over the positions `start … start + count − 1`. -/
structure Domain where
  id : ByteArray
  start : Nat
  count : Nat

/-- `CommitmentDomain(binding, owner, RangeIndexedDomain(start, start + count), hash="sha512")`, `owner ≥ −2`. -/
def Domain.make (binding : ByteArray) (owner : Int) (start count : Nat) : Domain :=
  ⟨hash "domain" [binding, uint (owner + 2).toNat, rangeIdentity start (start + count), uint count], start, count⟩

def Domain.leaf (d : Domain) (rank position : Nat) (schema : String) (value : ByteArray) : ByteArray :=
  hash "leaf" [d.id, uint rank, uint position, schema.toUTF8, value]

def Domain.node (d : Domain) (level index : Nat) (left right : ByteArray) : ByteArray :=
  hash "node" [d.id, uint level, uint index, left, right]

/-- The leaf at `rank` past a tree's `count` leaves, up to its `2^depth`: `H(frame "pad" [id, uint rank])`. -/
def Domain.pad (d : Domain) (rank : Nat) : ByteArray := hash "pad" [d.id, uint rank]

/-- `merkle_depth`. -/
def depth (count : Nat) : Nat := if count ≤ 1 then 0 else Nat.log2 (count - 1) + 1

/-- The root a value at `position` climbs to along `path`. -/
def climb (d : Domain) (position : Nat) (schema : String) (value : ByteArray) (path : Array ByteArray) : ByteArray :=
  Id.run do
  let rank := position - d.start
  let mut digest := d.leaf rank position schema value
  let mut cursor := rank
  for h : level in [0:path.size] do
    let sib := path[level]
    digest := if cursor % 2 = 0 then d.node level (cursor >>> 1) digest sib else d.node level (cursor >>> 1) sib digest
    cursor := cursor >>> 1
  return digest

/-- `verify_opening(domain, Commitment(root, domain.count), Opening(position, value, path), schema, limits)`: the root
64 bytes, the position in the domain's range, the path `depth count` siblings of 64 bytes, and the leaf climbs to the
root. (The limits' 64 MiB proof cap is above any path here: 64 siblings at most.) -/
def verifyOpening (d : Domain) (root : ByteArray) (position : Nat) (value : ByteArray) (schema : String)
    (path : Array ByteArray) : Bool :=
  root.size == 64 && d.count > 0 && d.start ≤ position && position < d.start + d.count &&
    path.size == depth d.count && path.all (·.size == 64) && climb d position schema value path == root

end FrameV3

end RemoteAudit
