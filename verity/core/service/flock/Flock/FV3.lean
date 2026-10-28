/-
# frame-v3 (PROTOCOL.md §16.6; `verity.ml.commitments.merkle`)

The tree every committed value is opened from, as the verifiers compute it; its spec is `Security/Definitions/Hash/Tree.lean`.
A frame hashes `FRAME ‖ be32 |tag| ‖ tag ‖ Σ (be64 |part| ‖ part)` (`input`, `merkle._message`), and an integer in a frame
is its minimal big-endian bytes, one zero byte for 0 (`uint`, `merkle._uint`). A tree pads its leaves to a power of two
with `"pad"` leaves and hashes `"node"`s level by level up to its root.

* `hash`, `leaf`, `root`: the frame over a hash `H`, SHA-512 unless the caller names another. C-Flock's SHA-256 statements
  (`Blake3Row`) name `Sha256.hash`, the one other hash a frame takes.
* `hashM`, `leafM`, `rootM`: the same over a frame-hash action of a monad. The oracle verifier (`Flock.With`) runs them
  through its oracle (`FV3.hashWith`, `leafWith`, `rootWith`); `hash`, `leaf` and `root` are them in `Id`
  (`hash_eq_hashM`, `leaf_eq_leafM`, `root_eq_rootM`).
* `FV3.Sha512`: frame-v3-sha512, the commitment format (`CommitmentDomain(…, hash="sha512")`), for every caller that
  isn't C-Flock's SHA-256 statement. A `Domain` over the positions `start … start + count − 1` (`Domain.make`), its leaf,
  node, pad and empty root, `levels`, `root`, `path`, `Domain.climb` and `Domain.verifyOpening`; and the registered rows'
  climb by 64-byte siblings (`node`, `climbL`, `sibs`, `climb`, `leafAs`), which `Flock.Registered` exports.
-/
import Flock.Hash
import Flock.Bytes
import Flock.Net

namespace Flock
namespace FV3

def FRAME : ByteArray := "veritor/protocol/merkle/frame/v3".toUTF8.push 0

/-- The bytes a frame hashes: `FRAME ‖ be32 |tag| ‖ tag ‖ Σ (be64 |part| ‖ part)`. -/
def input (tag : String) (parts : List ByteArray) : ByteArray :=
  let t := tag.toUTF8
  parts.foldl (fun acc p => acc ++ be64 p.size.toUInt64 ++ p) (FRAME ++ be32 t.size.toUInt32 ++ t)

def hash (tag : String) (parts : List ByteArray) (H : ByteArray → ByteArray := Sha512.hash) : ByteArray :=
  H (input tag parts)

/-- `_uint`: `x`'s minimal big-endian bytes, `(bit_length + 7) / 8` of them and at least one. -/
def uint (x : Nat) : ByteArray :=
  let n := (Nat.log2 x + 8) / 8
  ⟨(Array.range n).map fun i => (x >>> (8 * (n - 1 - i))).toUInt8⟩

section Monadic

variable {m : Type → Type} [Monad m]

/-- `hash` with the hash an action `sha` of `m`. -/
def hashM (sha : ByteArray → m ByteArray) (tag : String) (parts : List ByteArray) : m ByteArray :=
  sha (input tag parts)

/-- The leaf of `payload` at `i` (its rank and its position) under schema `schema`, `h` the frame's hash. -/
def leafM (h : String → List ByteArray → m ByteArray) (dom : ByteArray) (i : Nat) (schema : String) (payload : ByteArray) :
    m ByteArray :=
  h "leaf" [dom, uint i, uint i, schema.toUTF8, payload]

/-- The root over `leaves`, padded to `nextPow2` of their count, `h` the frame's hash. -/
def rootM (h : String → List ByteArray → m ByteArray) (dom : ByteArray) (leaves : Array ByteArray) : m ByteArray := do
  let n := leaves.size
  let width := nextPow2 n
  let mut level := leaves ++ (← (Array.range (width - n)).mapM fun k => h "pad" [dom, uint (n + k)])
  for depth in [0:log2ceil width] do
    level ← (Array.range (level.size / 2)).mapM fun i =>
      h "node" [dom, uint depth, uint i, level[2 * i]!, level[2 * i + 1]!]
  return level.getD 0 ByteArray.empty

end Monadic

def leaf (dom : ByteArray) (i : Nat) (schema : String) (payload : ByteArray) (H : ByteArray → ByteArray := Sha512.hash) :
    ByteArray :=
  hash "leaf" [dom, uint i, uint i, schema.toUTF8, payload] H

def root (dom : ByteArray) (leaves : Array ByteArray) (H : ByteArray → ByteArray := Sha512.hash) : ByteArray := Id.run do
  let n := leaves.size
  let width := nextPow2 n
  let mut level := leaves ++ (Array.range (width - n)).map fun k => hash "pad" [dom, uint (n + k)] H
  for depth in [0:log2ceil width] do
    level := (Array.range (level.size / 2)).map fun i => hash "node" [dom, uint depth, uint i, level[2 * i]!, level[2 * i + 1]!] H
  return level.getD 0 ByteArray.empty

theorem hash_eq_hashM (H : ByteArray → ByteArray) (tag : String) (parts : List ByteArray) :
    hash tag parts H = Id.run (hashM (m := Id) (pure ∘ H) tag parts) := rfl

theorem leaf_eq_leafM (H : ByteArray → ByteArray) (dom : ByteArray) (i : Nat) (schema : String) (payload : ByteArray) :
    leaf dom i schema payload H = Id.run (leafM (m := Id) (fun t ps => pure (hash t ps H)) dom i schema payload) := rfl

theorem root_eq_rootM (H : ByteArray → ByteArray) (dom : ByteArray) (leaves : Array ByteArray) :
    root dom leaves H = Id.run (rootM (m := Id) (fun t ps => pure (hash t ps H)) dom leaves) := rfl

/-! ## frame-v3-sha512 -/

namespace Sha512

/-- The frame under SHA-512. -/
def hash (tag : String) (parts : List ByteArray) : ByteArray := FV3.hash tag parts Flock.Sha512.hash

def TAGGED : ByteArray := "verity/tagged-sha512/v1".toUTF8.push 0

/-- `identity.tagged_sha512`: SHA-512 of `TAGGED ‖ be32 |tag| ‖ tag ‖ be64 |payload| ‖ payload`. -/
def taggedSha512 (tag : String) (payload : ByteArray) : ByteArray :=
  let t := tag.toUTF8
  Flock.Sha512.hash (TAGGED ++ be32 t.size.toUInt32 ++ t ++ be64 payload.size.toUInt64 ++ payload)

/-- `RangeIndexedDomain(start, stop).identity_sha512()`, step 1. -/
def rangeIdentity (start stop : Nat) : ByteArray :=
  taggedSha512 "veritor/indexed-domain/range/v1" s!"\{\"start\":{start},\"step\":1,\"stop\":{stop}}".toUTF8

/-- A domain over the positions `start … start + count − 1`; position `start + r` is rank `r`. -/
structure Domain where
  id : ByteArray
  start : Nat := 0
  count : Nat

/-- `CommitmentDomain(binding, owner, RangeIndexedDomain(start, start + count), hash="sha512")`; the binding is 32 or 64
bytes and `owner ≥ −2`, which the callers fix. -/
def Domain.make (binding : ByteArray) (owner : Int) (start count : Nat) : Domain :=
  { id := hash "domain" [binding, uint (owner + 2).toNat, rangeIdentity start (start + count), uint count], start, count }

namespace Domain

def leaf (d : Domain) (rank position : Nat) (schema : String) (value : ByteArray) : ByteArray :=
  hash "leaf" [d.id, uint rank, uint position, schema.toUTF8, value]

def node (d : Domain) (level index : Nat) (left right : ByteArray) : ByteArray :=
  hash "node" [d.id, uint level, uint index, left, right]

/-- The leaf at `rank` past the tree's `count` leaves, up to its `2^depth`. -/
def pad (d : Domain) (rank : Nat) : ByteArray := hash "pad" [d.id, uint rank]

def emptyRoot (d : Domain) : ByteArray := hash "empty" [d.id]

/-- `tree_levels`: the leaves padded to `2 ^ log2ceil count` (`merkle_depth`), then each level of nodes up to the root;
none for no leaves. -/
def levels (d : Domain) (leaves : Array ByteArray) : Array (Array ByteArray) := Id.run do
  if leaves.isEmpty then return #[]
  let width := nextPow2 leaves.size
  let mut level := leaves ++ (Array.range (width - leaves.size)).map fun k => d.pad (leaves.size + k)
  let mut out := #[level]
  for lvl in [0:log2ceil leaves.size] do
    level := (Array.range (level.size / 2)).map fun i => d.node lvl i level[2 * i]! level[2 * i + 1]!
    out := out.push level
  return out

/-- The root over the leaves (`MerkleTree.commitment.root`), the empty root for none. -/
def root (d : Domain) (leaves : Array ByteArray) : ByteArray :=
  match (d.levels leaves).back? with
  | some top => top[0]!
  | none => d.emptyRoot

end Domain

/-- `MerkleTree.open`'s path: the siblings of leaf `rank` in the levels `ls` (`Domain.levels`), leaf level first. -/
def path (ls : Array (Array ByteArray)) (rank : Nat) : Array ByteArray := Id.run do
  let mut out := #[]
  let mut r := rank
  for level in ls.pop do
    out := out.push level[r ^^^ 1]!
    r := r >>> 1
  return out

/-- The node over `l` and `r` at index `idx` of level `d`, in the tree of the domain whose id is `dom`. -/
def node (dom : ByteArray) (d idx : Nat) (l r : ByteArray) : ByteArray :=
  FV3.hash "node" [dom, FV3.uint d, FV3.uint idx, l, r] Flock.Sha512.hash

/-- The root a digest `cur` at index `idx` of level `d` climbs to through the siblings `sibs`, lowest first. -/
def climbL (dom : ByteArray) : ByteArray → Nat → Nat → List ByteArray → ByteArray
  | cur, _, _, [] => cur
  | cur, idx, d, sib :: rest =>
    climbL dom (if idx % 2 = 0 then node dom d (idx / 2) cur sib else node dom d (idx / 2) sib cur) (idx / 2) (d + 1) rest

/-- A path's `depth` siblings, 64 bytes each, leaf level first. -/
def sibs (path : ByteArray) (depth : Nat) : List ByteArray :=
  (List.range depth).map fun d => path.extract (64 * d) (64 * d + 64)

/-- The root a leaf at `i` climbs to through `path` (the siblings, leaf level first, 64 bytes each). -/
def climb (dom leaf : ByteArray) (i depth : Nat) (path : ByteArray) : ByteArray := climbL dom leaf i 0 (sibs path depth)

/-- The leaf at `pos` (its rank and its position) of a row whose hm96 leaf of schema `schema` is `payload`. -/
def leafAs (schema : String) (dom : ByteArray) (pos : Nat) (payload : ByteArray) : ByteArray :=
  FV3.leaf dom pos schema payload Flock.Sha512.hash

/-- The root a value at `position` climbs to along `path`, leaf level first. -/
def Domain.climb (d : Domain) (position : Nat) (schema : String) (value : ByteArray) (path : Array ByteArray) : ByteArray :=
  let rank := position - d.start
  climbL d.id (d.leaf rank position schema value) rank 0 path.toList

/-- `verify_opening(domain, Commitment(root, count), Opening(position, value, path), schema, limits)` without the limits:
the commitment's count is the domain's and its root 64 bytes, the position in the domain's range, the path
`log2ceil count` siblings (`merkle_depth`) of 64 bytes, and the leaf climbs to the root. The limits' `max_proof_bytes`
(64 MiB of value and path) is the caller's to enforce where its values can reach it. -/
def Domain.verifyOpening (d : Domain) (root : ByteArray) (count position : Nat) (value : ByteArray) (schema : String)
    (path : Array ByteArray) : Bool :=
  count == d.count && root.size == 64 && d.start ≤ position && position < d.start + d.count &&
    path.size == log2ceil d.count && path.all (·.size == 64) && d.climb position schema value path == root

end Sha512

end FV3
end Flock
