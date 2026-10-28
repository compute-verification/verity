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
* `FV3.Domain sc`: a domain over the positions `start … start + count − 1` of a frame under the `Scheme` `sc` (its hash,
  digest size and tagged hash), with its leaf, node, pad and empty root, `levels`, `root`, `climb` and `verifyOpening`;
  `FV3.path` is an opening's path from the levels.
* `FV3.Sha512`: frame-v3-sha512, the commitment format (`CommitmentDomain(…, hash="sha512")`), for every caller that
  isn't C-Flock's SHA-256 statement or PoUW's SHA-256 frames: its `scheme`, `Domain` (`FV3.Domain scheme`),
  `Domain.make`, `rangeIdentity` and `path`; and the registered rows' climb by 64-byte siblings (`node`, `climbL`,
  `sibs`, `climb`, `leafAs`), which `Flock.Registered` exports.
* `FV3.Sha256`: the same domain API under SHA-256, for PoUW's SHA-256 frames (Pearl-C's `h0`).
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

/-! ## frame-v3 domains -/

/-- A frame's hash: its digest function `H`, its digests' size in bytes, and the tagged hash (`identity.tagged_sha256`,
`tagged_sha512`) a range's identity is taken under. -/
structure Scheme where
  H : ByteArray → ByteArray
  bytes : Nat
  tagged : String → ByteArray → ByteArray

namespace Scheme

variable (sc : Scheme)

def hash (tag : String) (parts : List ByteArray) : ByteArray := FV3.hash tag parts sc.H

/-- `RangeIndexedDomain(start, stop)`'s identity under the scheme's tagged hash, step 1. -/
def rangeIdentity (start stop : Nat) : ByteArray :=
  sc.tagged "veritor/indexed-domain/range/v1" s!"\{\"start\":{start},\"step\":1,\"stop\":{stop}}".toUTF8

/-- The node over `l` and `r` at index `idx` of level `d`, in the tree of the domain whose id is `dom`. -/
def node (dom : ByteArray) (d idx : Nat) (l r : ByteArray) : ByteArray := sc.hash "node" [dom, uint d, uint idx, l, r]

/-- The root a digest `cur` at index `idx` of level `d` climbs to through the siblings `sibs`, lowest first. -/
def climbL (dom : ByteArray) : ByteArray → Nat → Nat → List ByteArray → ByteArray
  | cur, _, _, [] => cur
  | cur, idx, d, sib :: rest =>
    climbL dom (if idx % 2 = 0 then sc.node dom d (idx / 2) cur sib else sc.node dom d (idx / 2) sib cur) (idx / 2) (d + 1)
      rest

end Scheme

/-- A domain of a frame under `sc` over the positions `start … start + count − 1`; position `start + r` is rank `r`. -/
structure Domain (sc : Scheme) where
  id : ByteArray
  start : Nat := 0
  count : Nat

/-- `CommitmentDomain(binding, owner, RangeIndexedDomain(start, start + count), hash)`; the binding is 32 or 64 bytes and
`owner ≥ −2`, which the callers fix. -/
def Domain.make (sc : Scheme) (binding : ByteArray) (owner : Int) (start count : Nat) : Domain sc :=
  { id := sc.hash "domain" [binding, uint (owner + 2).toNat, sc.rangeIdentity start (start + count), uint count], start,
    count }

namespace Domain

variable {sc : Scheme}

def leaf (d : Domain sc) (rank position : Nat) (schema : String) (value : ByteArray) : ByteArray :=
  sc.hash "leaf" [d.id, uint rank, uint position, schema.toUTF8, value]

def node (d : Domain sc) (level index : Nat) (left right : ByteArray) : ByteArray := sc.node d.id level index left right

/-- The leaf at `rank` past the tree's `count` leaves, up to its `2^depth`. -/
def pad (d : Domain sc) (rank : Nat) : ByteArray := sc.hash "pad" [d.id, uint rank]

def emptyRoot (d : Domain sc) : ByteArray := sc.hash "empty" [d.id]

/-- `tree_levels`: the leaves padded to `2 ^ log2ceil count` (`merkle_depth`), then each level of nodes up to the root;
none for no leaves. -/
def levels (d : Domain sc) (leaves : Array ByteArray) : Array (Array ByteArray) := Id.run do
  if leaves.isEmpty then return #[]
  let width := nextPow2 leaves.size
  let mut level := leaves ++ (Array.range (width - leaves.size)).map fun k => d.pad (leaves.size + k)
  let mut out := #[level]
  for lvl in [0:log2ceil leaves.size] do
    level := (Array.range (level.size / 2)).map fun i => d.node lvl i level[2 * i]! level[2 * i + 1]!
    out := out.push level
  return out

/-- The root over the leaves (`MerkleTree.commitment.root`), the empty root for none. -/
def root (d : Domain sc) (leaves : Array ByteArray) : ByteArray :=
  match (d.levels leaves).back? with
  | some top => top[0]!
  | none => d.emptyRoot

/-- The root a value at `position` climbs to along `path`, leaf level first. -/
def climb (d : Domain sc) (position : Nat) (schema : String) (value : ByteArray) (path : Array ByteArray) : ByteArray :=
  let rank := position - d.start
  sc.climbL d.id (d.leaf rank position schema value) rank 0 path.toList

/-- `verify_opening(domain, Commitment(root, count), Opening(position, value, path), schema, limits)` without the limits:
the commitment's count is the domain's and its root one digest, the position in the domain's range, the path
`log2ceil count` siblings (`merkle_depth`) of one digest each, and the leaf climbs to the root. The limits'
`max_proof_bytes` (64 MiB of value and path) is the caller's to enforce where its values can reach it. -/
def verifyOpening (d : Domain sc) (root : ByteArray) (count position : Nat) (value : ByteArray) (schema : String)
    (path : Array ByteArray) : Bool :=
  count == d.count && root.size == sc.bytes && d.start ≤ position && position < d.start + d.count &&
    path.size == log2ceil d.count && path.all (·.size == sc.bytes) && d.climb position schema value path == root

end Domain

/-- `MerkleTree.open`'s path: the siblings of leaf `rank` in the levels `ls` (`Domain.levels`), leaf level first. -/
def path (ls : Array (Array ByteArray)) (rank : Nat) : Array ByteArray := Id.run do
  let mut out := #[]
  let mut r := rank
  for level in ls.pop do
    out := out.push level[r ^^^ 1]!
    r := r >>> 1
  return out

/-! ## frame-v3-sha512 -/

namespace Sha512

/-- The frame under SHA-512. -/
def hash (tag : String) (parts : List ByteArray) : ByteArray := FV3.hash tag parts Flock.Sha512.hash

def TAGGED : ByteArray := "verity/tagged-sha512/v1".toUTF8.push 0

/-- `identity.tagged_sha512`: SHA-512 of `TAGGED ‖ be32 |tag| ‖ tag ‖ be64 |payload| ‖ payload`. -/
def taggedSha512 (tag : String) (payload : ByteArray) : ByteArray :=
  let t := tag.toUTF8
  Flock.Sha512.hash (TAGGED ++ be32 t.size.toUInt32 ++ t ++ be64 payload.size.toUInt64 ++ payload)

def scheme : Scheme := ⟨Flock.Sha512.hash, 64, taggedSha512⟩

/-- `RangeIndexedDomain(start, stop).identity_sha512()`, step 1. -/
def rangeIdentity (start stop : Nat) : ByteArray := scheme.rangeIdentity start stop

/-- A frame-v3-sha512 domain; its leaf, node, pad, empty root, levels, root, climb and opening check are `FV3.Domain`'s. -/
abbrev Domain := FV3.Domain scheme

/-- `CommitmentDomain(binding, owner, RangeIndexedDomain(start, start + count), hash="sha512")`. -/
def Domain.make (binding : ByteArray) (owner : Int) (start count : Nat) : Domain :=
  FV3.Domain.make scheme binding owner start count

def path (ls : Array (Array ByteArray)) (rank : Nat) : Array ByteArray := FV3.path ls rank

/-- The node over `l` and `r` at index `idx` of level `d`, in the tree of the domain whose id is `dom`. -/
def node (dom : ByteArray) (d idx : Nat) (l r : ByteArray) : ByteArray :=
  FV3.hash "node" [dom, FV3.uint d, FV3.uint idx, l, r] Flock.Sha512.hash

/-- The root a digest `cur` at index `idx` of level `d` climbs to through the siblings `sibs`, lowest first. -/
def climbL (dom : ByteArray) : ByteArray → Nat → Nat → List ByteArray → ByteArray
  | cur, _, _, [] => cur
  | cur, idx, d, sib :: rest =>
    climbL dom (if idx % 2 = 0 then node dom d (idx / 2) cur sib else node dom d (idx / 2) sib cur) (idx / 2) (d + 1) rest

theorem climbL_eq (dom cur : ByteArray) (idx d : Nat) (sibs : List ByteArray) :
    climbL dom cur idx d sibs = scheme.climbL dom cur idx d sibs := by
  induction sibs generalizing cur idx d with
  | nil => rfl
  | cons sib rest ih => simp only [climbL, Scheme.climbL]; exact ih ..

/-- A path's `depth` siblings, 64 bytes each, leaf level first. -/
def sibs (path : ByteArray) (depth : Nat) : List ByteArray :=
  (List.range depth).map fun d => path.extract (64 * d) (64 * d + 64)

/-- The root a leaf at `i` climbs to through `path` (the siblings, leaf level first, 64 bytes each). -/
def climb (dom leaf : ByteArray) (i depth : Nat) (path : ByteArray) : ByteArray := climbL dom leaf i 0 (sibs path depth)

/-- The leaf at `pos` (its rank and its position) of a row whose hm96 leaf of schema `schema` is `payload`. -/
def leafAs (schema : String) (dom : ByteArray) (pos : Nat) (payload : ByteArray) : ByteArray :=
  FV3.leaf dom pos schema payload Flock.Sha512.hash

end Sha512

/-! ## frame-v3-sha256

PoUW's SHA-256 frames (`CommitmentDomain(…, hash="sha256")`, Pearl-C's `h0`), the one other hash a commitment takes. -/

namespace Sha256

/-- The frame under SHA-256. -/
def hash (tag : String) (parts : List ByteArray) : ByteArray := FV3.hash tag parts Flock.Sha256.hash

def TAGGED : ByteArray := "veritor/tagged-sha256/v1".toUTF8.push 0

/-- `identity.tagged_sha256`: SHA-256 of `TAGGED ‖ be32 |tag| ‖ tag ‖ be64 |payload| ‖ payload`. -/
def taggedSha256 (tag : String) (payload : ByteArray) : ByteArray :=
  let t := tag.toUTF8
  Flock.Sha256.hash (TAGGED ++ be32 t.size.toUInt32 ++ t ++ be64 payload.size.toUInt64 ++ payload)

def scheme : Scheme := ⟨Flock.Sha256.hash, 32, taggedSha256⟩

/-- `RangeIndexedDomain(start, stop).identity()`, step 1. -/
def rangeIdentity (start stop : Nat) : ByteArray := scheme.rangeIdentity start stop

/-- A frame-v3-sha256 domain; its leaf, node, pad, empty root, levels, root, climb and opening check are `FV3.Domain`'s. -/
abbrev Domain := FV3.Domain scheme

/-- `CommitmentDomain(binding, owner, RangeIndexedDomain(start, start + count), hash="sha256")`. -/
def Domain.make (binding : ByteArray) (owner : Int) (start count : Nat) : Domain :=
  FV3.Domain.make scheme binding owner start count

def path (ls : Array (Array ByteArray)) (rank : Nat) : Array ByteArray := FV3.path ls rank

end Sha256

end FV3
end Flock
