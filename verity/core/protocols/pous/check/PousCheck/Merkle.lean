import Flock.Hash
import Flock.HmRow

/-!
# The verification tree

The SHA-512 Merkle tree whose root binds the stored blocks:

~~~text
R(c)        = SHA512(P(8·|c|) || c)                                   sha512/row/v2, P = Flock.HmRow.rowPrefixV2
leaf(i)     = SHA512(0x00 || salt[32] || u64be(s) || u16be(v) || R(c_i))      the spec's (benchmarks/pous/p2_v1/reference.py)
packageLeaf = SHA512(0x00 || salt[24] || u64le(s) || u16le(v) || R(c_i))      the package's (protocols/pous/verifier.py)
node(l, r)  = SHA512(0x01 || l || r)
~~~

for block `i = s · blocks_per_segment + v`. The two leaves differ only in their wrapper:
the spec's binds the 32-byte setup salt and big-endian indices, the generic protocol's `tag_s` (its 24-byte salt and
`s` as one 32-byte little-endian integer) and `v` little-endian.

The tree has the shape of RFC 6962: over `n > 1` leaves it splits at the largest power of two strictly below `n`, in
leaf order, and one leaf's root is the leaf. `path` is RFC 6962's `PATH(i, D)`, the siblings from the leaf up, and
`verifyPath` RFC 9162 §2.1.3.2's inclusion check, which takes left and right from the index and the leaf count.
-/

namespace PousCheck.Merkle

open Flock

/-- Every digest is SHA-512's whole output. -/
def DIGEST : Nat := 64

/-- `x` as `width` bytes, big-endian (`x mod 256^width`). -/
def be (x width : Nat) : ByteArray := ⟨(Array.range width).map fun i => (x >>> (8 * (width - 1 - i))).toUInt8⟩

/-- `x` as `width` bytes, little-endian (`x mod 256^width`). -/
def le (x width : Nat) : ByteArray := ⟨(Array.range width).map fun i => (x >>> (8 * i)).toUInt8⟩

/-- A block's `sha512/row/v2` digest `R`: the block as a row of `8·|block|` bits. -/
def rowDigest (block : ByteArray) : ByteArray := Sha512.hash (HmRow.rowPrefixV2 (8 * block.size) ++ block)

/-- The spec's leaf of block `(s, v)` under the 32-byte setup salt: `0x00 ‖ salt ‖ u64be(s) ‖ u16be(v) ‖ R`, 107
bytes hashed (`reference.leaf`). -/
def leaf (salt : ByteArray) (s v : Nat) (block : ByteArray) : ByteArray :=
  Sha512.hash (ByteArray.mk #[0] ++ salt ++ be s 8 ++ be v 2 ++ rowDigest block)

/-- The generic protocol's leaf (`verifier.leaf_hash`): `0x00 ‖ tag_s ‖ u16le(v) ‖ R`, `tag_s` the 24-byte salt then
`u64le(s)`. -/
def packageLeaf (salt : ByteArray) (s v : Nat) (block : ByteArray) : ByteArray :=
  Sha512.hash (ByteArray.mk #[0] ++ salt ++ le s 8 ++ le v 2 ++ rowDigest block)

/-- An inner node. -/
def node (left right : ByteArray) : ByteArray := Sha512.hash (ByteArray.mk #[1] ++ left ++ right)

/-- The largest power of two strictly below `n`, for `n ≥ 2`. -/
def split (n : Nat) : Nat := 2 ^ Nat.log2 (n - 1)

theorem split_pos (n : Nat) : 0 < split n := Nat.two_pow_pos _

theorem split_lt {n : Nat} (h : 2 ≤ n) : split n < n := by
  have := @Nat.log2_self_le (n - 1) (by omega)
  unfold split; omega

/-- RFC 6962's `MTH` of the `n` leaves from `lo`: one leaf's root is the leaf. -/
def mth (leaves : Array ByteArray) (lo n : Nat) : ByteArray :=
  if h : n ≤ 1 then leaves[lo]?.getD ByteArray.empty
  else
    have := split_lt (n := n) (by omega)
    have := split_pos n
    node (mth leaves lo (split n)) (mth leaves (lo + split n) (n - split n))
termination_by n

/-- The root over every leaf, in block order; `none` for no leaf. -/
def root (leaves : Array ByteArray) : Option ByteArray :=
  if leaves.isEmpty then none else some (mth leaves 0 leaves.size)

/-- RFC 6962's `PATH(i, D[lo : lo + n])`: leaf `i`'s siblings, from the leaf up. -/
def pathFrom (leaves : Array ByteArray) (lo n i : Nat) : Array ByteArray :=
  if h : n ≤ 1 then #[]
  else
    have := split_lt (n := n) (by omega)
    have := split_pos n
    let k := split n
    if i < k then (pathFrom leaves lo k i).push (mth leaves (lo + k) (n - k))
    else (pathFrom leaves (lo + k) (n - k) (i - k)).push (mth leaves lo k)
termination_by n

/-- Leaf `i`'s audit path in the tree over `leaves`. -/
def path (leaves : Array ByteArray) (i : Nat) : Array ByteArray := pathFrom leaves 0 leaves.size i

/-- The exact sibling count `h(i, n)` of an RFC 9162 path for leaf `i` of `n`; `none` for `i ≥ n`. -/
def pathLength (n i : Nat) : Option Nat :=
  if i < n then some (go n n i) else none
where
  go : Nat → Nat → Nat → Nat
    | 0, _, _ => 0
    | fuel + 1, n, i =>
      if n ≤ 1 then 0
      else
        let k := split n
        if i < k then 1 + go fuel k i else 1 + go fuel (n - k) (i - k)

/-- RFC 9162's inner loop: shift `fn` and `sn` right while `fn` is even and nonzero. -/
def strip : Nat → Nat → Nat → Nat × Nat
  | 0, fn, sn => (fn, sn)
  | fuel + 1, fn, sn => if fn % 2 == 0 && fn != 0 then strip fuel (fn / 2) (sn / 2) else (fn, sn)

/-- RFC 9162 §2.1.3.2: `leaf` is leaf `i` of the `n`-leaf tree with `root`, by `path` (siblings from the leaf up).
A sibling of another length than a digest fails, as in `wire.verify_path`. -/
def verifyPath (root : ByteArray) (n i : Nat) (leaf : ByteArray) (path : Array ByteArray) : Bool := Id.run do
  if n ≤ i then return false
  let mut fn := i
  let mut sn := n - 1
  let mut r := leaf
  for p in path do
    if p.size != DIGEST || sn == 0 then return false
    if fn % 2 == 1 || fn == sn then
      r := node p r
      if fn % 2 == 0 then
        (fn, sn) := strip (fn + 1) fn sn
    else
      r := node r p
    fn := fn / 2
    sn := sn / 2
  return sn == 0 && r == root

end PousCheck.Merkle
