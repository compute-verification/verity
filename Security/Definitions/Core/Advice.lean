import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.Size
import Definitions.Core.Digest

/-!
# The advice root, with every string it hashes

The advice the developer reports, the registered compiler's input (`Verity.Compiler`), is committed under a hidden root
(`verity.ml.commitments.advice`, which partition v2, `verity.ml.circuits.partition_v2`, uses): the advice's rows, each
under a salt of the prover's choosing in an `hm96-sha512/row/v2` leaf, in one `frame-v3-sha512` tree. The commitment
commits the advice, never a partition chosen beside it: the partition is the compiler's output on the advice (partition
v2's rows are its Call and cut tables, from which the compiler reads each gate's proof block). This module models that root with each hash it takes, so that
a statement can name the strings a broken binding collides: `hashInputs`, every string hashed from the rows up to the
root.

`H` is the hash on byte strings, with core's 64-byte digests (`Verity.Digest`): SHA-512 in the deployment (the one-hash
rule). Nothing here assumes anything of `H`. A statement over this model holds for every `H`, and where it needs two
hashed strings to differ in digest, it concludes instead that two distinct strings of the commitments' `hashInputs`
share one.

Row `i` (bytes `row`) under salt `y` hashes, as `advice.py`'s `leaf` does:
* `x = H(rowPrefix ‖ row)`, the row's `sha512/row/v2` digest (`rowInput`);
* `c = H(saltPrefix ‖ y)`, hm96's salt digest (`saltInput`);
* `t = H(leafPrefix ‖ b ‖ c)` with `b = x ⊕ M·y`, hm96's tree leaf over the commit string `b ‖ c` (`hmInput`);
* `H(leaf i t)`, the frame-v3 leaf at rank `i` (`leafInput`).

Past the rows the leaves are pads `H(pad i)` up to a power of two, the node at level `k` and index `j` is
`H(node k j left right)`, and the root is the one node `merkle_depth(count)` levels up: frame-v3's `tree_levels`
(`nodeAt`, `root`).

Folded into the framing (`Framing`) as constants of the public statement, not modelled as hashes: the domain id (the
SHA-512 of the domain: its binding, owner `-1` and the public row count `R_b`) and the schema string, which the frame
messages carry; the row width, in `rowPrefix`; and hm96's key, through its digest in `leafPrefix` and its matrix in
`mask`. `advice.py` refuses a commitment of no rows, and `empty_root` is not modelled: here the tree of no rows is one
pad. Which rows the advice has is its encoding, partition v2's `rows`, padded with zero rows to `R_b` (`Rows`).
-/

namespace Verity.Advice

/-- The framing of the advice's hashes: every byte of their inputs besides the rows, the salts and the digests they
chain. Each is a constant of the public statement (its domain, the hm96 key, the row width). -/
structure Framing where
  /-- `sha512/row/v2`'s first block, its tag and the row width in bits: a row's digest is `H(rowPrefix ‖ row)`. -/
  rowPrefix : List UInt8
  /-- hm96's salt prefix, one block: a salt's digest is `H(saltPrefix ‖ y)`. -/
  saltPrefix : List UInt8
  /-- hm96's leaf prefix, its tag and the key's digest in one block: the tree leaf is `H(leafPrefix ‖ b ‖ c)`. -/
  leafPrefix : List UInt8
  /-- hm96's mask `M·y` of a salt `y`, under the key. -/
  mask : List UInt8 → Digest
  /-- frame-v3's leaf message at rank `i` over the value `t`: the leaf tag, the domain id, `i` as rank and position, the
  schema `hm96-sha512/row/v2`, and `t`. -/
  leaf : ℕ → Digest → List UInt8
  /-- frame-v3's pad message at rank `i`: the pad tag, the domain id and `i`. -/
  pad : ℕ → List UInt8
  /-- frame-v3's node message at level `k` and index `j` over two children: the node tag, the domain id, `k`, `j`, and
  the children. -/
  node : ℕ → ℕ → Digest → Digest → List UInt8

/-- **The framing's format facts**: a leaf message determines its value, a node message its level and children, and a
leaf, a pad and a node message never coincide. frame-v3 writes each message as a tag (`leaf`, `pad`, `node`) followed by
length-prefixed parts (`merkle._message`), so each fact holds of its messages; none is about `H`. -/
structure Framing.Separated (F : Framing) : Prop where
  leaf_inj : ∀ i, Function.Injective (F.leaf i)
  node_inj : ∀ {k j a b k' j' a' b'}, F.node k j a b = F.node k' j' a' b' → k = k' ∧ a = a' ∧ b = b'
  leaf_ne_pad : ∀ i t i', F.leaf i t ≠ F.pad i'
  leaf_ne_node : ∀ i t k j a b, F.leaf i t ≠ F.node k j a b
  pad_ne_node : ∀ i k j a b, F.pad i ≠ F.node k j a b

variable (H : List UInt8 → Digest) (F : Framing)

/-- The input of a row's `sha512/row/v2` digest `x`. -/
def rowInput (row : List UInt8) : List UInt8 := F.rowPrefix ++ row

/-- The input of hm96's salt digest `c`. -/
def saltInput (y : List UInt8) : List UInt8 := F.saltPrefix ++ y

/-- The input of hm96's tree leaf `t`: the leaf prefix, then the commit string `b ‖ c`, `b = x ⊕ M·y`. -/
def hmInput (row y : List UInt8) : List UInt8 :=
  F.leafPrefix ++
    (Digest.bytes (fun i => H (rowInput F row) i ^^^ F.mask y i) ++ (H (saltInput F y)).bytes)

/-- The input of the frame-v3 leaf of row `row` at rank `i` under salt `y`. -/
def leafInput (i : ℕ) (row y : List UInt8) : List UInt8 := F.leaf i (H (hmInput H F row y))

variable (rows : List (List UInt8)) (salt : ℕ → List UInt8)

/-- The leaf at position `i` of the tree over `rows`, row `i` under salt `salt i`: its frame-v3 leaf, or past the rows a
pad. -/
def leafAt (i : ℕ) : Digest :=
  if h : i < rows.length then H (leafInput H F i rows[i] (salt i)) else H (F.pad i)

/-- The node at height `k` and index `j`: the leaf `j` at height `0`, and at height `k + 1` the frame-v3 node at level `k`
over the nodes `2j` and `2j + 1` of height `k`. -/
def nodeAt : ℕ → ℕ → Digest
  | 0, j => leafAt H F rows salt j
  | k + 1, j => H (F.node k j (nodeAt k (2 * j)) (nodeAt k (2 * j + 1)))

/-- frame-v3's `merkle_depth`: the levels above `count` leaves, padded to a power of two. -/
def depth (count : ℕ) : ℕ := (count - 1).size

/-- The root of the advice tree over `rows`, row `i` under salt `salt i`. -/
def root : Digest := nodeAt H F rows salt (depth rows.length) 0

/-- The strings hashed below the node at height `k` and index `j`: at a row's leaf its four inputs, at a pad the pad's,
and at a node its message and everything below it. -/
def inputsAt : ℕ → ℕ → Finset (List UInt8)
  | 0, j =>
    if h : j < rows.length then
      {rowInput F rows[j], saltInput F (salt j), hmInput H F rows[j] (salt j), leafInput H F j rows[j] (salt j)}
    else {F.pad j}
  | k + 1, j =>
    insert (F.node k j (nodeAt H F rows salt k (2 * j)) (nodeAt H F rows salt k (2 * j + 1)))
      (inputsAt k (2 * j) ∪ inputsAt k (2 * j + 1))

/-- **What the advice root hashes**: every string hashed from the rows up to the root. -/
def hashInputs : Finset (List UInt8) := inputsAt H F rows salt (depth rows.length) 0

/-- The advice as rows, the advice list `a_1, …, a_n` encoded as partition v2's `rows` padded to the public row count, and
the decoder that reads the advice back: partition v2's `decode`. -/
structure Rows where
  rows : List (List Bool) → List (List UInt8)
  decode : List (List UInt8) → Option (List (List Bool))

/-- **The rows' format fact**: they decode to their advice, so two advice lists never have the same rows. -/
def Rows.Decodes (A : Rows) : Prop :=
  ∀ a, A.decode (A.rows a) = some a

/-- A commitment, the prover's side of an advice root: the advice list `a`, and the salt of each of its rows (the
prover's choice). Only its root goes public. -/
structure Committed where
  a : List (List Bool)
  salt : ℕ → List UInt8

namespace Committed

variable (A : Rows) (r : Committed)

/-- The commitment's root: the advice tree over its advice's rows under its salts. -/
def root : Digest := Advice.root H F (A.rows r.a) r.salt

/-- Every string the commitment hashes from its rows up to its root. -/
def hashInputs : Finset (List UInt8) := Advice.hashInputs H F (A.rows r.a) r.salt

end Committed

end Verity.Advice
