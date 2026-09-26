/-
PROTOCOL.md §14: the proof's own Merkle trees, as a parameter. A scheme fixes the digest size (the width of every cap entry
and sibling in the proof), the node hash, the leaf of an opened row, and the `merkle_hash` index the proof's `PcsParams`
must carry. The recorded statements use unsalted SHA-256; the target statement uses SHA-512 on every commitment path.
-/
import Flock.Field
import Flock.Hash
import Flock.Bytes

namespace Flock

structure MerkleScheme where
  /-- the leaf-scheme id the circuit file declares (META `leaf_scheme.id`) -/
  id : String
  digestLen : Nat
  /-- `PcsParams.merkle_hash`, the enum index in the proof -/
  pcsIndex : UInt32
  node : ByteArray → ByteArray → ByteArray
  leaf : Array F128 → ByteArray
  deriving Inhabited

namespace MerkleScheme

def rowBytes (row : Array F128) : ByteArray := row.foldl F128.pushBytes ByteArray.empty

/-- Unsalted SHA-256: leaf `SHA-256(row)`, node `SHA-256(l ‖ r)`, `HashKind::Sha256 = 0`. -/
def sha256 (id : String) : MerkleScheme :=
  { id, digestLen := 32, pcsIndex := 0, node := fun l r => Sha256.hash (l ++ r), leaf := fun row => Sha256.hash (rowBytes row) }

/-- Unsalted SHA-512 with 64-byte digests. Provisional: the id and the `HashKind` index are placeholders until PR #83
publishes the target statement's; the hiding (Halevi–Micali) leaf will replace the unsalted one. -/
def sha512 (id : String) : MerkleScheme :=
  { id, digestLen := 64, pcsIndex := 2, node := fun l r => Sha512.hash (l ++ r), leaf := fun row => Sha512.hash (rowBytes row) }

def all : List MerkleScheme :=
  [sha256 "flock-leaf/sha256-unsalted/v0", sha256 "flock-leaf/sha256-unsalted", sha512 "flock-leaf/sha512-unsalted"]

def find? (id : String) : Option MerkleScheme := all.find? (·.id == id)

end MerkleScheme

/-- The digest a query climbs to from `acc` at index `idx`, one sibling per level, leaf side first: the node of
`(acc, sibling)` when the index is even, of `(sibling, acc)` when it is odd. -/
def climb (ms : MerkleScheme) : ByteArray → Nat → List ByteArray → ByteArray
  | acc, _, [] => acc
  | acc, idx, s :: rest => climb ms (if idx % 2 = 0 then ms.node acc s else ms.node s acc) (idx / 2) rest

/-- Query at `pos` opens: its row has `lanes` elements, it has `d − c` siblings, and its leaf climbs to its cap entry. -/
def opens (ms : MerkleScheme) (cap : Array ByteArray) (c d lanes pos : Nat) (row : Array F128) (sibs : List ByteArray) : Bool :=
  decide (row.size = lanes) && decide (sibs.length = d - c) &&
    decide (climb ms (ms.leaf row) pos sibs = cap.getD (pos >>> (d - c)) ByteArray.empty)

/-- The `w` siblings of query `j`. -/
def sibsOf (siblings : Array ByteArray) (w j : Nat) : List ByteArray := (siblings.extract (j * w) (j * w + w)).toList

/-- §14: `rows` open to `cap` (depth `c`) at `positions` of a codeword of `2^d` rows, each of `lanes` elements, with
`d − c` siblings per query. -/
def merkleCheck (ms : MerkleScheme) (cap : Array ByteArray) (c d : Nat) (positions : Array Nat) (rows : Array (Array F128))
    (siblings : Array ByteArray) (lanes : Nat) : V Unit :=
  if cap.size != 2 ^ c || rows.size != positions.size || siblings.size != positions.size * (d - c) then
    throw "merkle: shape"
  else if cap.any (·.size != ms.digestLen) || siblings.any (·.size != ms.digestLen) then
    throw "merkle: digest width"
  else
    match (List.range positions.size).find? (fun j =>
        !opens ms cap c d lanes positions[j]! rows[j]! (sibsOf siblings (d - c) j)) with
    | some j => throw s!"merkle: query {j} does not open to the cap"
    | none => pure ()

end Flock
