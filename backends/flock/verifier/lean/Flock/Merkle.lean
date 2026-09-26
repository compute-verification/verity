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

/-- §14: `rows` open to `cap` (depth `c`) at `positions` of a codeword of `2^d` rows, each of `lanes` elements, with
`d − c` siblings per query, leaf side first at the bottom. -/
def merkleCheck (ms : MerkleScheme) (cap : Array ByteArray) (c d : Nat) (positions : Array Nat) (rows : Array (Array F128))
    (siblings : Array ByteArray) (lanes : Nat) : V Unit := do
  if cap.size != 2 ^ c || rows.size != positions.size || siblings.size != positions.size * (d - c) then
    throw "merkle: shape"
  if cap.any (·.size != ms.digestLen) || siblings.any (·.size != ms.digestLen) then throw "merkle: digest width"
  for j in [0:positions.size] do
    let row := rows[j]!
    if row.size != lanes then throw "merkle: row length"
    let mut acc := ms.leaf row
    let mut idx := positions[j]!
    for t in [0:d - c] do
      let sib := siblings[j * (d - c) + t]!
      acc := if idx % 2 == 0 then ms.node acc sib else ms.node sib acc
      idx := idx / 2
    if acc != cap[positions[j]! >>> (d - c)]! then throw s!"merkle: query {j} does not open to the cap"

end Flock
