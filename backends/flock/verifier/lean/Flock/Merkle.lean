/-
PROTOCOL.md §14: the proof's own Merkle trees, as a parameter. A scheme fixes the digest size (the width of every cap entry
and sibling in the proof), the node hash, the leaf of an opened row, and the `merkle_hash` index the proof's `PcsParams`
must carry. The recorded statements use unsalted SHA-256; the target statement uses SHA-512 on every commitment path.
-/
import Flock.Field
import Flock.Hash
import Flock.Bytes
import Flock.Hm96

namespace Flock

structure MerkleScheme where
  /-- the leaf-scheme id the circuit file declares (META `leaf_scheme.id`) -/
  id : String
  digestLen : Nat
  /-- `PcsParams.merkle_hash`, the enum index in the proof -/
  pcsIndex : UInt32
  node : ByteArray → ByteArray → ByteArray
  /-- an opened row's leaf, given its salt (empty for an unsalted scheme) -/
  leaf : Array F128 → ByteArray → ByteArray
  /-- the salt each opening carries (`opened_salts`), 0 for an unsalted scheme -/
  saltBytes : Nat := 0
  deriving Inhabited

namespace MerkleScheme

def rowBytes (row : Array F128) : ByteArray := row.foldl F128.pushBytes ByteArray.empty

/-- Unsalted SHA-256: leaf `SHA-256(row)`, node `SHA-256(l ‖ r)`, `HashKind::Sha256 = 0`. -/
def sha256 (id : String) : MerkleScheme :=
  { id, digestLen := 32, pcsIndex := 0, node := fun l r => Sha256.hash (l ++ r), leaf := fun row _ => Sha256.hash (rowBytes row) }

/-- Unsalted SHA-512 with 64-byte digests (PR #83 at `631567f7`, `HashKind::Sha512 = 2`). -/
def sha512 (id : String) : MerkleScheme :=
  { id, digestLen := 64, pcsIndex := 2, node := fun l r => Sha512.hash (l ++ r), leaf := fun row _ => Sha512.hash (rowBytes row) }

/-- `hm96-sha512/v1` (PR #83 from `2908d078`, `HashKind::Hm96Sha512 = 3`): the leaf hides its row, `SHA-512(leaf_prefix ‖ b ‖ c)`
for `x = SHA-512(row)` and the opening's 192-byte salt; nodes `SHA-512(l ‖ r)`. -/
def hm96Sha512 : MerkleScheme :=
  { id := "hm96-sha512/v1", digestLen := 64, pcsIndex := 3, saltBytes := 192, node := fun l r => Sha512.hash (l ++ r),
    leaf := fun row salt => Hm96.Default512.leaf (Sha512.hash (rowBytes row)) salt }

def all : List MerkleScheme :=
  [sha256 "flock-leaf/sha256-unsalted/v0", sha256 "flock-leaf/sha256-unsalted", sha512 "flock-leaf/sha512-unsalted", hm96Sha512]

def find? (id : String) : Option MerkleScheme := all.find? (·.id == id)

end MerkleScheme

/-- The digest a query climbs to from `acc` at index `idx`, one sibling per level, leaf side first: the node of
`(acc, sibling)` when the index is even, of `(sibling, acc)` when it is odd. -/
def climb (ms : MerkleScheme) : ByteArray → Nat → List ByteArray → ByteArray
  | acc, _, [] => acc
  | acc, idx, s :: rest => climb ms (if idx % 2 = 0 then ms.node acc s else ms.node s acc) (idx / 2) rest

/-- Query at `pos` opens: its row has `lanes` elements, it has `d − c` siblings, and its leaf (under `salt`) climbs to its
cap entry. -/
def opens (ms : MerkleScheme) (cap : Array ByteArray) (c d lanes pos : Nat) (row : Array F128) (salt : ByteArray)
    (sibs : List ByteArray) : Bool :=
  decide (row.size = lanes) && decide (sibs.length = d - c) && decide (salt.size = ms.saltBytes) &&
    decide (climb ms (ms.leaf row salt) pos sibs = cap.getD (pos >>> (d - c)) ByteArray.empty)

/-- The `w` siblings of query `j`. -/
def sibsOf (siblings : Array ByteArray) (w j : Nat) : List ByteArray := (siblings.extract (j * w) (j * w + w)).toList

/-- The salts `n` openings carry. -/
def saltCount (ms : MerkleScheme) (n : Nat) : Nat := if ms.saltBytes = 0 then 0 else n

/-- §14: `rows` open to `cap` (depth `c`) at `positions` of a codeword of `2^d` rows, each of `lanes` elements, with
`d − c` siblings per query and, for a salted scheme, one salt per query. -/
def merkleCheck (ms : MerkleScheme) (cap : Array ByteArray) (c d : Nat) (positions : Array Nat) (rows : Array (Array F128))
    (siblings : Array ByteArray) (salts : Array ByteArray) (lanes : Nat) : V Unit :=
  if cap.size != 2 ^ c || rows.size != positions.size || siblings.size != positions.size * (d - c)
      || salts.size != saltCount ms positions.size then
    throw "merkle: shape"
  else if cap.any (·.size != ms.digestLen) || siblings.any (·.size != ms.digestLen) then
    throw "merkle: digest width"
  else
    match (List.range positions.size).find? (fun j =>
        !opens ms cap c d lanes positions[j]! rows[j]! (salts.getD j ByteArray.empty) (sibsOf siblings (d - c) j)) with
    | some j => throw s!"merkle: query {j} does not open to the cap"
    | none => pure ()

end Flock
