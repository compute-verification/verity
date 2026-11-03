import Mathlib
import Definitions.Hash.Basic

/-!
# frame-v3 trees over a hash (trusted)

The served tree, `verity.ml.commitments.merkle` under `frame-v3-sha512` (`verity/ml/commitments/frame_v3/PROTOCOL.md`
§1 and §6), with the hash a parameter `H` (SHA-512 is the instance of record). A frame hashes
`FRAME ‖ u32be |tag| ‖ tag ‖ Σ (u64be |part| ‖ part)` (`frame`). A domain is the verifier's: its id (a digest the verifier
derives from the binding, owner, positions and count, never from an opening), its count, and its positions. The frame
leaf is `H(frame "leaf" [id, uint rank, uint position, schema, value])` and a node at level `lv` and index `i` is
`H(frame "node" [id, uint lv, uint i, l, r])`; the verifier's code is `PouwCommit.Domain.verify`.

An opening at `rank` is the leaf digest and the path's siblings from the leaf level up. The verifier accepts when
`rank < count`, the path has `merkle_depth count` siblings, and the climb reaches the committed root (`Tree.accepts`).
Leaf, node and root are 64-byte digests, which the verifier's size checks (`root.size`, each sibling's) refuse otherwise;
`Digest` carries that in its type. `Tree.hashed` lists the node messages the climb hashed, bottom first.

A whole tree, the committer's, is `Tree.root` with every message it hashes `Tree.msgs`: leaves, `pad` frames up to a
power of two and nodes, or the `empty` frame at count 0.

A range-indexed domain (`Domain.range`) is `CommitmentDomain(binding, owner, RangeIndexedDomain(0, count), "sha512")`:
its id is `H(frame "domain" [binding, uint (owner + 2), positions, uint count])`, `positions` the range's SHA-512
identity `tagged_sha512("veritor/indexed-domain/range/v1", {"start":0,"step":1,"stop":count})`, and rank `r`'s position
is `r`.
-/

namespace Hash.Tree

/-- `veritor/protocol/merkle/frame/v3\0`. -/
def FRAME : Msg := cstr "veritor/protocol/merkle/frame/v3"

/-- The message a frame hashes: `FRAME ‖ u32be |tag| ‖ tag ‖ Σ (u64be |part| ‖ part)`. -/
def frame (tag : String) (parts : List Msg) : Msg :=
  FRAME ++ be 4 (utf8 tag).length ++ utf8 tag ++ parts.flatMap fun p => be 8 p.length ++ p

/-- A frame-v3 domain as the verifier holds it: its id, its count, and the position of each rank. -/
structure Domain where
  id : Digest
  count : ℕ
  position : ℕ → ℕ

/-- The frame leaf's message over `value` at `rank`, under `schema`. -/
def leafMsg (D : Domain) (rank : ℕ) (schema : String) (value : Msg) : Msg :=
  frame "leaf" [D.id.bytes, uint rank, uint (D.position rank), utf8 schema, value]

/-- The node's message at level `lv` and index `i` over the children `l` and `r`. -/
def nodeMsg (D : Domain) (lv i : ℕ) (l r : Digest) : Msg :=
  frame "node" [D.id.bytes, uint lv, uint i, l.bytes, r.bytes]

/-- `merkle_depth`: `⌈log₂ count⌉`, and 0 for a count of at most 1. -/
def depth (count : ℕ) : ℕ := if count ≤ 1 then 0 else Nat.log2 (count - 1) + 1

/-- From digest `x` at index `cur` of level `lv`, up the siblings: the node messages hashed on the way, and the digest
reached. -/
def climb (H : Msg → Digest) (D : Domain) : ℕ → ℕ → Digest → List Digest → List Msg × Digest
  | _, _, x, [] => ([], x)
  | lv, cur, x, s :: ss =>
    let m := if cur % 2 = 0 then nodeMsg D lv (cur / 2) x s else nodeMsg D lv (cur / 2) s x
    let up := climb H D (lv + 1) (cur / 2) (H m) ss
    (m :: up.1, up.2)

/-- The verifier accepts the leaf digest `leaf` at `rank` with path `sibs` against `root`. -/
def accepts (H : Msg → Digest) (D : Domain) (root : Digest) (rank : ℕ) (leaf : Digest) (sibs : List Digest) : Prop :=
  rank < D.count ∧ sibs.length = depth D.count ∧ (climb H D 0 rank leaf sibs).2 = root

/-- The node messages the verifier hashed checking that opening, bottom first. -/
def hashed (H : Msg → Digest) (D : Domain) (rank : ℕ) (leaf : Digest) (sibs : List Digest) : List Msg :=
  (climb H D 0 rank leaf sibs).1

/-! ## A whole tree

The committer's tree over a domain's `count` values (`tree_levels` and `MerkleTree.commitment`): level 0 is the leaves
in rank order, then pads `H(frame "pad" [id, uint rank])` up to `2 ^ depth count`; each level above hashes its pairs
left to right as nodes; the root is the top level's one digest, and a domain of count 0 has the empty root
`H(frame "empty" [id])`. Here the value at each rank is `value rank`, under one schema. -/

/-- The pad's message at `rank`. -/
def padMsg (D : Domain) (rank : ℕ) : Msg := frame "pad" [D.id.bytes, uint rank]

/-- The empty root's message. -/
def emptyMsg (D : Domain) : Msg := frame "empty" [D.id.bytes]

/-- The leaves' messages, in rank order: `value r` under `schema` at each rank below the count. -/
def leaves (D : Domain) (schema : String) (value : ℕ → Msg) : List Msg :=
  (List.range D.count).map fun r => leafMsg D r schema (value r)

/-- Level 0's digests: the leaves', then the pads' up to `2 ^ depth count`. -/
def base (H : Msg → Digest) (D : Domain) (schema : String) (value : ℕ → Msg) : List Digest :=
  (List.range (2 ^ depth D.count)).map fun r =>
    H (if r < D.count then leafMsg D r schema (value r) else padMsg D r)

/-- One level's node messages at level `lv`, from index `i`: its digests' pairs, left to right. -/
def pairs (D : Domain) (lv : ℕ) : ℕ → List Digest → List Msg
  | i, l :: r :: xs => nodeMsg D lv i l r :: pairs D lv (i + 1) xs
  | _, _ => []

/-- From level `lv`'s digests, `d` levels up: the node messages hashed, bottom first, and the top level's digests. -/
def build (H : Msg → Digest) (D : Domain) : ℕ → ℕ → List Digest → List Msg × List Digest
  | _, 0, xs => ([], xs)
  | lv, d + 1, xs =>
    let up := build H D (lv + 1) d ((pairs D lv 0 xs).map H)
    (pairs D lv 0 xs ++ up.1, up.2)

/-- **The root** over `value`: the empty root at count 0, else the top level's digest `depth count` levels above
`base` (one digest, so `headD`'s default is never read). -/
def root (H : Msg → Digest) (D : Domain) (schema : String) (value : ℕ → Msg) : Digest :=
  if D.count = 0 then H (emptyMsg D)
  else (build H D 0 (depth D.count) (base H D schema value)).2.headD (H (emptyMsg D))

/-- **Every message the tree hashes**: the empty root's at count 0, else the leaves', the pads' and the nodes',
bottom first. -/
def msgs (H : Msg → Digest) (D : Domain) (schema : String) (value : ℕ → Msg) : List Msg :=
  if D.count = 0 then [emptyMsg D]
  else leaves D schema value ++ (List.range' D.count (2 ^ depth D.count - D.count)).map (padMsg D) ++
    (build H D 0 (depth D.count) (base H D schema value)).1

/-! ## A range-indexed domain -/

/-- `identity.tagged_sha512`'s message: `verity/tagged-sha512/v1\0 ‖ u32be |tag| ‖ tag ‖ u64be |payload| ‖ payload`. -/
def taggedMsg (tag : String) (payload : Msg) : Msg :=
  cstr "verity/tagged-sha512/v1" ++ be 4 (utf8 tag).length ++ utf8 tag ++ be 8 payload.length ++ payload

/-- `RangeIndexedDomain(0, count)`'s canonical manifest, `{"start":0,"step":1,"stop":count}`. -/
def rangeManifest (count : ℕ) : Msg := utf8 ("{\"start\":0,\"step\":1,\"stop\":" ++ toString count ++ "}")

/-- **A range-indexed domain** of `count` ranks under `binding` and `owner`, at the hash `H`: its id the verifier
derives from them, and rank `r` at position `r`. -/
def Domain.range (H : Msg → Digest) (binding : Digest) (owner : ℤ) (count : ℕ) : Domain where
  id := H (frame "domain" [binding.bytes, uint (owner + 2).toNat,
    (H (taggedMsg "veritor/indexed-domain/range/v1" (rangeManifest count))).bytes, uint count])
  count := count
  position r := r

end Hash.Tree
