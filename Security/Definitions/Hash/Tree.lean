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

end Hash.Tree
