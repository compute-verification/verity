import Definitions.Hash.Basic

/-!
# A binary Merkle path (trusted)

A tree of depth `d` over a hash `H : Msg → Digest`, its leaf the hash of a leaf input and each node the hash of a node
input built from its two children's digests (frame-v3-sha512's shape, `verity/ml/commitments/frame_v3/PROTOCOL.md` §6,
with the node's constant material in `tag`). An opening at position `i` is the leaf input and the sibling digests from
the leaf level up; bit `j` of `i` says whether the path's node at level `j` is a right child. The verifier recomputes
the root (`Path.root`) and accepts when the path has exactly `d` siblings, `i < 2^d`, and the root is the committed one
(`Path.accepts`). `Path.hashed` lists every input the recomputation hashed: the leaf input, then each node input.

What a leaf input is belongs to the scheme: a keyed row (`Definitions/Hash/Keyed.lean`) or the encoding of an SIS digest
(`Definitions/Hash/Sis.lean`). Children's digests are 64 bytes each, so a node input determines them.
-/

namespace Hash.Path

/-- A node's input: the tag, then the left child's digest, then the right's; `cur` is the right child when `right`. -/
def nodeIn (tag : Msg) (right : Bool) (cur sib : Digest) : Msg :=
  if right then tag ++ List.ofFn sib ++ List.ofFn cur else tag ++ List.ofFn cur ++ List.ofFn sib

/-- From digest `x` at position `i`, up the siblings `sibs`: the node inputs hashed on the way, and the digest reached. -/
def climb (H : Msg → Digest) (tag : Msg) : ℕ → Digest → List Digest → List Msg × Digest
  | _, x, [] => ([], x)
  | i, x, s :: ss =>
    let m := nodeIn tag (i % 2 = 1) x s
    let up := climb H tag (i / 2) (H m) ss
    (m :: up.1, up.2)

/-- The root an opening recomputes: the leaf input `leaf` hashed, then climbed. -/
def root (H : Msg → Digest) (tag : Msg) (i : ℕ) (leaf : Msg) (sibs : List Digest) : Digest :=
  (climb H tag i (H leaf) sibs).2

/-- Every input the recomputation hashed: the leaf input, then the node inputs from the leaf level up. -/
def hashed (H : Msg → Digest) (tag : Msg) (i : ℕ) (leaf : Msg) (sibs : List Digest) : List Msg :=
  leaf :: (climb H tag i (H leaf) sibs).1

/-- The verifier accepts an opening (`leaf`, `sibs`) at position `i` of a depth-`d` tree with root `r`. -/
def accepts (H : Msg → Digest) (tag : Msg) (d : ℕ) (r : Digest) (i : ℕ) (leaf : Msg) (sibs : List Digest) : Prop :=
  sibs.length = d ∧ i < 2 ^ d ∧ root H tag i leaf sibs = r

end Hash.Path
