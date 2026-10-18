import Definitions.Hash

/-!
# The hash area's guarantee statements (statement first, not yet proved)

Statement first (Daniel, 17 Oct): each statement is frozen before its proof, red-teamed, carded, and only then proved, by
a theorem `Hash.SecurityProofs.G : Hash.Guarantees.G` in `Security/Proofs/Hash/`. Until then each is under
`shape.frozen` as `unstated` in `Security/lean-audit.json`, and `Stated` reads them so that the lock records their text.

Both are deterministic and assume nothing about the hash (Daniel, 8 Oct): each conclusion is the property, or a
collision among the messages the two openings' checks themselves hashed, never an advantage bound.

* `RowBinding`: two openings of one served root at one rank, in the served format (frame-v3-sha512 tree, frame leaf,
  hm96-sha512 leaf under the verifier's key, `sha512/row-seg/v1` row digest), carry the same row, or the checks hashed a
  collision.
* `TreeBinding`: two openings of one frame-v3 root at one rank carry the same leaf digest, or the path checks hashed a
  collision of node messages.
-/

namespace Hash.Guarantees

/-- **Row binding, as a break tied to the run.** Two openings of one root at one rank, each accepted by the verifier of the
served row leaf (`RowLeaf.accepts`: a frame-v3 tree over the verifier's domain, its frame leaf binding the domain id,
rank, position and the schema `hm96-sha512/row-seg/v1`, its nodes their level and index; the hm96-sha512 leaf under the
verifier's key, bound through `H(key)` in its prefix; the `sha512/row-seg/v1` digest binding the row's bit length and
each segment's; every part length-framed or fixed-width), carry the same row, or two different messages that `H` hashed
in the two checks have one digest, or two different messages that `G` hashed in the two row digests have one digest.

The instance of record is `H = G = SHA-512`, where the last two disjuncts together are a SHA-512 collision among the
inputs the two openings hashed. The statement holds for any 64-byte `H` and `G`. -/
def RowBinding : Prop :=
  ∀ (H G : Msg → Digest) (V : RowLeaf.Verifier) (root : Digest) (rank : ℕ) (o₁ o₂ : RowLeaf.Opening),
    RowLeaf.accepts H G V root rank o₁ → RowLeaf.accepts H G V root rank o₂ →
    o₁.row = o₂.row ∨
      CollisionIn H (RowLeaf.outerHashed H G V rank o₁ ++ RowLeaf.outerHashed H G V rank o₂) ∨
      CollisionIn G (RowLeaf.innerHashed G o₁ ++ RowLeaf.innerHashed G o₂)

/-- **Frame-v3 tree binding, as a break tied to the run.** Two leaf digests accepted at one rank against one root, with a
rank below the domain's count and paths of `merkle_depth count` siblings (`Tree.accepts`), are equal, or two different
node messages (each framing the domain id, its level and index, and its children) that the two path checks hashed have
one digest under `H`. -/
def TreeBinding : Prop :=
  ∀ (H : Msg → Digest) (D : Tree.Domain) (root : Digest) (rank : ℕ) (l₁ l₂ : Digest) (sibs₁ sibs₂ : List Digest),
    Tree.accepts H D root rank l₁ sibs₁ → Tree.accepts H D root rank l₂ sibs₂ →
    l₁ = l₂ ∨ CollisionIn H (Tree.hashed H D rank l₁ sibs₁ ++ Tree.hashed H D rank l₂ sibs₂)

/-- **The hash area's statements are stated**: the tuple of `RowBinding` and `TreeBinding`, equal to itself. It says
nothing about the verifier: it is there so that the lock records each statement's text, and every definition it reads,
from the red team on. A statement leaves it when its own proof is listed. -/
def Stated : Prop :=
  let statements := (RowBinding, TreeBinding)
  statements = statements

end Hash.Guarantees
