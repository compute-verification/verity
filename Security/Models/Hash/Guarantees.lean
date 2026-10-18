import Definitions.Hash

/-!
# The hash area's guarantee statements (statement first, not yet proved)

Statement first (Daniel, 17 Oct): each statement is frozen before its proof, red-teamed, carded, and only then proved, by
a theorem `Hash.SecurityProofs.G : Hash.Guarantees.G` in `Security/Proofs/Hash/`. Until then each is under
`shape.frozen` as `unstated` in `Security/lean-audit.json`, and `Stated` reads them so that the lock records their text.

Both are deterministic and assume nothing about the hash (Daniel, 8 Oct): each conclusion is the property, or a
collision among the messages the two openings' checks themselves hashed, never an advantage bound.

* `RowBinding`: two openings of one root at one rank, in the `hm96-sha512/row-seg/v1` leaf format (frame-v3-sha512 tree,
  frame leaf, hm96-sha512 leaf under the verifier's key, `sha512/row-seg/v1` row digest), carry the same row, or the
  checks hashed a collision. It covers row-seg leaves only, not `hm96-sha512/row/v2` or `row/v1`.
* `TreeBinding`: two openings of one frame-v3 root at one rank carry the same leaf digest, or the path checks hashed a
  collision of node messages.

SHA-512 is the served hash (Daniel's ruling on the hash card, 18 Oct: BLAKE3 not adopted). A probability read from
either statement is per prover: the collision comes from that prover's (extracted) openings, so it is a bound on that
prover's link finders under `ecr/sha-512` (`FlockSoundness.Assumptions.SHA512CRExpected`), never one `δ_bind` for every
prover, which is false at SHA-512 (a prover that hard-codes one collision opens one leaf two ways). In particular neither
statement discharges a constant over all provers such as `Pouw.Assumptions.ServiceSoundStaged`'s `δb`.
-/

namespace Hash.Guarantees

/-- **Row binding, as a break tied to the run.** Two openings of one root at one rank, each accepted by the verifier of the
`hm96-sha512/row-seg/v1` row leaf (`RowLeaf.accepts`: a frame-v3 tree over the verifier's domain, its frame leaf binding the domain id,
rank, position and the schema `hm96-sha512/row-seg/v1`, its nodes their level and index; the hm96-sha512 leaf under the
verifier's key, bound through `H(key)` in its prefix; the `sha512/row-seg/v1` digest binding the row's bit length and
each segment's; every part length-framed or fixed-width), carry the same row, or two different messages that `H` hashed
in the two checks have one digest, or two different messages that `G` hashed in the two row digests have one digest.

The instance of record is `H = G = SHA-512`, where the last two disjuncts together are a SHA-512 collision among the
inputs the two openings hashed. It is stated for any 64-byte `H` and `G`.

Scope:
* **Leaves.** Only `hm96-sha512/row-seg/v1` leaves: PoUW's per-call A rows (role X) and the clear sampled audit's rows.
  Leaves under `hm96-sha512/row/v2` (PoUW's registered W rows, the ZK path's B rows, bit-row ports, registered advice) and
  `hm96-sha512/row/v1` (C-Flock's default ports) are outside it. A row/v2 statement is this one with `RowSeg.digest`
  replaced by `G (P(ℓ) ‖ row bytes)`, with `P(ℓ)` the 128-byte padded `"verity/sha512-row/v2\0" ‖ u64be ℓ`, the schema
  `hm96-sha512/row/v2`, and `innerHashed` that one message. A row/v1 statement adds role, word width and word count to
  the prefix.
* **Domain.** `V.domain` is free, and RowBinding holds for every domain. It does not say that a root is a rows tree. The
  schema doesn't separate them: PoUW's tile-digest trees write `hm96-sha512/row-seg/v1` too, with `x` the 64-byte digest
  itself, and there a row's leaf equals the leaf of its own row-seg digest with no collision. Only the domain's binding
  (`weight_binding`/`activation_binding` against `transcript_binding` in `verity/core/protocols/pouw/audit.py`)
  separates them, so a consumer establishes that `V.domain` is a rows domain from its binding.
* **Relation to C-Flock.** `FlockSoundness.Binding.ExplicitLeaf.rowSeg` already proves row-seg binding at the hm96 leaf,
  in explicit-pair form, over `SegRow l`, `List UInt8` digests and one hash. RowBinding's hm96 and row-seg part is the
  same property in this area's types, from a frame root. No lemma relates `RowSeg.digest` to `HmRow.segDigest` or
  `Hm96.leaf` to `Hm96.Default512` yet. A later bridge connects them, and states the break as a search over the listed
  messages (`Claims.findColl`'s shape) for a link finder. -/
def RowBinding : Prop :=
  ∀ (H G : Msg → Digest) (V : RowLeaf.Verifier) (root : Digest) (rank : ℕ) (o₁ o₂ : RowLeaf.Opening),
    RowLeaf.accepts H G V root rank o₁ → RowLeaf.accepts H G V root rank o₂ →
    o₁.row = o₂.row ∨
      CollisionIn H (RowLeaf.outerHashed H G V rank o₁ ++ RowLeaf.outerHashed H G V rank o₂) ∨
      CollisionIn G (RowLeaf.innerHashed G o₁ ++ RowLeaf.innerHashed G o₂)

/-- **Frame-v3 tree binding, as a break tied to the run.** Two leaf digests accepted at one rank against one root, with a
rank below the domain's count and paths of `merkle_depth count` siblings (`Tree.accepts`), are equal, or two different
node messages (each framing the domain id, its level and index, and its children) that the two path checks hashed have
one digest under `H`.

Its reader is the step from a serving root to the registered leaf. C-Flock takes the hm96 leaf per position as given
(`FlockSoundness.Binding.Layout.regLeaf`), and the frame-v3 tree above it is outside its proof. TreeBinding binds a root
that a record pins to that leaf, which is the content main doesn't have. It is not about C-Flock's own column trees
(`leafOfId`, nodes "SHA-512 of the pair"), so it feeds neither `TableCRL` nor `SessCRL`. -/
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
