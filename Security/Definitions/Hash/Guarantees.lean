import Definitions.Hash

/-!
# The hash area's guarantee statements

Each is proved by a theorem `Hash.SecurityProofs.G : Hash.Guarantees.G` in `Security/Proofs/Hash/`. All are
deterministic and assume nothing about the hash: each conclusion is the property, or a collision among
the messages the two openings' checks themselves hashed, never an advantage bound. So no proof takes an assumption.

* `RowBinding`: two openings of one root at one rank, in the hm96 bit-row leaf format the verifier holds (frame-v3-sha512
  tree, frame leaf, hm96-sha512 leaf under the verifier's key, and a `sha512/row-seg/v1` or `sha512/row/v2` row digest),
  carry the same row, or the checks hashed a collision. It covers row-seg and row/v2 leaves, not `hm96-sha512/row/v1`.
* `TreeBinding`: two openings of one frame-v3 root at one rank carry the same leaf digest, or the path checks hashed a
  collision of node messages.
* `EmbeddingReadBinding`: when `EmbeddingRead_v1` returns ok for a token in the vocabulary, the row it returns is the
  registered table's row at that token, or the checks hashed a collision. It is `RowBinding` at rank `tok`.

SHA-512 is the served hash, not BLAKE3. A probability read from
any of them is per prover: the collision comes from that prover's (extracted) openings, so it is a bound on that
prover's link finders under `ecr/sha-512` (`FlockSoundness.Assumptions.SHA512CRExpected`), never one `δ_bind` for every
prover, which is false at SHA-512 (a prover that hard-codes one collision opens one leaf two ways). In particular
none of them discharges a constant over all provers such as `Pouw.Assumptions.ServiceSoundStaged`'s `δb`.
-/

namespace Hash.Guarantees

/-- **Row binding, as a break tied to the run.** Two openings of one root at one rank, each accepted by the verifier of the
hm96 bit-row leaf in its format (`RowLeaf.accepts`: a frame-v3 tree over the verifier's domain, its frame leaf binding the
domain id, rank, position and the format's schema, its nodes their level and index; the hm96-sha512 leaf under the
verifier's key, bound through `H(key)` in its prefix; the format's row digest binding the row's bit length and its bits,
each segment's under `sha512/row-seg/v1`; every part length-framed or fixed-width), carry the same row, or two different
messages that `H` hashed in the two checks have one digest, or two different messages that `G` hashed in the two row
digests have one digest.

The instance of record is `H = G = SHA-512`, where the last two disjuncts together are a SHA-512 collision among the
inputs the two openings hashed. It is stated for any 64-byte `H` and `G`.

Scope:
* **Leaves.** The verifier's format (`RowLeaf.Format`): `hm96-sha512/row-seg/v1` leaves (`seg`: PoUW's per-call A rows,
  the vLLM row tables, A6's `x` port, the clear sampled audit's rows) or `hm96-sha512/row/v2` leaves (`v2`: H1's tables
  and registered weights, serving's step rows, the chunked templates' `w`, `in<g>` and `o0` rows, the remote audit's
  receipts). A row cut into several leaves (H1's chunk rows past K = 4096, n leaves a row) is bound leaf by leaf, at each
  leaf's rank; which leaves form a row is the layout's. Leaves under `hm96-sha512/row/v1` (C-Flock's default ports, M0's
  serving rows) are outside it: a row/v1 format adds role, word width and word count to the prefix.
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
      CollisionIn G (RowLeaf.innerHashed G V o₁ ++ RowLeaf.innerHashed G V o₂)

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

/-- **The embedding's row read binds its row, as a break tied to the run.** When `EmbeddingRead_v1` (`EmbeddingRead.ok`)
returns ok on the witness opening `o` for a token `tok` in the vocabulary, the row it returns (`EmbeddingRead.h`) is the
row of any opening `o₀` accepted at rank `tok` against the same root under the read's verifier (the registered table's
row at `tok`), or two different messages that `H` hashed in the two checks have one digest, or two different messages
that `G` hashed in the two row digests have one digest.

Scope:
* **What it is about.** The reference's outputs (`Definitions/Hash/EmbeddingRead.lean`), not the gates; the gates'
  tests (`verity/ml/commitments/tests/test_boolean_embedding.py`) hold them to the reference.
* **What the unit's wiring supplies.** `root`, `domain` and `key` are the registered table's: the root and domain id are
  registered reads of the table's root in the unit that holds the read, the domain's count is `V`, and the key is hm96's
  pinned key, which the table's registration uses. `RowBinding`'s note on domains applies: the domain's binding says it
  is a rows domain.
* **A token outside the vocabulary** returns the fill and ok whatever the witness, and the statement says nothing there. -/
def EmbeddingReadBinding : Prop :=
  ∀ (H G : Msg → Digest) (domain : Tree.Domain) (key : Msg) (root : Digest) (vocab words tok : ℕ)
    (o : RowLeaf.Opening),
    EmbeddingRead.ok H G domain key root vocab tok o →
    ∀ o₀ : RowLeaf.Opening, tok < vocab → RowLeaf.accepts H G (EmbeddingRead.verifier domain key) root tok o₀ →
      EmbeddingRead.h vocab words tok o = o₀.row ∨
        CollisionIn H (RowLeaf.outerHashed H G (EmbeddingRead.verifier domain key) tok o ++
          RowLeaf.outerHashed H G (EmbeddingRead.verifier domain key) tok o₀) ∨
        CollisionIn G (RowLeaf.innerHashed G (EmbeddingRead.verifier domain key) o ++
          RowLeaf.innerHashed G (EmbeddingRead.verifier domain key) o₀)

end Hash.Guarantees
